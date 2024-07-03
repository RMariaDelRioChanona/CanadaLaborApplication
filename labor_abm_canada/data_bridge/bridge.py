from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

from labor_abm_canada.data_bridge import (
    CANADA_LABOUR_FORCE,
    FILE_MOBILITY_NETWORK,
    FILE_OCCUPATIONS,
    FILE_TECHNOLOGIES,
    generic_loader,
)

# Default column names
CAPACITY_JOB_COLNAME = "CPY (Direct)_linear"
OPERATION_JOB_COLNAME = "Op FTE (Direct)_linear"


def generate_monthly_range(start_year: int, end_year: int) -> pd.DatetimeIndex:
    """
    Generates a monthly range between two years

    Args:
        start_year (int): The start year
        end_year (int): The end year

    Returns:
        pd.DatetimeIndex: A monthly range between the two years
    """
    return pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-01", freq="MS")


class DataBridge:
    """
    A class that builds labour model inputs starting from "raw" data sources.
    """

    def __init__(
        self,
        scenario: pd.DataFrame,
        occupations: pd.DataFrame,
        mobility_network: pd.DataFrame,
        technologies: pd.DataFrame,
        tech_shocks: Optional[pd.DataFrame] = None,
        tech_to_occupations: Optional[pd.DataFrame] = None,
        occupation_shocks: Optional[pd.DataFrame] = None,
    ):

        self._scenario = scenario
        self._occupations = occupations
        self._mobility_network = mobility_network
        self._technologies = technologies

        self._tech_shocks = tech_shocks
        self._tech_to_occupations = tech_to_occupations
        self._occupation_shocks = occupation_shocks

    @property
    def scenario(self):
        return self._scenario.copy()

    @property
    def occupations(self):
        return self._occupations.copy()

    @property
    def mobility_network(self):
        return self._mobility_network.copy()

    @property
    def technologies(self):
        return self._technologies.copy()

    @property
    def tech_shocks(self):
        if self._tech_shocks is None:
            self._tech_shocks = self.shocks_to_technologies()
        return self._tech_shocks.copy()

    @property
    def tech_to_occupations(self):
        if self._tech_to_occupations is None:
            self._tech_to_occupations = self.technologies_to_occupations()
        return self._tech_to_occupations.copy()

    @property
    def occupation_shocks(self):
        if self._occupation_shocks is None:
            self._occupation_shocks = self.shocks_to_occupations()
        return self._occupation_shocks.copy()

    @classmethod
    def from_standard_files(cls, scenario_file: str | Path):
        """
        Creates a DataBridge instance from standard files.
        """
        scenario = generic_loader(scenario_file)
        occupations = generic_loader(FILE_OCCUPATIONS)
        mobility_network = generic_loader(FILE_MOBILITY_NETWORK)
        technologies = generic_loader(FILE_TECHNOLOGIES)

        return cls(scenario, occupations, mobility_network, technologies)

    @classmethod
    def from_files(
        cls,
        scenario_file: str | Path,
        occupations_file: str | Path,
        mobility_network_file: str | Path,
        technologies_file: str | Path,
    ):
        """
        Creates a DataBridge instance from custom files.
        """
        scenario = generic_loader(scenario_file)
        occupations = generic_loader(occupations_file)
        mobility_network = generic_loader(mobility_network_file)
        technologies = generic_loader(technologies_file)

        return cls(scenario, occupations, mobility_network, technologies)

    @staticmethod
    def make_operation_jobs_cumulative(monthly_jobs: pd.DataFrame) -> pd.DataFrame:
        """
        Makes the losses of operation jobs cumulative

        Args:
            monthly_jobs (pd.DataFrame): The DataFrame with the monthly jobs data
        """

        # Step 1: Filter out the 'Op' job types
        op_jobs = monthly_jobs[monthly_jobs["job_type"].isin(["Op_created_jobs", "Op_destroyed_jobs"])]
        # Step 2: Sort by 'time' to ensure correct order for cumulative sum
        op_jobs = op_jobs.sort_values(by=["region", "variable", "model", "scenario", "job_type", "time"])
        # Step 3: Apply cumulative sum on the 'jobs' column within each group
        op_jobs["jobs"] = op_jobs.groupby(["region", "variable", "model", "scenario", "job_type"])["jobs"].cumsum()
        # Step 4: Extract the non-Op job types to concatenate them back together
        non_op_jobs_df = monthly_jobs[~monthly_jobs["job_type"].isin(["Op_created_jobs", "Op_destroyed_jobs"])]
        # Step 5: Concatenate back the modified Op jobs data with the non-Op jobs data
        updated_monthly_jobs_df = pd.concat([non_op_jobs_df, op_jobs], ignore_index=True)
        # Sorting to restore any order if necessary
        updated_monthly_jobs_df = updated_monthly_jobs_df.sort_values(
            by=["region", "variable", "model", "scenario", "job_type", "time"]
        )
        return updated_monthly_jobs_df

    @staticmethod
    def distribute_jobs_monthly(scenario: pd.DataFrame):
        """
        Distributes shocks uniformly month-after-month

        Args:
            scenario (pd.DataFrame): The DataFrame with the "primary" shocks for the labour ABM.
        """

        # List to store the new entries
        data = []

        # Job columns and their descriptive names
        job_columns = [
            ("CPY (Direct)_linear_positive", "CPY_created_jobs"),
            ("CPY (Direct)_linear_negative", "CPY_destroyed_jobs"),
            ("Op FTE (Direct)_linear_positive", "Op_created_jobs"),
            ("Op FTE (Direct)_linear_negative", "Op_destroyed_jobs"),
        ]

        # Iterate over the DataFrame
        for _, row in scenario.iterrows():
            start_year = int(row["time"])
            if start_year == 2021:
                end_year = 2025
            else:
                start_year += 1  # Adjust start year for next period
                end_year = start_year + 4

            # Generate the monthly range for each period
            monthly_range = generate_monthly_range(start_year, end_year)

            # Distribute jobs monthly for each type
            for job_column, job_type in job_columns:
                job_value = row[job_column]
                job_per_month_acc = job_value / len(monthly_range)  # Distribute evenly across all months

                for date in monthly_range:
                    data.append(
                        {
                            "region": row["region"],
                            "variable": row["variable"],
                            "model": row["model"],
                            "scenario": row["scenario"],
                            "time": date.strftime("%Y-%m"),  # Format as year-month
                            "unit": row["unit"],
                            "jobs": job_per_month_acc,
                            "job_type": job_type,
                        }
                    )

        monthly_jobs_df = pd.DataFrame(data)
        # Group by the relevant columns and sum the jobs for all regions
        national_jobs = monthly_jobs_df.groupby(
            ["variable", "model", "scenario", "time", "job_type", "unit"], as_index=False
        ).agg({"jobs": "sum"})

        # Add a 'region' column filled with 'National'
        national_jobs["region"] = "National"

        # Concatenate the national_jobs DataFrame back to the original monthly_jobs_df
        monthly_jobs_df = pd.concat([monthly_jobs_df, national_jobs], ignore_index=True)

        # Optionally, you might want to sort the DataFrame by time or other columns to maintain order
        monthly_jobs_df = monthly_jobs_df.sort_values(
            by=["region", "variable", "model", "scenario", "job_type", "time"]
        )

        # Convert the list to DataFrame
        return monthly_jobs_df

    def shocks_to_technologies(
        self,
        capacity_jobs_colname: str = CAPACITY_JOB_COLNAME,
        operation_jobs_colname: str = OPERATION_JOB_COLNAME,
    ) -> pd.DataFrame:
        """
        Transforms the scenario dataframe into a timeseries of job creation and destruction for capacity and
        operations.

        Args:
            capacity_jobs_colname (str): The column name for capacity jobs.
            operation_jobs_colname (str): The column name for operation jobs.
        """
        scenario = self.scenario

        columns_to_keep = [
            "region",
            "variable",
            "model",
            "scenario",
            "time",
            "unit",
            "value",
        ]
        columns_to_keep += [capacity_jobs_colname, operation_jobs_colname]
        scenario = scenario[columns_to_keep].copy()

        # Split the job columns into positive and negative
        scenario["CPY (Direct)_linear_positive"] = scenario[capacity_jobs_colname].apply(lambda x: x if x >= 0 else 0)
        scenario["CPY (Direct)_linear_negative"] = scenario[capacity_jobs_colname].apply(lambda x: x if x <= 0 else 0)
        scenario["Op FTE (Direct)_linear_positive"] = scenario[operation_jobs_colname].apply(
            lambda x: x if x >= 0 else 0
        )
        scenario["Op FTE (Direct)_linear_negative"] = scenario[operation_jobs_colname].apply(
            lambda x: x if x <= 0 else 0
        )

        # Aggregate rows with the same region, variable, and time
        scenario = scenario.groupby(["region", "variable", "model", "scenario", "time", "unit"], as_index=False).sum()

        # Rescale
        scenario["CPY (Direct)_linear_positive"].sum() / 1e6
        scenario["Op FTE (Direct)_linear_positive"].sum() / 1e6

        # Distribute the shocks uniformly, month-after-month
        monthly_jobs_df = self.distribute_jobs_monthly(scenario)

        monthly_jobs_df["jobs"][
            (monthly_jobs_df["region"] == "National") & (monthly_jobs_df["job_type"] == "CPY_created_jobs")
        ].sum() / 1e6
        monthly_jobs_df["jobs"][
            (monthly_jobs_df["region"] == "National") & (monthly_jobs_df["job_type"] == "Op_created_jobs")
        ].sum() / 1e6

        # Make the operation jobs cumulative
        monthly_jobs_df = self.make_operation_jobs_cumulative(monthly_jobs_df)

        # Replace the job_type values with more descriptive names
        monthly_jobs_df["job_type"] = monthly_jobs_df["job_type"].replace(
            {
                "CPY_created_jobs": "jobs",
                "Op_created_jobs": "jobs",
                "CPY_destroyed_jobs": "jobs",
                "Op_destroyed_jobs": "jobs",
            }
        )
        # Group by all columns except 'jobs' and sum the 'jobs' values
        grouped_jobs_df = (
            monthly_jobs_df.groupby(["region", "variable", "model", "scenario", "time", "unit", "job_type"])
            .sum()
            .reset_index()
        )

        return grouped_jobs_df

    def technologies_to_occupations(
        self,
    ):
        """
        Returns a map between technology shocks and occupation shocks.
        """
        # Load industry-to-occupation data
        df_occupations = self.occupations
        df_occupations["TOT_EMP"] = pd.to_numeric(df_occupations["TOT_EMP"], errors="coerce")
        df_occupations["TOT_EMP"] = df_occupations["TOT_EMP"].fillna(10.0)

        # Load mobility network
        df_network = self.mobility_network

        # Load technology data
        df_tech = self.technologies

        # Map NAICS to technology and get naics codes
        industry_technology_map = dict(zip(df_tech["NAICS6d"], df_tech["variable"]))
        industries = df_tech["NAICS6d"].unique().tolist()

        # We only care about those industries that are shocked
        df_occupations = df_occupations.loc[df_occupations["NAICS"].isin(industries)].copy()

        # We only care about those industries which are in the network (others can be diff classification does)
        occupations_bls_shock = df_occupations["OCC_CODE"].unique().tolist()
        occupations_network = df_network["OCC_target"].unique().tolist()
        # Take the intersection to build the set of occupations to filter the crosswalks
        occupations_to_keep = list(set(occupations_bls_shock).intersection(set(occupations_network)))

        occupations_to_remove = [
            "35-9090",
            "53-4020",
            "53-7110",
            "19-1090",
            "11-9060",
        ]  # Remove some occupations; # TODO: ask Maria about this
        occupations_to_keep = [occ for occ in occupations_to_keep if occ not in occupations_to_remove]

        df_occupations = df_occupations[df_occupations["OCC_CODE"].isin(occupations_to_keep)]  # Filter the occupations
        df_occupations = df_occupations[["NAICS", "OCC_CODE", "TOT_EMP"]]

        # Map NAICS to technology in the occupations dataframe
        df_occupations["variable"] = df_occupations["NAICS"].map(industry_technology_map)
        # Calculate total employment for each 'variable' and create a dictionary
        tech_totals = df_occupations.groupby("variable")["TOT_EMP"].sum().to_dict()
        # Create a new column with the total employment for each 'variable'
        df_occupations["tech_tot_emp"] = df_occupations["variable"].map(tech_totals)
        # Calculate the employment fraction for each row
        df_occupations["EMP_frac"] = df_occupations["TOT_EMP"] / df_occupations["tech_tot_emp"]
        # Keep only the relevant columns
        df_occupations = df_occupations[["variable", "OCC_CODE", "EMP_frac"]]

        return df_occupations

    def shocks_to_occupations(
        self,
    ):
        """
        Maps technology shocks to occupation shocks.
        """

        tech_shocks = self.tech_shocks
        tech_to_occupations = self.tech_to_occupations

        tech_shocks = tech_shocks[
            tech_shocks["region"] != "National"
        ]  # Remove national, better calculated at this level

        # Map technology shocks to occupations
        occupations_shocks = pd.merge(tech_shocks, tech_to_occupations, on="variable", how="left")
        occupations_shocks["jobs_by_occupation"] = occupations_shocks["jobs"] * occupations_shocks["EMP_frac"]

        # Create a new DataFrame for national sums by grouping without the region and summing the jobs.
        national_totals = (
            occupations_shocks.groupby(["time", "OCC_CODE"])
            .agg(total_jobs=("jobs_by_occupation", "sum"))
            .reset_index()
        )
        national_totals["region"] = "National"

        # Group by region, time, and occupation code and sum the jobs.
        regional_totals = (
            occupations_shocks.groupby(["region", "time", "OCC_CODE"])
            .agg(total_jobs=("jobs_by_occupation", "sum"))
            .reset_index()
        )
        regional_totals = regional_totals[regional_totals["region"] != "National"]

        # Concatenate the regional and national DataFrames.
        occupations_shocks = pd.concat([regional_totals, national_totals], ignore_index=True)

        return occupations_shocks

    def generate_model_data_inputs(
        self, region: str = "National", dict_lf_region: dict = CANADA_LABOUR_FORCE, **kwargs
    ):
        """
        Generates the data necessary to run the labour ABM.
        """
        df_mobility_network = self.mobility_network
        occupation_shocks = self.occupation_shocks

        df_mobility_network = df_mobility_network.rename({"trans_merge_alpha05": "weight"}, axis="columns")
        mobility_network = nx.from_pandas_edgelist(
            df_mobility_network, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph()
        )

        # Create dictionary for occupation codes to titles
        occ_codes_to_names = dict(zip(df_mobility_network["OCC_target"], df_mobility_network["OCC_TITLE_OCC"]))

        # Define and find strongly connected components
        largest_cc = max(nx.strongly_connected_components(mobility_network), key=len)
        mobility_network = mobility_network.subgraph(largest_cc).copy()
        nodes_order = list(mobility_network.nodes)
        assert nx.is_strongly_connected(mobility_network), "The mobility network is not strongly connected."

        # Create a DataFrame for node ID, OCC code, and OCC title
        node_details_df = pd.DataFrame(
            {
                "node_id": range(len(nodes_order)),
                "OCC_code": nodes_order,
                "OCC_title": [occ_codes_to_names.get(occ, "Unknown") for occ in nodes_order],
            }
        )

        # Filter data for specific region and rescale by labour force
        occupation_shocks_region = occupation_shocks[occupation_shocks["region"] == region]
        dict_soc_emp = dict(zip(df_mobility_network["OCC_target"], df_mobility_network["TOT_EMP_OCC"]))
        lab_force_usa = np.array([dict_soc_emp.get(node, 0) for node in nodes_order]).sum()
        scale_factor = dict_lf_region[region] / lab_force_usa
        df_mobility_network["TOT_EMP_OCC"] *= scale_factor
        dict_soc_emp = dict(zip(df_mobility_network["OCC_target"], df_mobility_network["TOT_EMP_OCC"]))

        # Create and populate the new DataFrame for employment data
        df_employment = pd.DataFrame({"OCC": nodes_order})
        times = sorted(occupation_shocks_region["time"].unique())
        df_pivot = occupation_shocks_region.pivot_table(
            index="OCC_CODE", columns="time", values="total_jobs", aggfunc="sum", fill_value=0.0
        )
        emp_cols = [f"emp {time}" for time in times]
        emp_dataframe = pd.DataFrame(0.0, index=df_employment.index, columns=emp_cols)

        df_employment = pd.concat([df_employment, emp_dataframe], axis=1).copy()

        # Calculate base employment and job additions
        for idx, row in df_employment.iterrows():
            occ = row["OCC"]
            base_emp = dict_soc_emp.get(occ, 0)
            job_additions = df_pivot.loc[occ] if occ in df_pivot.index else pd.Series(0, index=times)
            for time in times:
                df_employment.at[idx, f"emp {time}"] = base_emp + job_additions.get(time, 0.0)

            # Check for negative values and adjust

        def adjust_row(dataframe_row):
            min_value = dataframe_row[1:].min()  # Find minimum value excluding the 'OCC' column
            if min_value < 0:
                adjustment = 1.1 * abs(min_value)  # Calculate adjustment value
                dataframe_row[1:] += adjustment  # Adjust all employment columns
            return dataframe_row

            # Apply the adjustment to all rows with negative values

        for idx, row in df_employment.iterrows():
            if any(row[col] < 0 for col in df_employment.columns if col.startswith("emp")):
                df_employment.loc[idx] = adjust_row(row)

        # Generate network adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(mobility_network, weight="weight").todense()
        adjacency_matrix = np.array(adjacency_matrix)

        return df_employment, node_details_df, adjacency_matrix

    def generate_model_inputs(self, region, burn_in: int = 1_000, smooth: int = 3, **kwargs):
        """
        Generates the inputs required to run the labour abm.
        """

        # Load data inputs
        scenario, node_details, adjacency_matrix = self.generate_model_data_inputs(region=region, **kwargs)

        # Load scenario and convert to numpy array
        demand_scenario = scenario.drop(columns="OCC").values  # Drop OCC column and convert to numpy array

        # # list of times (for plotting purposes)
        time_columns = [col for col in scenario.columns if col.startswith("emp")]
        time_indices = [pd.to_datetime(col.split()[1], format="%Y-%m") for col in time_columns]

        # get n and convert to tensors
        n_occupations = adjacency_matrix.shape[0]
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        demand_scenario = torch.from_numpy(demand_scenario)
        t_scenario = demand_scenario.shape[1]

        t_max = burn_in + t_scenario

        # now get e, u, v accordingly
        e = demand_scenario[:, 0]

        u = 0.045 * e  # 5% of e
        v = 0.02 * e  # 5% of e
        # preserve labor force
        e = e - u
        sum_e_u = e + u
        L = sum_e_u.sum()

        # make target demand so that first it is constant so it converges and then scenario
        d_dagger = torch.zeros(n_occupations, burn_in + t_scenario)

        # Expand sum_e_u for broadcasting
        sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]
        # Populate d_dagger
        d_dagger[:, :burn_in] = sum_e_u_expanded.repeat(1, burn_in)

        d_dagger[:, burn_in:] = demand_scenario

        # perform smoothing
        # Convert d_dagger to numpy array for smoothing
        d_dagger_np = d_dagger.numpy()

        # Apply rolling window smoothing with minimum points 1
        df_d_dagger = pd.DataFrame(d_dagger_np.T).rolling(window=smooth, min_periods=1).mean().T

        # Convert back to tensor
        d_dagger = torch.from_numpy(df_d_dagger.values)

        # check positive demand
        assert torch.all(d_dagger > 0), "Scenarios must not have negative demand"

        # since no data use uniform wages
        wages = torch.ones(n_occupations)

        model_inputs = {
            "adjacency_matrix": adjacency_matrix,
            "initial_employment": e,
            "initial_unemployment": u,
            "initial_vacancies": v,
            "L": L,
            "n_occupations": n_occupations,
            "t_max": t_max,
            "wages": wages,
            "d_dagger": d_dagger,
            "time_indices": time_indices,
        }

        return model_inputs
