from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt
from matplotlib.ticker import MaxNLocator

FILE_PATH = Path(__file__).parent
PROJECT_PATH = FILE_PATH.parent.parent
DATA_PATH = PROJECT_PATH / "data"

FILE_OCCUPATIONS = DATA_PATH / "nat5d_6d_M2018_dl.xlsx"
FILE_MOBILITY_NETWORK = DATA_PATH / "edge_list_cc_mobility_merge.csv"
FILE_TECHNOLOGIES = DATA_PATH / "6digitNAICS_tech.csv"

CAPACITY_JOB_COLNAME = "CPY (Direct)_linear"
OPERATION_JOB_COLNAME = "Op FTE (Direct)_linear"


def generic_loader(file_path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Generic loader for data files. It uses the file extension to determine the appropriate loader.
    """
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix == ".xlsx":
        return pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")


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
    monthly_jobs_df = monthly_jobs_df.sort_values(by=["region", "variable", "model", "scenario", "job_type", "time"])

    # Convert the list to DataFrame
    return monthly_jobs_df


def generate_shock_timeseries(
    filename: str | Path,
    capacity_jobs_colname: str = CAPACITY_JOB_COLNAME,
    operation_jobs_colname: str = OPERATION_JOB_COLNAME,
) -> pd.DataFrame:
    """
    Transforms the original scenario dataframe into a timeseries of job creation and destruction for capacity and
    operations.

    Args:
        filename (str | Path): The path to the scenario file.
        capacity_jobs_colname (str): The column name for capacity jobs.
        operation_jobs_colname (str): The column name for operation jobs.
    """
    scenario = pd.read_csv(filename)

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
    scenario = scenario.loc[columns_to_keep].copy()

    # Split the job columns into positive and negative
    scenario["CPY (Direct)_linear_positive"] = scenario[capacity_jobs_colname].apply(lambda x: x if x >= 0 else 0)
    scenario["CPY (Direct)_linear_negative"] = scenario[capacity_jobs_colname].apply(lambda x: x if x <= 0 else 0)
    scenario["Op FTE (Direct)_linear_positive"] = scenario[operation_jobs_colname].apply(lambda x: x if x >= 0 else 0)
    scenario["Op FTE (Direct)_linear_negative"] = scenario[operation_jobs_colname].apply(lambda x: x if x <= 0 else 0)

    # Aggregate rows with the same region, variable, and time
    scenario = scenario.groupby(["region", "variable", "model", "scenario", "time", "unit"], as_index=False).sum()

    # Rescale
    scenario["CPY (Direct)_linear_positive"].sum() / 1e6
    scenario["Op FTE (Direct)_linear_positive"].sum() / 1e6

    # Distribute the shocks uniformly, month-after-month
    monthly_jobs_df = distribute_jobs_monthly(scenario)

    monthly_jobs_df["jobs"][
        (monthly_jobs_df["region"] == "National") & (monthly_jobs_df["job_type"] == "CPY_created_jobs")
    ].sum() / 1e6
    monthly_jobs_df["jobs"][
        (monthly_jobs_df["region"] == "National") & (monthly_jobs_df["job_type"] == "Op_created_jobs")
    ].sum() / 1e6

    # Make the operation jobs cumulative
    monthly_jobs_df = make_operation_jobs_cumulative(monthly_jobs_df)

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


path_data = "../data/CanadaData/"
path_data_bls = "../data/BLS/oesm18in4/"
path_fig = "../results/fig/"
path_params = "../data/parameters/"
path_tech = "../data/copper/"


file_naics_occ = "nat5d_6d_M2018_dl.xlsx"
file_network = "../data/networks/edgelist_cc_mobility_merge.csv"
file_tech_ind = "6digitNAICS_tech.csv"


def technologies_to_occupations(
    file_occupations: str | Path = FILE_OCCUPATIONS,
    file_mobility_network: str | Path = FILE_MOBILITY_NETWORK,
    file_technologies: str | Path = FILE_TECHNOLOGIES,
):
    """
    Returns a map between technology shocks and occupation shocks.
    """

    df_ind_occ = pd.read_excel(file_occupations)
    df_network = pd.read_csv(file_mobility_network)
    df_tech = pd.read_csv(file_technologies)

    dict_naics_tech = dict(zip(df_tech["NAICS6d"], df_tech["variable"]))
    inds = list(df_tech["NAICS6d"].unique())
    df_ind_occ["TOT_EMP"] = pd.to_numeric(df_ind_occ["TOT_EMP"], errors="coerce")

    # We only care about those industries that are shocked
    df_inds_occ_sub = df_ind_occ[df_ind_occ["NAICS"].isin(inds)]
    # We only care about those industries which are in the network (others can be diff classification does)
    occs_bls_shock = list(df_inds_occ_sub["OCC_CODE"].unique())
    occs_net = list(df_network["OCC_target"].unique())
    # set of occupations to filter the crosswalks
    set_occ_keep = list(set(occs_bls_shock).intersection(set(occs_net)))
    # being cautious
    for occ in ["35-9090", "53-4020", "53-7110", "19-1090", "11-9060"]:
        if occ in set_occ_keep:
            print(occ)
            set_occ_keep.remove(occ)
    df_inds_occ_sub = df_inds_occ_sub[df_inds_occ_sub["OCC_CODE"].isin(set_occ_keep)]
    df_inds_occ_sub["TOT_EMP"] = pd.to_numeric(df_inds_occ_sub["TOT_EMP"], errors="coerce")
    df_inds_occ_sub["TOT_EMP"].fillna(10, inplace=True)
    df_inds_occ_sub = df_inds_occ_sub[["NAICS", "OCC_CODE", "TOT_EMP"]]
    df_inds_occ_sub["variable"] = df_inds_occ_sub["NAICS"].map(dict_naics_tech)
    # Calculate total employment for each 'variable' and create a dictionary
    tech_totals = df_inds_occ_sub.groupby("variable")["TOT_EMP"].sum().to_dict()
    # Create a new column with the total employment for each 'variable'
    df_inds_occ_sub["tech_tot_emp"] = df_inds_occ_sub["variable"].map(tech_totals)
    # Calculate the employment fraction for each row
    df_inds_occ_sub["EMP_frac"] = df_inds_occ_sub["TOT_EMP"] / df_inds_occ_sub["tech_tot_emp"]
    df_inds_occ_sub = df_inds_occ_sub[["variable", "OCC_CODE", "EMP_frac"]]

    return df_inds_occ_sub
