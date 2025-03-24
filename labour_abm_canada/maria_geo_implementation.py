import copy
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

import sys
print(sys.executable)
sys.path.append('/home/maria/Dropbox/Macrocosm/CanadaLaborApplication')

from labour_abm_canada.configuration.configuration import LaborSettings, ModelConfiguration
from labour_abm_canada.data_bridge import OCC_NAME_DICT
from labour_abm_canada.data_bridge.bridge import DataBridge
from labour_abm_canada.model import labour_abm as lbm
from labour_abm_canada.regions.regions import Regions

scenario_filename = 'data/DataMarch2025/employment-noc-4-codes-urban-rural_clean_scenarios.csv'
network_occ_filename = 'data/DataMarch2025/occupational_network_cc_plusllm_connected_edgelist.csv'
network_geo_filename = 'data/DataMarch2025/mobility_2016_provinces_only_finescale.csv'
employment_filename = 'data/DataMarch2025/employment-noc-4-codes-counts-urban-rural_clean.csv'


df_geo = pd.read_csv(network_geo_filename)
df_emp = pd.read_csv(employment_filename)
df_occ = pd.read_csv(network_occ_filename)
df_sc = pd.read_csv(scenario_filename)

regions = df_geo['residence_1_year_ago_2016'].unique()
G = nx.from_pandas_edgelist(df_occ, source='NOC_Code_Source', target='NOC_Code_Target', edge_attr='weight', create_using=nx.Graph())

    # Get the sorted list of nodes
sorted_nodes = sorted(G.nodes())


# databridge = DataBridge.from_standard_files(scenario_file=scenario_filename)
# model_inputs = databridge.generate_model_inputs('National', burn_in=1_000, smooth=6)

DEFAULT_MODEL_PARAMS = Path(__file__).parent / "model-params.yaml"

model_parameters="model-params.yaml"

labor_settings = LaborSettings.from_yaml(model_parameters)

burn_in=1_000
n_test = 10

#######
# Functions that need updating in data bridge
#######

def generate_model_data_inputs_maria(
    scenario_filename='', 
    network_occ_filename=network_occ_filename, 
    network_geo_filename=network_geo_filename,
    employment_filename=employment_filename,
    # NOTE LUCA the two parameters below should/could be model params
    stay_province = 2/3,
    stay_occ = 1/2,
    burn_in: int = 1_000,
    model_parameters: str | Path = DEFAULT_MODEL_PARAMS,
    seed: int = 123):
    '''NOTE LUCA This functions should basically replace generate_model_data_inputs in the data_bridge.py file.
    Gets and processes the data necessary to run the labour ABM. Mainly the network, wages, 
    initial conditions (employment, unemployment, vacancies), and the demand scenario.
    '''

    # Build geography network
    df_geo = pd.read_csv(network_geo_filename)
    geo_adjacency_matrix = df_geo.iloc[:, 1:].values
    np.fill_diagonal(geo_adjacency_matrix, 0)
    row_sums = geo_adjacency_matrix.sum(axis=1)
    row_normalized_matrix = geo_adjacency_matrix / row_sums[:, np.newaxis]
    regions = df_geo['residence_1_year_ago_2016'].unique()

    P_geo = stay_province * np.diag(np.ones(len(regions))) + \
        (1 - stay_province) * row_normalized_matrix

    # Build occupation network
    df_occ = pd.read_csv(network_occ_filename)
    # Remove self loops
    df_occ = df_occ[df_occ['NOC_Code_Source'] != df_occ['NOC_Code_Target']]
    # Create a directed graph from the edgelist
    G = nx.from_pandas_edgelist(df_occ, source='NOC_Code_Source', target='NOC_Code_Target', edge_attr='weight', create_using=nx.Graph())
    # Get the sorted list of nodes
    sorted_nodes = sorted(G.nodes())
    # Get the adjacency matrix in the sorted order
    occ_adjacency_matrix = nx.to_numpy_array(G, nodelist=sorted_nodes)
    occ_adjacency_matrix = occ_adjacency_matrix / occ_adjacency_matrix.sum(axis=1)[:, np.newaxis]
    np.fill_diagonal(occ_adjacency_matrix, 0)
    P_occ = stay_occ * np.diag(np.ones(len(occ_adjacency_matrix))) + \
            (1 - stay_occ) * occ_adjacency_matrix

    # Combine the two networks
    adjacency_matrix = np.kron(P_geo, P_occ)

    # Employment 
    df_node_attributes = pd.read_csv(employment_filename)
    
    if scenario_filename == '':
        df_scenarios = ''
    else:
        df_scenarios = pd.read_csv(scenario_filename)


    return df_scenarios, df_node_attributes, adjacency_matrix
    # df_employment, node_details_df, adjacency_matrix

    # return scenario, node_details, adjacency_matrix


def generate_model_inputs_maria(burn_in: int = 1_000, scenario='',smooth: int = 3, **kwargs):
        """
        NOTE LUCA: This function should replace generate_model_inputs in the data_bridge.py file. 
        I've modified it so that if no scenario is given then it just runs the steady state 
        so far with scenario = '', however, I think we could just do if scenario is None then run steady state.
        Uses generate data_inputs to then 
        Generates the inputs required to run the labour abm.
        """

        # Load data inputs
        scenario, node_details, adjacency_matrix = generate_model_data_inputs_maria(**kwargs)

        n_occupations = adjacency_matrix.shape[0]
        adjacency_matrix = torch.from_numpy(adjacency_matrix)

        if isinstance(scenario, str):   
            if scenario == '':
                t_max = burn_in

                e = torch.from_numpy(np.array(node_details['employment']))
                u = 0.045 * e  # 5% of e
                v = 0.02 * e  # 5% of e
                e = e - u
                sum_e_u = e + u
                L = sum_e_u.sum()
                d_dagger = sum_e_u.unsqueeze(1).repeat(1, burn_in)
                time_indices = [date.strftime("%Y-%m-%d") for date in \
                                pd.date_range(start='2025-01-01', periods=t_max, freq='M')]
       
        else:
            df_scenario = scenario # assume it was dataframe, to clean up later
            
            e = torch.from_numpy(np.array(node_details['employment']))
            u = 0.045 * e  # 5% of e
            v = 0.02 * e  # 5% of e
            e = e - u
            sum_e_u = e + u

            L = sum_e_u.sum()

            # Load scenario and convert to numpy array
            demand_scenario = scenario.drop(columns=['node_id', 'code', 'region']).values  # Drop OCC column and convert to numpy array
            # # list of times (for plotting purposes)
            demand_scenario = torch.from_numpy(demand_scenario)
            print(demand_scenario.shape)
            t_scenario = demand_scenario.shape[1]
            t_max = burn_in + t_scenario

            d_dagger = torch.zeros(n_occupations, burn_in + t_scenario)
            sum_e_u_expanded = sum_e_u.unsqueeze(1)
            d_dagger[:, :burn_in] = sum_e_u_expanded.repeat(1, burn_in)

            d_dagger[:, burn_in:] = demand_scenario

            time_indices = [date.strftime("%Y-%m-%d") for date in \
                            pd.date_range(start='2020-01-01', periods=t_max, freq='M')]

        # get n and convert to tensors


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
            # NOTE LUCA: I've added this dictionary to store the node details
            # node_details will be useful for plotting might be worth adding in the runner or lab_abm
            "node_details":node_details
        }

        return model_inputs


def get_metrics_by_region_or_occupation(lab_abm, node_details, region=None, occupation=None):
    """
    #NOTE LUCA: This function is new. Since now we run the model nationally, but may still want to see 
    # local/occupation effects this function filters the results by region or occupation.
    Returns the metrics for the given region or occupation (or both).

    Parameters:
    lab_abm (LabourABM): The LabourABM instance.
    node_details (pd.DataFrame): DataFrame containing node details.
    region (str, optional): The region to filter by.
    occupation (str, optional): The occupation code to filter by.

    Returns:
    dict: A dictionary containing the filtered metrics.
    """
    # Filter nodes based on region and/or occupation
    if region and occupation:
        filtered_nodes = node_details[(node_details['region'] == region) &\
                                       (node_details['code'].astype(str) == occupation)]
    elif region:
        filtered_nodes = node_details[node_details['region'] == region]
    elif occupation:
        filtered_nodes = node_details[node_details['code'].astype(str) == occupation]
    else:
        filtered_nodes = node_details

    node_ids = filtered_nodes['node_id'].values.astype(int)

    # Get the metrics for the filtered nodes
    unemp = lab_abm.unemployment[node_ids, :].sum(dim=0)
    vacancies = lab_abm.vacancies[node_ids, :].sum(dim=0)
    employment = lab_abm.employment[node_ids, :].sum(dim=0)
    demand_scenario = lab_abm.demand_scenario[node_ids, :].sum(dim=0)

    return {
        'ids': node_ids,
        'unemployment': unemp,
        'vacancies': vacancies,
        'employment': employment,
        'demand_scenario': demand_scenario
    }


####
# Run the model
####

# NOTE LUCA, I struggled getting the model configuration to run, so I hardcoded the values for now.
# Once this is fixed it should be straightforward to do this in the repackaging

model_required_inputs = generate_model_inputs_maria(scenario_filename=scenario_filename)

model_configuration = ModelConfiguration(
    labor=labor_settings, t_max=1_060, n=model_required_inputs["n_occupations"],
)

# plt.imshow(np.log(model_required_inputs["adjacency_matrix"][:161, :161]))
# plt.show()


lab_abm = lbm.LabourABM.default_create(
        model_configuration=model_configuration,
        transition_matrix=model_required_inputs["adjacency_matrix"],
        initial_employment=model_required_inputs["initial_employment"],
        initial_unemployment=model_required_inputs["initial_unemployment"],
        initial_vacancies=model_required_inputs["initial_vacancies"],
        wages=model_required_inputs["wages"],
        demand_scenario=model_required_inputs["d_dagger"],
    )


lab_abm.run_model()

model_required_inputs['node_details']

times = model_required_inputs['time_indices']

# Aggregate data
total_unemployment = lab_abm.unemployment.sum(dim=0)[:]
total_vacancies = lab_abm.vacancies.sum(dim=0)[:]
total_employment = lab_abm.employment.sum(dim=0)[:]
total_demand = lab_abm.demand_scenario.sum(dim=0)[:]
d_dagger = lab_abm.demand_scenario[:, :]
D_dagger = d_dagger.sum(dim=0)
L = total_employment[0] + total_unemployment[0]  # Labor force at time zero

# Region/occ plot
region = 'AB_rural'

occupation = '1001'
metrics = get_metrics_by_region_or_occupation(lab_abm, model_required_inputs['node_details'], \
                                              region=region, occupation=occupation)

print("Unemployment:", metrics['unemployment'])
print("Vacancies:", metrics['vacancies'])
print("Employment:", metrics['employment'])
print("Demand Scenario:", metrics['demand_scenario'])

lab_abm.unemployment

model_required_inputs['node_details']['region'] == '1001'

metrics['ids']
metrics['employment']

#####
# Functions for plotting
#####
def plot_aggregate_metrics(times, D_dagger, total_unemployment, total_vacancies, L, region, T_steady=0):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times[T_steady:], D_dagger.numpy()[T_steady:] / L)
    axes[0].set_title('Total Demand (in terms of 2021 labor force)', fontsize=16)
    axes[0].set_ylabel('Count', fontsize=14)
    axes[0].grid(True)
    
    axes[1].plot(times[T_steady:], 100 * total_unemployment.numpy()[T_steady:] / L)
    axes[1].set_title('Unemployment Rate Over Time', fontsize=16)
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=14)
    axes[1].grid(True)
    
    axes[2].plot(times[T_steady:], 100 * total_vacancies.numpy()[T_steady:] / L)
    axes[2].set_title('Vacancy Rate', fontsize=16)
    axes[2].set_ylabel('Vacancy Rate (%)', fontsize=14)
    axes[2].set_xlabel('Time', fontsize=14)
    axes[2].grid(True)
    
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle(f'Aggregate Metrics for {region}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_occupation_metrics(times, lab_abm, occ, node_details, region, T_steady=0):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Filter node details for the given occupation and region
    filtered_nodes = node_details[(node_details['code'].astype(str) == occ) & (node_details['region'] == region)]
    if filtered_nodes.empty:
        print(f"No data found for occupation {occ} in region {region}")
        return

    node_id = filtered_nodes['node_id'].values[0]
    occupation_title = filtered_nodes['occupation'].values[0]

    axes[0].plot(times[T_steady:], lab_abm.demand_scenario[node_id, T_steady:].numpy())
    axes[0].set_title(f'Demand for Workers in {occupation_title}', fontsize=16)
    axes[0].set_ylabel('Count', fontsize=14)
    axes[0].grid(True)

    axes[1].plot(times[T_steady:], 100 * lab_abm.unemployment[node_id, T_steady:].numpy() / 
                 (lab_abm.unemployment[node_id, T_steady:].numpy() + lab_abm.employment[node_id, T_steady:].numpy()))
    axes[1].set_title(f'Unemployment Rate in {occupation_title}', fontsize=16)
    axes[1].set_ylabel('Unemployment Rate (%)', fontsize=14)
    axes[1].grid(True)

    axes[2].plot(times[T_steady:], 100 * lab_abm.vacancies[node_id, T_steady:].numpy() / 
                 (lab_abm.vacancies[node_id, T_steady:].numpy() + lab_abm.employment[node_id, T_steady:].numpy()))
    axes[2].set_title(f'Vacancy Rate in {occupation_title}', fontsize=16)
    axes[2].set_ylabel('Vacancy Rate (%)', fontsize=14)
    axes[2].set_xlabel('Time', fontsize=14)
    axes[2].grid(True)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.suptitle(f'Occupation Metrics for {occupation_title}\n in {region}', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Example usage
T_steady = 0  # Set the starting time index for steady state
regions_test = ["AB_urban", "ON_urban", "SK"]
subset_occ = ['1001', '9410', '8511', '7220']


times = model_required_inputs['time_indices']

# Plot aggregate metrics
plot_aggregate_metrics(times, D_dagger, total_unemployment, total_vacancies, L, region, T_steady)

# Plot occupation metrics for specific occupations
for occ in subset_occ:
    
    plot_occupation_metrics(times, lab_abm, occ, model_required_inputs['node_details'], region, T_steady)



###################################################




model_required_inputs = generate_model_inputs_maria()

model_configuration = ModelConfiguration(
    labor=labor_settings, t_max=1_000, n=model_required_inputs["n_occupations"],
)


lab_abm = lbm.LabourABM.default_create(
        model_configuration=model_configuration,
        transition_matrix=model_required_inputs["adjacency_matrix"],
        initial_employment=model_required_inputs["initial_employment"],
        initial_unemployment=model_required_inputs["initial_unemployment"],
        initial_vacancies=model_required_inputs["initial_vacancies"],
        wages=model_required_inputs["wages"],
        demand_scenario=model_required_inputs["d_dagger"],
    )


lab_abm.run_model()

lab_abm.node_details


results = dict()
results["configs"] = dict(
    scenario_filename=scenario_filename,
    region=region,
    burn_in=burn_in,
    seed=seed,
    model_params=model_configuration.dict(),
)

results["occupation_names"] = OCC_NAME_DICT


# Aggregate data
total_unemployment = lab_abm.unemployment.sum(dim=0)[:]
total_vacancies = lab_abm.vacancies.sum(dim=0)[:]
total_employment = lab_abm.employment.sum(dim=0)[:]
total_demand = lab_abm.demand_scenario.sum(dim=0)[:]
d_dagger = lab_abm.demand_scenario[:, :]
D_dagger = d_dagger.sum(dim=0)


plt.plot(total_unemployment/total_employment)
plt.show()



df_scenarios, df_node_attributes, adjacency_matrix = generate_model_data_inputs_maria(
    scenario_filename=scenario_filename,
    network_occ_filename=network_occ_filename,
    network_geo_filename=network_geo_filename,
    employment_filename=employment_filename,
    burn_in=1_000,
    model_parameters="model-params.yaml")


A = np.random.rand(n_test, n_test)
A = A / A.sum(axis=1)[:, None]
A = torch.tensor(A)
initial_employment = 10_000*torch.tensor(np.random.rand(n_test))
initial_unemployment = 0.045 * initial_employment
initial_vacancies = 0.045 * initial_employment
initial_employment = initial_employment - initial_unemployment
sum_e_u = initial_employment + initial_unemployment
L = sum_e_u.sum()

wages = torch.tensor(np.ones(n_test))

d_dagger = torch.zeros(n_test, burn_in )

# Expand sum_e_u for broadcasting
sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]
# Populate d_dagger
d_dagger[:, :burn_in] = sum_e_u_expanded.repeat(1, burn_in)


lab_abm = lbm.LabourABM.default_create(
        model_configuration=model_configuration,
        transition_matrix=adjacency_matrix,
        initial_employment=initial_employment,
        initial_unemployment=initial_unemployment,
        initial_vacancies=initial_vacancies,
        wages=wages,
        demand_scenario=d_dagger
    )



lab_abm.run_model()

# Aggregate data
total_unemployment = lab_abm.unemployment.sum(dim=0)[:]
total_vacancies = lab_abm.vacancies.sum(dim=0)[:]
total_employment = lab_abm.employment.sum(dim=0)[:]
total_demand = lab_abm.demand_scenario.sum(dim=0)[:]
d_dagger = lab_abm.demand_scenario[:, :]
D_dagger = d_dagger.sum(dim=0)


total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


run_model(scenario_filename=scenario_filename,
    region="National",
    model_parameters="model-params.yaml",
    burn_in=1_000,
    seed=123
)

params = [0.1, 0.2, 0.5, 1.0]

# for p in params:
#     params_updated = {"p": p}
#     # save this as a yaml
#     with open("yaml_path.yaml", "s") as outfile:
#         yaml.save(params_updated, outfile)

#     run_model(
#         scenario_filename="ABC/DEF",
#         model_parameters="yaml_path.yaml",
#         region="National",
#         burn_in=1_000,
#         seed=123
#     )