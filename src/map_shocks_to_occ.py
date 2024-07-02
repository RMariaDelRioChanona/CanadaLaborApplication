import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt
import networkx as nx
import numpy as np

path_data = "../data/CanadaData/"
path_data_bls = '../data/BLS/oesm18in4/'
path_fig = "../results/fig/"
path_params = "../data/parameters/"
path_tech = "../data/copper/"

file_shocks = "shock_timeseries_region_tech.csv"
file_tech_occ = "technologies_occ.csv"

df_shocks = pd.read_csv(path_tech + file_shocks)
df_shocks['jobs'][(df_shocks['time'] == '2055-12') & (df_shocks['region'] == 'National')].sum()

# remove national, better calculated at this level for consitency (some cancel out between regions)
df_shocks=df_shocks[df_shocks['region'] != 'National']

df_map = pd.read_csv(path_tech + file_tech_occ)

merged_df = pd.merge(df_shocks, df_map, on='variable', how='left')

merged_df['jobs_by_occupation'] = merged_df['jobs'] * merged_df['EMP_frac']

grouped_df = merged_df.groupby(['region', 'time', 'OCC_CODE']).agg(
    total_jobs=('jobs_by_occupation', 'sum')
).reset_index()

grouped_df=grouped_df[grouped_df['region'] != 'National']

# Create a new DataFrame for national sums by grouping without the region and summing the jobs.
national_totals = merged_df.groupby(['time', 'OCC_CODE']).agg(
    total_jobs=('jobs_by_occupation', 'sum')
).reset_index()
national_totals['region'] = 'National'

# Concatenate the regional and national DataFrames.
final_df = pd.concat([grouped_df, national_totals], ignore_index=True)


final_df[final_df['total_jobs'] < 0]

final_df[(final_df['time'] == '2055-12') & (final_df['region'] == 'National')]

final_df['total_jobs'][(final_df['time'] == '2030-12') & (final_df['region'] == 'British Columbia.a')]


final_df['total_jobs'][(final_df['time'] == '2030-12') & (final_df['region'] == 'British Columbia.a')].sum()

final_df['total_jobs'][(final_df['time'] == '2055-12') & (final_df['region'] == 'British Columbia.a')].sum()




############
# Now check with employment
############

dict_lf_region = {'Alberta.a':2.59e6, 
'British Columbia.a':2.93e6, 
'Manitoba.a':0.73e6, 
'New Brunswick.a':0.411e6,
       'Newfoundland and Labrador.a':0.26e6, 
       'Nova Scotia.a':0.526e6,
        'Ontario.a':4.2e6,
       'Ontario.b':4.2e6, 
       'Prince Edward Island.a':0.096e6, 
       'Quebec.a':2.35e6, 
       'Quebec.b':2.35e6,
       'Saskatchewan.a':0.613e6,
        'National':21e6}



file_path_name="../data/networks/edgelist_cc_mobility_merge.csv"
df_net = pd.read_csv(file_path_name)
dict_soc_name = dict(zip(df_net["OCC_target"], df_net["OCC_TITLE_OCC"]))

def process_region_employment(reg, final_df, dict_lf_region):
    # Load network data
    file_path_name = "../data/networks/edgelist_cc_mobility_merge.csv"
    df_net = pd.read_csv(file_path_name)
    df_net = df_net.rename({"trans_merge_alpha05": "weight"}, axis="columns")
    G = nx.from_pandas_edgelist(df_net, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph())

    # Create dictionary for occupation codes to titles
    dict_soc_name = dict(zip(df_net["OCC_target"], df_net["OCC_TITLE_OCC"]))

    # Define and find strongly connected components
    nodes_order = list(G.nodes)
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G_strongly = G.subgraph(largest_cc).copy()
    nodes_order = list(G_strongly.nodes)
    assert(nx.is_strongly_connected(G_strongly))

    # Create a DataFrame for node ID, OCC code, and OCC title
    node_details_df = pd.DataFrame({
        "node_id": range(len(nodes_order)),
        "OCC_code": nodes_order,
        "OCC_title": [dict_soc_name.get(occ, "Unknown") for occ in nodes_order]
    })

    # Filter data for specific region and calculate scale factor
    df_reg = final_df[final_df['region'] == reg]
    dict_soc_emp = dict(zip(df_net["OCC_target"], df_net["TOT_EMP_OCC"]))
    lab_force_usa = np.array([dict_soc_emp.get(node, 0) for node in nodes_order]).sum()
    scale_factor = dict_lf_region[reg] / lab_force_usa
    df_net['TOT_EMP_OCC'] *= scale_factor
    dict_soc_emp = dict(zip(df_net["OCC_target"], df_net["TOT_EMP_OCC"]))

    # Create and populate the new DataFrame for employment data
    new_df = pd.DataFrame({"OCC": nodes_order})
    times = sorted(df_reg['time'].unique())
    df_pivot = df_reg.pivot_table(index='OCC_CODE', columns='time', values='total_jobs', aggfunc='sum', fill_value=0)
    emp_cols = {f'emp {time}': 0 for time in times}
    new_df = pd.concat([new_df, pd.DataFrame(columns=emp_cols)], sort=False).fillna(0)

    for index, row in new_df.iterrows():
        occ = row['OCC']
        base_emp = dict_soc_emp.get(occ, 0)
        job_additions = df_pivot.loc[occ] if occ in df_pivot.index else pd.Series(0, index=times)
        for time in times:
            new_df.at[index, f'emp {time}'] = base_emp + job_additions.get(time, 0)

    # Check for negative values and adjust
    def adjust_row(row):
        min_value = row[1:].min()  # Find minimum value excluding the 'OCC' column
        if min_value < 0:
            adjustment = 1.1 * abs(min_value)  # Calculate adjustment value
            row[1:] += adjustment  # Adjust all employment columns
        return row

    # Apply the adjustment to all rows with negative values
    for index, row in new_df.iterrows():
        if any(row[col] < 0 for col in new_df.columns if col.startswith('emp')):
            new_df.loc[index] = adjust_row(row)


    # Calculate adjacency matrix
    A = nx.adjacency_matrix(G_strongly, weight="weight").todense()
    A = np.array(A)
    
    return new_df, A, node_details_df

# Example usage
reg = 'British Columbia.a'
new_df, A, node_details_df = process_region_employment(reg, final_df, dict_lf_region)


node_details_df.to_csv("../data/networks/node_occ_name.csv")

def export_data(A, new_df, region):
    # Define file paths
    adj_matrix_path = f"../data/networks/occ_mobility_fromusa.csv"
    df_path = f"../data/copper/scenario_{region.replace('.', '_')}.csv"
    
    # Export adjacency matrix A
    np.savetxt(adj_matrix_path, A, delimiter=",")
    
    # Export new_df
    new_df.to_csv(df_path, index=False)
    
    print(f"Data exported for region: {region}")

# Example usage within the loop
for r in dict_lf_region.keys():
    print(r)
    new_df, A, node_details = process_region_employment(r, final_df, dict_lf_region)
    export_data(A, new_df, r)


df_sas, A, node_details = process_region_employment('Saskatchewan.a', final_df, dict_lf_region)

negative_occ_list = []
for index, row in df_sas.iterrows():
    if any(row[col] < 0 for col in df_sas.columns if col.startswith('emp')):
        negative_occ_list.append(row['OCC'])

negative_occ_list[0]

# Function to adjust the row to avoid negative values
def adjust_row(row):
    min_value = row[1:].min()  # Find minimum value excluding the 'OCC' column
    if min_value < 0:
        adjustment = 1.1 * abs(min_value)  # Calculate adjustment value
        row[1:] += adjustment  # Adjust all employment columns
    return row

# Apply the adjustment to all rows with negative values
for occ in negative_occ_list:
    df_sas.loc[df_sas['OCC'] == occ] = df_sas[df_sas['OCC'] == occ].apply(adjust_row, axis=1)


df_sas[df_sas['OCC'] == negative_occ_list[0]].min()

export_data(A, new_df)

def load_data(region):
    # Define file paths
    adj_matrix_path = f"../data/networks/occ_mobility_fromusa.csv"
    df_path = f"../data/copper/scenario_{region.replace('.', '_')}.csv"
    
    # Load adjacency matrix A
    A = np.loadtxt(adj_matrix_path, delimiter=",")
    
    # Load new_df and convert to numpy array
    new_df = pd.read_csv(df_path)
    time_columns = [col for col in new_df.columns if col.startswith('emp')]
    times = [col.split()[1] for col in time_columns]

    
    data_array = new_df.drop(columns='OCC').values  # Drop OCC column and convert to numpy array
    
    return A, data_array, times

# Example usage
region = 'British Columbia.a'
A_loaded, data_array, times_list = load_data(region)


### TODO get datetime/years

# Example usage
reg = 'British Columbia.a'

for r in dict_lf_region.keys():
    print(r)
    new_df, A,  node_details_df  = process_region_employment(r, final_df, dict_lf_region)


new_df, A = process_region_employment("Ontario.a", final_df, dict_lf_region)

new_df, A = process_region_employment("Saskatchewan.a", final_df, dict_lf_region)
new_df.min()


