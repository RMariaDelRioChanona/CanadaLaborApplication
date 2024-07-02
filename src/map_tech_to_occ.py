import pandas as pd
import seaborn as sns
from matplotlib import pylab as plt

path_data = "../data/CanadaData/"
path_data_bls = '../data/BLS/oesm18in4/'
path_fig = "../results/fig/"
path_params = "../data/parameters/"
path_tech = "../data/copper/"


file_naics_occ = "nat5d_6d_M2018_dl.xlsx"
file_network="../data/networks/edgelist_cc_mobility_merge.csv"
file_tech_ind = "6digitNAICS_tech.csv"

df_ind_occ = pd.read_excel(path_data_bls + file_naics_occ)
df_network = pd.read_csv(file_network)

df_tech = pd.read_csv(path_tech + file_tech_ind)

df_tech.head()

dict_naics_tech = dict(zip(df_tech['NAICS6d'], df_tech['variable']))
inds = list(df_tech['NAICS6d'].unique())


df_ind_occ['TOT_EMP'] = pd.to_numeric(df_ind_occ['TOT_EMP'], errors='coerce')

# Calculate the minimum of TOT_EMP excluding NaNs
min_tot_emp = df_ind_occ['TOT_EMP'].min()

# We only care about those industries that are shocked
df_inds_occ_sub = df_ind_occ[df_ind_occ['NAICS'].isin(inds)]
# We only care about those industries which are in the network (others can be diff classification does)
occs_bls_shock = list(df_inds_occ_sub['OCC_CODE'].unique())
occs_net = list(df_network['OCC_target'].unique())
# set of occupations to filter the crosswalks
set_occ_keep = list(set(occs_bls_shock).intersection(set(occs_net)))

# being cautious
for occ in ['35-9090', '53-4020', '53-7110', '19-1090', '11-9060']:
    if occ in set_occ_keep:
        print(occ)
        set_occ_keep.remove(occ)

df_inds_occ_sub = df_inds_occ_sub[df_inds_occ_sub['OCC_CODE'].isin(set_occ_keep)]
df_inds_occ_sub['TOT_EMP'] = pd.to_numeric(df_inds_occ_sub['TOT_EMP'], errors='coerce')
df_inds_occ_sub['TOT_EMP'].fillna(10, inplace=True)
df_inds_occ_sub = df_inds_occ_sub[['NAICS', 'OCC_CODE', 'TOT_EMP' ]]
df_inds_occ_sub['variable'] = df_inds_occ_sub['NAICS'].map(dict_naics_tech)


# Calculate total employment for each 'variable' and create a dictionary
tech_totals = df_inds_occ_sub.groupby('variable')['TOT_EMP'].sum().to_dict()

# Create a new column with the total employment for each 'variable'
df_inds_occ_sub['tech_tot_emp'] = df_inds_occ_sub['variable'].map(tech_totals)

# Calculate the employment fraction for each row
df_inds_occ_sub['EMP_frac'] = df_inds_occ_sub['TOT_EMP'] / df_inds_occ_sub['tech_tot_emp']

df_inds_occ_sub[['variable','OCC_CODE', 'EMP_frac']].to_csv(path_tech + "technologies_occ.csv", index=False)





