import numpy as np
import torch
import scipy.stats as stats
import pandas as pd
import copy
from matplotlib import pylab as plt
import sys
import os
src_dir = os.path.join(os.getcwd(), '../src')
# src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)
import labor_abm as lbm
import utils as ut
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

path_data = "../data/CanadaData/"
path_fig = "../results/fig/"

df_gdp = pd.read_csv(path_data + "u_v_gdp_seasonal.csv")
df_gdp = pd.read_csv(path_data + "u_v_gdp_seasonal_barometer.csv")

df_gdp[df_gdp['REF_DATE'] == '2015-01-01']
df_gdp = df_gdp.iloc[:-1]

len(df_gdp) + 20

seed =123
L = 20000
N = 2
T_steady = 200
T_smooth = 10


###########
# Run complete network
##########
(
    delta_u,
    delta_v,
    gamma_u,
    gamma_v,
    lam,
    beta_u,
    beta_e
) = ut.baseline_parameters_jtj_multapps()

(
    A,
    e,
    u,
    v,
    L,
    N,
    sum_e_v,
    wages
)=ut.network_and_employment(network="merge")

###########
# Set target demand
###########

T, d_dagger = ut.target_from_gdp_e_u(df_gdp, e, u,col_name="cycle_bb_mix", N=N, T_steady=T_steady, T_smooth=T_smooth)

plt.plot(d_dagger.sum(axis=0))
plt.show()

plt.plot(d_dagger[0, :])
plt.show()


#### run model

lab_abm = lbm.LabourABM(
    N,
    T,
    seed,
    delta_u,
    delta_v,
    gamma_u,
    gamma_v,
    lam,
    beta_e,
    beta_u,
    A,
    d_dagger,
    e,
    u,
    v,
    wages,
)

e, u, v, spon_sep, state_sep, spon_vac, state_vac, jtj, utj = lab_abm.run_model()

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger



# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
# plt.plot(100*total_unemployment.numpy()[210:]/L, 100*total_vacancies.numpy()[210:]/L, 'o-', label='all')
plt.plot(100*total_unemployment.numpy()[210:280]/L, 100*total_vacancies.numpy()[210:280]/L, 'o-', label='model 2015- Covid')
plt.plot(100*total_unemployment.numpy()[280:]/L, 100*total_vacancies.numpy()[280:]/L, 'o-', label='model postCovid')
# plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
# plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.show()



# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model 2015- Covid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
# plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
# plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)

plt.show()



##
lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)

plt.plot(d_dagger[0, :]/10000)
plt.show()

len(df_gdp)

plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])
# plt.plot(df_gdp['unemployment'], df_gdp['vacancy'])
plt.show()

gamma_u*3
delta_u*1.2

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)

lab_abm.run_model()

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger

df_gdp['REF_DATE'].iloc[111:]

dates = pd.date_range(start='2015-01-01', periods=len(total_unemployment.numpy()[20:]), freq='M')


# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates,100*total_unemployment.numpy()[20:]/L, 'o-',label='Total Unemployment')
plt.plot(dates, df_gdp['unemployment'], 'o-' ,label='unemp for real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_ureal_umodel_dddager_gdp_0.png')
plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates, 100*total_vacancies.numpy()[20:]/L,'o-' ,label='Total Vacancies')
plt.plot(dates, df_gdp['vacancy'], 'o-' ,label='vacanciesfor real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_vreal_vmodel_dddager_gdp_0.png')
plt.show()



# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model 2015- Covid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
# plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
# plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_model_bcurve_0.png')
plt.show()


# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model preCovid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180], 'o-', label='model preCovid')
plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:],'o-',  label='model postCovid')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_model_bcurve_0_comp.png')
plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(total_unemployment.numpy()/L, 'o-',label='Total Unemployment')
plt.plot(total_vacancies.numpy()/L,'o-' ,label='Total Vacancies')
# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()


############
# Model tweek params
#############
N, T, L, seed, delta_u, delta_v, gamma_u, gamma_v,\
        lam, beta_u, beta_e, A, e, u, v, d_dagger, wages = ut.from_gdp_uniform_across_occ(L, N, T_steady, T_smooth, df_gdp)

gamma_u*3
delta_u*1.2


#### with quick calib params

delta_u, delta_v, gamma_u, gamma_v = [0.03175170735208581,
  0.009298833869991242,
  0.35918423605862815,
  0.8167276800747527]

# lab_abm = lbm.LabourABM(N, T, seed, 1.6*delta_u, 1.3*delta_v, 1.5*gamma_u, 1.5*gamma_v, lam, \
#                         beta_e, beta_u, A, d_dagger, e, u, v, wages)


lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)

lab_abm.run_model()

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


dates = pd.date_range(start='2015-01-01', periods=len(total_unemployment.numpy()[20:]), freq='M')


# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates,100*total_unemployment.numpy()[20:]/L, 'o-',label='Total Unemployment')
plt.plot(dates, df_gdp['unemployment'], 'o-' ,label='unemp for real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_ureal_umodel_dddager_gdp_1.png')
plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates, 100*total_vacancies.numpy()[20:]/L,'o-' ,label='Total Vacancies')
plt.plot(dates, df_gdp['vacancy'], 'o-' ,label='vacanciesfor real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_vreal_vmodel_dddager_gdp_1.png')
plt.show()



# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[:]/L, 100*total_vacancies.numpy()[:]/L, 'o-', label='all')
# plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
# plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)

plt.show()


# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model preCovid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180], 'o-', label='real preCovid')
plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:],'o-',  label='real postCovid')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_model_bcurve_1_comp.png')
plt.show()


##########
# Use business barometer
#########


N, T, L, seed, delta_u, delta_v, gamma_u, gamma_v,\
        lam, beta_u, beta_e, A, e, u, v, d_dagger, wages = ut.from_gdp_uniform_across_occ(\
            L, N, T_steady, T_smooth, df_gdp, col_name='cycle_bb_mix')

gamma_u*3
delta_u*1.2


lab_abm = lbm.LabourABM(N, T, seed, 1.6*delta_u, 1.8*delta_v, 1.2*gamma_u, 1.0*gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)


lab_abm.run_model()

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


dates = pd.date_range(start='2015-01-01', periods=len(total_unemployment.numpy()[20:]), freq='M')


# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model preCovid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180], 'o-', label='model preCovid')
plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:],'o-',  label='model postCovid')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_model_bcurve_2_comp.png')
plt.show()



# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates,100*total_unemployment.numpy()[20:]/L, 'o-',label='Total Unemployment')
plt.plot(dates, df_gdp['unemployment'], 'o-' ,label='unemp for real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_ureal_umodel_dddager_gdp_2.png')
plt.show()




# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates, 100*total_vacancies.numpy()[20:]/L,'o-' ,label='Total Vacancies')
plt.plot(dates, df_gdp['vacancy'], 'o-' ,label='vacanciesfor real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_vreal_vmodel_dddager_gdp_2.png')
plt.show()



# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model 2015- Covid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
# plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
# plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_model_bcurve_2.png')
plt.show()


# Plot the data Beveridge curve
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-', label='model preCovid')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-', label='model postCovid')
plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180], 'o-', label='model preCovid')
plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:],'o-',  label='model postCovid')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_model_bcurve_2_comp.png')
plt.show()



# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(total_unemployment.numpy()[120:]/L, total_vacancies.numpy()[120:]/L, 'o-')
# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.show()


len(total_unemployment.numpy())

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[120:180]/L, 100*total_vacancies.numpy()[120:180]/L, 'o-')
plt.plot(100*total_unemployment.numpy()[180:]/L, 100*total_vacancies.numpy()[180:]/L, 'o-')
plt.plot(df_gdp['unemployment'].iloc[111:180], df_gdp['vacancy'].iloc[111:180])
plt.plot(df_gdp['unemployment'].iloc[180:], df_gdp['vacancy'].iloc[180:])

# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.legend()
plt.grid(True)
plt.show()


# Set up your figure
plt.figure(figsize=(10, 5))

# Setup for colormap
n = len(total_unemployment[120:])  # Number of points
colors = plt.cm.viridis(np.linspace(0, 1, n))  # Colormap from blue to green

# Normalizing the indices for mapping to the colormap
norm = Normalize(vmin=0, vmax=n-1)
mapper = ScalarMappable(norm=norm, cmap='viridis')

# Plot each data point individually
for i in range(n):
    plt.plot(total_unemployment[120+i]/L, total_vacancies[120+i]/L, 'o-', color=mapper.to_rgba(i))

# Set titles and labels
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vacancies')
plt.grid(True)

# Adding colorbar to show the time progression
plt.colorbar(mapper, label='Time progression')
plt.savefig(path_fig + "can_bev.png")



