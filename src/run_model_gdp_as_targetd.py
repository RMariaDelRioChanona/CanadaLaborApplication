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

df = pd.read_csv(path_data + "u_v_gdp_seasonal_barometer.csv")
# Start form 2015 when vac data start, unitl last time period where we have some NaNs
df = df[df['REF_DATE'] >= '2015-04-01']
df = df.iloc[:-1]


u_real= torch.tensor(df['unemployment'].values)
v_real= torch.tensor(df['vacancy'].values)

# timeseries used for target demand
gdp = torch.tensor(df['GDP_Cycle12'].values)
bb = torch.tensor(df['BBarometer_Cycle'].values)
bb_gdp = torch.tensor(df['GBP_BB3MA_minmax_interact'].values)
gdp2 = gdp**2
bb2 = bb**2
ts1, ts2, ts3, ts4, ts5 = gdp, gdp2, bb, bb2, bb_gdp
timeseries = [ts1, ts2, ts3, ts4, ts5]



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
    beta_e,
    a0, 
    a1, 
    a2, 
    a3, 
    a4, 
    a5
) = ut.parameters_calibrated_inc_target_demand()

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

T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[a0, a1, a2, a3, a4, a5], T_steady=T_steady, T_smooth=T_smooth)
        
lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, beta_u, A, d_dagger, e, u, v, wages)

_ = lab_abm.run_model()



total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger

dates = pd.date_range(start='2015-01-01', periods=len(total_unemployment.numpy()[20:]), freq='M')

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[T_steady + T_smooth:]/L, 'o-',label='Model Unemployment')
plt.plot(u_real, 'o-' ,label='Canada Unemployment')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_ureal_umodel_dddager_fit.png')
plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_vacancies.numpy()[T_steady + T_smooth:]/L, 'o-',label='Model Unemployment')
plt.plot(v_real, 'o-' ,label='Canada Unemployment')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(path_fig + 'canada_vreal_vmodel_dddager_fit.png')
plt.show()

