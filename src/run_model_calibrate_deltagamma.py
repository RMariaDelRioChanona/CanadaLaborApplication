import copy
import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from matplotlib import pylab as plt

src_dir = os.path.join(os.getcwd(), "../src")
# src_dir = os.path.join(os.getcwd(), 'src')
sys.path.append(src_dir)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import labor_abm as lbm
import utils as ut

path_data = "../data/CanadaData/"
path_fig = "../results/fig/"

df_gdp = pd.read_csv(path_data + "u_v_gdp_seasonal.csv")
df_gdp = pd.read_csv(path_data + "u_v_gdp_seasonal_barometer.csv")

# df_gdp[df_gdp['REF_DATE'] == '2015-01-01']


n_samples = 100
seed = 123
threshold = 0.1

ut.generate_seeds(123, 100)

accepted_parameters, tested_params, mse_list, mse_av = ut.calibrate_delta_gamma(df_gdp, n_samples, seed, threshold)

mse_av_array = np.array(mse_av)

# Get indices of sorted mse_av in ascending order
sorted_indices = np.argsort(mse_av_array)

# Use these indices to reorder tested_params correspondingly
sorted_tested_params = [tested_params[i] for i in sorted_indices]


len(df_gdp) + 20

L = 20000
N = 2
T_steady = 10
T_smooth = 10


######

N, T, L, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_u, beta_e, A, e, u, v, d_dagger, wages = (
    ut.from_gdp_uniform_across_occ(L, N, T_steady, T_smooth, df_gdp)
)

lab_abm = lbm.LabourABM(
    N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, beta_u, A, d_dagger, e, u, v, wages
)

plt.plot(d_dagger[0, :] / 10000)
plt.show()

len(df_gdp)
