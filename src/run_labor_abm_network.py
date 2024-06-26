import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

# from matplotlib import pylab as plt
import labor_abm as lbm
import utils as ut

from matplotlib import pylab as plt

#################
# Basic test
#################

seed = 123
T = 500

(
    delta_u,
    delta_v,
    gamma_u,
    gamma_v,
    lam,
    beta_u,
    beta_e
) = ut.baseline_parameters()


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


d_dagger = ut.set_d_dagger_uniform(T, sum_e_v)


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


# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(total_unemployment.numpy()/L, 'o-',label='Total Unemployment')
# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
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
#
print(100*total_unemployment.numpy()[-1]/L)