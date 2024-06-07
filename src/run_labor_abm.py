import numpy as np
import torch
import scipy.stats as stats
import pandas as pd
import copy
from matplotlib import pylab as plt
import labor_abm as lbm
import utils as ut


#################
# Basic test
#################

N, T, L, seed, delta_u, delta_v, gamma_u, gamma_v, lam\
    , beta_u, beta_e, A, e, u, v, d_dagger, wages = ut.test_2node_scenario()

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)

lab_abm.run_model()

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


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

#### Complete network


N, T, L, seed, delta_u, delta_v, gamma_u, gamma_v, lam\
    , beta_u, beta_e, A, e, u, v, d_dagger, wages = ut.test_2node_scenario_complete()

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)

lab_abm.run_model()

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


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

total_unemployment/L
total_vacancies/L




N = 10
T = 100
L = 1000
seed = 111
delta_u =  0.016
delta_v =  0.017
gamma_u = 10 * delta_u
gamma_v = gamma_u
lam = 0.001
beta_u = 10
beta_e = 1


T = 5
A, e, u, v, d_dagger, wages = ut.create_symmetric_A_euv_d_daggers(N, T, L, seed)

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_u, gamma_u, gamma_v, lam, \
                        beta_u, beta_e, A, d_dagger, e, u, v, wages)
lab_abm.run_model()

sum_e_u = torch.rand(10)

sum_e_u.size()

d_test = sum_e_u.unsqueeze(1).repeat(1, 20)
d_dagger[:, 0]


# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employmnet = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger
total_demand
total_employmnet


total_vacancies.numpy()

plt.plot(total_demand.numpy(), label='Total demand')

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(total_unemployment.numpy(), label='Total Unemployment')
plt.plot(total_vacancies.numpy(), label='Total Vacancies')
# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()

lab_abm.employment.size()


lab_abm.run_model()