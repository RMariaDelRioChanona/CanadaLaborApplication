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
from scipy.optimize import differential_evolution, minimize
from tqdm import tqdm
import random

path_data = "../data/CanadaData/"
path_fig = "../results/fig/"
path_params = "../data/parameters/"

## parameters and networks
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


(
    lam,
    beta_u,
    beta_e
) = ut.baseline_parameters_jtj_multapps_from_data()

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


df = pd.read_csv(path_data + "u_v_gdp_seasonal_barometer.csv")
# Start form 2015 when vac data start, unitl last time period where we have some NaNs
df = df[df['REF_DATE'] >= '2015-04-01']
df = df.iloc[:-1]


u_real= torch.tensor(df['unemployment'].values)
v_real= torch.tensor(df['vacancy'].values)
gdp = torch.tensor(df['GDP_Cycle12'].values)
bb = torch.tensor(df['BBarometer_Cycle'].values)
bb_gdp = torch.tensor(df['GBP_BB3MA_minmax_interact'].values)
gdp2 = gdp**2
bb2 = bb**2

ts1, ts2, ts3, ts4, ts5 = gdp, gdp2, bb, bb2, bb_gdp
timeseries = [ts1, ts2, ts3, ts4, ts5]
y1, y2 = u_real, v_real


#### Make function here for calibration since aprameters start feeding in

# Define a scale for each parameter to ensure they are in a similar range
scale_factors = [1000, 1000, 100, 100, 10, 10, 10, 10, 10, 10]  # Adjust these factors based on your parameter magnitudes
scale_factors_model_only = scale_factors[:4]

def scale_parameters(params, scale_factors):
    return [p * s for p, s in zip(params, scale_factors)]

def unscale_parameters(params, scale_factors):
    return [p / s for p, s in zip(params, scale_factors)]

def opt_function_full(x_var_scaled, scale_factors, u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages, T_steady=10, T_smooth=200, scale=True):
    if scale:
        x_var = unscale_parameters(x_var_scaled, scale_factors)
    else:
        x_var = x_var_scaled
    delta_u, delta_v, gamma_u, gamma_v, a0, a1, a2, a3, a4, a5 = x_var
    sum_a1_to_a5 = a0 + a1 + a2 + a3 + a4 + a5
    print(delta_u, delta_v, gamma_u, gamma_v, a0, a1, a2, a3, a4, a5)

    if sum_a1_to_a5 <= 0:
        return 1000

    else:
        T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[a0, a1, a2, a3, a4, a5], T_steady=T_steady, T_smooth=T_smooth)
        
        lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, beta_u, A, d_dagger, e, u, v, wages)

        _ = lab_abm.run_model()

        u_out = 100 * lab_abm.unemployment.sum(axis=0) / L
        v_out = 100 * lab_abm.vacancies.sum(axis=0) / L

        u_out = u_out[T_steady + T_smooth:]
        v_out = v_out[T_steady + T_smooth:]

        # Ensure correct shapes and convert to numpy arrays
        assert u_out.shape == u_real.shape
        u_out_np = u_out.numpy()
        v_out_np = v_out.numpy()
        u_real_np = u_real.numpy()
        v_real_np = v_real.numpy()

        # Calculate average values of u_real and v_real
        avg_u_real = np.mean(u_real_np)

        # Mean squared error for u, normalized by average of u_real
        mse_u = np.mean((u_real_np - u_out_np) ** 2) / avg_u_real

        # Mean squared error for v, normalized by average of v_real
        mask_valid = ~np.isnan(v_real_np)
        avg_v_real = np.mean(v_real_np[mask_valid])
        mse_v = np.mean((v_real_np[mask_valid] - v_out_np[mask_valid]) ** 2) / avg_v_real

        # Regularization terms
        reg_a_sum =  (a0 + a1 + a2 + a3 + a4 + a5 - 1) ** 2  # Penalize sum(a1, a2, ..., a5) being far from 1

        # Combine objective function and regularization
        f_obj = float(mse_u + mse_v + reg_a_sum)
        print("error ", mse_u, mse_v, f_obj)

    # except Exception as e:
    #     print(f"Exception occurred: {e}")
    #     f_obj = float('inf')  # Assign a large value in case of error

    if np.isnan(f_obj):
        f_obj = float('inf')  # Assign a large value if NaN occurs

    return f_obj



def opt_function_params(x_var_scaled, scale_factors, u_real, v_real, a0, a1, a2, a3, a4, a5, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages, T_steady=10, T_smooth=200, scale=True):
    if scale:
        x_var = unscale_parameters(x_var_scaled, scale_factors)
    else:
        x_var = x_var_scaled
    delta_u, delta_v, gamma_u, gamma_v = x_var
    sum_a1_to_a5 = a0 + a1 + a2 + a3 + a4 + a5
    print(delta_u, delta_v, gamma_u, gamma_v)

    if sum_a1_to_a5 <= 0:
        return 1000

    else:
        T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[a0, a1, a2, a3, a4, a5], T_steady=T_steady, T_smooth=T_smooth)
        
        lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, beta_u, A, d_dagger, e, u, v, wages)

        _ = lab_abm.run_model()

        u_out = 100 * lab_abm.unemployment.sum(axis=0) / L
        v_out = 100 * lab_abm.vacancies.sum(axis=0) / L

        u_out = u_out[T_steady + T_smooth:]
        v_out = v_out[T_steady + T_smooth:]

        # Ensure correct shapes and convert to numpy arrays
        assert u_out.shape == u_real.shape
        u_out_np = u_out.numpy()
        v_out_np = v_out.numpy()
        u_real_np = u_real.numpy()
        v_real_np = v_real.numpy()

        # Calculate average values of u_real and v_real
        avg_u_real = np.mean(u_real_np)

        # Mean squared error for u, normalized by average of u_real
        mse_u = np.mean((u_real_np - u_out_np) ** 2) / avg_u_real

        # Mean squared error for v, normalized by average of v_real
        mask_valid = ~np.isnan(v_real_np)
        avg_v_real = np.mean(v_real_np[mask_valid])
        mse_v = np.mean((v_real_np[mask_valid] - v_out_np[mask_valid]) ** 2) / avg_v_real

        # Regularization terms
        reg_a_sum =  (a0 + a1 + a2 + a3 + a4 + a5 - 1) ** 2  # Penalize sum(a1, a2, ..., a5) being far from 1

        # Combine objective function and regularization
        f_obj = float(mse_u + mse_v + reg_a_sum)
        print("error ", mse_u, mse_v, f_obj)

    # except Exception as e:
    #     print(f"Exception occurred: {e}")
    #     f_obj = float('inf')  # Assign a large value in case of error

    if np.isnan(f_obj):
        f_obj = float('inf')  # Assign a large value if NaN occurs

    return f_obj


####

# Based on model calibrated for US
model_params_us = [0.016, 0.012, 0.16, 0.16]
# Based on Canada specific observations
candidate_params = [0.02, 0.01, 0.5, 0.5]
candidate_params_2 = [0.025, 0.02, 0.2, 0.4]
model_starting_options = [model_params_us, candidate_params]


# Using uniform across gdp and bb time series
timeseries_params_linear = [0, 0.5, 0.5, 0, 0, 0]
timeseries_params_mix = [0, 1, 0.2,0.2,0.2,0.2,0.2]
timeseries_params_mix_below = [-0.02, 0.2,0.2,0.2,0.2,0.2]
timeseries_params_linear_below = [-0.01, 0.33, 0.33, 0, 0, 0.33]

timeseries_options = [timeseries_params_linear, timeseries_params_mix, timeseries_params_mix_below, timeseries_params_linear_below ]
model_bounds =  [[0.01, 0.06], [0.003, 0.03], [0.1, 0.95], [0.1, 0.95]]
timeseries_bounds = [[-0.1, 0.1], [-1, 1], [-1, 1], [-1, 1], [-1, 1], [-1, 1]]
bounds = np.array(model_bounds  + timeseries_bounds )

starting_params_1 = model_starting_options[1] + timeseries_options[2]
# starting_params_1 = candidate_params_2 + timeseries_options[3]

# res_de = differential_evolution(opt_function_full, bounds=bounds, args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
#                                 maxiter=5, popsize=20, mutation=(0.5, 1), recombination=0.7, seed=42, x0=starting_params_1)


starting_params_1_scaled = scale_parameters(starting_params_1, scale_factors)

bounds = [(0.01, 0.05), (0.005, 0.025), (0.1, 0.95), (0.1, 0.95), 
          (-0.05, 0.05), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]

bounds_scaled = [(low * scale, high * scale) for (low, high), scale in zip(bounds, scale_factors)]

# Define constraints to enforce bounds
constraints = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 0.01},  # delta_u >= 0.01
    {'type': 'ineq', 'fun': lambda x: 0.05 - x[0]},  # delta_u <= 0.05
    {'type': 'ineq', 'fun': lambda x: x[1] - 0.005},  # delta_v >= 0.005
    {'type': 'ineq', 'fun': lambda x: 0.025 - x[1]},  # delta_v <= 0.025
    {'type': 'ineq', 'fun': lambda x: x[2] - 0.1},  # gamma_u >= 0.1
    {'type': 'ineq', 'fun': lambda x: 0.95 - x[2]},  # gamma_u <= 0.95
    {'type': 'ineq', 'fun': lambda x: x[3] - 0.1},  # gamma_v >= 0.1
    {'type': 'ineq', 'fun': lambda x: 0.95 - x[3]},  # gamma_v <= 0.95
    {'type': 'ineq', 'fun': lambda x: x[4] + 0.05},  # a0 >= -0.05
    {'type': 'ineq', 'fun': lambda x: 0.05 - x[4]},  # a0 <= 0.05
    {'type': 'ineq', 'fun': lambda x: x[5] + 1},  # a1 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[5]},  # a1 <= 1
    {'type': 'ineq', 'fun': lambda x: x[6] + 1},  # a2 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[6]},  # a2 <= 1
    {'type': 'ineq', 'fun': lambda x: x[7] + 1},  # a3 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[7]},  # a3 <= 1
    {'type': 'ineq', 'fun': lambda x: x[8] + 1},  # a4 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[8]},  # a4 <= 1
    {'type': 'ineq', 'fun': lambda x: x[9] + 1},  # a5 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 - x[9]}   # a5 <= 1
]

# Scale the constraints for COBYLA
constraints_scaled = [
    {'type': 'ineq', 'fun': lambda x: x[0] - 0.01 * scale_factors[0]},  # delta_u >= 0.01
    {'type': 'ineq', 'fun': lambda x: 0.05 * scale_factors[0] - x[0]},  # delta_u <= 0.05
    {'type': 'ineq', 'fun': lambda x: x[1] - 0.005 * scale_factors[1]},  # delta_v >= 0.005
    {'type': 'ineq', 'fun': lambda x: 0.025 * scale_factors[1] - x[1]},  # delta_v <= 0.025
    {'type': 'ineq', 'fun': lambda x: x[2] - 0.1 * scale_factors[2]},  # gamma_u >= 0.1
    {'type': 'ineq', 'fun': lambda x: 0.95 * scale_factors[2] - x[2]},  # gamma_u <= 0.95
    {'type': 'ineq', 'fun': lambda x: x[3] - 0.1 * scale_factors[3]},  # gamma_v >= 0.1
    {'type': 'ineq', 'fun': lambda x: 0.95 * scale_factors[3] - x[3]},  # gamma_v <= 0.95
    {'type': 'ineq', 'fun': lambda x: x[4] + 0.05 * scale_factors[4]},  # a0 >= -0.05
    {'type': 'ineq', 'fun': lambda x: 0.05 * scale_factors[4] - x[4]},  # a0 <= 0.05
    {'type': 'ineq', 'fun': lambda x: x[5] + 1 * scale_factors[5]},  # a1 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 * scale_factors[5] - x[5]},  # a1 <= 1
    {'type': 'ineq', 'fun': lambda x: x[6] + 1 * scale_factors[6]},  # a2 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 * scale_factors[6] - x[6]},  # a2 <= 1
    {'type': 'ineq', 'fun': lambda x: x[7] + 1 * scale_factors[7]},  # a3 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 * scale_factors[7] - x[7]},  # a3 <= 1
    {'type': 'ineq', 'fun': lambda x: x[8] + 1 * scale_factors[8]},  # a4 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 * scale_factors[8] - x[8]},  # a4 <= 1
    {'type': 'ineq', 'fun': lambda x: x[9] + 1 * scale_factors[9]},  # a5 >= -1
    {'type': 'ineq', 'fun': lambda x: 1 * scale_factors[9] - x[9]}   # a5 <= 1
]

res_cobyla_full = minimize(opt_function_full, x0=starting_params_1_scaled, 
                           args=(scale_factors, u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                           method='COBYLA', constraints=constraints_scaled, options={'maxiter': 1000})

optimal_params_full_scaled = res_cobyla_full.x
optimal_params_full = unscale_parameters(optimal_params_full_scaled, scale_factors)

print('Optimal solution (full):', optimal_params_full)
print('Objective value (full):', res_cobyla_full.fun)

# Save the results of the full optimization
params_full_df = pd.DataFrame([optimal_params_full], columns=[
    'delta_u', 'delta_v', 'gamma_u', 'gamma_v', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'
])
params_full_df.to_csv(os.path.join(path_params, 'parameters_all.csv'), index=False)

# Focus on a subset of parameters
x0_model = optimal_params_full[:4]
a0s, a1s, a2s, a3s, a4s, a5s = optimal_params_full[4:]
x0_model_scaled = scale_parameters(x0_model, scale_factors_model_only)

# Constraints for the fine-grained optimization (only the first four parameters)
constraints_model_only_scaled = constraints_scaled[:8]

res_cobyla_finegrained = minimize(opt_function_params, x0=x0_model_scaled, 
                                  args=(scale_factors_model_only, u_real, v_real, a0s, a1s, a2s, a3s, a4s, a5s, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                                  method='COBYLA', constraints=constraints_model_only_scaled, options={'maxiter': 1000})

optimal_params_finegrained_scaled = res_cobyla_finegrained.x
optimal_params_finegrained = unscale_parameters(optimal_params_finegrained_scaled, scale_factors_model_only)

# Combine fine-grained parameters with a0 to a5 for saving
optimal_params_finegrained_full = optimal_params_finegrained + [a0s, a1s, a2s, a3s, a4s, a5s]

print('Optimal solution (fine-grained):', optimal_params_finegrained)
print('Objective value (fine-grained):', res_cobyla_finegrained.fun)

# Save the results of the fine-grained optimization
params_finegrained_df = pd.DataFrame([optimal_params_finegrained_full], columns=[
    'delta_u', 'delta_v', 'gamma_u', 'gamma_v', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'
])
params_finegrained_df.to_csv(os.path.join(path_params, 'parameters_finegrained.csv'), index=False)



####### CODE ENDS HERE

#### objective value final 0.21887919326468977 (original)

#Other 0.28340218508431464
#######################3

delta_u, delta_v, gamma_u, gamma_v = optimal_params_full[:4]
a0, a1, a2, a3, a4, a5 = optimal_params_full[4:]

T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[a0, a1, a2, a3, a4, a5], T_steady=10, T_smooth=200)

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, \
                        beta_u, A, d_dagger, e, u, v, wages)

_ = lab_abm.run_model()


# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger



plt.plot(d_dagger.sum(axis=0).numpy())
plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(100*total_unemployment.numpy()[210:]/L, 100*total_vacancies.numpy()[210:]/L,'o-',label='Model')
# plt.plot(u_real, v_real, 'o-', label='real')
# plt.title('Aggregated Unemployment and Vacancies Over Time')
# plt.xlabel('unemp')
# plt.ylabel('vac')
# plt.legend()
# plt.grid(True)
# plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[210:]/L, 'o-',label='Total Unemployment')
plt.plot(u_real, 'o-', label='u real')
plt.plot(100*total_vacancies.numpy()[210:]/L,'o-' ,label='Total Vacancies')
plt.plot(v_real, 'o-', label='v_real')
# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()


#########



res_cobyla = minimize(opt_function_full, x0=starting_params_1_scaled, \
                      args=(scale_factors,u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='COBYLA', constraints=constraints, options={'maxiter': 1000})



optimal_params_scaled = res_cobyla.x
optimal_params = unscale_parameters(optimal_params_scaled, scale_factors)


print('Optimal solution:', optimal_params)
print('Objective value:', res_cobyla.fun)

x0_model = optimal_params[:4]
a0s, a1s, a2s, a3s, a4s, a5s = scale_parameters(optimal_params[4:])
x0_model_scaled = scale_parameters(x0_model, scale_factors)


res_cobyla = minimize(opt_function_params, x0=x0_model_scaled, 
                      args=(scale_factors_model_only, u_real, v_real, a0s, a1s, a2s, a3s, a4s, a5s, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='COBYLA', constraints=constraints, options={'maxiter': 1000})


optimal_params_scaled = res_cobyla.x
optimal_params = unscale_parameters(optimal_params_scaled, scale_factors_model_only)
print('Optimal solution:', optimal_params)



res_cobyla = minimize(opt_function_full, x0=starting_params_1_scaled, \
                      args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='COBYLA', constraints=constraints, options={'maxiter': 1000})

print('Optimal solution:', res_cobyla.x)
print('Objective value:', res_cobyla.fun)


res_slsqp = minimize(opt_function_full, x0=starting_params_1_scaled, \
                      args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='SLSQP', bounds=bounds_scaled, options={'maxiter': 1000})


optimal_params_scaled = res_slsqp.x
optimal_params = unscale_parameters(optimal_params_scaled, scale_factors)


print('Optimal solution:', optimal_params)
print('Objective value:', res_slsqp.fun)

x0 = res_slsqp.x






bounds = [(0.01, 0.05), (0.005, 0.025), (0.1, 0.95), (0.1, 0.95), 
          (-0.05, 0.05), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
bounds_scaled = [(low * scale, high * scale) for (low, high), scale in zip(bounds, scale_factors)]

res_slsqp = minimize(opt_function_full, x0=starting_params_1_scaled, \
                      args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='SLSQP', bounds=bounds_scaled, options={'maxiter': 1000})

optimal_params_scaled = res_slsqp.x
optimal_params = unscale_parameters(optimal_params_scaled, scale_factors)


print('Optimal solution:', optimal_params)
print('Objective value:', res_slsqp.fun)

res_slsqp.x

res_lbfgsb = minimize(opt_function_full, x0=starting_params_1, \
                      args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})

print('Optimal solution:', res_lbfgsb.x)
print('Objective value:', res_lbfgsb.fun)






len(starting_params_1)

res_cobyla = minimize(opt_function_full, x0=starting_params_1, \
                      args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='cobyla', bounds=bounds)



#####

delta_u, delta_v, gamma_u, gamma_v = optimal_params[:4]
a01, a1, a2, a3, a4, a5 = optimal_params[4:]

T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[a01, a1, a2, a3, a4, a5], T_steady=100, T_smooth=100)

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, \
                        beta_u, A, d_dagger, e, u, v, wages)

_ = lab_abm.run_model()


# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger



plt.plot(d_dagger.sum(axis=0).numpy())
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[210:]/L, 100*total_vacancies.numpy()[210:]/L,'o-',label='Model')
plt.plot(u_real, v_real, 'o-', label='real')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('unemp')
plt.ylabel('vac')
plt.legend()
plt.grid(True)
plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment.numpy()[210:]/L, 'o-',label='Total Unemployment')
plt.plot(u_real, 'o-', label='u real')
plt.plot(100*total_vacancies.numpy()[210:]/L,'o-' ,label='Total Vacancies')
plt.plot(v_real, 'o-', label='v_real')
# plt.plot(total_demand.numpy(), label='Total demand')
plt.title('Aggregated Unemployment and Vacancies Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.show()


######
# Run Model
#####

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
plt.plot(100*total_unemployment[210:].numpy()/L, 'o-',label='Total Unemployment')
plt.plot(u_real, 'o-',label='Total Unemployment')
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


####
# Evaluation function
####

n_params = 10 # 4 for deltas and gammas, 6 for time series ones. 
model_params_bounds = [[0.001, 0.05], [0.001, 0.05], [0.001, 0.99], [0.001, 0.99]]
t_s_bounds = [[-0.2,1] for i in range(6)]
bounds = np.array(model_params_bounds + t_s_bounds )



def opt_function_timeseries(x_var, u_real, v_real, delta_u, delta_v, gamma_u, gamma_v, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages, T_steady=100, T_smooth=100):

    a0, a1, a2, a3, a4, a5 = x_var

    sum_a1_to_a5 = a0 + a1 + a2 + a3 + a4 + a5

    # On case wnating to enforce D = L
    # a0, a1, a2, a3, a4, a5 = [a / sum_a1_to_a5 for a in (a0, a1, a2, a3, a4, a5)]

    # if parameters don't make sense skip evaluation
    if sum_a1_to_a5 <= 0:
        return 1000

    else:

        T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[a0, a1, a2, a3, a4, a5],\
                                                            T_steady=T_steady, T_smooth=T_smooth)
        
        lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, \
                                beta_u, A, d_dagger, e, u, v, wages,)

        # for now intermediates not needed, with better data curtail spontaneos separations by voluntary quit rate 
        # (ignore this comment for now)
        _ = lab_abm.run_model()

        # mult 100 to make it percentage
        u_out = 100 * lab_abm.unemployment.sum(axis=0)/L
        # TODO calculate the vacancy rate properly
        v_out = 100 * lab_abm.vacancies.sum(axis=0)/L

        # ignore parts that are steady state 
        u_out = u_out[T_steady + T_smooth:]
        v_out = v_out[T_steady + T_smooth:]

        assert u_out.shape == u_real.shape

        # Calculate average values of u_real and v_real
        avg_u_real = torch.mean(u_real)
        

        # Mean squared error for u, normalized by average of u_real
        mse_u = torch.mean((u_real - u_out) ** 2) / avg_u_real

        # Mean squared error for v, normalized by average of v_real
        mask_valid = ~torch.isnan(v_real)
        avg_v_real = torch.mean(v_real[mask_valid])
        mse_v = torch.mean(((v_real - v_out) ** 2)[mask_valid]) / avg_v_real
        

        # Regularization terms
        reg_a0 = a0 ** 2  # Penalize a0 being far from 0
        reg_a_sum = (a1 + a2 + a3 + a4 + a5 - 1) ** 2  # Penalize sum(a1, a2, ..., a5) being far from 1


        # Combine objective function and regularization
        f_obj = float(mse_u + mse_v + reg_a0 + reg_a_sum)

        return f_obj



####
# run F
####

# resting code runs
all_params_tofit = [delta_u, delta_v, gamma_u, gamma_v, 0,0.2,0.2,0.2,0.2,0.2]
error = opt_function_full(all_params_tofit, u_real, v_real, lam, beta_u, beta_e, A, e, u, v, \
                     L, N, sum_e_v, wages, T_steady=100, T_smooth=100)

ts_params_fit = [0,0.2,0.2,0.2,0.2,0.2]
error = opt_function_timeseries(ts_params_fit,  u_real, v_real, delta_u, delta_v, gamma_u, gamma_v, lam, beta_u, beta_e, A, e, u, v, \
                     L, N, sum_e_v, wages, T_steady=100, T_smooth=100)


### Fit timeseries only
model_params_bounds = [[0.01, 0.03], [0.008, 0.025], [0.1, 0.99], [0.01, 0.99]]
t_s_bounds = [[-0.3, 1] for i in range(5)]
# modified so that a0 is very close to 0
bounds = np.array([[-0.01, 0.01]] + t_s_bounds)


maxiter = 10
popsize = 20

# Estimate the total number of function evaluations
maxfev = maxiter * popsize * len(bounds)

# Initialize tqdm progress bar with the total number of evaluations
progress_bar = tqdm(total=maxfev, desc='Optimization Progress')

# Wrapper function to count function evaluations
def wrapped_opt_function_timeseries(x, *args):
    wrapped_opt_function_timeseries.nfev += 1
    progress_bar.update(1)
    return opt_function_timeseries(x, *args)

# Initialize evaluation counter
wrapped_opt_function_timeseries.nfev = 0

# Run differential evolution with the wrapped objective function
res_de = differential_evolution(wrapped_opt_function_timeseries, bounds=bounds, 
                                args=(u_real, v_real, delta_u, delta_v, gamma_u, gamma_v, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                                maxiter=maxiter, popsize=popsize, mutation=(0.5, 1), recombination=0.7, seed=42)

# Close the progress bar
progress_bar.close()

# Print final optimization result
print("Optimization result:", res_de)


res_de = differential_evolution(opt_function_timeseries, bounds=bounds, args=(u_real, v_real, delta_u, delta_v, gamma_u, gamma_v,\
                                                                              lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                                maxiter=10, popsize=20, mutation=(0.5, 1), recombination=0.7, seed=42)


params_test = [0.011794077746262852, 0.0248227, 0.95438, 0.370, 0.8612, 0.9866144, 0.606300, 0.678468, -0.252169, 0.6875257]


delta_u, delta_v, gamma_u, gamma_v, a0, a1, a2, a3, a4, a5 = params_test


T, d_dagger = ut.target_from_parametrized_timeseries(timeseries, e, u, parameters=[0,0.2,0.2,0.2,0.2,0.2],\
                                                    T_steady=100, T_smooth=100)

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, \
                        beta_u, A, d_dagger, e, u, v, wages)
_ = lab_abm.run_model()

np.sum([a0, a1, a2, a3, a4, a5 ])

# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


plt.plot(d_dagger.sum(axis=0))
plt.show()

plt.plot(total_unemployment/L)
plt.show()

plt.plot(total_vacancies/L)
plt.show()

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment[210:].numpy()/L, 'o-',label='Total Unemployment')

print('opt_func_call')
error = opt_function(np.array(params_test), u_real, v_real, lam, beta_u, beta_e, A, e, u, v, \
                     L, N, sum_e_v, wages, T_steady=100, T_smooth=100)


model_params_bounds = [[0.01, 0.03], [0.008, 0.025], [0.1, 0.99], [0.01, 0.99]]
t_s_bounds = [[-0.3,1] for i in range(5)]
# modified so that a0 is very close to 0
bounds = np.array(model_params_bounds + [-0.01, 0.01] + t_s_bounds )

print("Before differential_evolution call")
print("Bounds shape:", bounds.shape)

res_de = differential_evolution(opt_function, bounds=bounds, args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                                maxiter=10, popsize=20, mutation=(0.5, 1), recombination=0.7, seed=42)

print("After differential_evolution call")
print("res_de:", res_de)

# Refine with COBYLA
x0 = res_de.x

res_cobyla = minimize(opt_function_full, x0=[0.016, 0.012, 0.16, 0.16,0,0.2,0.2,0.2,0.2,0.2], \
                      args=(u_real, v_real, lam, beta_u, beta_e, A, e, u, v, L, N, sum_e_v, wages),
                      method='cobyla', bounds=bounds)



# Aggregate the data
total_unemployment = lab_abm.unemployment.sum(axis=0)  # Sum across the first dimension
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger


# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(100*total_unemployment[210:].numpy()/L, 'o-',label='Total Unemployment')
plt.plot(u_real, 'o-',label='Total Unemployment')
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

# df_gdp[df_gdp['REF_DATE'] == '2015-01-01']


df_gdp

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

N, T, L, seed, delta_u, delta_v, gamma_u, gamma_v,\
        lam, beta_u, beta_e, A, e, u, v, d_dagger, wages = ut.from_gdp_uniform_across_occ(L, N, T_steady, T_smooth, df_gdp)

lab_abm = lbm.LabourABM(N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, \
                        beta_e, beta_u, A, d_dagger, e, u, v, wages)

plt.plot(d_dagger[0, :]/10000)
plt.show()

len(df_gdp)