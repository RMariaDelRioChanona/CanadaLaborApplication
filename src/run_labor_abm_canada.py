import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from matplotlib import pylab as plt

import labor_abm as lbm
import utils as ut

# Set paths
path_data = "../data/CanadaData/"
path_fig = "../results/fig/"
path_params = "../data/parameters/"

# Set seed for reproducibility
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

# Define regions
all_regions = [
    "Alberta.a",
    "British Columbia.a",
    "Manitoba.a",
    "New Brunswick.a",
    "Newfoundland and Labrador.a",
    "Nova Scotia.a",
    "Ontario.a",
    "Ontario.b",
    "Prince Edward Island.a",
    "Quebec.a",
    "Quebec.b",
    "Saskatchewan.a",
    "National",
]

# Define region
region = "Saskatchewan.a"
region = "British Columbia.a"

# Load data
T_steady = 1000
df = pd.read_csv(path_data + "u_v_gdp_seasonal_barometer.csv")
df = df[df["REF_DATE"] >= "2015-04-01"]
df = df.iloc[:-1]

# Load parameters
delta_u, delta_v, gamma_u, gamma_v, lam, beta_u, beta_e = ut.parameters_calibrated()

# Overwrite some parameters
gamma_u = 0.12
gamma_v = 0.12

# Load occupation data
df_nodes = pd.read_csv("../data/networks/node_occ_name.csv")
dict_nodeid_title = dict(zip(df_nodes["node_id"], df_nodes["OCC_title"]))

# Generate network and scenario data
A, e, u, v, L, N, T, wages, d_dagger, times = ut.network_and_scenario(region, T_steady=T_steady, smooth=6)

# Initialize labor ABM
lab_abm = lbm.LabourABM(
    N, T, seed, delta_u, delta_v, gamma_u, gamma_v, lam, beta_e, beta_u, A, d_dagger, e, u, v, wages
)

# Run the model
_ = lab_abm.run_model()

# Aggregate data
total_unemployment = lab_abm.unemployment.sum(axis=0)
total_vacancies = lab_abm.vacancies.sum(axis=0)
total_employment = lab_abm.employment.sum(axis=0)
total_demand = lab_abm.d_dagger.sum(axis=0)
d_dagger = lab_abm.d_dagger
D_dagger = d_dagger.sum(axis=0)


# Define plotting functions
def plot_aggregate_metrics(times, D_dagger, total_unemployment, total_vacancies, L, region, T_steady):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, D_dagger.numpy()[T_steady:] / L)
    axes[0].set_title("Total Demand (in terms of 2021 labor force)", fontsize=16)
    axes[0].set_ylabel("Count", fontsize=14)
    axes[0].grid(True)

    axes[1].plot(times, 100 * total_unemployment.numpy()[T_steady:] / L)
    axes[1].set_title("Unemployment Rate Over Time", fontsize=16)
    axes[1].set_ylabel("Unemployment Rate (%)", fontsize=14)
    axes[1].grid(True)

    axes[2].plot(times, 100 * total_vacancies.numpy()[T_steady:] / L)
    axes[2].set_title("Vacancy Rate", fontsize=16)
    axes[2].set_ylabel("Vacancy Rate (%)", fontsize=14)
    axes[2].set_xlabel("Time", fontsize=14)
    axes[2].grid(True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12)

    plt.suptitle(f"Aggregate Metrics for {region}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_occupation_metrics(times, lab_abm, occ, dict_nodeid_title, region, T_steady):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(times, lab_abm.d_dagger[occ, T_steady:])
    axes[0].set_title(f"Demand for Workers in {dict_nodeid_title[occ]}", fontsize=16)
    axes[0].set_ylabel("Count", fontsize=14)
    axes[0].grid(True)

    axes[1].plot(
        times,
        lab_abm.unemployment[occ, T_steady:]
        / (lab_abm.unemployment[occ, T_steady:] + lab_abm.employment[occ, T_steady:]),
    )
    axes[1].set_title(f"Unemployment Rate in {dict_nodeid_title[occ]}", fontsize=16)
    axes[1].set_ylabel("Unemployment Rate (%)", fontsize=14)
    axes[1].grid(True)

    axes[2].plot(
        times,
        lab_abm.vacancies[occ, T_steady:] / (lab_abm.vacancies[occ, T_steady:] + lab_abm.employment[occ, T_steady:]),
    )
    axes[2].set_title(f"Vacancy Rate in {dict_nodeid_title[occ]}", fontsize=16)
    axes[2].set_ylabel("Vacancy Rate (%)", fontsize=14)
    axes[2].set_xlabel("Time", fontsize=14)
    axes[2].grid(True)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=12)

    plt.suptitle(f"Occupation Metrics for {dict_nodeid_title[occ]}\n in {region}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Plot aggregate metrics
plot_aggregate_metrics(times, D_dagger, total_unemployment, total_vacancies, L, region, T_steady)

# Plot occupation metrics for a specific occupation
# 423	423	49-9080	Wind Turbine Service Technicians
# 28	28	17-2070	Electrical and Electronics Engineers
# 56	51-8010	Power Plant Operators, Distributors, and Dispatchers
# 513	513	19-4050	Nuclear Technicians

subset_occ = [28, 56, 423, 513]

for occ in subset_occ:
    plot_occupation_metrics(times, lab_abm, occ, dict_nodeid_title, region, T_steady)
