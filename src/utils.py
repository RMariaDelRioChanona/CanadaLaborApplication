import networkx as nx
import numpy as np
import pandas as pd
import torch

from labor_abm import LabourABM

########
# functions for parameters
########


def parameters_calibrated():
    # Load parameters from the CSV file
    csv_path = "../data/parameters/parameters_finegrained.csv"
    data = pd.read_csv(csv_path)

    # Read the first row of the CSV file
    delta_u = data.loc[0, "delta_u"]
    delta_v = data.loc[0, "delta_v"]
    gamma_u = data.loc[0, "gamma_u"]
    gamma_v = data.loc[0, "gamma_v"]

    # Static values
    lam = 0.01
    beta_u = 10
    beta_e = 1

    return (
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
    )


def parameters_calibrated_inc_target_demand():
    # Load parameters from the CSV file
    csv_path = "../data/parameters/parameters_finegrained.csv"
    data = pd.read_csv(csv_path)

    # Read the first row of the CSV file
    delta_u = data.loc[0, "delta_u"]
    delta_v = data.loc[0, "delta_v"]
    gamma_u = data.loc[0, "gamma_u"]
    gamma_v = data.loc[0, "gamma_v"]
    a0 = data.loc[0, "a0"]
    a1 = data.loc[0, "a1"]
    a2 = data.loc[0, "a2"]
    a3 = data.loc[0, "a3"]
    a4 = data.loc[0, "a4"]
    a5 = data.loc[0, "a5"]

    # include parameters for target demand

    # Static values
    lam = 0.01
    beta_u = 10
    beta_e = 1

    return (delta_u, delta_v, gamma_u, gamma_v, lam, beta_u, beta_e, a0, a1, a2, a3, a4, a5)


def baseline_parameters_original_model():
    delta_u = 0.016
    delta_v = 0.012
    gamma_u = 10 * delta_u
    gamma_v = gamma_u
    lam = 0.00
    beta_u = 1
    beta_e = 1

    return (
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
    )


def baseline_parameters_jtj_multapps():
    delta_u = 0.016
    delta_v = 0.012
    gamma_u = 10 * delta_u
    gamma_v = gamma_u
    lam = 0.01
    beta_u = 10
    beta_e = 1

    return (
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
    )


def baseline_parameters_jtj_multapps_from_data():
    lam = 0.01
    beta_u = 10
    beta_e = 1

    return (
        lam,
        beta_u,
        beta_e,
    )


def baseline_parameters_jtj():
    delta_u = 0.016
    delta_v = 0.012
    gamma_u = 10 * delta_u
    gamma_v = gamma_u
    lam = 0.01
    beta_u = 1
    beta_e = 1

    return (
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
    )


###########
# Networks and employment
###########


def network_and_scenario(region, T_steady=200, smooth=3):
    adj_matrix_path = f"../data/networks/occ_mobility_fromusa.csv"
    df_path = f"../data/copper/scenario_{region.replace('.', '_')}.csv"

    # Load adjacency matrix A
    A = np.loadtxt(adj_matrix_path, delimiter=",")

    # Load new_df and convert to numpy array
    new_df = pd.read_csv(df_path)
    demand_scenario = new_df.drop(columns="OCC").values  # Drop OCC column and convert to numpy array
    # list of times (for plotting purposes)
    time_columns = [col for col in new_df.columns if col.startswith("emp")]
    times = [col.split()[1] for col in time_columns]
    times = [pd.to_datetime(col.split()[1], format="%Y-%m") for col in time_columns]
    # get n and convert to tensors
    N = A.shape[0]
    A = torch.from_numpy(A)
    demand_scenario = torch.from_numpy(demand_scenario)
    T_scenario = demand_scenario.shape[1]

    T = T_steady + T_scenario

    # now get e, u, v accordingly
    e = demand_scenario[:, 0]

    u = 0.045 * e  # 5% of e
    v = 0.02 * e  # 5% of e
    # preserve labor force
    e = e - u
    sum_e_u = e + u
    L = sum_e_u.sum()

    # make target demand so that first it is constant so it converges and then scenario
    d_dagger = torch.zeros(N, T_steady + T_scenario)

    # Expand sum_e_u for broadcasting
    sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]
    # Populate d_dagger
    d_dagger[:, :T_steady] = sum_e_u_expanded.repeat(1, T_steady)

    d_dagger[:, T_steady:] = demand_scenario

    # perform smoothing
    # Convert d_dagger to numpy array for smoothing
    d_dagger_np = d_dagger.numpy()

    # Apply rolling window smoothing with minimum points 1
    df_d_dagger = pd.DataFrame(d_dagger_np.T).rolling(window=smooth, min_periods=1).mean().T

    # Convert back to tensor
    d_dagger = torch.from_numpy(df_d_dagger.values)

    # check positive demand
    assert torch.all(d_dagger > 0), "Scenarios must not have negative demand"

    # since no data use uniform wages
    wages = torch.ones(N)

    return A, e, u, v, L, N, T, wages, d_dagger, times


def network_and_employment(file_path_name="../data/networks/edgelist_cc_mobility_merge.csv", network="merge"):
    df = pd.read_csv(file_path_name)
    dict_soc_emp = dict(zip(df["OCC_target"], df["TOT_EMP_OCC"]))

    if network == "merge":
        df = df.rename({"trans_merge_alpha05": "weight"}, axis="columns")
        G = nx.from_pandas_edgelist(df, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph())
    elif network == "cc":
        df = df.rename({"trans_prob_cc": "weight"}, axis="columns")
        G = nx.from_pandas_edgelist(df, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph())

    # Note that nodes are ordered differently with network x
    nodes_order = list(G.nodes)
    [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    print("nodes dropped ", set(nodes_order).difference(largest_cc))
    S = [G.subgraph(c).copy() for c in nx.strongly_connected_components(G)]
    G_strongly = S[0]
    nodes_order = list(G_strongly.nodes)
    assert nx.is_strongly_connected(G_strongly)

    ### employment part
    e = torch.tensor([dict_soc_emp[node] for node in nodes_order])
    # initiliaze with values a bit below unemp and vacancy rate (since delta's will increase uand v a bit)
    u = 0.045 * e  # 5% of e
    v = 0.02 * e  # 5% of e

    # still preserve total labor force
    e = e - u
    sum_e_u = e + u
    L = sum_e_u.sum()

    A = nx.adjacency_matrix(G_strongly, weight="weight").todense()
    A = np.array(A)
    n = A.shape[0]

    A = torch.from_numpy(A)

    wages = torch.ones(n)

    return A, e, u, v, L, n, sum_e_u, wages


##################
# Target demand
##################


def from_timeseries_to_tensors(df, col_names=["cycle12"]):
    """read dataframe and output the tensors of time series with signals"""

    col_name1, col_name2, col_name3, col_name4, col_name5 = col_names[:5]

    ts1 = torch.tensor(df[col_name1].values)
    ts2 = torch.tensor(df[col_name2].values)
    ts3 = torch.tensor(df[col_name3].values)
    ts4 = torch.tensor(df[col_name4].values)
    ts5 = torch.tensor(df[col_name5].values)

    return ts1, ts2, ts3, ts4, ts5


def target_from_gdp_e_u(df_gdp, e, u, col_name="cycle12", L=20000, N=2, T_steady=90, T_smooth=10):

    sum_e_u = e + u
    # getting cycle shape and length
    cycle = torch.tensor(df_gdp[col_name])
    T_cycle = len(cycle)
    # getting the first value of cycle for smoothing (ease into cycle)
    ini_val_cycle = cycle[0]
    dif_1 = ini_val_cycle - 1
    increment = dif_1 / T_smooth
    smoothing = torch.tensor([1 + increment * n_s for n_s in range(T_smooth)])

    d_dagger = torch.zeros(N, T_steady + T_smooth + T_cycle)
    # Expand sum_e_u for broadcasting
    sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]

    # Expand smoothing for each of the 534 units
    smoothing_expanded = smoothing.unsqueeze(0).repeat(N, 1)  # shape becomes [534, 10]

    # Populate d_dagger
    d_dagger[:, :T_steady] = sum_e_u_expanded.repeat(1, T_steady)  # Steady state
    d_dagger[:, T_steady : T_steady + T_smooth] = sum_e_u_expanded * smoothing_expanded  # Smoothing transition
    d_dagger[:, T_steady + T_smooth :] = sum_e_u_expanded * cycle.unsqueeze(0).repeat(N, 1)  # Apply cycle

    T = T_steady + T_cycle + T_smooth

    wages = torch.rand([1, 1])

    return T, d_dagger


def target_from_gdp_e_u_and_shock(df_gdp, df_shock, e, u, col_name="cycle12", L=20000, N=2, T_steady=90, T_smooth=10):

    sum_e_u = e + u
    # getting cycle shape and length
    cycle = torch.tensor(df_gdp[col_name])
    T_cycle = len(cycle)
    # getting the first value of cycle for smoothing (ease into cycle)
    ini_val_cycle = cycle[0]
    dif_1 = ini_val_cycle - 1
    increment = dif_1 / T_smooth
    smoothing = torch.tensor([1 + increment * n_s for n_s in range(T_smooth)])

    d_dagger = torch.zeros(N, T_steady + T_smooth + T_cycle)
    # Expand sum_e_u for broadcasting
    sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]

    # Expand smoothing for each of the 534 units
    smoothing_expanded = smoothing.unsqueeze(0).repeat(N, 1)  # shape becomes [534, 10]

    # Populate d_dagger
    d_dagger[:, :T_steady] = sum_e_u_expanded.repeat(1, T_steady)  # Steady state
    d_dagger[:, T_steady : T_steady + T_smooth] = sum_e_u_expanded * smoothing_expanded  # Smoothing transition
    d_dagger[:, T_steady + T_smooth :] = sum_e_u_expanded * cycle.unsqueeze(0).repeat(N, 1)  # Apply cycle

    T = T_steady + T_cycle + T_smooth

    wages = torch.rand([1, 1])

    return T, d_dagger


def target_from_parametrized_timeseries(timeseries, e, u, parameters=[0, 1, 1, 1, 1, 1], T_steady=90, T_smooth=10):
    """Get pytorch tensors that are timeserieswith canada indicators to compute target demand. First a constant one + smoothing into
    the actual signals
    e and u are pytorch tensors with initial employment used also to split target demand accoridngly
    """
    N = e.shape[0]
    sum_e_u = e + u
    # getting cycle shape and length
    ts1, ts2, ts3, ts4, ts5 = timeseries
    T_cycle = len(ts1)
    a0, a1, a2, a3, a4, a5 = parameters

    cycle = a0 + a1 * ts1 + a2 * ts2 + a3 * ts3 + a4 * ts4 + a5 * ts5
    # getting the first value of cycle for smoothing (ease into cycle)
    ini_val_cycle = cycle[0]
    dif_1 = ini_val_cycle - 1
    increment = dif_1 / T_smooth
    smoothing = torch.tensor([1 + increment * n_s for n_s in range(T_smooth)])

    d_dagger = torch.zeros(N, T_steady + T_smooth + T_cycle)
    # Expand sum_e_u for broadcasting
    sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]

    # Expand smoothing for each of the 534 units
    smoothing_expanded = smoothing.unsqueeze(0).repeat(N, 1)  # shape becomes [534, T_smooth]

    # Populate d_dagger
    d_dagger[:, :T_steady] = sum_e_u_expanded.repeat(1, T_steady)  # Steady state

    d_dagger[:, T_steady : T_steady + T_smooth] = sum_e_u_expanded * smoothing_expanded  # Smoothing transition
    d_dagger[:, T_steady + T_smooth :] = sum_e_u_expanded * cycle.unsqueeze(0).repeat(N, 1)  # Apply cycle

    T = T_steady + T_cycle + T_smooth
    return T, d_dagger


################
# Networks
################


def network_and_employment_fromadj(
    file_adj="../data/networks/occupational_mobility_network.csv",
    file_emp="../data/networks/ipums_employment_2016.csv",
):
    df = pd.read_csv(file_emp)
    e = torch.tensor(df["IPUMS_CPS_av_monthly_employment_whole_period"])
    A = np.genfromtxt(file_adj, delimiter=",")

    u = 0.016 * e  # 0.0463 * e  # 5% of e
    v = 0.012 * e  # 0.019 * e  # 5% of e
    # u = 10*(0.0001*e)**2/e
    # v = 10*(0.0001*e)**2/e

    # get lab force
    sum_e_u = e + u
    L = sum_e_u.sum()

    n = A.shape[0]

    A = torch.from_numpy(A)

    # print("A[3, 5]" ,A[3, 5])
    # print( "e[5]", e[5])

    wages = torch.ones(n)

    return A, e, u, v, L, n, sum_e_u, wages


# def network_and_employment(file_path_name="../data/networks/edgelist_cc_mobility_merge.csv",\
#                            network="merge"):
#     df = pd.read_csv(file_path_name)
#     dict_soc_emp = dict(zip(df["OCC_target"], df["TOT_EMP_OCC"]))

#     if network == "merge":
#         df = df.rename({"trans_merge_alpha05": "weight"}, axis="columns")
#         G = nx.from_pandas_edgelist(
#         df, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph()
#         )
#     elif network == "cc":
#         df = df.rename({"trans_prob_cc": "weight"}, axis="columns")
#         G = nx.from_pandas_edgelist(
#         df, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph()
#         )

#     # Note that nodes are ordered differently with network x
#     nodes_order = list(G.nodes)
#     [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]
#     largest_cc = max(nx.strongly_connected_components(G), key=len)
#     print("nodes dropped ", set(nodes_order).difference(largest_cc))
#     S = [G.subgraph(c).copy() for c in nx.strongly_connected_components(G)]
#     G_strongly = S[0]
#     nodes_order = list(G_strongly.nodes)
#     assert(nx.is_strongly_connected(G_strongly))
#     print(dict_soc_emp)

#     ### employment part
#     e = torch.tensor([dict_soc_emp[node] for node in nodes_order])
#     # u = 0.08 * e#0.0463 * e  # 5% of e
#     # v = 0.05 * e #0.019 * e  # 5% of e
#     u = 10*(0.0001*e)**2/e
#     v = 10*(0.0001*e)**2/e


#     # still preserve total labor force
#     e = e - u
#     sum_e_u = e + u
#     L = sum_e_u.sum()


#     A = nx.adjacency_matrix(G_strongly, weight="weight").todense()
#     A = np.array(A)
#     n = A.shape[0]


#     A = torch.from_numpy(A)

#     wages = torch.ones(n)

#     return A, e, u, v, L, n, sum_e_u, wages


def set_d_dagger_uniform(T, sum_e_u):
    d_dagger = sum_e_u.unsqueeze(1).repeat(1, T)
    return d_dagger


def employment_merge_mob_cc(file_path_name="../data/networks/edgelist_cc_mobility_merge_joris.csv"):
    df = pd.read_csv(file_path_name)
    dict_soc_emp = dict(zip(df["OCC_target"], df["TOT_EMP_OCC"]))


def career_changers():
    df = pd.read_csv("../data/networks/career_changers_mobility_edgelist.csv")
    # df_emp =
    # df_cc = pd.read_csv(path_data +file_cc )
    # df_emp = pd.read_csv(path_emp + file_employment)
    #  make dictionaries at different levels
    dict_soc_emp = dict(zip(df_emp["OCC_CODE"], df_emp["TOT_EMP"]))
    dict_soc_emp_major = dict(zip(df_emp["OCC_CODE"], df_emp["TOT_EMP_major"]))
    dict_soc_emp_minor = dict(zip(df_emp["OCC_CODE_minor"], df_emp["TOT_EMP_minor"]))
    df_cc["TOT_EMP"] = df_cc["OCC_source"].map(dict_soc_emp)

    df_cc[np.isnan(df_cc["TOT_EMP"])]
    df_cc = df_cc.rename({"trans_prob_cc": "weight"}, axis="columns")
    G = nx.from_pandas_edgelist(df_cc, "OCC_source", "OCC_target", edge_attr="weight", create_using=nx.DiGraph())
    # Note that nodes are ordered differently with network x
    nodes_order = list(G.nodes)
    nx.is_strongly_connected(G)
    [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    print("nodes dropped ", set(nodes_order).difference(largest_cc))
    S = [G.subgraph(c).copy() for c in nx.strongly_connected_components(G)]
    G_strongly = S[0]
    nodes_order = list(G_strongly.nodes)
    A_cc = nx.adjacency_matrix(G_strongly, weight="weight").todense()
    A_cc = np.array(A_cc)
    n = A_cc.shape[0]


def test_2node_scenario_complete():
    N = 2
    T = 1000
    L = 2000
    seed = 111
    delta_u = 0.016
    delta_v = 0.012
    gamma_u = 10 * delta_u
    gamma_v = gamma_u
    lam = 0.01
    beta_u = 10
    beta_e = 1

    A = test_2_nodes_complete()
    e = torch.tensor([1000, 1000])
    # u = 0.05 * e  # 5% of e
    # v = 0.05 * e  # 5% of e
    u = 0.0463 * e  # 5% of e
    v = 0.019 * e  # 5% of e
    # still preserve total labor force
    e = e - u
    # Create d_dagger using broadcasting
    sum_e_u = e + u
    d_dagger = sum_e_u.unsqueeze(1).repeat(1, T)

    wages = torch.rand([1, 1])

    return (
        N,
        T,
        L,
        seed,
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
        A,
        e,
        u,
        v,
        d_dagger,
        wages,
    )


def target_from_gdp_e_u(df_gdp, e, u, col_name="cycle12", L=20000, N=2, T_steady=90, T_smooth=10):
    sum_e_u = e + u
    # getting cycle shape and length
    cycle = torch.tensor(df_gdp[col_name])
    T_cycle = len(cycle)
    # getting the first value of cycle for smoothing (ease into cycle)
    ini_val_cycle = cycle[0]
    dif_1 = ini_val_cycle - 1
    increment = dif_1 / T_smooth
    smoothing = torch.tensor([1 + increment * n_s for n_s in range(T_smooth)])

    # d_dagger = torch.zeros([N, T_steady + T_smooth + T_cycle])
    # # repeting constant employment across time
    # d_dagger[:, :T_steady] = sum_e_u.unsqueeze(1).repeat(1, T_steady)
    # # now emp (emp per occ) smoothing
    # d_dagger[:, T_steady : T_steady + T_smooth] = sum_e_u * smoothing.unsqueeze(0).repeat(N, 1)
    # # now constant emp (emp per occ) cycle
    # d_dagger[:, T_steady + T_smooth :] = sum_e_u * cycle.unsqueeze(0).repeat(N, 1)

    d_dagger = torch.zeros(N, T_steady + T_smooth + T_cycle)
    # Expand sum_e_u for broadcasting
    sum_e_u_expanded = sum_e_u.unsqueeze(1)  # shape becomes [534, 1]

    # Expand smoothing for each of the 534 units
    smoothing_expanded = smoothing.unsqueeze(0).repeat(N, 1)  # shape becomes [534, 10]

    # Populate d_dagger
    d_dagger[:, :T_steady] = sum_e_u_expanded.repeat(1, T_steady)  # Steady state
    d_dagger[:, T_steady : T_steady + T_smooth] = sum_e_u_expanded * smoothing_expanded  # Smoothing transition
    d_dagger[:, T_steady + T_smooth :] = sum_e_u_expanded * cycle.unsqueeze(0).repeat(N, 1)  # Apply cycle

    T = T_steady + T_cycle + T_smooth

    A = test_2_nodes_complete()

    wages = torch.rand([1, 1])

    return T, d_dagger


def from_gdp_uniform_across_occ(L, N, T_steady, T_smooth, df_gdp, col_name="cycle12"):
    emp_per_occ = L / N
    # Use col_name='cycle_bb_mix' for using also business barometer
    # at some point this should be update to non uniform distribution
    e = torch.tensor([emp_per_occ for i in range(N)])
    u = 0.0463 * e  # 5% of e
    v = 0.019 * e  # 5% of e
    e = e - u
    sum_e_u = e + u
    # getting cycle shape and length
    cycle = torch.tensor(df_gdp[col_name])
    T_cycle = len(cycle)
    # getting the first value of cycle for smoothing (ease into cycle)
    ini_val_cycle = cycle[0]
    dif_1 = ini_val_cycle - 1
    increment = dif_1 / T_smooth
    smoothing = torch.tensor([1 + increment * n_s for n_s in range(T_smooth)])

    d_dagger = torch.zeros([N, T_steady + T_smooth + T_cycle])
    # repeting constant employment across time
    d_dagger[:, :T_steady] = sum_e_u.unsqueeze(1).repeat(1, T_steady)
    # now emp (emp per occ) smoothing
    d_dagger[:, T_steady : T_steady + T_smooth] = emp_per_occ * smoothing.unsqueeze(0).repeat(N, 1)
    # now constant emp (emp per occ) cycle
    d_dagger[:, T_steady + T_smooth :] = emp_per_occ * cycle.unsqueeze(0).repeat(N, 1)

    T = T_steady + T_cycle + T_smooth
    seed = 111
    delta_u = 0.016
    delta_v = 0.012
    gamma_u = 10 * delta_u
    gamma_v = gamma_u
    lam = 0.01
    beta_u = 10
    beta_e = 1

    A = test_2_nodes_complete()

    wages = torch.rand([1, 1])

    return (
        N,
        T,
        L,
        seed,
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
        A,
        e,
        u,
        v,
        d_dagger,
        wages,
    )


def from_gdp_uniform_across_occ_nodeltasgammas(df_gdp, col_name="cycle12", L=20000, N=2, T_steady=10, T_smooth=10):
    emp_per_occ = L / N
    # at some point this should be update to non uniform distribution
    e = torch.tensor([emp_per_occ for i in range(N)])
    u = 0.0463 * e  # 5% of e
    v = 0.019 * e  # 5% of e
    e = e - u
    sum_e_u = e + u
    # getting cycle shape and length
    cycle = torch.tensor(df_gdp[col_name])
    T_cycle = len(cycle)
    # getting the first value of cycle for smoothing (ease into cycle)
    ini_val_cycle = cycle[0]
    dif_1 = ini_val_cycle - 1
    increment = dif_1 / T_smooth
    smoothing = torch.tensor([1 + increment * n_s for n_s in range(T_smooth)])

    d_dagger = torch.zeros([N, T_steady + T_smooth + T_cycle])
    # repeting constant employment across time
    d_dagger[:, :T_steady] = sum_e_u.unsqueeze(1).repeat(1, T_steady)
    # now emp (emp per occ) smoothing
    d_dagger[:, T_steady : T_steady + T_smooth] = emp_per_occ * smoothing.unsqueeze(0).repeat(N, 1)
    # now constant emp (emp per occ) cycle
    d_dagger[:, T_steady + T_smooth :] = emp_per_occ * cycle.unsqueeze(0).repeat(N, 1)

    T = T_steady + T_cycle + T_smooth
    seed = 111
    lam = 0.01
    beta_u = 10
    beta_e = 1

    A = test_2_nodes_complete()

    wages = torch.rand([1, 1])

    return N, T, L, seed, lam, beta_u, beta_e, A, e, u, v, d_dagger, wages


def create_symmetric_A_euv_d_daggers(n, t, l, seed=None):
    """Create a random symmetric matrix with 1 on the diagonal and entries between 0 and 1,
    and create vectors e, u, v, and matrix d_dagger where each column is e + u.
    """
    # Set the seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Create a symmetric matrix
    random_matrix = torch.rand(n, n)
    symmetric_matrix = 0.5 * (random_matrix + random_matrix.T)
    torch.diagonal(symmetric_matrix).fill_(1)  # Ensure the diagonal is set to 1

    # Create vectors e, u, and v
    e = torch.rand(n) * l
    u = 0.05 * e  # 5% of e
    v = 0.05 * e  # 5% of e

    # Create d_dagger using broadcasting
    sum_e_u = e + u
    d_dagger = sum_e_u.unsqueeze(1).repeat(1, t)  # Repeat the sum across t columns

    wages = torch.rand(n)

    return symmetric_matrix, e, u, v, d_dagger, wages


def test_2_nodes():
    toy_adjacency = [[0.8, 0.2], [0.3, 0.7]]
    return torch.tensor(toy_adjacency)


def test_2node_scenario():
    N = 2
    T = 3000
    L = 2000
    seed = 111
    delta_u = 0.016
    delta_v = 0.012
    gamma_u = 10 * delta_u
    gamma_v = gamma_u
    lam = 0.01
    beta_u = 10
    beta_e = 1

    A = test_2_nodes()
    e = torch.tensor([1200, 800])
    u = 0.05 * e  # 5% of e
    v = 0.05 * e  # 5% of e
    # u = 0.0451 * e  # 5% of e
    # v = 0.0186 * e  # 5% of e
    # still preserve total labor force
    e = e - u
    # Create d_dagger using broadcasting
    sum_e_u = e + u
    d_dagger = sum_e_u.unsqueeze(1).repeat(1, T)

    wages = torch.rand([1, 1])

    return (
        N,
        T,
        L,
        seed,
        delta_u,
        delta_v,
        gamma_u,
        gamma_v,
        lam,
        beta_u,
        beta_e,
        A,
        e,
        u,
        v,
        d_dagger,
        wages,
    )


def test_2_nodes_complete():
    toy_adjacency = [[0.5, 0.5], [0.5, 0.5]]
    return torch.tensor(toy_adjacency)


########
# functions for calibration
########


def generate_seeds(initial_seed, n_samples):
    np.random.seed(initial_seed)  # Seed the random number generator
    seeds = np.random.randint(0, 2**32 - 1, size=n_samples)  # Generate n_samples random seeds
    return seeds


def sample_delta_gammas():
    delta_u = np.random.uniform(0.01, 0.1)
    delta_v = np.random.uniform(0.005, 0.05)
    gamma_u = np.random.uniform(delta_u * 5, 1)
    gamma_v = np.random.uniform(delta_v * 5, 1)

    return delta_u, delta_v, gamma_u, gamma_v


###### TODO, make a function that returns baseline params
##### TODO make a function that reads gpd/baromets/etc signal and makes
# a smoothed target demand


def sample_target_demand(df_gdp, column="cycle12", seed=123):
    # TODO remove seed set
    torch.seed(seed)
    cycle = torch.tensor(df_gdp[column])
    n = len(cycle)
    cycle_new = torch.tensor(n)
    for i in range(n):
        cycle_new[n] = cycle[n] + torch.random()


def time_series_l2_diff(series_data, series_model):
    # mean squared error between time series
    return torch.sqrt(torch.sum((series_data - series_model) ** 2))


def batch_time_series_l2_diff(series_batch_data, series_batch_model):
    # mean squared error of multiple time series (given as n_timeseries, n_time) tensor
    return torch.sqrt(torch.sum((series_batch_data - series_batch_model) ** 2, dim=1))


def time_series_mse(series_data, series_model):
    # Convert pandas series to torch tensors, assuming series_data might contain NaNs
    series_data_tensor = torch.tensor(series_data.values, dtype=torch.float32)[:-1]
    series_model_tensor = series_model[20:-1]
    # print(len(series_data_tensor))
    # print(len(series_model_tensor))

    # Create a mask that is True for non-NaN entries and False for NaN entries
    mask = ~torch.isnan(series_data_tensor)

    # Apply the mask to filter out NaNs from both the data and the model predictions
    series_data_filtered = series_data_tensor[mask]
    series_model_filtered = series_model_tensor[mask]

    # Compute the mean squared error, normalizing by the number of valid (non-NaN) data points
    mse = torch.mean((series_data_filtered - series_model_filtered) ** 2)

    if mse.isnan():
        print(series_model_tensor)
        print(series_data_filtered)

    return mse


def calibrate_delta_gamma(df_gdp, n_samples, seed, threshold):
    # start wtih say threshold = 0.01
    n_seeds = generate_seeds(seed, n_samples)
    unemployment_rate_data = df_gdp["unemployment"]
    vacancy_rate_data = df_gdp["vacancy"]
    accepted_parameters = []
    tested_params = []
    mse_list = []
    mse_av = []

    for s in n_seeds:
        (
            N,
            T,
            L,
            seed,
            lam,
            beta_u,
            beta_e,
            A,
            e,
            u,
            v,
            d_dagger,
            wages,
        ) = from_gdp_uniform_across_occ_nodeltasgammas(df_gdp)

        delta_u, delta_v, gamma_u, gamma_v = sample_delta_gammas()

        lab_abm = LabourABM(
            N,
            T,
            s,
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

        lab_abm.run_model()

        (
            unemployment_rate,
            employment_rate,
            vacancy_rate,
            sep_ratio,
            vac_ratio,
            jtj_over_utj,
        ) = lab_abm.calculate_rates()

        mse_u = time_series_mse(unemployment_rate_data, unemployment_rate)
        mse_v = time_series_mse(vacancy_rate_data, vacancy_rate)
        mse_list.append([mse_u, mse_v])
        mse_av.append(0.5 * (mse_u + mse_v))
        tested_params.append([delta_u, delta_v, gamma_u, gamma_v])

        if mse_u < threshold and mse_v < threshold:
            accepted_parameters.append([delta_u, delta_v, gamma_u, gamma_v])
            mse_list.append([mse_u, mse_v])

    return accepted_parameters, tested_params, mse_list, mse_av
