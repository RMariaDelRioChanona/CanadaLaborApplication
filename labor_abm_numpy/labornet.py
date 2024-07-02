"""Network Labor Model
@rmaria del rio-chanona
"""

import copy
from sys import argv

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import Delaunay
from shapely.geometry import Polygon

############################
# Code for running the agent-based model (solving the analytical equations)
# for running the agent-based model without approximations see code in Julia
############################


def matching_probability(sij, vj):
    """probability of an application sent to occupation j is successful"""
    # get supply of workers for j
    sj = np.sum(sij, axis=0)
    sj_inv = np.array([1.0 / s for s in sj])
    # get labor market tightness of an occupation
    θj = np.multiply(vj, sj_inv)
    with np.errstate(divide="ignore", invalid="ignore"):
        θj = np.multiply(vj, sj_inv)
        θj_inv = np.array([1.0 / θ for θ in θj])
    θj_inv[np.isnan(θj_inv)] = 0
    θj[np.isnan(θj)] = 0
    # get probability of application to occupation j being succesfull
    pj = np.multiply(θj, 1 - np.exp(-θj_inv))
    return pj


def matching_probability_multiple_apps(sij, sij_e, sij_u, vj, β_u, β_e, Q):
    """probability of an application sent to occupation j is successful
    AND accepted by the applicant
    """
    # get supply of workers for j
    sj = np.sum(sij, axis=0)
    sj_inv = np.array([1.0 / s for s in sj])
    # get labor market tightness of an occupation
    θj = np.multiply(vj, sj_inv)
    θj_inv = np.array([1.0 / θ for θ in θj])
    # get probability of application to occupatoin j being succesfull
    pj = np.multiply(θj, 1 - np.exp(-θj_inv))
    # get probability an offer is received by worker in occupation i
    ρi = np.sum(np.multiply(Q, pj), axis=1)

    # get probability a worker receives >=1 job offer
    apps_array_u = np.array([i for i in range(1, β_u + 1)])
    apps_array_e = np.array([i for i in range(1, β_u + 1)])
    # NOTE by passing apps_array_u/e as argument can also increase speed, but may be negligible
    # Old lines below. TODO delete.
    # hi_u = np.sum([stats.binom.pmf(l, β_u, ρi) for l in range(1, β_u+1)], axis = 0)
    # hi_e = np.sum([stats.binom.pmf(l, β_e, ρi) for l in range(1, β_e+1)], axis = 0)
    # hi_u_inv = np.array([1./h for h in hi_u])
    # hi_e_inv = np.array([1./h for h in hi_e])
    binomial_u = stats.binom.pmf(apps_array_u[:, None], β_u, ρi)
    binomial_e = stats.binom.pmf(apps_array_e[:, None], β_e, ρi)

    hi_u = np.sum(binomial_u, axis=0)
    hi_e = np.sum(binomial_e, axis=0)
    hi_u_inv = 1.0 / hi_u
    hi_e_inv = 1.0 / hi_e

    # get probability a worker accepts offer in occupation i
    #     ϕi_u = np.sum(np.multiply(hi_u_inv, [(1/l)*stats.binom.pmf(l, β_u, ρi) for l in range(1, β_u+1)]), axis = 0)
    #     ϕi_e = np.sum(np.multiply(hi_e_inv, [(1/l)*stats.binom.pmf(l, β_e, ρi) for l in range(1, β_e+1)]), axis = 0)
    ϕi_u = np.sum(np.multiply(hi_u_inv, (1.0 / apps_array_u)[:, None] * binomial_u), axis=0)
    ϕi_e = np.sum(np.multiply(hi_e_inv, (1.0 / apps_array_e)[:, None] * binomial_e), axis=0)

    # get probability a vacancy with >=1 application is filled
    tempj_u = np.sum(np.multiply(ϕi_u, sij_u), axis=0)
    tempj_e = np.sum(np.multiply(ϕi_e, sij_e), axis=0)
    ψj = np.multiply(sj_inv, (tempj_u + tempj_e))  ## CHANGE TEMP NAMES

    # get probability of application to occupation j being succesful AND accepted
    ppj = np.multiply.reduce((θj, 1 - np.exp(-θj_inv), ψj))
    return ppj


def fire_and_hire_workers(parameters, variables, d_dagger_t, A, matching_probability):
    """
    Function that updates the number of employed and unemployed workers and
    job vacancies in each occupation. The update considers the spontanous
    separations and opening of vacancies, as well as the directed separations
    and openings and flow of workers.
    Note that, when the demand shock is too large additional vacancies may
    be closed.
    Args:
        parameters(list): δ_u, δ_v, γ_u, γ_v (floats)
        variables (list): with employment, u, v (np arrays) with employment
            unemployment and vacancies for each occupation
        d_dagger_t(np arrays):target demand for each occupation
        A(np arrays): adjacency matrix
        matching_probability(function): probability of app in j being succesfull
    """
    assert len(parameters) == 4, "wrong number of parameters"
    δ_u, δ_v, γ_u, γ_v = parameters

    employment, u, v = variables
    print("e5, u5, v5 ", employment[5], u[5], v[5])
    n = len(employment)
    # spontanous separations and job opening
    separated_workers = δ_u * employment
    opened_vacancies = δ_v * employment
    # state dependent separations and job openings (directed effort)
    # compute difference between realized and target demand
    Δ_demand = employment + v + -d_dagger_t
    # get extra separations and openings, bounded by employment of occupation
    g_u = np.minimum(employment, γ_u * np.maximum(np.zeros(n), Δ_demand))
    g_v = np.minimum(employment, γ_v * np.maximum(np.zeros(n), -Δ_demand))
    # get number of separations and job openings
    separated_workers = separated_workers + (1 - δ_u) * g_u
    opened_vacancies = opened_vacancies + (1 - δ_v) * g_v

    print("spon sep ", separated_workers[5])
    print("state sep ", (1 - δ_u) * g_u[5])
    # Search and matching
    Av = np.multiply(v, A)
    # get matrix q_ij, probability of applying from i to j
    Q = Av / np.sum(Av, axis=1, keepdims=1)
    print("Q[3,5]", Q[3, 5])
    # expected number of job applications from i to j
    sij = np.multiply(u[:, None], Q)

    print("sij 3,5", sij[3, 5])
    sj = (sij).sum(axis=0)
    print("sj 5", sj[5])
    # expected probability of application being succesfull
    pj = matching_probability(sij, v)
    # expected flow of workers
    print("job offers ", pj[5])
    F = np.multiply(sij, pj)  # sij .* pj
    print("F[3,5], ", F[3, 5])

    # getting hired workers and transitioning workers
    hired_workers = np.sum(F, axis=0)
    exported_workers = np.sum(F, axis=1)

    # print("vacancies ", v)
    # print("expected applications ", sij.sum(axis=0))
    # print("matches ", hired_workers)
    # print("outflow unemployment", hired_workers.sum())
    # print("separations ", separated_workers.sum())
    # print("open vac", opened_vacancies.sum())
    # print("sep ", separated_workers)
    # print("exp ", exported_workers)
    # print("u ", u)

    # check everything okay with code
    assert min(hired_workers) >= 0
    assert min(exported_workers) >= 0

    # print("unemployed = ",u.sum())
    # print("vacancies = ",v.sum())
    # print("probabilities = ",pj.sum())
    # print("hired workers = ",hired_workers.sum())
    # print("exported_workers = ",exported_workers.sum())
    print("got employed Fij_u.sum(axis=0)[5]", exported_workers[5])
    print("sep workers[5]", separated_workers[5])
    print("u[5]", u[5])

    # Update e, w and v
    u += separated_workers - exported_workers
    employment += hired_workers - separated_workers
    v += opened_vacancies - hired_workers

    # make sure variables are postive (accounting for floating point error)
    # assert(minimum(v) >= 0.0 || isapprox(v, zeros(n); atol=1e-15, rtol=0))
    # assert(minimum(u) >= 0.0 || isapprox(u, zeros(n); atol=1e-15, rtol=0))
    # assert(minimum(employment) >= 0.0 || isapprox(employment,
    #         zeros(n); atol=1e-15, rtol=0))
    # now that we know variables are non negative, correct floating point error
    v = np.maximum(1e-15 * np.ones(n), v)
    u = np.maximum(1e-15 * np.ones(n), u)
    employment = np.maximum(1e-15 * np.ones(n), employment)

    variables = [employment, u, v]
    changes = [separated_workers, exported_workers]
    return variables, changes


def fire_and_hire_workers_onjobsearch(parameters, employment, u, v, d_dagger_t, A, matching_probability):
    """
    Function that updates the number of employed and unemployed workers and
    job vacancies in each occupation. The update considers the spontanous
    separations and opening of vacancies, as well as the directed separations
    and openings and flow of workers.
    Note that, when the demand shock is too large additional vacancies may
    be closed.
    Args:
        δ_u, δ_v, γ_u, γ_v, λ (floats): parameters
        employment, u, v, d_dagger_t(np arrays): arrays with the employment
            unemployment, vacancies and target demand for each occupation
        A(np arrays): adjacency matrix
        matching_probability(function): probability of app in j being succesfull
    """
    assert len(parameters) == 5, "wrong number of parameters"
    δ_u, δ_v, γ_u, γ_v, λ = parameters

    employment, u, v = variables
    n = len(employment)
    # spontanous separations and job opening
    separated_workers = δ_u * employment
    opened_vacancies = δ_v * employment
    # state dependent separations and job openings (directed effort)
    # compute difference between realized and target demand
    Δ_demand = employment + v + -d_dagger_t
    # get extra separations and openings, bounded by employment of occupation
    g_u = np.minimum(employment, γ_u * np.maximum(np.zeros(n), Δ_demand))
    g_v = np.minimum(employment, γ_v * np.maximum(np.zeros(n), -Δ_demand))
    # get number of separations and job openings
    separated_workers = separated_workers + (1 - δ_u) * g_u
    opened_vacancies = opened_vacancies + (1 - δ_v) * g_v

    # Search and matching
    Av = np.multiply(v, A)
    # get matrix q_ij, probability of applying from i to j
    Q = Av / np.sum(Av, axis=1, keepdims=1)
    # expected number of job applications from i to j
    sij_u = np.multiply(u[:, None], Q)  # as a results of unemployed workers
    sij_e = λ * np.multiply(employment[:, None], Q)  # as a result of employed wrkrs
    sij = sij_u + sij_e
    # expected probability of application being succesfull
    pj = matching_probability(sij, v)
    # expected flow of workers
    F_u = np.multiply(sij_u, pj)  # sij .* pj
    F_e = np.multiply(sij_e, pj)

    # getting hired workers and transitioning workers
    # previously unemployed workers
    hired_workers_u = np.sum(F_u, axis=0)
    exported_workers_u = np.sum(F_u, axis=1)
    # previously employed workers
    hired_workers_e = np.sum(F_e, axis=0)
    exported_workers_e = np.sum(F_e, axis=1)
    # check everything okay with code
    assert min(hired_workers_u) >= 0
    assert min(exported_workers_u) >= 0
    assert min(hired_workers_e) >= 0
    assert min(exported_workers_e) >= 0
    exported_workers = esported_workers_u + exported_workers_e

    # Update e, w and v
    u += separated_workers - exported_workers_u
    employment += hired_workers_u + hired_workers_e - exported_workers_e - separated_workers
    v += opened_vacancies - hired_workers_u - hired_workers_e

    # Correct floating point error
    v = np.maximum(1e-15 * np.ones(n), v)
    u = np.maximum(1e-15 * np.ones(n), u)
    employment = np.maximum(1e-15 * np.ones(n), employment)

    variables = [employment, u, v]
    changes = [separated_workers, exported_workers_u]

    return variables, changes


def fire_and_hire_workers_multiple_apps(parameters, variables, d_dagger_t, A, matching_probability):
    """
    Function that updates the number of employed and unemployed workers and
    job vacancies in each occupation. The update considers the spontanous
    separations and opening of vacancies, as well as the directed separations
    and openings and flow of workers.
    Note that, when the demand shock is too large additional vacancies may
    be closed.
    Args:
        δ_u, δ_v, γ_u, γ_v, λ, β_u, β_e (floats): parameters
        employment, u, v, d_dagger_t(np arrays): arrays with the employment
            unemployment, vacancies and target demand for each occupation
        A(np arrays): adjacency matrix
        matching_probability(function): probability of app in j being succesfull
    """
    assert len(parameters) == 7, "wrong number of parameters"
    δ_u, δ_v, γ_u, γ_v, λ, β_u, β_e = parameters

    employment, u, v = variables

    n = len(employment)
    # spontanous separations and job opening
    separated_workers = δ_u * employment
    opened_vacancies = δ_v * employment
    # state dependent separations and job openings (directed effort)
    # compute difference between realized and target demand
    Δ_demand = employment + v + -d_dagger_t
    # get extra separations and openings, bounded by employment of occupation
    g_u = np.minimum(employment, γ_u * np.maximum(np.zeros(n), Δ_demand))
    g_v = np.minimum(employment, γ_v * np.maximum(np.zeros(n), -Δ_demand))
    # get number of separations and job openings
    separated_workers = separated_workers + (1 - δ_u) * g_u
    opened_vacancies = opened_vacancies + (1 - δ_v) * g_v

    # Search and matching
    Av = np.multiply(v, A)
    # get matrix q_ij, probability of applying from i to j
    Q = Av / np.sum(Av, axis=1, keepdims=1)

    # expected number of job applications from i to j
    sij_u = β_u * np.multiply(u[:, None], Q)  # as a result of unemployed workers
    sij_e = λ * β_e * np.multiply(employment[:, None], Q)  # as a result of employed workers
    sij = sij_u + sij_e
    # expected probability of application being succesfull
    pj = matching_probability_multiple_apps(sij, sij_e, sij_u, v, β_u, β_e, Q)
    # expected flow of workers
    F_u = np.multiply(sij_u, pj)  # sij .* pj
    F_e = np.multiply(sij_e, pj)

    # getting hired workers and transitioning workers
    # previously unemployed workers
    hired_workers_u = np.sum(F_u, axis=0)
    exported_workers_u = np.sum(F_u, axis=1)
    # previously employed workers
    hired_workers_e = np.sum(F_e, axis=0)
    exported_workers_e = np.sum(F_e, axis=1)
    # check everything okay with code
    assert min(hired_workers_u) >= 0
    assert min(exported_workers_u) >= 0
    assert min(hired_workers_e) >= 0
    assert min(exported_workers_e) >= 0
    exported_workers = exported_workers_u + exported_workers_e

    # Update e, w and v
    u += separated_workers - exported_workers_u
    employment += hired_workers_u + hired_workers_e - exported_workers_e - separated_workers
    v += opened_vacancies - hired_workers_u - hired_workers_e

    # Correct floating point error
    v = np.maximum(1e-15 * np.ones(n), v)
    u = np.maximum(1e-15 * np.ones(n), u)
    employment = np.maximum(1e-15 * np.ones(n), employment)

    variables = [employment, u, v]
    changes = [separated_workers, exported_workers_u]

    return variables, changes


def fire_and_hire_workers_wages(parameters, variables, d_dagger_t, A, matching_probability):
    """
    Function that updates the number of employed and unemployed workers and
    job vacancies in each occupation. The update considers the spontanous
    separations and opening of vacancies, as well as the directed separations
    and openings and flow of workers.
    Note that, when the demand shock is too large additional vacancies may
    be closed.
    Args:
        δ_u, δ_v, γ_u, γ_v, λ, β_u, β_e (floats): parameters
        employment, u, v, d_dagger_t(np arrays): arrays with the employment
            unemployment, vacancies and target demand for each occupation
        A(np arrays): adjacency matrix
        matching_probability(function): probability of app in j being succesfull
    """
    assert len(parameters) == 9, "wrong number of parameters"
    δ_u, δ_v, γ_u, γ_v, λ, β_u, β_e, wages, wage_function = parameters

    employment, u, v = variables
    n = len(employment)
    # spontanous separations and job opening
    separated_workers = δ_u * employment
    opened_vacancies = δ_v * employment
    # state dependent separations and job openings (directed effort)
    # compute difference between realized and target demand
    Δ_demand = employment + v + -d_dagger_t
    # get extra separations and openings, bounded by employment of occupation
    g_u = np.minimum(employment, γ_u * np.maximum(np.zeros(n), Δ_demand))
    g_v = np.minimum(employment, γ_v * np.maximum(np.zeros(n), -Δ_demand))
    # get number of separations and job openings
    separated_workers = separated_workers + (1 - δ_u) * g_u
    opened_vacancies = opened_vacancies + (1 - δ_v) * g_v

    # Search and matching
    vw = np.multiply(v, wage_fun(wages, wage_function, n))
    Av = np.multiply(vw, A)
    # get matrix q_ij, probability of applying from i to j
    Q = Av / np.sum(Av, axis=1, keepdims=1)
    # expected number of job applications from i to j
    sij_u = β_u * np.multiply(u[:, None], Q)  # as a result of unemployed workers
    sij_e = λ * β_e * np.multiply(employment[:, None], Q)  # as a result of employed workers
    sij = sij_u + sij_e
    # expected probability of application being succesfull
    pj = matching_probability_multiple_apps(sij, sij_e, sij_u, v, β_u, β_e, Q)
    # expected flow of workers
    F_u = np.multiply(sij_u, pj)  # sij .* pj
    F_e = np.multiply(sij_e, pj)

    # getting hired workers and transitioning workers
    # previously unemployed workers
    hired_workers_u = np.sum(F_u, axis=0)
    exported_workers_u = np.sum(F_u, axis=1)
    # previously employed workers
    hired_workers_e = np.sum(F_e, axis=0)
    exported_workers_e = np.sum(F_e, axis=1)
    # check everything okay with code
    assert min(hired_workers_u) >= 0
    assert min(exported_workers_u) >= 0
    assert min(hired_workers_e) >= 0
    assert min(exported_workers_e) >= 0
    exported_workers = exported_workers_u + exported_workers_e

    # Update e, w and v
    u += separated_workers - exported_workers_u
    employment += hired_workers_u + hired_workers_e - exported_workers_e - separated_workers
    v += opened_vacancies - hired_workers_u - hired_workers_e

    # Correct floating point error
    v = np.maximum(1e-15 * np.ones(n), v)
    u = np.maximum(1e-15 * np.ones(n), u)
    employment = np.maximum(1e-15 * np.ones(n), employment)

    variables = [employment, u, v]
    changes = [separated_workers, exported_workers_u]

    return variables, changes


def wage_fun(wages, wage_function, n):
    if wage_function == "simple":
        return wages
    if wage_function == "log":
        return np.log(wages)
    if wage_function == None:
        return np.ones((1, n))


def run_numerical_solution(
    fire_and_hire_workers,
    t_sim,
    parameters,
    variables_0,
    target_demand_function,
    D_0,
    D_f,
    t_shock,
    k,
    t_halfsig,
    matching,
    A_matrix,
    τ,
):
    """Iterates the firing and hiring process for n_steady steps. Then runs the simulation with the
    restructured employment for n_sim time steps.
    returns unemployment rate, w, v, e and W lists.
    Args:
    fire_and_hire_workers(function): function that updates w, v, e and W.
    A_matrix(array): adjacency matrix of the network
    t_sin(int): number of times the simulation is run (after steady state)
    delta_u, delta_v, gamma_u, gamma_v(Float64): parameters of network
    employment_0(array(n_occ, 1)): number of employed workers at initial time.
    """
    employment_0, unemployment_0, vacancies_0 = variables_0
    assert len(employment_0) == len(vacancies_0) == len(unemployment_0)
    n_occ = len(employment_0)
    initial_variables = []
    # setting initial conditions
    employment = copy.deepcopy(employment_0)
    unemployment = copy.deepcopy(unemployment_0)
    vacancies = copy.deepcopy(vacancies_0)
    new_variables = [employment, unemployment, vacancies]
    # defining arrays where information is stored
    E = np.zeros([t_sim, n_occ])
    U = np.zeros([t_sim, n_occ])
    V = np.zeros([t_sim, n_occ])
    D = np.zeros([t_sim, n_occ])
    D[0, :] = D_0

    # recording initial conditions
    E[0, :] = employment_0
    U[0, :] = unemployment_0
    V[0, :] = vacancies_0
    Variables = [E, U, V]
    U_all = np.zeros([t_sim, t_sim, n_occ])

    for t in range(1, t_sim):
        # for t in range(t_sim-1):
        print("time ", t)
        # compute target demand for given time step
        d_dagger_t = target_demand_function(t, D_0, D_f, t_shock, k, t_halfsig)
        D[t, :] = d_dagger_t
        # update main variables and get the number of separations
        new_variables, changes = fire_and_hire_workers(parameters, new_variables, d_dagger_t, A_matrix, matching)
        separated, exported = changes
        # the number of separations = unemployed workers with 1 t.s. of unemp
        U_all[t, 0, :] = separated
        # fill in expected number of unemployed workers with given job spell
        # note that max job spell is time of simulation so far
        # fill in for more than 1 time step
        for n in range(1, t + 1):
            # job spell is those of previous job spell - the ones hired
            # note Variables[1] = unemployed list
            U_all[t, n, :] = U_all[t - 1, n - 1, :] * (1 - exported / (Variables[1][t - 1, :]))
        # store information in arrays
        Variables[0][t, :] = new_variables[0]
        Variables[1][t, :] = new_variables[1]
        Variables[2][t, :] = new_variables[2]

    return Variables, U_all, D


def calibrate_sigmoid(shock_duration, automation_level=0.9999):
    """
    Args
        shock_duration: time when automation level is reached
        tolerance: float between 0 << t < 1, level of automation that is
        reached
    return: sigmoid growth rate (k) and half life
    """
    sigmoid_half_life = shock_duration / 2
    k = -np.log(1 / automation_level - 1) / sigmoid_half_life
    return sigmoid_half_life, k


def labor_restructure(D_0, automation_fraction, demand_scale=1):
    """Given an initial distribution of labor demand and an automation fraction
    of each occupation, returns the employment distribution after shock
    assumes employment stays constant
    Args:
        D_0: original demand
        automation_frection: fraction that each occupation gets automated
        demand_scale: how much demand increases or decreases. If 1 remains
        constant
    """
    # getting number of occupations and labor force
    n = len(D_0)
    L = sum(D_0)
    # number of hour worked. Has no effect
    x0 = 8
    # total number of hours per occupation
    h0 = x0 * D_0
    # number of working hours in each occupation after automation [t]
    hf = np.multiply(h0, (np.ones(n) - automation_fraction))
    # new working time [t/e]
    xf = sum(hf) / L
    # new employment demand of workers [e]
    D_f = np.array([demand_scale * hf[i] / xf for i in range(n)])
    return D_f


def target_demand_automation(t, d_0, d_final, t_shock, k, t_halfsig):
    """function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: vector of initial demand of occupation, minimum of sigmoid
        d_final: vector of final demand of occupation, maximum of sigmoid
        k:growth rate
        t_halfsig: half growth time
    Returns
        d_dagger(Array{Float64,2}): demand of occupation at time t
    """
    if t < t_shock:
        return d_0
    else:
        # set half life considering the time at which the shock starts
        t0 = t_halfsig + t_shock
        # note if different adoption rate could introduce elementwise mult.
        d_dagger = d_0 + (d_final - d_0) * 1 / (1 + np.exp(-k * (t - t0)))
        return d_dagger


def target_demand_cycle(t, d_0, d_final, t_shock, amplitude, period):
    """function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: vector of initial demand of occupation
        d_final: (ignored)
        amplitude: amplitude of business cycle
        period: period for full business cycle
    Returns
        d_dagger(Array{Float64,2}): demand of occupation at time t
    """
    if t < t_shock:
        return d_0
    else:
        # start cycle when shock starts
        t0 = t + t_shock
        d_dagger = d_0 * (1 - amplitude * np.sin((2 * np.pi / period) * t0))
        return d_dagger


def target_demand_gfc(t, d_0, d_final, t_shock, alpha, target_period):
    """function that target demand of time t of sigmoid shock with parameters
    Args:
        d_0: vector of initial demand of occupation
        d_final: (np.array): array with historical unemployment/gdp
        alpha: amplitude rescaling of shock
        period: period for full business cycle
    Returns
        d_dagger(Array{Float64,2}): demand of occupation at time t
    """
    # check time steps of series
    cycle_original_timesteps = len(d_final)
    # do new time scale with required steps
    demand_rescale = np.array(
        [d_final[int(t * cycle_original_timesteps / target_period)] for t in range(target_period)]
    )
    # control amplitude
    demand_rescale = demand_rescale * (1 - alpha) + alpha * 1.0
    if t < t_shock:
        return d_0
    else:
        # start cycle when shock starts
        return demand_rescale[t % target_period] * d_0


def bussiness_cycle_period(τ, start_year=2008, finish_year=2018):
    """
    calibrates how many time steps should bussiness cycle last
    """
    delta_years = finish_year - start_year
    # how much of a business cycle do the years 2008-2018 account for
    # assume it unemployment started at mid point and is now at minimum
    cycle_fraction = 3 / 4
    # time step in weeks NOTE tau + 1 since counting starts at 0
    # TODO
    Δ_t = 27 / (τ + 1)
    # one year equals
    one_year_ts = 52 / Δ_t  # time steps
    delta_years_in_ts = delta_years * one_year_ts
    # since there has not been a full cycle divide by cycle frac
    # delta_years_in_ts * cycle_fraction = cycle_duration
    cycle_duration = delta_years_in_ts / cycle_fraction
    # +1 to account for index starting at 0
    return int(cycle_duration) + 1


def add_self_loops(A, r):
    """Add homogenous self (r) loops to matrix A
    preserve column normalization (sum to 1)
    """
    n = A.shape[0]
    A_new = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                A_new[i, j] = r
            else:
                A_new[i, j] = A[i, j] * (1 - r)
    return A_new


def add_self_loops_heterogenous(A, r):
    """Add heterogenous self (r[i]) loops to matrix A
    preserve column normalization (sum to 1)
    """
    n = A.shape[0]
    A_new = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                A_new[i, j] = r[i]
            else:
                A_new[i, j] = A[i, j] * (1 - r[i])
    return A_new


############################
# Code for calculating percentage change in unemployment
############################


def u_longterm_from_jobspell(U_ltm, τ):
    # NOTE -1 since python starts counting on 1
    return np.sum(U_ltm[:, τ - 1 :, :], axis=1)


def period_occ_u_rate(E, U, time_start, time_end):
    """gives the period unemployment rate of an occupation
    E: employment rate of occupation per time
    U: unemployment rate of occupation per time
    time_start(int): time from which the average starts
    time_end(int): time from which the average ends
    """
    e = E[time_start:time_end, :]
    u = U[time_start:time_end, :]
    return 100 * sum(u) / sum(e + u)


def period_occ_ltu_rate(E, U, U_lt, time_start, time_end):
    """gives the period longterm unemploymetn rate of anoccupation
    E: employed of occupation per time
    U: unemployed of occupation per time
    U_lt: longterm unemployed of occupation per time
    time_start(int): time from which the average starts
    time_end(int): time from which the average ends
    """
    e = E[time_start:time_end, :]
    u = U[time_start:time_end, :]
    ltu = U_lt[time_start:time_end, :]
    return 100 * sum(ltu) / sum(e + u)


def percentage_change_u(E, U, time_start_1, time_end_1, time_start_2, time_end_2):
    u_initial_num = period_occ_u_rate(E, U, time_start_1, time_end_1)
    u_transition_num = period_occ_u_rate(E, U, time_start_2, time_end_2)
    return 100 * (u_transition_num - u_initial_num) / u_initial_num


def percentage_change_ltu(E, U, U_lt, time_start_1, time_end_1, time_start_2, time_end_2):
    ltu_initial_num = period_occ_ltu_rate(E, U, U_lt, time_start_1, time_end_1)
    ltu_transition_num = period_occ_ltu_rate(E, U, U_lt, time_start_2, time_end_2)
    return 100 * (ltu_transition_num - ltu_initial_num) / ltu_initial_num


############################
# Code used for calibration (measuring area surrounded by Beveridge curve)
############################
def PolyArea(x, y):
    """Calculate polygon area"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/
        # derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def find_edges_with(i, edge_set):
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


def stitch_boundaries(edges):
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


def get_polygon(u, v):
    """Get's polygon of Beveridge Curve"""
    assert len(u) == len(v)
    # get number of points
    points = np.zeros([len(u), 2])
    # unemployment in x coordinate and vacancies in y coordinate
    for i in range(len(u)):
        points[i, 0] = u[i]
        points[i, 1] = v[i]
    # get edges and boundary
    edges = alpha_shape(points, alpha=1, only_outer=True)
    bound_edges = stitch_boundaries(edges)
    # for simple polygons (no crosses)
    if len(bound_edges) == 1:
        # get points in array
        x_array = np.zeros(2 * len(bound_edges[0]))
        y_array = np.zeros(2 * len(bound_edges[0]))

        points_alpha = np.zeros([2 * len(bound_edges[0]), 2])
        count = 0
        for i, j in bound_edges[0]:
            if count + 2 <= len(x_array):
                x_array[count] = points[[i, j], 0][0]
                y_array[count] = points[[i, j], 1][0]
                points_alpha[count][0] = points[[i, j], 0][0]
                points_alpha[count][1] = points[[i, j], 1][0]
                count += 1
                points_alpha[count][0] = points[[i, j], 0][1]
                points_alpha[count][1] = points[[i, j], 1][1]
                x_array[count] = points[[i, j], 0][1]
                y_array[count] = points[[i, j], 1][1]
                count += 1
        poly = Polygon(points_alpha)
        # plt.plot(x_array, y_array, "o--", label="array", alpha=0.8)
        # plt.legend()
        # plt.show()
    # if polygon is not simple use buffer (since empirical curve is simple)
    # non-simple polygons tend to be a bad fit, still good to take into account
    else:
        poly = Polygon(points)
        poly = poly.buffer(0)
    return poly


def evaluate_cost(u0, v0, poly_empirical, t_shock, cycle_duration):
    """take unemployment and vacancy rate, preprocess them, takes alpha
    shape and compares overlap with empirical one
    u0(np.array): unemployment rate
    v0(np.array): vacancy rate
    """
    # discard transient state that it takes to get into the cycle
    tend_transition = t_shock + 15
    u0 = u0[tend_transition : tend_transition + cycle_duration]
    v0 = v0[tend_transition : tend_transition + cycle_duration]

    poly_fit = get_polygon(u0, v0)
    try:  # if polygon is invalid have cost 1
        poly_intersect = poly_empirical.intersection(poly_fit)
        poly_union = poly_empirical.union(poly_fit)
        area_inter = poly_intersect.area / poly_union.area
    except:
        area_inter = 0
    cost = 1 - area_inter
    print("cost function", cost, "\n")
    return cost


def find_best_parameters(δ_u_list, δ_v_list, τ_list, cycle_amp_list, employment, real_bev_poly, t_shock, t_sim, A):
    """function that takes lists of parameters and runs through them
    exhaustively to find the best parameter
    δ_u_list(list):list of parameters
    δ_v_list(list):list of parameters
    τ_list(list):list of parameters
    cycle_amp_list(list):list of parameters
    employment_0(np.array): vector if initial employment
    real_bev_poly(polygon): polygon of empirical Beveridge curve to match
    t_shock(int): time at which the shock (the sin wave) happens
    t_sim(int): simulation time
    A(np.array): network adjacency matrix
    """
    # recording cost and evaluation index of interation
    min_cost = 1
    index_evaluation = 1
    # number of evaluations
    n_evaluations = len(δ_u_list) * len(δ_v_list) * len(τ_list) * len(cycle_amp_list)

    df_error = pd.DataFrame()
    df_error["d_u"] = np.zeros(n_evaluations)
    df_error["d_v"] = np.zeros(n_evaluations)
    df_error["tau"] = np.zeros(n_evaluations)
    df_error["cycle_amp"] = np.zeros(n_evaluations)
    df_error["error"] = np.zeros(n_evaluations)
    # data frame with errors
    for cycle_amp in cycle_amp_list:
        for τ in τ_list:
            cycle_period = bussiness_cycle_period(τ)  # + 1 # now implemented in fuction
            shock_params = cycle_amp, cycle_period
            # r = calibrate_selfloop(τ)
            for δ_u in δ_u_list:
                for δ_v in δ_v_list:
                    # setting deltas
                    γ_u = 10 * δ_u
                    γ_v = 10 * δ_u
                    print(
                        "du ",
                        δ_u,
                        " dv ",
                        δ_v,
                        " tau ",
                        τ,
                        " cycle_period ",
                        cycle_period,
                        "cycle_amp ",
                        cycle_amp,
                    )
                    # setting initial conditions
                    employment_0 = employment[:]
                    unemployment_0 = δ_u * employment_0
                    vacancies_0 = δ_v * employment_0
                    L = np.sum(employment_0 + unemployment_0)
                    # initial demand and target demand
                    D_0 = employment_0 + unemployment_0
                    # set final demand equal to initial demand
                    D_f = employment_0 + unemployment_0
                    # shock parameters (cycle amplitude and period)
                    cycle_period = bussiness_cycle_period(τ)

                    # run model
                    # TODO replace δ etc. with parameters
                    U, V, E, U_all, D = run_numerical_solution(
                        fire_and_hire_workers,
                        t_sim,
                        δ_u,
                        δ_v,
                        γ_u,
                        γ_v,
                        employment_0,
                        unemployment_0,
                        vacancies_0,
                        target_demand_cycle,
                        D_0,
                        D_f,
                        t_shock,
                        cycle_amp,
                        cycle_period,
                        matching_probability,
                        A,
                        τ,
                    )

                    unemployment_rate = 100 * U.sum(axis=1) / L
                    vacancy_rate = 100 * V.sum(axis=1) / (V.sum(axis=1) + E.sum(axis=1))
                    # uncomment following lines to plot
                    # plt.plot(unemployment_rate, vacancy_rate, label="occupational mobility network")
                    # plt.xlabel("unemployment rate (%)")
                    # plt.ylabel("vacancy rate (%)")
                    # plt.legend()
                    # plt.show()

                    #####
                    # run cost functions
                    #####
                    # #adding u, v to csv
                    cost = evaluate_cost(unemployment_rate, vacancy_rate, real_bev_poly, t_shock, cycle_period)
                    #####
                    # save errors
                    #####
                    # # getting the cost given by python
                    df_error["d_u"].loc[index_evaluation] = δ_u
                    df_error["d_v"].loc[index_evaluation] = δ_v
                    df_error["tau"].loc[index_evaluation] = τ
                    df_error["cycle_amp"].loc[index_evaluation] = cycle_amp
                    df_error["error"].loc[index_evaluation] = cost

                    if cost < min_cost:
                        best_du = copy.copy(δ_u)
                        best_dv = copy.copy(δ_v)
                        best_tau = copy.copy(τ)
                        best_amp = copy.copy(cycle_amp)
                        min_cost = cost
                    print("cost = ", cost)
                    index_evaluation += 1

    print("best du ", best_du, "dv ", best_dv, "tau ", τ, "amp ", best_amp, "cost ", cost)
    return df_error


def find_best_parameters_gfc(
    δ_u_list, δ_v_list, τ_list, cycle_amp_list, employment, demand_rate, real_bev_poly, t_shock, t_sim, A
):
    """function that takes lists of parameters and runs through them
    exhaustively to find the best parameter. Done for the GFC beveridge curve
    takes into account the vacancies and employment as target demand
    δ_u_list(list):list of parameters
    δ_v_list(list):list of parameters
    τ_list(list):list of parameters
    cycle_amp_list(list):list of parameters
    employment_0(np.array): vector if initial employment
    demand_rate(np.array): array of demand rate (should oscilate around 1)
    real_bev_poly(polygon): polygon of empirical Beveridge curve to match
    t_shock(int): time at which the shock (the sin wave) happens
    t_sim(int): simulation time
    A(np.array): network adjacency matrix
    """
    # recording cost and evaluation index of interation
    min_cost = 1
    index_evaluation = 1
    # number of evaluations
    n_evaluations = len(δ_u_list) * len(δ_v_list) * len(τ_list) * len(cycle_amp_list)

    df_error = pd.DataFrame()
    df_error["d_u"] = np.zeros(n_evaluations)
    df_error["d_v"] = np.zeros(n_evaluations)
    df_error["tau"] = np.zeros(n_evaluations)
    df_error["cycle_amp"] = np.zeros(n_evaluations)
    df_error["error"] = np.zeros(n_evaluations)
    # data frame with errors
    for cycle_amp in cycle_amp_list:
        for τ in τ_list:
            cycle_period = bussiness_cycle_period(τ)  # + 1 # now implemented in fuction
            shock_params = cycle_amp, cycle_period
            # r = calibrate_selfloop(τ)
            for δ_u in δ_u_list:
                for δ_v in δ_v_list:
                    # setting deltas
                    γ_u = 10 * δ_u
                    γ_v = 10 * δ_u
                    print(
                        "du ",
                        δ_u,
                        " dv ",
                        δ_v,
                        " tau ",
                        τ,
                        " cycle_period ",
                        cycle_period,
                        "cycle_amp ",
                        cycle_amp,
                    )
                    # setting initial conditions
                    employment_0 = employment[:]
                    unemployment_0 = δ_u * employment_0
                    vacancies_0 = δ_v * employment_0
                    L = np.sum(employment_0 + unemployment_0)
                    # initial demand and target demand
                    D_0 = employment_0 + unemployment_0
                    # shock parameters (cycle amplitude and period)
                    cycle_period = bussiness_cycle_period(τ)

                    # run model
                    U, V, E, U_all, D = run_numerical_solution(
                        fire_and_hire_workers,
                        t_sim,
                        δ_u,
                        δ_v,
                        γ_u,
                        γ_v,
                        employment_0,
                        unemployment_0,
                        vacancies_0,
                        target_demand_gfc,
                        D_0,
                        demand_rate,
                        t_shock,
                        cycle_amp,
                        cycle_period,
                        matching_probability,
                        A,
                        τ,
                    )

                    unemployment_rate = 100 * U.sum(axis=1) / L
                    vacancy_rate = 100 * V.sum(axis=1) / (V.sum(axis=1) + E.sum(axis=1))
                    # uncomment following lines to plot
                    # plt.plot(unemployment_rate, vacancy_rate, label="occupational mobility network")
                    # plt.xlabel("unemployment rate (%)")
                    # plt.ylabel("vacancy rate (%)")
                    # plt.legend()
                    # plt.show()

                    #####
                    # run cost functions
                    #####
                    # #adding u, v to csv
                    cost = evaluate_cost(unemployment_rate, vacancy_rate, real_bev_poly, t_shock, cycle_period)
                    #####
                    # save errors
                    #####
                    # # getting the cost given by python
                    df_error["d_u"].loc[index_evaluation] = δ_u
                    df_error["d_v"].loc[index_evaluation] = δ_v
                    df_error["tau"].loc[index_evaluation] = τ
                    df_error["cycle_amp"].loc[index_evaluation] = cycle_amp
                    df_error["error"].loc[index_evaluation] = cost

                    if cost < min_cost:
                        best_du = copy.copy(δ_u)
                        best_dv = copy.copy(δ_v)
                        best_tau = copy.copy(τ)
                        best_amp = copy.copy(cycle_amp)
                        min_cost = cost
                    print("cost = ", cost)
                    index_evaluation += 1

    print("best du ", best_du, "dv ", best_dv, "tau ", τ, "amp ", best_amp, "cost ", cost)
    return df_error


def calibrate_selfloop(τ, occ_mobility=0.19, unemployment_rate=0.06):
    """
    occupational mobility default value is 19%
    unemployment rate is average unemployment rate since 2000
    τ units are in time steps, six percent
    """
    week_duration = 27 / (τ + 1)
    Δ_t = 52 / week_duration  # time step duration in a year
    x = 1 - occ_mobility
    r = (x ** (1 / Δ_t) + unemployment_rate - 1) / unemployment_rate
    return r
