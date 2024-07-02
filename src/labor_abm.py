import copy

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch


class LabourABM:
    def __init__(
        self,
        # number of occupations
        N: int,
        # number of time steps to run model
        T: int,
        # seed for initial A_hat
        seed: int,
        # Parameters
        # separation and opening rate
        δ_u: float,
        δ_v: float,
        # adaptation rate u and v
        γ_u: float,
        γ_v: float,
        # probability of on-the-job search
        λ: float,
        # n applications by employed and unemployed
        β_e: float,
        β_u: float,
        # Ease of Transition matrix (assumed known for now)
        A: torch.Tensor,
        # Scenario/target demand
        d_dagger: torch.Tensor,
        # Employment data
        initial_employment: torch.Tensor,  # Initial employment data
        initial_unemployment: torch.Tensor,  # Initial unemployment data
        initial_vacancies: torch.Tensor,  # Initial vacancies data
        wages: torch.Tensor,
        device: torch.device = torch.device("cpu"),  # Default to CPU
        dtype: torch.dtype = torch.float32,  # Default to float32
    ):
        self.N = N
        self.T = T
        self.seed = seed
        self.separation_rate = δ_u
        self.opening_rate = δ_v
        self.adaptation_u = γ_u
        self.adaptation_v = γ_v
        self.otjob_search_prob = λ
        self.n_applications_emp = β_e
        self.n_applications_unemp = β_u
        self.A = A.to(device=device, dtype=dtype)
        self.d_dagger = d_dagger.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

        # Initialize tensors to hold data for all time steps and set initial values
        self.employment = torch.zeros((N, T), device=device, dtype=dtype)
        self.unemployment = torch.zeros((N, T), device=device, dtype=dtype)
        self.vacancies = torch.zeros((N, T), device=device, dtype=dtype)

        # Intialize arrays to keep track of what is happening
        self.spon_separations = torch.zeros((N, T), device=device, dtype=dtype)
        self.state_separations = torch.zeros((N, T), device=device, dtype=dtype)
        self.spon_vacancies = torch.zeros((N, T), device=device, dtype=dtype)
        self.state_vacancies = torch.zeros((N, T), device=device, dtype=dtype)
        # point to the target occ
        self.from_job_to_occ = torch.zeros((N, T), device=device, dtype=dtype)
        self.from_unemp_to_occ = torch.zeros((N, T), device=device, dtype=dtype)

        # Set initial values at time t = 0
        self.employment[:, 0] = initial_employment
        self.unemployment[:, 0] = initial_unemployment
        self.vacancies[:, 0] = initial_vacancies

        self.wages = wages.to(device=device, dtype=dtype)

    # @classmethod
    # def from_data(
    #     cls,
    #     employment: pd.DataFrame,
    #     A: pd.DataFrame,
    #     d_dagger: pd.DataFrame,
    #     parameters: pd.DataFrame
    # ) -> Self:
    #     d_dagger =  torch.tensor(d_dagger["d_dagger"].values, dtype=torch.float32)
    #     wages = torch.tensor(d_dagger["wages"].values, dtype=torch.float32)
    #     A_tensor = torch.tensor(A.values, dtype=torch.float32)
    #     employment = torch.zeros(N, N, dtype=torch.float32)
    #     unemployment = torch.zeros(N, N, dtype=torch.float32)
    #     vacancies = torch.zeros(N, N, dtype=torch.float32)
    #     employment[:, 0] = torch.tensor(employment["employment"].values, dtype=torch.float32)
    #     unemployment[:, 0] = torch.tensor(employment["unemployment"].values, dtype=torch.float32)
    #     vacancies[:, 0] = torch.tensor(employment["vacancies"].values, dtype=torch.float32)

    #     # Extracting parameters from the 'parameters' DataFrame
    #     seed = int(parameters['seed'].values[0])
    #     N = int(parameters['N'].values[0])
    #     T = int(parameters['T'].values[0])
    #     δ_u = float(parameters['delta_u'].values[0])
    #     δ_v = float(parameters['delta_v'].values[0])
    #     γ_u = float(parameters['gamma_u'].values[0])
    #     γ_v = float(parameters['gamma_v'].values[0])
    #     λ = float(parameters['lambda'].values[0])
    #     β_e = float(parameters['beta_e'].values[0])
    #     β_u = float(parameters['beta_u'].values[0])

    #     return cls(N, T, seed, δ_u, δ_v, γ_u, γ_v, λ, β_e, β_u, A_tensor, d_dagger,
    #                employment, unemployment, vacancies)

    def initialize_variables(self):
        e = self.employment[:, 0].clone()
        u = self.unemployment[:, 0].clone()
        v = self.vacancies[:, 0].clone()

        # Variables that keep trakc on things
        spon_sep = self.spon_separations[:, 0].clone()
        state_sep = self.state_separations[:, 0].clone()
        spon_vac = self.spon_vacancies[:, 0].clone()
        state_vac = self.state_vacancies[:, 0].clone()
        jtj = self.from_job_to_occ[:, 0].clone()
        utj = self.from_unemp_to_occ[:, 0].clone()

        return e, u, v, spon_sep, state_sep, spon_vac, state_vac, jtj, utj

    # Separation and job opennings
    def spontaneous_separations(self, e):
        return self.separation_rate * e

    def spontaneous_openings(self, e):
        return self.opening_rate * e

    def state_dep_separations(self, diff_demand):
        return (1 - self.separation_rate) * self.adaptation_u * torch.maximum(torch.zeros(self.N), diff_demand)

    def state_dep_openings(self, diff_demand):
        return (1 - self.opening_rate) * self.adaptation_v * torch.maximum(torch.zeros(self.N), -diff_demand)

    # Search process
    def calc_attractiveness_vacancy(self):
        """wage(np.array) average wage in category
        A ease of transitioning between categories
        """
        attractiveness = torch.mul(self.wages, self.A)
        return attractiveness

    def calc_probability_applying(self, v):
        """How attractive a is a vacancy depending on skills, geography, wage, etc"""
        attractiveness = self.calc_attractiveness_vacancy()
        Av = torch.mul(v, attractiveness)
        Q = Av / torch.sum(Av, axis=1, keepdims=True)
        return Q

    def calc_applicants_and_applications_sent(self, e, u, v):
        """expected number of applications sent from one category to another"""
        Q = self.calc_probability_applying(v)

        # applicants
        aij_e = self.otjob_search_prob * torch.mul(e[:, None], Q)
        aij_u = torch.mul(u[:, None], Q)

        # sij(e) = β_e*ei*qij
        sij_e = self.n_applications_emp * aij_e
        # sij(u) = β_u*ui*qij
        sij_u = self.n_applications_unemp * aij_u

        return aij_e, aij_u, sij_e, sij_u

    # Matching process
    def calc_job_offers(self, v, sj):
        """Calculate number of vacancies that received at least one applicant"""
        v_inv = torch.reciprocal(v)
        v_inv[torch.isinf(v_inv)] = 0

        prob_job_offer = 1 - torch.exp(-torch.multiply(sj, v_inv))

        job_offers = torch.multiply(v, prob_job_offer)

        return job_offers

    def calc_prob_workers_with_offers(self, v, sj):
        job_offers = self.calc_job_offers(v, sj)
        # (beta_apps - l); where l = 0 to beta
        active_applications_from_u = torch.repeat_interleave(sj, self.n_applications_unemp).reshape(
            self.N, self.n_applications_unemp
        ) - torch.tensor(range(self.n_applications_unemp))
        active_applications_from_e = torch.repeat_interleave(sj, self.n_applications_emp).reshape(
            self.N, self.n_applications_emp
        ) - torch.tensor(range(self.n_applications_emp))

        # In rare cases where few applications are receives sj < beta. We prevent this from going negative
        active_applications_from_u = torch.clamp(active_applications_from_u, min=0.000000001)
        active_applications_from_e = torch.clamp(active_applications_from_e, min=0.000000001)

        # probability of app being drawn; job_offers / (beta_apps - l); where l = 0 to beta
        # clamp since job_offers <= beta_apps - l; at the cross it means app is for sure selected
        prob_app_selected_u = torch.clamp(torch.mul(job_offers[:, None], 1.0 / active_applications_from_u), max=1)
        prob_app_selected_e = torch.clamp(torch.mul(job_offers[:, None], 1.0 / active_applications_from_e), max=1)
        # probability no application is selected
        prob_no_app_selected_u = 1 - prob_app_selected_u
        prob_no_app_selected_e = 1 - prob_app_selected_e

        # prob none of those apps is drawn
        no_offer_u = torch.prod(prob_no_app_selected_u, axis=1)
        no_offer_e = torch.prod(prob_no_app_selected_e, axis=1)
        # worker getting offer
        got_offer_u = 1 - no_offer_u
        got_offer_e = 1 - no_offer_e

        return got_offer_e, got_offer_u

    # simulation
    def time_step(self, e, u, v, t):
        # workers separationa and job openings
        # print("time ", t)
        # print("e5, u5, v5 ", e[5], u[5], v[5])
        d = e + v
        diff_demand = d - self.d_dagger[:, t]
        spon_sep = self.spontaneous_separations(e)
        state_sep = self.state_dep_separations(diff_demand)
        spon_vac = self.spontaneous_openings(e)
        state_vac = self.state_dep_openings(diff_demand)
        # print("spon sep ", spon_sep[5])
        # print("state sep ", state_sep[5])
        separated_workers = spon_sep + state_sep
        opened_vacancies = spon_vac + state_vac

        # search
        aij_e, aij_u, sij_e, sij_u = self.calc_applicants_and_applications_sent(e, u, v)
        # print("siju 3,5", sij_u[3,5])
        ### NOTE / to-do, make sj come be calculated inside a function
        sj = (sij_u + sij_e).sum(axis=0)
        # print("sj 5", sj[5])
        # matching
        ### NOTE modifying below to debug
        prob_offer_e, prob_offer_u = self.calc_prob_workers_with_offers(v, sj)
        # what about acceptance?
        Fij_e = aij_e * prob_offer_e
        Fij_u = aij_u * prob_offer_u
        Fij = Fij_u + Fij_e
        # NOTE Uncomment blow below to run baseline
        ####################
        # job_offers = self.calc_job_offers(v, sj)
        # # print("job offers ", job_offers[5]/sj[5])
        # Fij_u = sij_u * job_offers/sj
        # Fij_e = sij_e * job_offers/sj
        # Fij = Fij_u + Fij_e
        # # print("F[3,5], ", Fij[3,5])
        # # print("F_e[3,5] should be 0 , ", Fij_e[3,5])
        #####################

        # TODO check right sum
        jtj = Fij_e.sum(axis=1)
        utj = Fij_u.sum(axis=1)
        # updating values
        # print("got employed Fij_u.sum(axis=1)[5]", Fij_u.sum(axis=1)[5])
        # print("sep workers[5]", separated_workers[5])
        # print("u[5]", u[5])
        e += -separated_workers + Fij.sum(axis=0) - Fij_e.sum(axis=1)
        # NOTE tell Anna about axis=1 below
        u += separated_workers - Fij_u.sum(axis=1)
        v += opened_vacancies - Fij.sum(axis=0)

        return e, u, v, spon_sep, state_sep, spon_vac, state_vac, jtj, utj

    def run_model(self):
        (
            e,
            u,
            v,
            spon_sep,
            state_sep,
            spon_vac,
            state_vac,
            jtj,
            utj,
        ) = self.initialize_variables()

        for t in range(1, self.T):
            (
                e,
                u,
                v,
                spon_sep,
                state_sep,
                spon_vac,
                state_vac,
                jtj,
                utj,
            ) = self.time_step(e, u, v, t)
            self.employment[:, t] = e
            self.unemployment[:, t] = u
            self.vacancies[:, t] = v

            # other things to keep track
            self.spon_separations[:, t] = spon_sep
            self.state_separations[:, t] = state_sep
            self.spon_vacancies[:, t] = spon_vac
            self.state_vacancies[:, t] = state_vac

            self.from_job_to_occ[:, t] = jtj
            self.from_unemp_to_occ[:, t] = utj

        return e, u, v, spon_sep, state_sep, spon_vac, state_vac, jtj, utj

    def calculate_aggregates(self):
        total_unemployment = self.unemployment.sum(axis=0)
        total_vacancies = self.vacancies.sum(axis=0)
        total_employment = self.employment.sum(axis=0)
        total_demand = self.d_dagger.sum(axis=0)
        # spontaneous and state dep separations
        total_spon_sep = self.spon_separations.sum(axis=0)
        total_state_sep = self.state_separations.sum(axis=0)
        total_spon_vac = self.spon_vacancies.sum(axis=0)
        total_state_vac = self.state_vacancies.sum(axis=0)
        # job to job transitions
        jtj = self.from_job_to_occ[:, 0].clone()
        utj = self.from_unemp_to_occ[:, 0].clone()

        return (
            total_unemployment,
            total_vacancies,
            total_employment,
            total_demand,
            total_spon_sep,
            total_state_sep,
            total_spon_vac,
            total_state_vac,
            jtj,
            utj,
        )

    def calculate_rates(self):
        (
            total_unemployment,
            total_vacancies,
            total_employment,
            total_demand,
            total_spon_sep,
            total_state_sep,
            total_spon_vac,
            total_state_vac,
            jtj,
            utj,
        ) = self.calculate_aggregates()

        unemployment_rate = 100 * total_unemployment / (total_employment + total_unemployment)
        employment_rate = 100 - unemployment_rate
        vacancy_rate = 100 * total_vacancies / (total_vacancies + total_employment)

        # rates in terms of separations
        sep_ratio = total_spon_sep / total_state_sep
        vac_ratio = total_spon_vac / total_state_vac

        # rates in terms of job transitions
        jtj_over_utj = jtj / utj

        return (
            unemployment_rate,
            employment_rate,
            vacancy_rate,
            sep_ratio,
            vac_ratio,
            jtj_over_utj,
        )


N = 10
T = 100
seed = 111
delta_u = 0.016
delta_v = 0.017
gamma_u = 10 * delta_u
gamma_v = gamma_u
lam = 0.01
beta_u = 10
beta_e = 1


def create_symmetric_A_euv_d_daggers(n, t, seed=None):
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
    e = torch.rand(n)
    u = 0.05 * e  # 5% of e
    v = 0.05 * e  # 5% of e

    # Create d_dagger using broadcasting
    sum_e_u = e + u
    d_dagger = sum_e_u.unsqueeze(1).repeat(1, t)  # Repeat the sum across t columns

    wages = torch.rand(n)

    return symmetric_matrix, e, u, v, d_dagger, wages


# A, e, u, v, d_dagger, wages = create_symmetric_A_euv_d_daggers(N, T, seed)

# lab_abm = LabourABM(N, T, seed, delta_u, delta_u, gamma_u, gamma_v, lam, beta_u, beta_e, A, d_dagger, e, u, v, wages)

# lab_abm.run_model()

# initial_employment = torch.rand(N)

# # probability of on-the-job search

# # n applications by employed and unemployed
# beta_e = float,
# beta_u = float,
# # Ease of Transition matrix (assumed known for now)
# A =  torch.Tensor,
# # Scenario/target demand
# d_dagger = torch.Tensor,
# # Employment data
# initial_employment: torch.Tensor,  # Initial employment data
# initial_unemployment: torch.Tensor,  # Initial unemployment data
# initial_vacancies: torch.Tensor,  # Initial vacancies data
# wages: torch.Tensor,


# LabourABM()

# # example of running a configuration

# employment = pd.read_csv("employment.csv")
# transition_matrix = pd.read_csv("transition_matrix.csv")

# labour = LabourABM.from_data(employment, transition_matrix)

# labour.run_simulation()

# simulation_length = 100

# unemployment_rate = np.zeros(100)

# for i in range(simulation_length):
#     labour.update()
#     unemployment_rate[i] = labour.get_unemployment_rate()
