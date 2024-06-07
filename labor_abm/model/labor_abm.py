import torch

from labor_abm.configuration.model_configuration import ModelConfiguration
from labor_abm.model.aggregates import Aggregates


def calc_job_offers(v, sj):
    """Calculate number of vacancies that received at least one applicant"""
    v_inv = torch.reciprocal(v)
    v_inv[torch.isinf(v_inv)] = 0

    prob_job_offer = 1 - torch.exp(-torch.multiply(sj, v_inv))

    job_offers = torch.multiply(v, prob_job_offer)

    return job_offers


class LaborABM:
    def __init__(
        self,
        n: int,
        t_max: int,
        separation_rate: float,
        opening_rate: float,
        adaptation_rate_u: float,
        adaptation_rate_v: float,
        otjob_search_prob: float,
        n_applications_emp: int,
        n_applications_unemp: int,
        transition_matrix: torch.Tensor,
        demand_scenario: torch.Tensor,
        employment: torch.Tensor,
        unemployment: torch.Tensor,
        vacancies: torch.Tensor,
        spontaneous_separations: torch.Tensor,
        state_separations: torch.Tensor,
        spontaneous_vacancies: torch.Tensor,
        state_vacancies: torch.Tensor,
        from_job_to_occ: torch.Tensor,
        from_unemp_to_occ: torch.Tensor,
        wages: torch.Tensor,
    ):
        self.n = n
        self.t_max = t_max
        self.separation_rate = separation_rate
        self.opening_rate = opening_rate
        self.adaptation_rate_u = adaptation_rate_u
        self.adaptation_rate_v = adaptation_rate_v
        self.otjob_search_prob = otjob_search_prob
        self.n_applications_emp = n_applications_emp
        self.n_applications_unemp = n_applications_unemp
        self.transition_matrix = transition_matrix
        self.demand_scenario = demand_scenario
        self.employment = employment
        self.unemployment = unemployment
        self.vacancies = vacancies
        self.wages = wages

        self.spontaneous_separations = spontaneous_separations
        self.state_separations = state_separations
        self.spontaneous_vacancies = spontaneous_vacancies
        self.state_vacancies = state_vacancies
        self.from_job_to_occ = from_job_to_occ
        self.from_unemp_to_occ = from_unemp_to_occ

    @classmethod
    def default_create(
        cls,
        model_configuration: ModelConfiguration,
        transition_matrix: torch.Tensor,
        demand_scenario: torch.Tensor,
        wages: torch.Tensor,
        initial_employment: torch.Tensor,
        initial_unemployment: torch.Tensor,
        initial_vacancies: torch.Tensor,
    ):
        unemployment = torch.zeros((model_configuration.n, model_configuration.t_max))
        unemployment[:, 0] = initial_unemployment
        employment = torch.zeros((model_configuration.n, model_configuration.t_max))
        employment[:, 0] = initial_employment
        vacancies = torch.zeros((model_configuration.n, model_configuration.t_max))
        vacancies[:, 0] = initial_vacancies

        return cls(
            n=model_configuration.n,
            t_max=model_configuration.t_max,
            separation_rate=model_configuration.labor.separation_rate,
            opening_rate=model_configuration.labor.opening_rate,
            adaptation_rate_u=model_configuration.labor.adaptation_rate_u,
            adaptation_rate_v=model_configuration.labor.adaptation_rate_v,
            otjob_search_prob=model_configuration.labor.otjob_search_prob,
            n_applications_emp=model_configuration.labor.n_applications_emp,
            n_applications_unemp=model_configuration.labor.n_applications_unemp,
            transition_matrix=transition_matrix,
            demand_scenario=demand_scenario,
            employment=employment,
            unemployment=unemployment,
            vacancies=vacancies,
            wages=wages,
            spontaneous_separations=torch.zeros(
                (model_configuration.n, model_configuration.t_max)
            ),
            state_separations=torch.zeros(
                (model_configuration.n, model_configuration.t_max)
            ),
            spontaneous_vacancies=torch.zeros(
                (model_configuration.n, model_configuration.t_max)
            ),
            state_vacancies=torch.zeros(
                (model_configuration.n, model_configuration.t_max)
            ),
            from_job_to_occ=torch.zeros(
                (model_configuration.n, model_configuration.t_max)
            ),
            from_unemp_to_occ=torch.zeros(
                (model_configuration.n, model_configuration.t_max)
            ),
        )

    def compute_spontaneous_separations(self, e: torch.Tensor):
        return self.separation_rate * e

    def state_dep_separations(self, diff_demand):
        return (
            (1 - self.separation_rate)
            * self.adaptation_rate_u
            * torch.maximum(torch.zeros(self.n), diff_demand)
        )

    def spontaneous_openings(self, e):
        return self.opening_rate * e

    def state_dep_openings(self, diff_demand):
        return (
            (1 - self.opening_rate)
            * self.adaptation_rate_v
            # * torch.maximum(torch.zeros(self.N), -diff_demand)
            * torch.clip(-diff_demand, min=0)
        )

    def calc_attractiveness_vacancy(self):
        """wage(np.array) average wage in category
        A ease of transitioning between categories
        """
        attractiveness = torch.mul(self.wages, self.transition_matrix)
        return attractiveness

    def calc_probability_applying(self, v):
        """How attractive a is a vacancy depending on skills, geography, wage, etc"""
        attractiveness = self.calc_attractiveness_vacancy()
        Av = torch.mul(v, attractiveness)
        Q = Av / torch.sum(Av, dim=1, keepdim=True)
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

    def calc_prob_workers_with_offers(self, v, sj):
        job_offers = calc_job_offers(v, sj)
        # (beta_apps - l); where l = 0 to beta
        active_applications_from_u = torch.repeat_interleave(
            sj, self.n_applications_unemp
        ).reshape(self.n, self.n_applications_unemp) - torch.tensor(
            range(self.n_applications_unemp)
        )
        active_applications_from_e = torch.repeat_interleave(
            sj, self.n_applications_emp
        ).reshape(self.n, self.n_applications_emp) - torch.tensor(
            range(self.n_applications_emp)
        )
        # prob of an app x not being drawn
        # 1 - job_offers / (beta_apps - l); where l = 0 to beta
        prob_no_app_selected_u = 1 - torch.mul(
            job_offers[:, None], 1.0 / active_applications_from_u
        )
        prob_no_app_selected_e = 1 - torch.mul(
            job_offers[:, None], 1.0 / active_applications_from_e
        )
        # prob none of those apps is drawn
        no_offer_u = torch.prod(prob_no_app_selected_u, dim=1)
        no_offer_e = torch.prod(prob_no_app_selected_e, dim=1)
        # worker getting offer
        got_offer_u = 1 - no_offer_u
        got_offer_e = 1 - no_offer_e

        return got_offer_e, got_offer_u

    def time_step(self, t: int):
        # workers separationa and job openings
        d = self.employment[:, t - 1] + self.vacancies[:, t - 1]
        diff_demand = d - self.demand_scenario[:, t]
        spon_sep = self.compute_spontaneous_separations(self.employment[:, t - 1])
        state_sep = self.state_dep_separations(diff_demand)
        spon_vac = self.spontaneous_openings(self.employment[:, t - 1])
        state_vac = self.state_dep_openings(diff_demand)

        separated_workers = spon_sep + state_sep
        opened_vacancies = spon_vac + state_vac

        # search
        aij_e, aij_u, sij_e, sij_u = self.calc_applicants_and_applications_sent(
            self.employment[:, t - 1],
            self.unemployment[:, t - 1],
            self.vacancies[:, t - 1],
        )
        ### NOTE / to-do, make sj come be calculated inside a function
        sj = (sij_u + sij_e).sum(dim=0)

        # matching
        prob_offer_e, prob_offer_u = self.calc_prob_workers_with_offers(
            self.vacancies[:, t - 1], sj
        )
        # what about acceptance?
        Fij_u = aij_u * prob_offer_u
        Fij_e = aij_e * prob_offer_e
        Fij = Fij_u + Fij_e

        # TODO check right sum
        jtj = Fij_e.sum(dim=1)
        utj = Fij_u.sum(dim=1)
        # updating values
        # e += -separated_workers + Fij.sum(dim=0) - Fij_e.sum(dim=1)
        # u += separated_workers - Fij_u.sum(dim=0)
        # v += opened_vacancies - Fij.sum(dim=0)

        self.employment[:, t] = (
            self.employment[:, t - 1]
            - separated_workers
            + Fij.sum(dim=0)
            - Fij_e.sum(dim=1)
        )

        self.unemployment[:, t] = (
            self.unemployment[:, t - 1] + separated_workers - Fij_u.sum(dim=0)
        )

        self.vacancies[:, t] = (
            self.vacancies[:, t - 1] + opened_vacancies - Fij.sum(dim=0)
        )

        self.spontaneous_separations[:, t] = spon_sep
        self.state_separations[:, t] = state_sep
        self.spontaneous_vacancies[:, t] = spon_vac
        self.state_vacancies[:, t] = state_vac

        self.from_job_to_occ[:, t] = jtj
        self.from_unemp_to_occ[:, t] = utj

    def run_model(self):
        for t in range(1, self.t_max):
            self.time_step(t)

    @property
    def aggregates(self) -> Aggregates:
        return Aggregates(
            total_unemployment=self.unemployment.sum(dim=0),
            total_vacancies=self.vacancies.sum(dim=0),
            total_employment=self.employment.sum(dim=0),
            total_demand=self.demand_scenario.sum(dim=0),
            total_spontaneous_separations=self.spontaneous_separations.sum(dim=0),
            total_state_separations=self.state_separations.sum(dim=0),
            total_spontaneous_vacancies=self.spontaneous_vacancies.sum(dim=0),
            total_state_vacancies=self.state_vacancies.sum(dim=0),
            job_to_job_transitions=self.from_job_to_occ[:, 0].clone(),
            unemployment_to_job_transitions=self.from_unemp_to_occ[:, 0].clone(),
        )
