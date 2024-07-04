from functools import cached_property
from typing import Tuple

import torch

from labor_abm_canada.configuration.configuration import ModelConfiguration


def calc_job_offers(vacancies: torch.Tensor, number_applications: torch.Tensor) -> torch.Tensor:
    """Calculate the number of job offers for each occupation.
    The probability that a job is offered is given by 1-exp(-n/v) for each vacancy, where n is the number of
    applications and v is the number of vacancies. The number of job offers is then given by n * p, where n is the
    number of vacancies and p is the probability defined above.

    Parameters
    ----------
    vacancies : torch.Tensor
        Vector of vacancy levels for each occupation.
    number_applications : torch.Tensor
        Matrix of the number of applications sent by workers to each vacancy for each occupation.

    Returns
    -------
    torch.Tensor
        Matrix of the number of job offers for each occupation.
    """
    v_inv = torch.reciprocal(vacancies)
    v_inv[torch.isinf(v_inv)] = 0

    prob_job_offer = 1 - torch.exp(-torch.multiply(number_applications, v_inv))

    job_offers = torch.multiply(vacancies, prob_job_offer)

    return job_offers


class LabourABM:

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
        """
        A class to represent an Agent Based Model of the labour market.

        Attributes
        ----------
        n : int
            Number of occupations in the model.
        t_max : int
            Number of time steps to run the model.
        separation_rate : float
            Rate of spontaneous separations (firings).
        opening_rate : float
            Rate of spontaneous job openings.
        adaptation_rate_u : float
            Rate of state-dependent separations.
        adaptation_rate_v : float
            Rate of state-dependent job openings.
        otjob_search_prob : float
            Probability of an employed worker to search for a job outside their occupation.
        n_applications_emp : int
            Number of applications sent by employed workers.
        n_applications_unemp : int
            Number of applications sent by unemployed workers.
        transition_matrix : torch.Tensor
            Matrix of transition probabilities between occupations.
        demand_scenario : torch.Tensor
            Vector of demand for each occupation at each time step.
        employment : torch.Tensor
            Matrix of employment levels for each occupation at each time step.
            (must be initialised)
        unemployment : torch.Tensor
            Matrix of unemployment levels for each occupation at each time step.
            (must be initialised)
        vacancies : torch.Tensor
            Matrix of vacancy levels for each occupation at each time step.
            (must be initialised)
        wages : torch.Tensor
            Vector of average wages for each occupation.
        spontaneous_separations : torch.Tensor
            Matrix of spontaneous separations for each occupation at each time step.
        state_separations : torch.Tensor
            Matrix of state-dependent separations for each occupation at each time step.
        spontaneous_vacancies : torch.Tensor
            Matrix of spontaneous vacancies for each occupation at each time step.
        state_vacancies : torch.Tensor
            Matrix of state-dependent vacancies for each occupation at each time step.
        from_job_to_occ : torch.Tensor
            Matrix of job-to-job transitions for each occupation at each time step.
        from_unemp_to_occ : torch.Tensor
            Matrix of unemployment-to-job transitions for each occupation at each time step.


        Methods
        -------
        compute_spontaneous_separations(e)
            Compute the number of spontaneous separations for each occupation given the employment levels.
        compute_spontaneous_openings(e)
            Compute the number of spontaneous job openings for each occupation given the employment levels.
        state_dep_separations(diff_demand)
            Compute the number of state-dependent separations for each occupation given the difference between
            current demand and target demand.
        state_dep_openings(diff_demand)
            Compute the number of state-dependent job openings for each occupation given the difference between
            current demand and target demand.
        calc_attractiveness_vacancy()
            Calculate the attractiveness of each vacancy for each occupation.
        calc_probability_applying(v)
            Calculate the probability of applying to each vacancy for each occupation.
        calc_applicants_and_applications_sent(e, u, v)
            Calculate the number of applicants and applications sent for each occupation.
        calc_prob_workers_with_offers(v, sj)
            Calculate the probability of workers receiving job offers for each occupation.
        time_step(t)
            Run one time step of the model.
        run_model()
            Run the model for t_max time steps.
        calculate_aggregates()
            Calculate the aggregate variables of the model.
        calculate_rates()
            Calculate the rates of unemployment, employment, and job vacancies.
        """

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
    ) -> "LabourABM":
        """
        Default initialiser of the LabourABM class.
        It initialises the model with the given configuration and initial values for employment, unemployment,
        and vacancies.

        Parameters
        ----------
        model_configuration : ModelConfiguration
            Configuration of the model.
        transition_matrix : torch.Tensor
            Matrix of transition probabilities between occupations.
        demand_scenario : torch.Tensor
            Vector of demand for each occupation at each time step.
        wages : torch.Tensor
            Vector of average wages for each occupation.
        initial_employment : torch.Tensor
            Vector of initial employment levels for each occupation.
        initial_unemployment : torch.Tensor
            Vector of initial unemployment levels for each occupation.
        initial_vacancies : torch.Tensor
            Vector of initial vacancy levels for each occupation.

        Returns
        -------
        LabourABM
            A new instance of the LabourABM class.

        """
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
            spontaneous_separations=torch.zeros((model_configuration.n, model_configuration.t_max)),
            state_separations=torch.zeros((model_configuration.n, model_configuration.t_max)),
            spontaneous_vacancies=torch.zeros((model_configuration.n, model_configuration.t_max)),
            state_vacancies=torch.zeros((model_configuration.n, model_configuration.t_max)),
            from_job_to_occ=torch.zeros((model_configuration.n, model_configuration.t_max)),
            from_unemp_to_occ=torch.zeros((model_configuration.n, model_configuration.t_max)),
        )

    def compute_spontaneous_separations(self, employment: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of spontaneous separations for each occupation given the employment levels.
        Given by the separation rate times the employment level.

        Parameters
        ----------
        employment : torch.Tensor
            Vector of employment levels for each occupation.

        Returns
        -------
        torch.Tensor
            Vector of spontaneous separations for each occupation

        """
        return self.separation_rate * employment

    def compute_spontaneous_openings(self, employment: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of spontaneous job openings for each occupation given the employment levels.
        Given by the opening rate times the employment level.

        Parameters
        ----------
        employment : torch.Tensor
            Vector of employment levels for each occupation.

        Returns
        -------
        torch.Tensor
            Vector of spontaneous job openings for each occupation
        """
        return self.opening_rate * employment

    def state_dep_separations(self, diff_demand: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of state-dependent separations for each occupation given the difference between
        current demand and target demand.

        Parameters
        ----------
        diff_demand : torch.Tensor
            Vector of differences between current demand and target demand for each occupation.

        Returns
        -------
        torch.Tensor
            Vector of state-dependent separations for each occupation
        """
        return (1 - self.separation_rate) * self.adaptation_rate_u * torch.clip(diff_demand, min=0)

    def state_dep_openings(self, diff_demand: torch.Tensor) -> torch.Tensor:
        """
        Compute the number of state-dependent job openings for each occupation given the difference between
        current demand and target demand.

        Parameters
        ----------
        diff_demand : torch.Tensor
            Vector of differences between current demand and target demand for each occupation.

        Returns
        -------
        torch.Tensor
            Vector of state-dependent job openings for each occupation

        """
        return (1 - self.opening_rate) * self.adaptation_rate_v * torch.clip(-diff_demand, min=0)

    # Search process
    @cached_property
    def vacancy_attractiveness(self) -> torch.Tensor:
        """
        Calculate the attractiveness of each vacancy for each occupation.
        The attractiveness is given by the product of the wages and the transition matrix,
        so that the attractiveness a_i = w_i * q_ij.
        """
        attractiveness = torch.mul(self.wages, self.transition_matrix)
        return attractiveness

    def calc_probability_applying(self, vacancies: torch.Tensor) -> torch.Tensor:
        """Computes the probability of applying to each vacancy for each occupation.
        It is proportional to the product of the attractiveness of the vacancy and the number of vacancies.

        Parameters
        ----------
        vacancies : torch.Tensor
            Vector of vacancy levels for each occupation.

        Returns
        -------
        torch.Tensor
            Matrix of probabilities of applying to each vacancy for each occupation."""
        attractiveness = self.vacancy_attractiveness
        unnorm_prob = torch.mul(vacancies, attractiveness)
        prob = unnorm_prob / torch.sum(unnorm_prob, dim=1, keepdim=True)
        return prob

    def calc_applicants_and_applications_sent(
        self, employment: torch.Tensor, unemployment: torch.Tensor, vacancies: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the number of applicants and applications sent for each occupation.

        Parameters
        ----------
        employment : torch.Tensor
            Vector of employment levels for each occupation.
        unemployment : torch.Tensor
            Vector of unemployment levels for each occupation.
        vacancies : torch.Tensor
            Vector of vacancy levels for each occupation.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - employed_applicant_fraction : torch.Tensor
                Matrix of the fraction of employed workers applying to each vacancy for each occupation.
            - unemployed_application_fraction : torch.Tensor
                Matrix of the fraction of unemployed workers applying to each vacancy for each occupation.
            - number_applications_employed : torch.Tensor
                Matrix of the number of applications sent by employed workers to each vacancy for each occupation.
            - number_applications_unemployed : torch.Tensor
                Matrix of the number of applications sent by unemployed workers to each vacancy for each occupation.
        """
        prob_applying = self.calc_probability_applying(vacancies)

        # applicants
        employed_applicant_fraction = self.otjob_search_prob * torch.mul(employment[:, None], prob_applying)
        unemployed_application_fraction = torch.mul(unemployment[:, None], prob_applying)

        # sij(e) = β_e*ei*qij
        number_applications_employed = self.n_applications_emp * employed_applicant_fraction
        # sij(u) = β_u*ui*qij
        number_applications_unemployed = self.n_applications_unemp * unemployed_application_fraction

        return (
            employed_applicant_fraction,
            unemployed_application_fraction,
            number_applications_employed,
            number_applications_unemployed,
        )

    def calc_prob_workers_with_offers(
        self, vacancies: torch.Tensor, number_applications: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the probability of workers receiving job offers for each occupation.

        Parameters
        ----------
        vacancies : torch.Tensor
            Vector of vacancy levels for each occupation.
        number_applications : torch.Tensor
            Matrix of the number of applications sent by workers to each vacancy for each occupation.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - got_offer_e : torch.Tensor
                Vector of the probability of employed workers receiving job offers for each occupation.
            - got_offer_u : torch.Tensor
                Vector of the probability of unemployed workers receiving job offers for each occupation
        """

        job_offers = calc_job_offers(vacancies, number_applications)
        # (beta_apps - l); where l = 0 to beta
        active_applications_from_u = torch.repeat_interleave(number_applications, self.n_applications_unemp).reshape(
            self.n, self.n_applications_unemp
        ) - torch.tensor(range(self.n_applications_unemp))
        active_applications_from_e = torch.repeat_interleave(number_applications, self.n_applications_emp).reshape(
            self.n, self.n_applications_emp
        ) - torch.tensor(range(self.n_applications_emp))
        # prob of an app x not being drawn
        # 1 - job_offers / (beta_apps - l); where l = 0 to beta
        prob_no_app_selected_u = 1 - torch.mul(job_offers[:, None], 1.0 / active_applications_from_u)
        prob_no_app_selected_e = 1 - torch.mul(job_offers[:, None], 1.0 / active_applications_from_e)
        # prob none of those apps is drawn
        no_offer_u = torch.prod(prob_no_app_selected_u, dim=1)
        no_offer_e = torch.prod(prob_no_app_selected_e, dim=1)
        # worker getting offer
        got_offer_u = 1 - no_offer_u
        got_offer_e = 1 - no_offer_e

        return got_offer_e, got_offer_u

    # simulation
    def time_step(self, t: int):
        """
        Run one time step of the model.

        The following steps are performed:
        - Compute the number of spontaneous separations and job openings.
        - Compute the number of state-dependent separations and job openings, by first computing the current realised
        demand (sum of employment and vacancies), then computing the difference between the realised demand and the
        target demand, dictated by the demand scenario.
        - Compute the number of applicants and applications sent for each occupation.
        - Compute the probability of workers receiving job offers for each occupation.
        - Match workers and update the employment, unemployment, and vacancies levels for each occupation.

        Parameters
        ----------
        t : int
            Current time step.
        """
        # workers separations and job openings
        d = self.employment[:, t - 1] + self.vacancies[:, t - 1]
        diff_demand = d - self.demand_scenario[:, t]
        spon_sep = self.compute_spontaneous_separations(self.employment[:, t - 1])
        state_sep = self.state_dep_separations(diff_demand)
        spon_vac = self.compute_spontaneous_openings(self.employment[:, t - 1])
        state_vac = self.state_dep_openings(diff_demand)
        separated_workers = spon_sep + state_sep
        opened_vacancies = spon_vac + state_vac

        # job search
        aij_e, aij_u, sij_e, sij_u = self.calc_applicants_and_applications_sent(
            self.employment[:, t - 1],
            self.unemployment[:, t - 1],
            self.vacancies[:, t - 1],
        )
        sj = (sij_u + sij_e).sum(dim=0)

        # matching
        prob_offer_e, prob_offer_u = self.calc_prob_workers_with_offers(self.vacancies[:, t - 1], sj)
        # what about acceptance?
        fij_e = aij_e * prob_offer_e
        fij_u = aij_u * prob_offer_u
        fij = fij_u + fij_e
        # NOTE Uncomment blow below to run baseline
        ####################
        # job_offers = self.calc_job_offers(v, sj)
        # # print("job offers ", job_offers[5]/sj[5])
        # fij_u = sij_u * job_offers/sj
        # fij_e = sij_e * job_offers/sj
        # fij = fij_u + fij_e
        # # print("F[3,5], ", fij[3,5])
        # # print("F_e[3,5] should be 0 , ", fij_e[3,5])
        #####################

        # TODO check right sum
        jtj = fij_e.sum(dim=1)
        utj = fij_u.sum(dim=1)

        # update

        self.employment[:, t] = self.employment[:, t - 1] - separated_workers + fij.sum(dim=0) - fij_e.sum(dim=1)

        self.unemployment[:, t] = self.unemployment[:, t - 1] + separated_workers - fij_u.sum(dim=0)

        self.vacancies[:, t] = self.vacancies[:, t - 1] + opened_vacancies - fij.sum(dim=0)

        self.spontaneous_separations[:, t] = spon_sep
        self.state_separations[:, t] = state_sep
        self.spontaneous_vacancies[:, t] = spon_vac
        self.state_vacancies[:, t] = state_vac

        self.from_job_to_occ[:, t] = jtj
        self.from_unemp_to_occ[:, t] = utj

    def run_model(self):
        """Run the model for t_max time steps."""
        for t in range(1, self.t_max):
            self.time_step(t)

    def calculate_aggregates(self):
        """Calculate the aggregate variables of the model.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            - total_unemployment : torch.Tensor
                Vector of total unemployment levels at each time step.
            - total_vacancies : torch.Tensor
                Vector of total vacancy levels at each time step.
            - total_employment : torch.Tensor
                Vector of total employment levels at each time step.
            - total_demand : torch.Tensor
                Vector of total demand levels at each time step.
            - total_spontaneous_separations : torch.Tensor
                Vector of total spontaneous separations at each time step.
            - total_state_separations : torch.Tensor
                Vector of total state-dependent separations at each time step.
            - total_spontaneous_vacancies : torch.Tensor
                Vector of total spontaneous vacancies at each time step.
            - total_state_vacancies : torch.Tensor
                Vector of total state-dependent vacancies at each time step.
            - job_to_job_transitions : torch.Tensor
                Vector of job-to-job transitions at each time step.
            - unemployment_to_job_transitions : torch.Tensor
                Vector of unemployment-to-job transitions at each time step."""

        total_unemployment = (self.unemployment.sum(dim=0),)
        total_vacancies = (self.vacancies.sum(dim=0),)
        total_employment = (self.employment.sum(dim=0),)
        total_demand = (self.demand_scenario.sum(dim=0),)
        total_spontaneous_separations = (self.spontaneous_separations.sum(dim=0),)
        total_state_separations = (self.state_separations.sum(dim=0),)
        total_spontaneous_vacancies = (self.spontaneous_vacancies.sum(dim=0),)
        total_state_vacancies = (self.state_vacancies.sum(dim=0),)
        job_to_job_transitions = (self.from_job_to_occ[:, 0].clone(),)
        unemployment_to_job_transitions = (self.from_unemp_to_occ[:, 0].clone(),)

        return (
            total_unemployment,
            total_vacancies,
            total_employment,
            total_demand,
            total_spontaneous_separations,
            total_state_separations,
            total_spontaneous_vacancies,
            total_state_vacancies,
            job_to_job_transitions,
            unemployment_to_job_transitions,
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


def create_symmetric_a_euv_d_daggers(
    n: int, t: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a random symmetric matrix with 1 on the diagonal and entries between 0 and 1,
    and create vectors e, u, v, and matrix d_dagger where each column is e + u.
    """

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
