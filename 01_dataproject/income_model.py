import numpy as np


class IncomeModelClass:
    """Life-cycle simulation model of income and human capital."""

    def __init__(self, seed=1234):
        """Set model parameters and random number generator."""

        # a. random number generator
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # b. population and age
        self.N = 50_000
        self.age_start = 18
        self.age_retirement = 65

        # c. education groups
        self.education_names = np.array([
            'short',
            'medium',
            'long'
        ])

        # d. education probabilities
        self.p_education = np.array([
            0.40,
            0.35,
            0.25
        ])

        # e. years of education
        self.years_education = np.array([
            1,
            3,
            5
        ])

        # f. initial human capital
        self.h_initial = np.array([
            1.00,
            1.20,
            1.55
        ])

        # g. growth of human capital while employed
        self.delta_education = np.array([
            0.010,
            0.020,
            0.030
        ])

        # h. depreciation while unemployed
        self.delta = 0.06

        # i. standard deviation of human-capital shock
        self.sigma_psi = 0.10

        # j. labour-market transition probabilities
        self.lambda_job = 0.60
        self.sigma_job = 0.05

        # k. income parameters
        self.student_grant = 0.45
        self.replacement_rate = 0.60
        self.benefit_floor = 0.35

        # l. standard deviation of temporary wage shock
        # zero in the baseline model
        self.sigma_wage = 0.0

    def draw_education(self):
        """Draw an education group for each individual."""

        # a. draw education index
        self.education_index = self.rng.choice(
            3,
            size=self.N,
            p=self.p_education
        )

        # b. assign education names
        self.education = self.education_names[
            self.education_index
        ]

        # c. assign years of education
        self.education_years = self.years_education[
            self.education_index
        ]

        # d. assign initial human capital
        self.human_capital_initial = self.h_initial[
            self.education_index
        ]

        # e. assign education-specific human-capital growth
        self.human_capital_growth = self.delta_education[
            self.education_index
        ]

    def setup_life_cycle(self):
        """Set up ages and labour-market entry ages."""

        # a. create age grid
        self.ages = np.arange(
            self.age_start,
            self.age_retirement
        )

        # b. number of periods in the simulation
        self.T = len(self.ages)

        # c. calculate age of labour-market entry
        self.labour_market_entry_age = (
            self.age_start
            + self.education_years
        )

    def simulate_employment(self):
        """Simulate employment status over the life cycle."""

        # a. initialize employment matrix
        # False = not employed
        # True = employed
        self.employed = np.zeros(
            (self.N, self.T),
            dtype=bool
        )

        # b. loop over ages
        for t, age in enumerate(self.ages):

            # i. individuals entering the labour market
            entering = (
                age == self.labour_market_entry_age
            )

            # individuals enter as unemployed
            self.employed[entering, t] = False

            # ii. skip first period
            if t == 0:
                continue

            # iii. individuals already in the labour market
            active = (
                age > self.labour_market_entry_age
            )

            # iv. employment status in previous period
            employed_last = self.employed[:, t-1]

            # v. random transition draw
            random_draw = self.rng.uniform(
                size=self.N
            )

            # vi. unemployed individuals who find a job
            find_job = (
                active
                & ~employed_last
                & (random_draw < self.lambda_job)
            )

            # vii. employed individuals who keep their job
            keep_job = (
                active
                & employed_last
                & (random_draw >= self.sigma_job)
            )

            # viii. update employment status
            self.employed[find_job, t] = True
            self.employed[keep_job, t] = True

    def simulate_human_capital(self):
        """Simulate human capital over the life cycle."""

        # a. initialize human-capital matrix
        self.human_capital = np.zeros(
            (self.N, self.T)
        )

        # b. human capital at age 18
        self.human_capital[:, 0] = (
            self.human_capital_initial
        )

        # c. loop over transitions between ages
        for t in range(1, self.T):

            # previous age determines the transition
            previous_age = self.ages[t-1]

            # i. individuals in education in previous period
            in_education_previous = (
                previous_age
                < self.labour_market_entry_age
            )

            # human capital is unchanged while in education
            self.human_capital[
                in_education_previous, t
            ] = self.human_capital[
                in_education_previous, t-1
            ]

            # ii. individuals in the labour market
            # in the previous period
            active_previous = (
                previous_age
                >= self.labour_market_entry_age
            )

            # iii. draw mean-one lognormal shock
            psi = self.rng.lognormal(
                -0.5 * self.sigma_psi**2,
                self.sigma_psi,
                size=self.N
            )

            # iv. employed in previous period
            employed_previous = (
                active_previous
                & self.employed[:, t-1]
            )

            self.human_capital[
                employed_previous, t
            ] = (
                self.human_capital[
                    employed_previous, t-1
                ]
                * (
                    1
                    + self.human_capital_growth[
                        employed_previous
                    ]
                )
                * psi[employed_previous]
            )

            # v. unemployed in previous period
            unemployed_previous = (
                active_previous
                & ~self.employed[:, t-1]
            )

            self.human_capital[
                unemployed_previous, t
            ] = (
                self.human_capital[
                    unemployed_previous, t-1
                ]
                * (1 - self.delta)
                * psi[unemployed_previous]
            )

    def simulate_income(self):
        """Simulate income over the life cycle."""

        # a. initialize income matrix
        self.income = np.zeros(
            (self.N, self.T)
        )

        # b. store underlying income from most recent job
        last_job_income = np.zeros(self.N)

        # c. loop over ages
        for t, age in enumerate(self.ages):

            # i. individuals in education
            in_education = (
                age < self.labour_market_entry_age
            )

            self.income[in_education, t] = (
                self.student_grant
            )

            # ii. individuals in labour market
            in_labour_market = (
                age >= self.labour_market_entry_age
            )

            # iii. employed individuals
            employed = (
                in_labour_market
                & self.employed[:, t]
            )

            # iv. draw mean-one temporary wage shock
            wage_shock = self.rng.lognormal(
                -0.5 * self.sigma_wage**2,
                self.sigma_wage,
                size=self.N
            )

            # v. labour income
            self.income[employed, t] = (
                self.human_capital[employed, t]
                * wage_shock[employed]
            )

            # vi. store underlying wage before
            # the temporary wage shock
            last_job_income[employed] = (
                self.human_capital[employed, t]
            )

            # vii. unemployed individuals
            unemployed = (
                in_labour_market
                & ~self.employed[:, t]
            )

            # viii. unemployed with previous employment
            previously_employed = (
                unemployed
                & (last_job_income > 0)
            )

            self.income[
                previously_employed, t
            ] = (
                self.replacement_rate
                * last_job_income[
                    previously_employed
                ]
            )

            # ix. unemployed who have never been employed
            never_employed = (
                unemployed
                & (last_job_income == 0)
            )

            self.income[never_employed, t] = (
                self.benefit_floor
            )

    def simulate(self):
        """Run the complete life-cycle simulation."""

        # a. reset random number generator
        self.rng = np.random.default_rng(
            self.seed
        )

        # b. draw education
        self.draw_education()

        # c. set up life-cycle timing
        self.setup_life_cycle()

        # d. simulate employment
        self.simulate_employment()

        # e. simulate human capital
        self.simulate_human_capital()

        # f. simulate income
        self.simulate_income()