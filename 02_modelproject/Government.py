from types import SimpleNamespace

import numpy as np

from scipy import optimize

from Consumer import ConsumerClass


class GovernmentClass(ConsumerClass):
    """ a government raising revenue from the consumer in Consumer.py

    Two kinds of instrument:

    1) a lump-sum tax T, which reduces income
    2) product taxes tau1, tau2, tau3, which raise the prices

    The consumer is the ConsumerClass, so everything from there is inherited.

    """

    def __init__(self,par=None):

        # a. I set up the default consumer parameters
        self.setup()

        # b. I add the government parameters
        self.setup_government()

        # c. I overwrite parameters if I provide an alternative calibration
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

        # d. I store the prices and income before taxes are introduced
        self.sync_pre_tax()

    def setup_government(self):
        """ add the tax instruments to the parameters """

        par = self.par

        # a. I initialize the lump-sum tax
        par.T = 0.0

        # b. I initialize the three product-tax rates
        par.tau1 = 0.0
        par.tau2 = 0.0
        par.tau3 = 0.0

    def sync_pre_tax(self):
        """ store the current prices and income as the situation without taxes """

        par = self.par

        # a. I store the original prices
        par.p1_pre = par.p1
        par.p2_pre = par.p2
        par.p3_pre = par.p3

        # b. I store the original income
        par.I_pre = par.I

    ##############################
    # 1. what the consumer faces #
    ##############################

    def set_taxes(self,T=0.0,tau1=0.0,tau2=0.0,tau3=0.0):
        """ set taxes and update consumer prices and disposable income """

        par = self.par

        # a. I store the current tax instruments
        par.T = T
        par.tau1 = tau1
        par.tau2 = tau2
        par.tau3 = tau3

        # b. I calculate the prices paid by the consumer
        # A product tax tau_j raises the consumer price to (1+tau_j)*p_j.
        par.p1 = (1+tau1)*par.p1_pre
        par.p2 = (1+tau2)*par.p2_pre
        par.p3 = (1+tau3)*par.p3_pre

        # c. I reduce disposable income by the lump-sum tax
        par.I = par.I_pre - T

    ##########################################
    # 2. revenue and consumer utility        #
    ##########################################

    def tax_revenue(self,opt=None):
        """ calculate total tax revenue at the current taxes """

        par = self.par

        # a. I solve the consumer problem if I have not already supplied a solution
        if opt is None:
            opt = self.solve(
                do_print=False
            )

        # b. I calculate the quantities purchased after taxes
        x1,x2,x3 = self.quantities(
            opt.s1,
            opt.w
        )

        # c. I calculate total government revenue
        #
        # I use PRE-TAX prices because the product-tax payment is
        # tau_j * p_j_pre * x_j.
        R = (
            par.T
            +
            par.tau1*par.p1_pre*x1
            +
            par.tau2*par.p2_pre*x2
            +
            par.tau3*par.p3_pre*x3
        )

        return R

    def revenue_and_utility(self,tau,goods=(2,)):
        """ calculate revenue and utility from a common product-tax rate """

        # a. I initially set all product-tax rates equal to zero
        tau1 = 0.0
        tau2 = 0.0
        tau3 = 0.0

        # b. I apply the tax rate only to the goods I want to tax
        if 1 in goods:
            tau1 = tau

        if 2 in goods:
            tau2 = tau

        if 3 in goods:
            tau3 = tau

        # c. I update the prices faced by the consumer
        self.set_taxes(
            T=0.0,
            tau1=tau1,
            tau2=tau2,
            tau3=tau3
        )

        # d. I solve the consumer problem after the tax is introduced
        opt = self.solve(
            do_print=False
        )

        # e. I calculate government revenue
        R = self.tax_revenue(
            opt=opt
        )

        # f. I record consumer utility after tax
        u = opt.u

        return R,u

    def revenue_and_utility_lump_sum(self,T):
        """ calculate revenue and utility from a lump-sum tax """

        # a. I introduce the lump-sum tax and set all product taxes to zero
        self.set_taxes(
            T=T,
            tau1=0.0,
            tau2=0.0,
            tau3=0.0
        )

        # b. I solve the consumer problem with the lower disposable income
        opt = self.solve(
            do_print=False
        )

        # c. I calculate government revenue
        R = self.tax_revenue(
            opt=opt
        )

        # d. I record consumer utility
        u = opt.u

        return R,u

    ##########################################
    # 3. hitting a given revenue requirement #
    ##########################################

    def max_revenue(self,goods=(2,),tau_max=10.0,N=1001):
        """ find the largest revenue on a grid of tax rates """

        # a. I construct a grid of possible tax rates
        tau_grid = np.linspace(
            0.0,
            tau_max,
            N
        )

        # b. I allocate an array for revenue
        R_grid = np.empty(
            N
        )

        # c. I calculate revenue at every tax rate
        for i,tau in enumerate(tau_grid):

            R_grid[i] = self.revenue_and_utility(
                tau,
                goods=goods
            )[0]

        # d. I locate the tax rate that gives the highest revenue
        i_max = np.argmax(
            R_grid
        )

        # e. I store the revenue-maximizing tax rate and revenue
        tau = tau_grid[i_max]
        R = R_grid[i_max]

        return tau,R

    def find_tax_rate(self,R_target,goods=(2,),bracket=(1e-10,1.0)):
        """ find the product-tax rate that raises a given amount of revenue """

        # a. I define the root as revenue minus the required revenue
        def objective(tau):

            R,_ = self.revenue_and_utility(
                tau,
                goods=goods
            )

            return R-R_target

        # b. I use Brent's method to find the tax rate
        try:

            result = optimize.root_scalar(
                objective,
                bracket=bracket,
                method='brentq'
            )

            tau = result.root

        # c. I return NaN if the revenue target cannot be reached in the bracket
        except ValueError:

            tau = np.nan

        return tau