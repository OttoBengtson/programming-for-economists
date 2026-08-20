from types import SimpleNamespace

import numpy as np

from scipy import optimize


class ConsumerClass:
    """ a consumer with nested CES preferences over three goods

    Good 1 is food. Goods 2 and 3 are bus trips and train trips, and they sit
    together in a nest.

    The problem is written in *nested* budget shares, in the same two steps as
    the nests themselves:

        s1 = the share of income spent on food
        w  = the share of the remaining (travel) budget spent on the bus

    This implies the three ordinary budget shares:

        s1 = food
        s2 = (1-s1)*w
        s3 = (1-s1)*(1-w)

    Since s1 and w can both be chosen freely between 0 and 1, every feasible
    choice can be represented by a point in the unit square [0,1] x [0,1].
    This makes the problem suitable for L-BFGS-B, which handles simple bounds.

    """

    def __init__(self,par=None):

        # a. set all default parameters
        self.setup()

        # b. overwrite selected parameters if the user provides them
        # Example:
        # ConsumerClass(par={'sigma_B':3.0})
        if not par is None:
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup(self):
        """ set the baseline parameters """

        # SimpleNamespace lets us store related objects together.
        # We can then write par.alpha instead of having many loose variables.
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preference weights
        par.alpha = 0.60 # weight on food in the upper CES nest
        par.beta = 0.50 # weight on bus in the lower CES nest

        # b. substitution parameters
        # sigma_A determines substitutability between food and travel
        # sigma_B determines substitutability between bus and train
        par.sigma_A = 0.80
        par.sigma_B = 0.40

        # c. prices and income
        par.p1 = 1.0 # price of food
        par.p2 = 1.0 # price of a bus trip
        par.p3 = 1.5 # price of a train trip
        par.I = 10.0 # income

        # d. numerical setting
        # CES utility may involve negative powers.
        # Therefore, exact zeros can create numerical problems.
        # We replace zero by this very small positive number inside .ces().
        par.s_min = 1e-12

    def __str__(self):
        """ print the parameters """

        par = self.par

        lines = ['ConsumerClass']
        lines.append(f'  alpha = {par.alpha:.4f}, beta = {par.beta:.4f}')
        lines.append(f'  sigma_A = {par.sigma_A:.4f}, sigma_B = {par.sigma_B:.4f}')
        lines.append(f'  p1 = {par.p1:.4f}, p2 = {par.p2:.4f}, p3 = {par.p3:.4f}')
        lines.append(f'  I = {par.I:.4f}')

        return '\n'.join(lines)

    ###################
    # 1. the CES nest #
    ###################

    def ces(self,z1,z2,w,sigma):
        """ the CES aggregate of two inputs

        Computes

            (w*z1**rho + (1-w)*z2**rho)**(1/rho)

        where

            rho = 1 - 1/sigma

        Args:

            z1 (float or ndarray): first input
            z2 (float or ndarray): second input
            w (float): weight on the first input
            sigma (float): substitution parameter, must not be 1

        Returns:

            (float or ndarray): the CES aggregate

        """

        par = self.par

        # a. sigma = 1 would give rho = 0 and therefore division by zero
        assert not np.isclose(
            sigma,
            1.0
        ), 'sigma = 1 gives rho = 0 and a division by zero'

        # b. avoid exact zeros
        # This matters especially when sigma < 1, because rho is negative
        # and we would otherwise raise zero to a negative power.
        z1 = np.maximum(z1,par.s_min)
        z2 = np.maximum(z2,par.s_min)

        # c. transform sigma into the CES exponent rho
        rho = 1-1/sigma

        # d. calculate the CES aggregate
        return (
            w*z1**rho
            + (1-w)*z2**rho
        )**(1/rho)

    def utility(self,x1,x2,x3):
        """ nested CES utility of a bundle of quantities

        The utility function has two levels:

        1. Bus and train are combined into a travel composite xB.
        2. Food and the travel composite are combined into total utility.

        Args:

            x1 (float or ndarray): quantity of food
            x2 (float or ndarray): number of bus trips
            x3 (float or ndarray): number of train trips

        Returns:

            (float or ndarray): utility

        """

        par = self.par

        # a. lower nest: combine bus and train
        # beta is the weight on bus
        # sigma_B determines how easily bus and train can substitute
        xB = self.ces(
            x2,
            x3,
            par.beta,
            par.sigma_B
        )

        # b. upper nest: combine food and travel
        # alpha is the weight on food
        # sigma_A determines substitution between food and travel
        u = self.ces(
            x1,
            xB,
            par.alpha,
            par.sigma_A
        )

        return u

    ###############################
    # 2. the nested budget shares #
    ###############################

    def shares(self,s1,w):
        """ the three budget shares implied by the nested shares

        s1 is the share spent directly on food.

        The remaining share, 1-s1, is the travel budget.
        w determines how this travel budget is split between bus and train.

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of travel budget spent on bus

        Returns:

            (tuple): shares spent on food, bus and train

        """

        # food gets s1
        s2 = (1-s1)*w

        # train gets whatever remains of the travel budget
        s3 = (1-s1)*(1-w)

        return s1,s2,s3

    def quantities(self,s1,w):
        """ convert budget shares into physical quantities

        A budget share tells us how much money is spent on a good.

        Example:
            expenditure on food = s1 * I

        Quantity is expenditure divided by price:

            x1 = s1 * I / p1

        Args:

            s1 (float or ndarray): share of income spent on food
            w (float or ndarray): share of travel budget spent on bus

        Returns:

            (tuple): quantities of food, bus trips and train trips

        """

        par = self.par

        # a. convert the two nested shares into the three ordinary shares
        s1,s2,s3 = self.shares(s1,w)

        # b. expenditure divided by price gives quantity
        x1 = s1*par.I/par.p1
        x2 = s2*par.I/par.p2
        x3 = s3*par.I/par.p3

        return x1,x2,x3

    def value_of_choice(self,s1,w):
        """ utility associated with a given pair of nested budget shares

        This function links the choice variables (s1,w) to utility:

            shares
              ↓
            quantities
              ↓
            utility

        Args:

            s1 (float or ndarray): share spent on food
            w (float or ndarray): share of travel budget spent on bus

        Returns:

            (float or ndarray): utility

        """

        # a. translate budget shares into quantities
        x1,x2,x3 = self.quantities(
            s1,
            w
        )

        # b. evaluate the consumer's utility from these quantities
        u = self.utility(
            x1,
            x2,
            x3
        )

        return u

    def objective(self,s):
        """ minus utility, for scipy.optimize

        scipy.optimize is designed to MINIMIZE functions.

        The consumer wants to MAXIMIZE utility, so we instead minimize

            -utility.

        The input s is an array:

            s[0] = s1
            s[1] = w

        """

        return -self.value_of_choice(
            s[0],
            s[1]
        )

    #################
    # 3. solving it #
    #################

    def solve_grid(self,N=200,do_print=True):
        """ solve by a 2-dimensional grid search over the nested shares

        We create N possible values of s1 and N possible values of w.

        This gives N x N possible combinations.

        Example:
            N = 100
            -> 100 x 100
            -> 10,000 utility evaluations

        Every point is feasible because the nested-share formulation
        automatically satisfies the budget constraint.

        Args:

            N (int): number of grid points for each choice variable
            do_print (bool): print the solution

        Returns:

            (SimpleNamespace): grids, utilities and optimal choice

        """

        opt = SimpleNamespace()

        # a. create one-dimensional grids
        # Both choice variables can lie between 0 and 1.
        s1_vec = np.linspace(
            0.0,
            1.0,
            N
        )

        w_vec = np.linspace(
            0.0,
            1.0,
            N
        )

        # b. combine the two vectors into a two-dimensional grid
        #
        # If N = 100, both arrays have shape (100,100).
        #
        # Each location (i,j) now represents one combination:
        #
        #     s1_grid[i,j]
        #     w_grid[i,j]
        #
        s1_grid,w_grid = np.meshgrid(
            s1_vec,
            w_vec,
            indexing='ij'
        )

        # c. calculate utility at every point
        #
        # Our functions support numpy arrays, so we do not need
        # two nested Python loops. NumPy evaluates the entire grid.
        u_grid = self.value_of_choice(
            s1_grid,
            w_grid
        )

        # d. locate the largest utility value
        #
        # np.argmax gives the position of the maximum if the
        # 2-dimensional array were written as one long vector.
        index_flat = np.argmax(
            u_grid
        )

        # e. convert this flat position back into row and column
        i,j = np.unravel_index(
            index_flat,
            u_grid.shape
        )

        # f. read the optimal nested shares from the grid
        s1_best = s1_grid[i,j]
        w_best = w_grid[i,j]

        # g. convert nested shares into the three actual budget shares
        s1_best,s2_best,s3_best = self.shares(
            s1_best,
            w_best
        )

        # h. utility at the best grid point
        u_best = u_grid[i,j]

        # i. store the solution
        opt.s1 = s1_best
        opt.w = w_best

        opt.s2 = s2_best
        opt.s3 = s3_best

        opt.u = u_best

        # j. store the full grids as well
        # We need them later for the 3D surface and contour plots.
        opt.s1_grid = s1_grid
        opt.w_grid = w_grid
        opt.u_grid = u_grid

        # k. number of utility evaluations
        # There are N choices of s1 for each of N choices of w.
        opt.nfev = N**2

        # l. optionally show the result
        if do_print:

            print(f's1 = {opt.s1:.6f}')
            print(f'w  = {opt.w:.6f}')
            print('')
            print('Budget shares:')
            print(f's1 = {opt.s1:.6f}')
            print(f's2 = {opt.s2:.6f}')
            print(f's3 = {opt.s3:.6f}')
            print('')
            print(f'utility = {opt.u:.6f}')
            print(f'function evaluations = {opt.nfev:,}')

        return opt

    def solve(self,s0=None,do_print=True,**kwargs):
        """ solve the consumer problem using L-BFGS-B

        Instead of checking every point on a grid, L-BFGS-B searches
        intelligently for the maximum.

        Because scipy minimizes, .objective() returns negative utility.

        The only restrictions are:

            0 <= s1 <= 1
            0 <= w  <= 1

        These are simple bounds, which is exactly what L-BFGS-B handles.

        Args:

            s0 (ndarray): initial guess for (s1,w)
            do_print (bool): print the solution
            kwargs: additional arguments passed to optimize.minimize

        Returns:

            (SimpleNamespace): solution, convergence path and scipy result

        """

        opt = SimpleNamespace()

        # a. choose starting point
        #
        # If nothing else is given, we start in the middle of the square.
        if s0 is None:
            s0 = np.array([
                0.5,
                0.5
            ])

        # Ensure s0 is a NumPy array of floating-point numbers.
        s0 = np.asarray(
            s0,
            dtype=float
        )

        # b. prepare a list that records the convergence path
        #
        # The callback below is only called AFTER the optimizer takes a step.
        # Therefore we manually add the initial point first.
        path = [
            s0.copy()
        ]

        # c. solve the minimization problem
        #
        # self.objective = -utility
        #
        # bounds say both variables must stay inside [0,1].
        #
        # callback stores every accepted point visited by the optimizer.
        res = optimize.minimize(
            self.objective,
            s0,
            method='L-BFGS-B',
            bounds=(
                (0.0,1.0),
                (0.0,1.0)
            ),
            callback=lambda sk: path.append(
                sk.copy()
            ),
            **kwargs
        )

        # d. read the optimal nested shares
        s1_best = res.x[0]
        w_best = res.x[1]

        # e. convert them into the three ordinary budget shares
        s1_best,s2_best,s3_best = self.shares(
            s1_best,
            w_best
        )

        # f. calculate utility at the final solution
        u_best = self.value_of_choice(
            s1_best,
            w_best
        )

        # g. store everything we will need later
        opt.s1 = s1_best
        opt.w = w_best

        opt.s2 = s2_best
        opt.s3 = s3_best

        opt.u = u_best

        # Convert the list of visited points into a NumPy array.
        # Each row is one point (s1,w) visited by the optimizer.
        opt.path = np.array(
            path
        )

        # Keep scipy's entire result object.
        # This contains useful information such as:
        #
        # res.success
        # res.nit    = number of iterations
        # res.nfev   = number of function evaluations
        # res.message
        #
        opt.res = res

        # h. optionally print the result
        if do_print:

            print(f'success = {res.success}')
            print(f'message = {res.message}')
            print('')

            print('Nested shares:')
            print(f's1 = {opt.s1:.6f}')
            print(f'w  = {opt.w:.6f}')
            print('')

            print('Budget shares:')
            print(f's1 = {opt.s1:.6f}')
            print(f's2 = {opt.s2:.6f}')
            print(f's3 = {opt.s3:.6f}')
            print(f'sum = {opt.s1 + opt.s2 + opt.s3:.6f}')
            print('')

            print(f'utility = {opt.u:.6f}')
            print(f'iterations = {res.nit}')
            print(f'function evaluations = {res.nfev}')

        return opt