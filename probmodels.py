#!/usr/local/env python
#
# Jeffrey Jose | Mar 21, 2017
#
# Probability Models


from __future__ import division
import sys, os

import numpy as np

from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel

# Defaults
ZERO_INFLATION_PERCENTAGE = .1

METHOD = 'nm'

class GLMBaseModel(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, **kwargs):

        if not exog:
            exog = np.zeros_like(endog)

        super(GLMBaseModel, self).__init__(endog, exog, **kwargs)


class GenericSpikePoisson(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, spike=0, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        self.spike = spike

        super(GenericLikelihoodModel, self).__init__(endog, exog, **kwargs)


    def _zip_pmf(self, x, lambda_, pi=ZERO_INFLATION_PERCENTAGE):

        if pi < 0 or pi > 1 or lambda_ <= 0:
            return np.zeros_like(x)
        else:
            return pi*(x == self.spike) + (1-pi)*stats.poisson.pmf(x, lambda_)


    def nloglikeobs(self, params):

        pi, lambda_ = params

        return -np.log(self._zip_pmf(self.endog, pi=pi, lambda_=lambda_))


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method=METHOD, **kwargs):

        if not start_params:
            lambda_start = self.endog.mean()

            excess_items = (self.endog == self.spike ).mean() - stats.poisson.pmf(self.spike, lambda_start)

            start_params = np.array([excess_items, lambda_start])

        return super(GenericSpikePoisson, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=method, **kwargs)


class ZeroInflatedPoisson(GenericSpikePoisson):

    def __init__(self, endog, exog=None, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        super(ZeroInflatedPoisson, self).__init__(endog, exog, spike=0, **kwargs)


class Poisson(GLMBaseModel):


    def _zip_pmf(self, x, lambda_):

        return stats.poisson.pmf(x, lambda_)


    def nloglikeobs(self, params):

        lambda_ = params

        return -np.log(self._zip_pmf(self.endog, lambda_=lambda_))


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method=METHOD, **kwargs):

        if not start_params:

            lambda_start = self.endog.mean()

            start_params = np.array([lambda_start])

        return super(Poisson, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=method, **kwargs)


#####################################################################################

class NegativeBinomial(GLMBaseModel):


    def _zip_pmf(self, x, n, p):

        return stats.nbinom.pmf(x, n, p)


    def nloglikeobs(self, params):

        n, p = params

        return -np.log(self._zip_pmf(self.endog, n=n, p=p))


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, method=METHOD, **kwargs):

        if not start_params:

            n_start = self.endog.mean()
            p_start = self.endog.mean()

            start_params = np.array([n_start, p_start])

        return super(NegativeBinomial, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=method, **kwargs)



class GenericSpikeNegativeBinomial(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, spike=0, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        self.spike = spike

        super(GenericLikelihoodModel, self).__init__(endog, exog, **kwargs)


    def _zip_pmf(self, x, n, p, pi=ZERO_INFLATION_PERCENTAGE):

        if pi < 0 or pi > 1:
            return np.zeros_like(x)
        else:
            return pi*(x == self.spike) + (1-pi)*stats.nbinom.pmf(x, n, p)


    def nloglikeobs(self, params):

        pi, n, p = params

        #print(pi, n, p, np.sum(-np.log(self._zip_pmf(self.endog, pi=pi, n=n, p=p))))
        return -np.log(self._zip_pmf(self.endog, pi=pi, n=n, p=p))


    def fit(self, start_params=None, maxiter=50000, maxfun=5000, method=METHOD, **kwargs):

        if not start_params:

            n_start = self.endog.mean()
            p_start = self.endog.mean()

            excess_items = ZERO_INFLATION_PERCENTAGE

            start_params = np.array([excess_items, n_start, p_start])

        result = super(GenericSpikeNegativeBinomial, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, method=method, **kwargs)

        # This is because statsmodel doesnt handle things properly, and craps on jupyter notebook
        # Shove in fake data.
        result.normalized_cov_params = np.array([[0,0,0], [0,0,0], [0,0,0]])

        return result


class ZeroInflatedNegativeBinomial(GenericSpikeNegativeBinomial):

    def __init__(self, endog, exog=None, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        super(ZeroInflatedNegativeBinomial, self).__init__(endog, exog, spike=0, **kwargs)



if __name__ == '__main__':

    x = x = [0, 5, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 3, 0, 0, 1, 2, 0, 6, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 16, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 0, 0, 0, 0, 9, 0, 0, 1, 0, 0, 0, 0, 0, 5, 0, 0, 3, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 2, 0, 0, 0, 1, 2, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 4, 0, 0, 6, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 62, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2, 0, 0, 0, 2, 2, 3, 0, 3, 0, 0, 0, 1, 0, 0, 0, 12, 0, 0, 6, 0, 0, 0, 2, 0, 0, 1, 16, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 3, 1, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 3, 1, 28, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 15, 12, 0, 4, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 25, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 5, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 1, 5, 2, 0, 0, 0, 1, 5, 0, 0, 0, 18, 4, 1, 0, 0, 1, 0, 0, 0, 5, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 9, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 2, 1, 0, 79, 1, 5, 0, 0, 8, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 2, 1, 16, 0, 0, 5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 10, 16, 0, 1, 5, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 22, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 12, 8, 0, 0, 0, 1, 0, 2, 0, 1, 0, 5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 7, 0, 0, 1, 0, 0, 0, 1, 2, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 6, 4, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 10, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 1, 2, 7, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 4, 4, 0, 0, 6, 4, 14, 0, 1, 0, 4, 16, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 1, 0, 2, 1, 4, 5, 0, 16, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 5, 0, 0, 2, 0, 1, 0, 0, 0, 34, 0, 0, 0, 1, 0, 0, 0, 0, 4, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 20, 1, 0, 2, 0, 0, 1, 1, 1, 1, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 7, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 0, 8, 2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 3, 0, 27, 4, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 16, 3, 0, 0, 0, 9, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 1, 0, 0, 3, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 6, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 5, 0, 0, 0, 1, 5, 1, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 11, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0, 1, 9, 0, 0, 0, 0, 0, 8, 0, 1, 0, 2, 0, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 3, 3, 2, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 1, 4, 3, 0, 0, 0, 2, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0, 1, 2, 0, 14, 50, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3, 0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 21, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 4, 1, 0, 0, 15, 0, 0, 4, 0, 9, 0, 9, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 6, 0, 2, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 11, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 1, 1, 6, 0, 1, 0, 25, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 25, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 14, 0, 1, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 0, 7, 0, 2, 0, 1, 0, 0, 1, 0, 0, 7, 34, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 1, 1, 2, 18, 0, 7, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 5, 0, 1, 0, 0, 1, 0, 0, 4, 0, 40, 2, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 3, 1, 2, 0, 0, 0, 3, 0, 1, 0, 0, 1, 14, 1, 1, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 3, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 3, 1, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 6, 0, 0, 5, 0, 0, 1, 3, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 2, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 9, 0, 1, 0, 2, 1, 2, 0, 0, 9, 0, 0, 0, 1, 10, 5, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 12, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 3, 1, 0, 11, 1, 0, 2, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 5, 1, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 7, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 3, 1, 12, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 4, 0, 0, 1, 10, 0, 4, 1, 0, 0, 0, 0, 17, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 0, 0, 0, 0, 0, 0, 0, 6, 0, 1, 4, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    model = ZeroInflatedNegativeBinomial(x)
    result = model.fit()
    result.summary()

