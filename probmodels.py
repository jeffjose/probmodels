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
ZERO_INFLATION_PERCENTAGE = 0.3
POISSON_LAMBDA            = 2.0

NEGATIVEBINOMIAL_N = 1
NEGATIVEBINOMIAL_P = 1


class GLMBase(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, **kwargs):

        if not exog:
            exog = np.zeros_like(endog)

        super(GLMBase, self).__init__(endog, exog, **kwargs)


class GenericSpikePoisson(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, spike=0, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        self.spike = spike

        super(GenericLikelihoodModel, self).__init__(endog, exog, **kwargs)


    def _zip_pmf(self, x, pi=ZERO_INFLATION_PERCENTAGE, lambda_=POISSON_LAMBDA):

        if pi < 0 or pi > 1 or lambda_ <= 0:
            return np.zeros_like(x)
        else:
            return pi*(x == self.spike) + (1-pi)*stats.poisson.pmf(x, lambda_)


    def nloglikeobs(self, params):

        pi, lambda_ = params

        return -np.log(self._zip_pmf(self.endog, pi=pi, lambda_=lambda_))


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):

        if not start_params:
            lambda_start = self.endog.mean()

            excess_items = (self.endog == self.spike ).mean() - stats.poisson.pmf(self.spike, lambda_start)

            start_params = np.array([excess_items, lambda_start])

        return super(GenericSpikePoisson, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)


class ZeroInflatedPoisson(GenericSpikePoisson):

    def __init__(self, endog, exog=None, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        super(ZeroInflatedPoisson, self).__init__(endog, exog, spike=0, **kwargs)


class Poisson(GLMBase):


    def _zip_pmf(self, x, lambda_=POISSON_LAMBDA):

        return stats.poisson.pmf(x, lambda_)


    def nloglikeobs(self, params):

        lambda_ = params

        return -np.log(self._zip_pmf(self.endog, lambda_=lambda_))


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):

        if not start_params:

            lambda_start = self.endog.mean()

            start_params = np.array([lambda_start])

        return super(Poisson, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)


#####################################################################################

class NegativeBinomial(GLMBase):


    def _zip_pmf(self, x, n, p):

        return stats.nbinom.pmf(x, n, p)


    def nloglikeobs(self, params):

        n, p = params

        return -np.log(self._zip_pmf(self.endog, n=n, p=p))


    def fit(self, start_params=None, maxiter=10000, maxfun=5000, **kwargs):

        if not start_params:

            n_start = self.endog.mean()
            p_start = self.endog.mean()

            start_params = np.array([n_start, p_start])

        return super(NegativeBinomial, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)



class GenericSpikeNegativeBinomial(GenericLikelihoodModel):

    def __init__(self, endog, exog=None, spike=0, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        self.spike = spike

        super(GenericLikelihoodModel, self).__init__(endog, exog, **kwargs)


    def _zip_pmf(self, x, n, p, pi=ZERO_INFLATION_PERCENTAGE):

        print(n,p,pi)

        if pi < 0 or pi > 1:
            return np.zeros_like(x)
        else:
            return pi*(x == self.spike) + (1-pi)*stats.nbinom.pmf(x, n, p)


    def nloglikeobs(self, params):

        pi, n, p = params

        return -np.log(self._zip_pmf(self.endog, pi=pi, n=n, p=p))


    def fit(self, start_params=None, maxiter=50000, maxfun=5000, **kwargs):

        if not start_params:

            n_start = self.endog.mean()
            p_start = self.endog.mean()

            excess_items = ZERO_INFLATION_PERCENTAGE

            start_params = np.array([excess_items, n_start, p_start])

        return super(GenericSpikeNegativeBinomial, self).fit(start_params=start_params, maxiter=maxiter, maxfun=maxfun, **kwargs)


class ZeroInflatedNegativeBinomial(GenericSpikeNegativeBinomial):

    def __init__(self, endog, exog=None, **kwargs):

        if exog is None:
            exog = np.zeros_like(endog)

        super(ZeroInflatedNegativeBinomial, self).__init__(endog, exog, spike=0, **kwargs)


