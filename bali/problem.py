#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class Problem

Can represent both inverse problems and Bayesian model selection problems.
For an inverse problem, initialize with one model (either the model itself or
a list with one model). For a selection problem, initialize with a list of 
models
"""

import numpy as np
from bali.gpe import GpeSquaredExponential, GpeMatern
from bali.gpe_helpers import ll_to_lbme

class Problem:
    def __init__(self, models, data):
        if type(models) == list:
            self.models = models
        else:
            self.models = [models]
        self.data = data
        
        self.n_output = data.size
        self.n_models = len(self.models)
        [m.inflate_variance(self.n_output) for m in self.models]
        
    def evaluate_model(self, x, k=0):
        return self.models[k].evaluate(x)
    
    def compute_likelihood(self, k=0):
        m = self.models[k]
        m.tabulate()
        
        d = self.data
        y = m.y
        v = m.variance
        
        likelihood = 1./np.sqrt(np.prod(2*np.pi*v)) *\
            np.exp(-np.sum((y-d)**2/(2*v), axis=1))
        return likelihood

    def compute_loglikelihood(self, k=0):
        m = self.models[k]
        m.tabulate()
            
        d = self.data
        y = m.y
        v = m.variance
        
        loglikelihood = -0.5*np.sum(np.log(2*np.pi*v)) - \
            np.sum((y-d)**2/(2*v), axis=1)
        return loglikelihood

    def compute_likelihoods(self):
        lls = [self.compute_likelihood(k) for k in range(self.n_models)]
        return lls
    
    def compute_loglikelihoods(self):
        lls = [self.compute_loglikelihood(k) for k in range(self.n_models)]
        return lls
    
    def compute_lbme(self):
        lls = self.compute_loglikelihoods()
        lbmes = [ll_to_lbme(ll) for ll in lls]
        return np.array(lbmes)
    
    def suggest_gpe(self, k=0, gpe_type='squared_exponential'):
        m = self.models[k]
        std = m.x.std(axis=0)[:, np.newaxis]
        l = std * np.array([0.01, 10])        
        norm = np.linalg.norm(self.data)
        sigma2 = norm * np.array([0.01, 100])        
        n_output = self.n_output
        
        if gpe_type == 'squared_exponential':
            gpe = GpeSquaredExponential(l, sigma2, n_output)
        elif gpe_type == 'matern':
            nu = [0.5, 10]
            gpe = GpeMatern(l, sigma2, nu, n_output)
        
        return gpe    
    
    def suggest_gpes(self, gpe_type='squared_exponential'):
        return [self.suggest_gpe(k, gpe_type) for k in range(self.n_models)]

