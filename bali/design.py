"""
Class SequentialDesign

Takes a problem (either inverse problem or selection problem) and a gpe.
Calling iterate(n) will perform n steps in the sequential design.

The following properties can be changed:
    n_subsample - sample size for subsampling
    allocation  - String defining the allocation strategy. Possible values:
                  'acq', 'random', 'alternation'
    always_update_acq - if set to true, then the acquisition function will be
                    updated for all models in each iteration (otherwise, only)
                    one acquisition function will be updated
    
To switch the acquisition function, call set_acquisition_function(acq).
Possible values: 'inverse', 'random', 'min_variance'

After finishing the sequential design, the output can be accessed as follows:
    error_ll() [only for inverse problems] computes the error in solving the
                inverse problem in terms of kl-div. This will tabulate the model
                function (=run it on all input points). Only do this, if the
                model function is fast.
    error_lbme() [only for selection problems] computes thes error in solve
                the selection problem in terms of kl-div. This will also tabulate
                the model. Only do this, if the model function is fast.
                
    ll - is a list that contains the loglikelihood for each model and for each
         iteration. ll[k] contains loglikelihoods for model k and has size
         (n_iter x n_x)
    lbme - is an array containing log-bmes for all iterations. It has size
         (n_iter x n_models)
    nodes - is a list of nodes-objects. nodes[k] contains the evaluation points
         of model k
    k_list - is the list of model indices (which model was evaluated in which iteration)
             [only makes sense for selection problems]
"""
from bali.nodes import Nodes
from bali.gpe_helpers import compute_errors, ll_to_lbme
import numpy as np
import warnings


class SequentialDesign:
    def __init__(self, problem, gpe, acq = 'inverse', allocation = 'acq'):
        self.problem = problem
        if type(gpe) == list:
            self.prior_gpe = gpe
        else:
            self.prior_gpe = [gpe]
        
        self.n_subsample = 500
        self.set_acquisition_function(acq)
        self.allocation = allocation
        
        n_models = problem.n_models
        self.nodes = [Nodes() for i in range(n_models)]
        self.gpe = [g.condition_to(n) for (g,n) in zip(self.prior_gpe, self.nodes)]
        self.ll = [self.estimate_loglikelihood(k) for k in range(n_models)]
        self.last_ll = [ll[-1,:][np.newaxis,:] for ll in self.ll]
        self.lbme = np.array([ll_to_lbme(ll) for ll in self.last_ll])[np.newaxis,:]
        self.acq_ready = [False] * n_models
        self.acq_max = np.zeros(n_models)
        self.next_x = [None] * n_models
        self.always_update_acq = False
        self.k_list = []

    def set_acquisition_function(self, acq):
        if acq == 'inverse':
            self.acquisition_function = acq_inverse
        elif acq == 'random':
            self.acquisition_function = acq_random
        elif acq == 'min_variance':
            self.acquisition_function = acq_min_variance   
            
    def allocation_strategy(self):
        if self.allocation == 'acq':
            return np.argmax(self.acq_max)
        if self.allocation == 'random':
            return np.random.choice(self.problem.n_models)
        if self.allocation == 'alternation':
            return len(self.k_list)%self.problem.n_models
        
    def estimate_loglikelihood(self, k):
        x = self.problem.models[k].x
        data = self.problem.data
        variance = self.problem.models[k].variance
        return self.gpe[k].estimate_loglikelihood(x, data, variance)[np.newaxis,:]

    def iterate(self, n=1):
        for i_iteration in range(n):
            print('.', end = '')
            
            for k in range(self.problem.n_models):
                self.update_acq(k)
            
            new_k = self.allocation_strategy()
            self.k_list.append(new_k)
            
            self.advance_model(new_k)
            for k in range(self.problem.n_models):
                self.ll[k] = np.append(self.ll[k], self.last_ll[k], axis=0)
            self.lbme = np.append(self.lbme, np.array([ll_to_lbme(ll) for ll in self.last_ll])[np.newaxis,:], axis=0)                
            
        print('')
        
    def advance_model(self, k):
        new_x = self.next_x[k]
        new_y = self.problem.evaluate_model(new_x, k)
        self.nodes[k].append(new_x, new_y)
        self.gpe[k] = self.prior_gpe[k].condition_to(self.nodes[k])
        ll = self.estimate_loglikelihood(k)
        self.last_ll[k] = ll
        self.acq_ready[k] = False
        
    def subsample(self, k):
        nodes = self.nodes[k]
        model = self.problem.models[k]
        if nodes.n == 0:
            candidates = model.x
        else:
            mask = np.array([not any(np.equal(nodes.x,xx).all(1)) for xx in model.x])        
            candidates = model.x[mask]
        
        n_c = candidates.shape[0]
        if n_c < self.n_subsample:
            return candidates
        else:
            idx = np.random.choice(n_c, self.n_subsample, replace = False)
            return candidates[idx, :]        
        
    def update_acq(self, k):
        if (not self.always_update_acq) and self.acq_ready[k]:
            return

        sub_x = self.subsample(k)
        gpe_list = self.gpe[k].discretize(sub_x)
        
        new_sub_idx, self.acq_max[k] = self.optimal_node_and_max(gpe_list, k)
        self.next_x[k] = sub_x[new_sub_idx]
        self.acq_ready[k] = True

    def optimal_node_and_max(self, gpe_list, k):
        data = self.problem.data
        variance = self.problem.models[k].variance
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore') 
            acq = [self.acquisition_function(gpe, data, variance) for gpe in gpe_list]
        acq_average = np.mean(acq, axis=0)
        optimal_node = np.argmax(acq_average)
        acq_max = np.max(acq_average)
        return optimal_node, acq_max
    
    def error_ll(self, k=0):
        ll_true = self.problem.compute_loglikelihood(k)
        return compute_errors(ll_true, self.ll[k])
                 
    def error_lbme(self):
        lbme_true = self.problem.compute_lbme()
        return compute_errors(lbme_true, self.lbme)
    
    def loglikelihood_function(self, k=0):
        d = self.problem.data
        v = self.problem.models[k].variance
        n_input = self.problem.models[k].x.shape[1]
        def ll(x):
            x = np.array(x).reshape((-1, n_input))
            return self.gpe[k].estimate_loglikelihood(x, d, v)
        return ll  
                         
def acq_inverse(gpe, data, variance):
    n_x = gpe.n_x
    n_output = gpe.n_output
    c_is_2d = (gpe.c.ndim == 2)
    current_l = gpe.estimate_likelihood(data, variance)
    var_gpe_prior = gpe.extract_variance()
    
    if c_is_2d:
        var_gpe_prior = np.diag(gpe.c)
    else:
        var_gpe_prior = np.full((n_x, n_output), np.nan)
        for i_output in range(n_output):
            var_gpe_prior[:, i_output] = np.diag(gpe.c[:, :, i_output])

    
    acq = np.full(n_x, -np.inf)
    all_indices = np.arange(n_x)
    if c_is_2d:
        index_list = all_indices[np.diag(gpe.c) != 0]
    else:
        index_list = all_indices
    
    for this_idx in index_list:
        if c_is_2d:
            Q = gpe.c[this_idx, this_idx]
            q = gpe.c[:, this_idx]
            var_gpe = var_gpe_prior - q*q/Q
            var_gpe = var_gpe[:, np.newaxis] * np.ones((1, n_output))
        else:
            var_gpe = np.full((n_x, n_output), np.nan)
            for i_output in range(n_output):
                Q = gpe.c[this_idx, this_idx, i_output]
                q = gpe.c[:, this_idx, i_output]
                var_gpe[:, i_output] = var_gpe_prior[:, i_output] - q*q/Q
                
        # sort out uncorrelated points
        if c_is_2d:
            idx_normal = index_list[gpe.c[index_list, this_idx] != 0]
        else:
            idx_normal = index_list
            
        # define properties of f(y_0)**2
        if c_is_2d:
            invc = (gpe.c[this_idx, this_idx] /
                    gpe.c[idx_normal, this_idx])[:, np.newaxis]
        else:
            invc = (gpe.c[this_idx, this_idx] /
                    gpe.c[idx_normal, this_idx])
        
        c_f = np.abs(invc)
        m_ff = (data - gpe.m[idx_normal, :]) * invc
        v_ff = 0.5*(var_gpe[idx_normal, :]+variance)*(invc**2)
        c_ff = 0.5*(c_f**2)/np.sqrt(2*np.pi*v_ff)

        # define properties of p(y_0)
        n_normal = idx_normal.size
        c_p = np.ones((n_normal, n_output))
        m_p = np.zeros((n_normal, n_output))
        v_p = np.ones((n_normal, 1)) * gpe.c[this_idx, this_idx]                
        
        c_total = integral_of_multiplied_normals(
            c_ff, m_ff, v_ff, c_p, m_p, v_p)
        c_var = np.prod(c_total, axis=1) - current_l[idx_normal]**2
        c_var = c_var.clip(min=0)
        acq[this_idx] = c_var.sum()/n_x
    return acq

def integral_of_multiplied_normals(c1, m1, v1, c2, m2, v2):
    e = np.exp(-(m1-m2)**2 / (2*(v1+v2)))
    c = c1*c2/np.sqrt(2*np.pi*(v1+v2))*e
    return c

def acq_random(gpe, data, variance):
    return np.random.random(gpe.n_x)

def acq_min_variance(gpe, data, variance):
    c_is_2d = (gpe.c.ndim == 2)
    if c_is_2d:
        c = gpe.c
        acq = [c[i,i] * c[i,:]@c[i,:] for i in range(gpe.n_x)]
    else:
        acq = np.zeros(gpe.n_x)
        for i_output in range(gpe.n_output):
            c = gpe.c[:, :, i_output]
            new_term = [c[i,i] * c[i,:]@c[i,:] for i in range(gpe.n_x)]
            acq += new_term
    return acq