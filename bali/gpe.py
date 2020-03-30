"""
Class Gpe

My implementation of Gaussian processes. Contains methods for identifying
hyperparameters.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import cholesky
from copy import deepcopy

from bali.gpe_helpers import log_mvnpdf, \
                        covariance_squared_exponential,\
                        covariance_matern, \
                        DiscreteGpe, \
                        linearize_gpe
from bali.nodes import Nodes


# numerical corrections in discretize und bei estimate_loglikelihood
# prior_var sollte eigentlich prior_cov aufrufen

class AbstractGpe:
    def __init__(self, n_output, lb, ub):
        self.n_output = n_output

        self.mu = (0.5 * (lb + ub)).flatten()
        self.sigma = ((ub - lb)/4).flatten()
        self.n_parameters = lb.size
        self.previous_starting_point = self.get_prior_center()
        self.nodes = Nodes()
        self.xi_is_ready = False
        self.hyperparameter_handling = 'map'
        self.n_walkers = 10

    def process_anisotropy(self, l, anisotropy):
        l = np.atleast_2d(l).T       
        if anisotropy:
            # if l is "flat", blow it up into matrix shape
            l = l * np.ones((2,anisotropy))
            
        self.n_l = l.shape[1]
        return l

    def condition_to(self, nodes):
        new = deepcopy(self)
        new.append_nodes(nodes)
        return new
    
    def append_nodes(self, nodes):
        self.nodes.join(nodes)
        self.xi_is_ready = False
        
    def parameters_to_xi(self, parameters):
        return (np.log(parameters) - self.mu) / self.sigma

    def xi_to_parameters(self, xi):
        return np.exp(self.sigma * xi + self.mu)

    def prior_mean(self, xi, x):
        n = x.shape[0]
        return np.zeros((n, self.n_output))

    def prior_cov(self, xi, x1, x2):
        raise NotImplementedError(
            'Function cov not implemented. Please use MixMatern or MixSquaredExponential')

    def prior_var(self, xi, x):
        parameters = self.xi_to_parameters(xi)
        sigma_squared = parameters[self.n_l]
        n_sample = x.shape[0]
        return np.full(n_sample, sigma_squared * (1+1e-7))
            
    def make_xi_list(self):
        if self.xi_is_ready:
            return
        
        if self.hyperparameter_handling == 'map':
            self.xi_list = [self.get_map_xi()]
            self.xi_is_ready = True
        else:
            self.xi_list = self.xi_mcmc()
            self.xi_is_ready = True
        
    def get_map_xi(self, start_from_previous = True):
        
        if self.nodes.n == 0:
            return self.get_prior_center()
        
        log_posterior_fun = self.create_log_posterior()
        def obj_fun(xi):
            return -log_posterior_fun(xi)
        
        if start_from_previous:
            starting_point = self.previous_starting_point
        else:
            starting_point = self.get_prior_center()

        result = minimize(obj_fun, starting_point)
        self.previous_starting_point = result.x
        return result.x
    
    def xi_mcmc(self):
        import emcee
        
        log_posterior = self.create_log_posterior()
        sampler = emcee.EnsembleSampler(
            self.n_walkers, self.n_parameters, log_posterior)
        
        xi_map = self.get_map_xi()
        noise = np.random.normal(
            scale=0.001, size = (self.n_walkers, self.n_parameters))
        p0 = xi_map[np.newaxis, :] + noise
        
        pos, prob, state = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        sampler.run_mcmc(pos, 1)

        return sampler.flatchain

    def get_prior_center(self):
        return np.zeros(self.n_parameters)

    def create_log_posterior(self):
        n_nodes = self.nodes.n
        if n_nodes == 0:
            def log_posterior(xi):
                return self.log_prior(xi)
        else:
            def log_posterior(xi):
                try:
                    log_likelihood = self.node_loglikelihood(xi, self.nodes)
                    log_prior = self.log_prior(xi)
                    value = log_likelihood + log_prior
                except np.linalg.LinAlgError:
                    value = -np.inf
                return value

        return log_posterior

    def log_prior(self, xi):
        return np.sum(norm.logpdf(xi))

    def node_loglikelihood(self, xi, nodes):
        c = self.prior_cov(xi, nodes.x)
        m_full = self.prior_mean(xi, nodes.x)

        loglikelihood = np.sum([log_mvnpdf(y, m, c) for (y, m) in zip(nodes.y.T, m_full.T)])
        return loglikelihood
    
    def discretize(self, x):
        self.make_xi_list()
        
        discrete_gpe_list = []
        for xi in self.xi_list:
            m = self.prior_mean(xi, x)
            c = self.prior_cov(xi, x)
            
            if self.nodes.n > 0:
                Q = self.prior_cov(xi, self.nodes.x, self.nodes.x)
                R = cholesky(Q)
                q = self.prior_cov(xi, x, self.nodes.x)
                r = np.linalg.solve(R.T, q.T).T
                deviation = self.nodes.y - self.prior_mean(xi, self.nodes.x)
                
                m = m + np.linalg.solve(R, r.T).T @ deviation
                c = c - r@r.T
                
                # some correction for numerical errors
                # 1) no negative diagonal entries (=variances) in c
                n_x = c.shape[0]
                diag_idx = np.eye(n_x, dtype=bool)
                negative_idx = c<0
                c[diag_idx & negative_idx] = 0
                # 2) enforce interpolation
                # 3) rows and columns in c must be zero on updated points
            
            discrete_gpe_list.append(DiscreteGpe(m, c))
            
        if self.hyperparameter_handling == 'linearize':
            return linearize_gpe(discrete_gpe_list)
        
        return discrete_gpe_list

    def estimate_loglikelihood(self, x, data, variance):
        m_list, v_list = self.compute_m_and_v(x)
        ll_list = []
        
        for m,v in zip(m_list, v_list):
            var_nu = v + variance
            
            loglikelihood = -0.5*np.sum(np.log(2*np.pi*var_nu), axis=1) - \
                np.sum((m-data)**2/(2*var_nu), axis=1)
                
            ll_list.append(loglikelihood)
        
        if len(ll_list) == 1:
            return ll_list[0]
        else:
            ll_array = np.array(ll_list)    
            l_array = np.exp(ll_array)
            return np.log(l_array.mean(axis = 0))
            
    def compute_m_and_v(self, x):
        self.make_xi_list()
        m_list = []
        v_list = []
        
        for xi in self.xi_list:
            v = self.prior_var(xi, x)
            m = self.prior_mean(xi, x)
            
            if self.nodes.n > 0:
                Q = self.prior_cov(xi, self.nodes.x, self.nodes.x)
                q = self.prior_cov(xi, x, self.nodes.x)
                
                deviation = self.nodes.y - self.prior_mean(xi, self.nodes.x)
                
                # Kriging (with some math-magic)
                R = cholesky(Q)
                r = np.linalg.solve(R.T, q.T).T
                v = v - np.sum(r**2, axis=1)
                m = m + np.linalg.solve(R, r.T).T @ deviation
                
                # 1) remove negative diagonal entries
                negative_idx = (v<0)
                v[negative_idx] = 0
                # 2) set rows and columns of updated points to zero
                # todo
                # 3) enforce interpolation
                # todo
            
            v = v[:, np.newaxis]
            m_list.append(m)
            v_list.append(v)
        return m_list, v_list
    
    def convert_to_ublb(self,p):
        if np.size(p) == 1:
            return np.array([p,p])
        else:
            return np.array(p)
    

class GpeMatern(AbstractGpe):
    
    def __init__(self, l, sigma_squared, nu, n_output, anisotropy = None):
        # There are two ways for making these anisotropic:
        # 1) pass multiple l-bounds, e.g. [[0.1, 10], [1, 100], [1, 100]] -> will create three l-parameters
        # 2) pass single l-bound and set anisotropic to the number of input-dimensions, e.g. anisotropic = 3

        l = self.convert_to_ublb(l)
        l = self.process_anisotropy(l, anisotropy)
        sigma_squared = self.convert_to_ublb(sigma_squared)
        nu = self.convert_to_ublb(nu)

        lb = np.log(np.column_stack((l[[0]], sigma_squared[0], nu[0])))
        ub = np.log(np.column_stack((l[[1]], sigma_squared[1], nu[1])))
        
        super().__init__(n_output, lb, ub)

    def prior_cov(self, xi, x1, x2 = None):
        parameters = self.xi_to_parameters(xi)
        l = parameters[:self.n_l]
        sigma_squared, nu = parameters[self.n_l:]
        
        return covariance_matern(l, sigma_squared, nu, x1, x2)


class GpeSquaredExponential(AbstractGpe):    
    def __init__(self, l, sigma_squared, n_output, anisotropy = None):
        l = self.convert_to_ublb(l)
        l = self.process_anisotropy(l, anisotropy)
        sigma_squared = self.convert_to_ublb(sigma_squared)

        lb = np.log(np.column_stack((l[[0]], sigma_squared[0])))
        ub = np.log(np.column_stack((l[[1]], sigma_squared[1])))

        super().__init__(n_output, lb, ub)

    def prior_cov(self, xi, x1, x2=None):
        parameters = self.xi_to_parameters(xi)
        l = parameters[:self.n_l]
        sigma_squared = parameters[self.n_l:]
        
        return covariance_squared_exponential(l, sigma_squared, x1, x2)