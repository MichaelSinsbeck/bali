"""
Class Model

normalize_x should only be True, if model is tabulated
"""
import numpy as np

class Model:
    def __init__(self, x, function, variance, normalize_x = False):
        if x.ndim == 1:
            self.x = x[:,np.newaxis].copy()
        else:
            self.x = x.copy()
        self.function = function
        self.variance = variance
        
        if normalize_x:
            mean = self.x.mean(axis=0)[np.newaxis,:]
            std = self.x.std(axis=0)[np.newaxis,:]
            self.x = (self.x - mean) / std
        
        if callable(self.function):
            self.is_tabulated = False
        else:
            self.y = self.function
            self.is_tabulated = True
            
    def inflate_variance(self, n_output):
        self.variance = np.ones(n_output) * np.array(self.variance).flatten()

    def evaluate(self, x):
        if callable(self.function):
            return self.function(x)
        
        for this_x, this_y in zip(self.x, self.y):
            if np.all(this_x == x):
                return this_y
        raise ValueError(
            'Value {} is not in table'.format(x))
            
    def tabulate(self):
        if not self.is_tabulated:
            self.y = np.array([self.function(this_x) for this_x in self.x])
            self.is_tabulated = True

