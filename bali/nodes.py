"""
Class Nodes
"""
import numpy as np

# x must be of size (n_nodes, n_input)
# y must be of size (n_nodes, n_output)
class Nodes():
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y
        if x is None:
            self.n = 0
        else:
            self.n = x.shape[0]
        
    def append(self, x, y):
        if self.x is None:
            self.x = np.atleast_2d(x)
            self.y = np.atleast_2d(y)
        else:
            self.x = np.append(self.x, np.atleast_2d(x), axis = 0)
            self.y = np.append(self.y, np.atleast_2d(y), axis = 0)
        self.n = self.x.shape[0]
        
    def join(self, other_nodes):
        if other_nodes.n > 0:
            self.append(other_nodes.x, other_nodes.y)
        
    def first(self,n):
        # returns the first n nodes for quick conditioning
        # e.g. gpe.condition_to(nodes.first(4))
        n = Nodes()
        n.append(self.x[:n,:], self.y[:n,:])
        return n