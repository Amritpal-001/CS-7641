import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall

class Isomap(object):
    def __init__(self): # nothing should be implemented here
        pass
    
    def pairwise_dist(self, x, y):
    
        raise NotImplementedError

    
    def manifold_distance_matrix(self, x, K):
    
        raise NotImplementedError


    def multidimensional_scaling(self, dist_matrix, d):
        
        raise NotImplementedError


    def __call__(self, data, K, d):
        
        raise NotImplementedError
