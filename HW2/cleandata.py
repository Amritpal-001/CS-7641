import numpy as np

class CleanData(object):
    def __init__(self): # No need to implement
        pass
    
    def pairwise_dist(self, x, y):

        raise NotImplementedError
    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs):

        raise NotImplementedError            
