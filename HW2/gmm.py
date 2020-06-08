import numpy as np
from tqdm import tqdm

class GMM(object):
    def __init__(self): # No need to implement
        pass
    
    def softmax(self,logits):
    
        raise NotImplementedError

    def logsumexp(self,logits):

        raise NotImplementedError

    def _init_components(self, points, K, **kwargs):

        raise NotImplementedError


    def _ll_joint(self, points, pi, mu, sigma, **kwargs):

        raise NotImplementedError

    def _E_step(self, points, pi, mu, sigma, **kwargs):

        raise NotImplementedError

    def _M_step(self, points, gamma, **kwargs):

        raise NotImplementedError

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):

        raise NotImplementedError
