import numpy as np
from kmeans import KMeans
from tqdm import tqdm


class GMM(object):
    def __init__(self):  # No need to implement
        pass

    def softmax(self, logits):  # [5pts]
        """
        Args:
            logits: N x D numpy array
        Return:
            logits: N x D numpy array
        """

        return np.exp(logits - self.logsumexp(logits))

    def logsumexp(self, logits):  # [5pts]
        """
        Args:
            logits: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logits[i,:])
        """
        max_sum = np.max(logits,axis=1)
        logits_scaled = logits - max_sum[:,None]
        return (np.log(np.sum(np.exp(logits_scaled.astype(np.float64)),axis=1, dtype=np.float64).astype(np.float64) +
                       1e-64)+max_sum).\
            reshape((logits.shape[0], 1))

    def _init_components(self, points, K, **kwargs):  # [5pts]
        """
        Args:
            points: NxD numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        """
        sigma = np.zeros((K, points.shape[1], points.shape[1]))
        pi = np.array([1 / K for i in range(K)])
        clusters_idx, mu, _ = KMeans()(points, K, max_iters=10000, verbose=False)
        for k in range(K):
            n_k = len(np.where(clusters_idx == k))
            mu_k = mu[k]
            sigma[k] = np.dot(pi[k] * mu_k.T, mu_k) / n_k
        print("sigma shape".format(sigma.shape))
        return pi, mu, sigma

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):  # [10pts]
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])

        Hint for undergraduate: Assume that each dimension of our multivariate gaussian are independent.
              This allows you to write treat it as a product of univariate gaussians.
        """
        K = len(mu)
        ll = np.zeros((points.shape[0], K))
        for j in range(K):
            normal_log = self.log_multivariate_normal(points, mu[j], sigma[j])+1e-64
            print("normal log = {}".format(normal_log))
            ll[:, j] = np.log(pi[j]+1e-64, dtype=np.float64) + normal_log
        return ll

    def log_multivariate_normal(self, x, mu, cov):
        x_scaled = x - mu
        #         print("x-mu= shape {}".format(x_scaled.shape))
        A = np.dot((x_scaled.astype(np.float64)), np.linalg.pinv(cov.astype(np.float64)))
        #         print("Shape A {}".format(A.shape))
        B = (A.T * x_scaled.T).sum(axis=0).astype(np.float64)
        #         print("Shape B {}".format(B.shape))
        return -B / 2 - np.log(1e-64 + np.sqrt(((2 * np.pi) ** x.shape[1]) * (np.linalg.det(cov))))

    def _E_step(self, points, pi, mu, sigma, **kwargs):  # [5pts]
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        return self.softmax(self._ll_joint(points, pi, mu, sigma))

    def _M_step(self, points, gamma, **kwargs):  # [10pts]
        """
        Args:
            points: NxD numpy array, the observations
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:  There are formulas in the slide.
        """
        pi = gamma.sum(axis=0) / len(points)
        print("pi shape {}".format(pi.shape))
        assert len(pi) == gamma.shape[1]
        mu = np.dot(gamma.T, points)/gamma.sum(axis=0).reshape((-1,1))
        assert mu.shape == (gamma.shape[1], points.shape[1])
        print("mu shape {}".format(mu.shape))
        sigma = np.zeros((len(mu), points.shape[1], points.shape[1]))
        for k in range(gamma.shape[1]):
            x_scaled = points - mu[k]
            print("x_scaled shape {}".format(x_scaled.shape))
            assert x_scaled.shape == points.shape
            A = gamma[:, k].T * x_scaled.T
            print("A shape {}".format(A.shape))
            assert A.shape == (points.shape[1], points.shape[0])
            sigma[k] = np.dot(A, x_scaled) / gamma[:, k].sum(0)
        print("sigma shape ={}".format(sigma.shape))
        assert sigma.shape == (gamma.shape[1], points.shape[1], points.shape[1])
        return pi, mu, sigma

    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)       
        Hint: You do not need to change it. For each iteration, we process E and M steps, then 
        """
        pi, mu, sigma = self._init_components(points, K, **kwargs)
        pbar = tqdm(range(max_iters))
        for it in pbar:
            # E-step
            gamma = self._E_step(points, pi, mu, sigma)

            # M-step
            pi, mu, sigma = self._M_step(points, gamma)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(points, pi, mu, sigma)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)
