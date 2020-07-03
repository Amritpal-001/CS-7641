from matplotlib import pyplot as plt
import numpy as np


class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X):  # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N x D arrays) as well as color images (N x D x 3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N x D array corresponding to an image (N x D x 3 if color image)
        Return:
            U: N x N (N x N x 3, for color images)
            S: min(N, D) x 1 (min(N, D) x 3, for color images)
            V: D x D (D x D x 3, for color images)
        """

        if X.ndim == 2:
            U, S, V = np.linalg.svd(X)
        else:
            U = np.zeros((X.shape[0], X.shape[0], 3))
            S = np.zeros((np.minimum(X.shape[0], X.shape[1]), 3))
            V = np.zeros((X.shape[1], X.shape[1], 3))

            for c in range(3):
                u_cur, s_cur, v_cur = np.linalg.svd(X[:, :, c])
                U[:, :, c] = u_cur
                S[:, c] = s_cur
                V[:, :, c] = v_cur
        return U, S, V

    def rebuild_svd(self, U, S, V, k):  # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N x N (N x N x 3, for color images)
            S: min(N, D) x 1 (min(N, D) x 3, for color images)
            V: D x D (D x D x 3, for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N x D array of reconstructed image (N x D x 3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        if U.ndim == 2:
            S = S[:k]
            U = U[:, :k]
            V = V[:k, :]
            return np.matmul(U, np.matmul(np.diag(S), V))
        else:
            X_rebuild = np.zeros((U.shape[0], V.shape[0], 3))
            for i in range(3):
                X_rebuild[:, :, i] = np.matmul(U[:, :k, i], np.matmul(np.diag(S[:k, i]), V[:k, :, i]))
            return X_rebuild

    def compression_ratio(self, X, k):  # [5pts]
        """
        Compute compression of an image: (num stored values in original)/(num stored values in compressed)
        Args:
            X: N x D array corresponding to an image (N x D x 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """

        return (k * (X.shape[0] + X.shape[1] + 1)) / (X.shape[0] * X.shape[1])

    def recovered_variance_proportion(self, S, k):  # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D) x 1 (min(N, D) x 3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: int (array of 3 ints for color image) corresponding to proportion of recovered variance
        """
        return np.power(S[:k], 2).sum(axis=0, dtype=np.float) / np.power(S, 2).sum(axis=0, dtype=np.float)
