import numpy as np


class CleanData(object):
    def __init__(self):  # No need to implement
        pass

    def pairwise_dist(self, x, y):  # [0pts] - copy from kmeans
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        diff = y.T - x
        print('diff {}'.format(diff.shape))
        return np.sqrt(np.sum(np.square(diff), axis=1))

    #         return np.linalg.norm(x-y)

    def find_neightbors(self, points, curr_point, K):
        dist = np.zeros(points.shape[0])
        dist = self.pairwise_dist(points, curr_point)
        #         print(dist)
        index = dist.argsort()
        neighbors = points[index]
        return neighbors, index

    def clean_class(self, label, complete_points, incomplete_points):
        inc_class_0 = incomplete_points[incomplete_points[:, -1] == label]
        comp_class_0 = complete_points[complete_points[:, -1] == label]
        for i in np.argwhere(np.isnan(inc_class_0)):
            #             print("el {}".format(inc_class_0[i[0]]))
            dist = self.pairwise_dist(np.delete(comp_class_0, i[1], axis=1), np.delete(inc_class_0[i[0]], i[1], axis=0))
            #             print("dist {}".format(dist))
            #             print("argsort {}".format(np.argsort(dist)))
            #             print("k nn {}".format(comp_class_0[np.argsort(dist)]))
            #             print("mean for col {}".format(np.mean(comp_class_0[np.argsort(dist),i[1]])))
            inc_class_0[i[0], i[1]] = np.mean(comp_class_0[np.argsort(dist), i[1]])
        return inc_class_0

    def __call__(self, incomplete_points, complete_points, K, **kwargs):  # [10pts]
        """
        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points: N_complete x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_incomplete + N_complete) x (D-1) X D numpy array of length K, containing both complete points and recently filled points

        Hints: (1) You want to find the k-nearest neighbors within each class separately;
               (2) There are missing values in all of the features. It might be more convenient to address each feature at a time.
        """
        inc_class_0 = self.clean_class(complete_points=complete_points, label=0, incomplete_points=incomplete_points)
        inc_class_1 = self.clean_class(complete_points=complete_points, label=1, incomplete_points=incomplete_points)
        return np.concatenate([complete_points, inc_class_0, inc_class_1], axis=0)
