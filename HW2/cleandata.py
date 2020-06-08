class CleanData(object):
    def __init__(self): # No need to implement
        pass
    
    def pairwise_dist(self, x, y): # [0pts] - copy from kmeans
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between 
            x[i, :] and y[j, :]
        """
        raise NotImplementedError
    
    def __call__(self, incomplete_points,  complete_points, K, **kwargs): # [10pts]
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
        raise NotImplementedError