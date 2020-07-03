import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N x 1, the prediction of labels
            label: numpy array of length N x 1, the ground truth of labels
        Return:
            a float value
        '''
        return np.sqrt(np.mean((pred - label) ** 2))

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: numpy array of length N, the 1-D observations
            degree: the max polynomial degree
        Return:
            feat: numpy array of shape Nx(degree+1), remember to include
            the bias term. feat is in the format of:
            [[1.0, x1, x1^2, x1^3, ....,],
             [1.0, x2, x2^2, x2^3, ....,],
             ......
            ]
        """

        feat = np.zeros((len(x), degree + 1))
        for d in range(degree + 1):
            feat[:, d] = x ** d
        return feat

    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        return xtest.dot(weight)

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        return np.dot(np.dot(np.linalg.pinv(np.dot(xtrain.T, xtrain)), xtrain.T), ytrain)

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weights = np.zeros((xtrain.shape[1], 1))
        for e in range(epochs):
            weights += learning_rate * np.dot(xtrain.T, ytrain - np.dot(xtrain, weights)) / len(ytrain)
        return weights

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weights = np.zeros((xtrain.shape[1]))
        for i in range(epochs):
            for idx in range(len(xtrain)):
                delta = xtrain[idx] * (ytrain[idx] - np.dot(xtrain[idx], weights)) / len(ytrain)
                weights += learning_rate * delta
        return weights

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        return np.dot(np.dot(np.linalg.pinv(np.dot(xtrain.T, xtrain) + c_lambda * np.eye(xtrain.shape[1])), xtrain.T),
                      ytrain)

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weights = np.zeros((xtrain.shape[1], 1))
        for e in range(epochs):
            derivative = np.dot(xtrain.T, ytrain - np.dot(xtrain, weights)) / len(ytrain)
            weights += weights * (2 * c_lambda * learning_rate) + learning_rate * derivative
        return weights

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        weights = np.zeros((xtrain.shape[1]))
        for i in range(epochs):
            for idx in range(len(xtrain)):
                derivative = xtrain[idx] * (ytrain[idx] - np.dot(xtrain[idx], weights))
                weights += weights * ( c_lambda * learning_rate) + learning_rate * derivative / len(ytrain)
        return weights

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args:
            X: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            y: Nx1 numpy array, the true labels
            kfold: integer, size of the fold for the data split
            c_lambda: floating number
        Return:
            mean_error: the mean of the RMSE for each fold
        """
        raise NotImplementedError