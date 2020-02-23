"""
Data creation methods
"""

import numpy as np
import numpy.random as rand
from sklearn import datasets


def lin_data(X, noise, theta):
    """
    Synthetic data generator for linear regression.
    
    Algorithm:
        Set Y = X * theta, then add Gaussian noise to each response.
        
    Args:
        X: (n x d matrix) Covariate matrix
        noise: (float) Standard deviation of the noise to be added to the response
        theta: (d x 1 vector) True vector of regression coefficients
        
    Returns:
        X: n x d covariate matrix
        Y: n x 1 noisy response vector
    """
    Y = np.matmul(X, theta)
    e = rand.normal(0, noise, len(Y))
    Y += e
    return Y
