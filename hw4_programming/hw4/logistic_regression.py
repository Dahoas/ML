import numpy as np

def setup_data():
    """ Setup training dataset

        Returns tuple of length 2: (X_train, y_train)
        where X_train is an Nx2 ndarray and y_train is an Nx1 ndarry.
    """
    X_train = np.array([[0.0, 3.0], [1.0, 3.0], [0.0, 1.0], [1.0, 1.0]])
    y_train = np.array([[1], [1], [0], [0]])
    
    return (X_train, y_train)


### FEEL FREE TO WRITE ANY HELPER FUNCTIONS HERE


def objective(w, X, y, lamb):
    """ Given the parameters, w, the data, X and y, and the hyper parameter,
        lambda, return the objective function for L2-regularized logistic
        regression.

        Specifically, the objective function is the negative log likelihood
        plus the negative log prior. This is the objective function that
        we will want to minimize with respect to the weights, w.

        w: Weight vector in Mx1 numpy ndarray6yy
        X: Design matrix in NxM numpy ndarray
        y: True output data in Nx1 numpy ndarray
        lamb: Scalar lambda value (sorry 'lambda' is a Python keyword)

        Returns: scalar value for the negative log likelihood
    """
    ### YOUR CODE HERE
    return 0

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS HERE


def gradient_descent(X, y, lamb, alpha, w0, num_iter):
    """ Implement gradient descent on the objective function using the
        parameters specified below. Return a list of weight vectors for
        each iteration starting with the initial w0.

        X: Design matrix in NxM numpy ndarray
        y: True output data in Nx1 numpy ndarray
        lamb: Scalar lambda value (sorry 'lambda' is a Python keyword)
        alpha: Scalar learning rate
        w0: Initial weight vector in Mx1 numpy ndarray

        Returns: List of (num_iter+1) weight vectors, starting with w0
        and then the weight vectors after each of the num_iter iterations..
        Each element in the list should be an Mx1 numpy ndarray.
    """
    ### YOUR CODE HERE
    return None

### FEEL FREE TO WRITE ANY HELPER FUNCTIONS HERE


def newtons_method(X, y, lamb, w0, num_iter):
    """ Implement Newton's method on the objective function using the
        parameters specified below. Return a list of weight vectors for
        each iteration starting with the initial w0.

        X: Design matrix in NxM numpy ndarray
        y: True output data in Nx1 numpy ndarray
        lamb: Scalar lambda value (sorry 'lambda' is a Python keyword)
        w0: Initial weight vector in Mx1 numpy ndarray

        Returns: List of (num_iter+1) weight vectors, starting with w0
        and then the weight vectors after each of the num_iter iterations..
        Each element in the list should be an Mx1 numpy ndarray.
    """
    ### YOUR CODE HERE
    return None


