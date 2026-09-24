import numpy as np

def compute_MAE_loss(y, tx, w):
    e = y - tx @ w

    return np.mean(np.abs(e))

def compute_subgradient_mae(y, tx, w):
    """Compute a subgradient of the MAE at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the subgradient of the MAE at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute subgradient gradient vector for MAE
    # ***************************************************
    N = y.shape[0]
    e = y - tx @ w
    grad = 1 / N * (-tx.T @ np.sign(e))
    return grad
