import numpy as np


def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def mse_derivative(y_pred, y_true):
    y_true = np.atleast_2d(y_true)
    y_pred = np.atleast_2d(y_pred)
    return 2 * (y_pred - y_true) / y_true.shape[0]
