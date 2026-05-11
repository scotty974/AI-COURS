import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def identity(z):
    return z


def identity_derivative(z):
    return np.ones_like(z)


activation_map = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "identity": identity,
}

activation_derivative_map = {
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "relu": relu_derivative,
    "identity": identity_derivative,
}
