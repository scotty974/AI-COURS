import numpy as np
from utils.activation import activation_map, activation_derivative_map


class Layers:
    def __init__(self, inputs_size, output_size, activation=None):
        limit = np.sqrt(2.0 / inputs_size)
        self.weight = np.random.randn(inputs_size, output_size) * limit
        self.bias = np.zeros(output_size)
        self.activation = activation
        self.z = None
        self.inputs = None

    def forward(self, inputs):
        self.inputs = np.atleast_2d(inputs)
        self.z = np.dot(self.inputs, self.weight) + self.bias
        if self.activation:
            return activation_map[self.activation](self.z)
        return self.z

    def backward(self, grad_out, learning_rate):
        grad_out = np.atleast_2d(grad_out)
        if self.activation:
            grad_out = grad_out * activation_derivative_map[self.activation](self.z)

        grad_w = np.dot(self.inputs.T, grad_out)
        grad_b = np.sum(grad_out, axis=0)
        grad_input = np.dot(grad_out, self.weight.T)

        self.weight -= learning_rate * grad_w
        self.bias -= learning_rate * grad_b

        return grad_input
