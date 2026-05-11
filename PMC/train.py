from core.neural import Models
import numpy as np
from core.layers import Layers
from utils.loss import mse_loss, mse_derivative

# Dataset XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Modèle
models = Models()

models.add_layers(Layers(2, 256, "relu"))
models.add_layers(Layers(256, 128, "relu"))
models.add_layers(Layers(128, 1, "sigmoid"))

# Training
epochs = 10000

for epoch in range(epochs):

    total_loss = 0

    for i in range(len(X)):

        models.fit(X[i], y[i])

        # forward
        output = models.forward()

        # loss
        loss = mse_loss(output, y[i])
        total_loss += loss

        # gradient
        grad = mse_derivative(output, y[i])

        # backward
        models.backward(grad, lr=0.1)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(X)}")

print("\n=== TEST XOR ===")

for i in range(len(X)):

    models.fit(X[i], y[i])

    output = models.forward()

    print(f"Input: {X[i]} -> Output: {output} | Expected: {y[i]}")