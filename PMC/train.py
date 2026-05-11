import numpy as np
from core.neural import Models
from core.layers import Layers
from utils.loss import mse_loss, mse_derivative

np.random.seed(42)

# Dataset XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
])

models = Models()
models.add_layers(Layers(2, 8, "tanh"))
models.add_layers(Layers(8, 1, "sigmoid"))

epochs = 5000
lr = 0.1

for epoch in range(epochs):
    models.fit(X, y)
    output = models.forward()
    loss = mse_loss(output, y)
    grad = mse_derivative(output, y)
    models.backward(grad, lr=lr)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")

print("\n=== TEST XOR ===")
models.fit(X, y)
output = models.forward()
for i in range(len(X)):
    print(f"Input: {X[i]} -> Output: {output[i][0]:.4f} | Expected: {y[i][0]}")
