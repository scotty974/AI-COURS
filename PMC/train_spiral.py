import numpy as np
import matplotlib.pyplot as plt
from core.neural import Models
from core.layers import Layers
from utils.loss import mse_loss, mse_derivative

np.random.seed(0)


def make_spiral(n_points=200, noise=0.2):
    """2-class spiral dataset."""
    X, y = [], []
    for cls in range(2):
        r = np.linspace(0.05, 1.0, n_points)
        t = np.linspace(cls * np.pi, cls * np.pi + 3 * np.pi, n_points) + np.random.randn(n_points) * noise
        X.append(np.stack([r * np.sin(t), r * np.cos(t)], axis=1))
        y.append(np.full((n_points, 1), cls))
    X = np.vstack(X)
    y = np.vstack(y).astype(float)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


X, y = make_spiral(n_points=200, noise=0.15)

models = Models()
models.add_layers(Layers(2, 32, "tanh"))
models.add_layers(Layers(32, 32, "tanh"))
models.add_layers(Layers(32, 1, "sigmoid"))

epochs = 8000
lr = 0.05

for epoch in range(epochs):
    models.fit(X, y)
    out = models.forward()
    loss = mse_loss(out, y)
    grad = mse_derivative(out, y)
    models.backward(grad, lr=lr)
    if epoch % 500 == 0:
        acc = np.mean((out > 0.5).astype(float) == y)
        print(f"Epoch {epoch:5d} | Loss: {loss:.4f} | Acc: {acc:.3f}")

models.fit(X, y)
out = models.forward()
final_acc = np.mean((out > 0.5).astype(float) == y)
print(f"\nFinal accuracy: {final_acc:.3f}")

# Decision boundary
xx, yy = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.2, 1.2, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
models.fit(grid, np.zeros((len(grid), 1)))
zz = models.forward().reshape(xx.shape)

plt.figure(figsize=(6, 6))
plt.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.7)
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolor="k", s=20)
plt.title(f"Spiral 2D | acc={final_acc:.3f}")
plt.savefig("spiral_boundary.png", dpi=100, bbox_inches="tight")
print("Saved spiral_boundary.png")
