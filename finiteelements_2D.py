import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh, eigs
from scipy import sparse

N = 40
xi = -2
xf = 2
dx = (xf - xi) / (N - 1)
X, Y = np.meshgrid(np.linspace(xi, xf, N, dtype=float),
                   np.linspace(xi, xf, N, dtype=float))

#B = -1
De = 1
a = 2
re = 1

def get_potential(x, y):
    #return B / np.sqrt(x ** 2 + y ** 2)
    return De * (1 - np.exp(-a * (np.sqrt(x ** 2 + y ** 2) - re))) ** 2

V = get_potential(X, Y)

diag = np.ones(N)
diags = np.array([diag, -2 * diag, diag])
D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
T = (-1 / (dx ** 2)) * sparse.kronsum(D, D)
U = sparse.diags(V.reshape(N ** 2), (0))
H = T + U
print(diags)

w, v = eigsh(H, k=10, which="SM")

def get_e(n):
    return v.T[n].reshape((N, N))

print(w)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(projection="3d")

fig2.suptitle("Morse potential in two dimensions")

ax2.plot_surface(X, Y, V)

fig.suptitle("Probability Density Functions for Hydrogen Eigenstates")

ax[0, 0].contourf(X, Y, get_e(0) ** 2, 20)
ax[0, 1].contourf(X, Y, get_e(1) ** 2, 20)
ax[0, 2].contourf(X, Y, get_e(2) ** 2, 20)
ax[1, 0].contourf(X, Y, get_e(3) ** 2, 20)
ax[1, 1].contourf(X, Y, get_e(4) ** 2, 20)
ax[1, 2].contourf(X, Y, get_e(5) ** 2, 20)

ax[0, 0].set_title("Eigenstate 1")
ax[0, 1].set_title("Eigenstate 2")
ax[0, 2].set_title("Eigenstate 3")
ax[1, 0].set_title("Eigenstate 4")
ax[1, 1].set_title("Eigenstate 5")
ax[1, 2].set_title("Eigenstate 6")

ax[0, 0].set_aspect("equal")
ax[0, 1].set_aspect("equal")
ax[0, 2].set_aspect("equal")
ax[1, 0].set_aspect("equal")
ax[1, 1].set_aspect("equal")
ax[1, 2].set_aspect("equal")

plt.show()