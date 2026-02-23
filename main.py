import numpy as np
import matplotlib.pyplot as plt

A = 1
B = 2
dx = 0.01

def differential_equation(x, y, yd):
    return (A - (B / abs(x))) * y

def runge_kutta(x0, y0, y0d, xf):
    if x0 == xf:
        return np.array([]), np.array([])
    s = 1
    if xf < x0:
        s = -1
    x1 = x0
    y1 = y0
    y1d = y0d
    X1 = [x1]
    Y1 = [y1]
    while (x1 < xf and x0 < xf) or (x1 > xf and x0 > xf):
        k1 = y1d
        l1 = differential_equation(x1, y1, y1d)
        k2 = y1d + s * (l1 * dx / 2)
        l2 = differential_equation(x1 + s * (dx / 2), y1 + s * (k1 * dx / 2), k2)
        k3 = y1d + s * (l2 * dx / 2)
        l3 = differential_equation(x1 + s * (dx / 2), y1 + s * (k2 * dx / 2), k3)
        k4 = y1d + s * (l3 * dx)
        l4 = differential_equation(x1 + s * dx, y1 + s * (k3 * dx), k4)
        y1d += s * (l1 + 2 * l2 + 2 * l3 + l4) * (dx / 6)
        y1 += s * (k1 + 2 * k2 + 2 * k3 + k4) * (dx / 6)
        x1 += s * dx
        X1.append(x1)
        Y1.append(y1)
    X1 = np.array(X1)
    Y1 = np.array(Y1)
    return X1, Y1

def get_potential(x):
    return B / abs(x)

fig, ax = plt.subplots(ncols=2, figsize=(8, 6))

boundary_x = -5

Xl, Yl = runge_kutta(-0.01, 0, -1, -5)
Xr, Yr = runge_kutta(-0.01, 0, -1, -1)
Xl = Xl[::-1]
Yl = Yl[::-1]
X = np.concatenate([Xl, Xr])
Y = np.concatenate([Yl, Yr])


ax[0].plot(X, Y)

plt.show()