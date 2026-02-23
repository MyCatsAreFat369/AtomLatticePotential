import numpy as np
from scipy.sparse.linalg import eigsh, eigs
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import eigh_tridiagonal

N = 2001
spacing = 10
x0 = -spacing / 2
xf = spacing / 2
dx = (x0 - xf) / (N - 1)
X = np.linspace(xf + dx, x0 - dx, N - 2)

#B = 100
v0 = 1
mu = 0.4

def get_potential(x):
    #return np.exp(x)
    #return np.exp(-(x + 10) ** 2 / (2 * 0.5 ** 2))
    #return B / abs(x)
    return v0 * (np.cosh(mu) ** 2) * ((np.tanh(x) + np.tanh(mu)) ** 2)

class Eigenstate:
    def __init__(self, xi):
        d = 1 / (dx ** 2) + get_potential(X - xi)
        e = -1 / (2 * dx ** 2) * np.ones(N - 3)
        w, v = eigh_tridiagonal(d, e)
        self.eigenvalues = w
        self.eigenvectors = v

    def get_e(self, n):
        return self.eigenvectors.T[n]

    def get_pdf(self, n):
        sum = 0
        Y1 = self.get_e(n)
        for y in Y1:
            sum += (y ** 2) * dx
        return (Y1 ** 2) / sum

pdfcount = 3
es = Eigenstate(0)
eigenstates = []
for i in range(0, pdfcount):
    eigenstates.append(es.get_e(i))

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))

energies1 = es.eigenvalues

for i in range(0, pdfcount):
    ax[0, 0].plot(X, es.get_pdf(i))

ax[0, 1].plot(X, get_potential(X))

ax[0, 2].bar(np.arange(0, pdfcount, 1), energies1[0:pdfcount])

#total_pdf = np.zeros(len(eigenstates[0].get_pdf(0)))
#for e in eigenstates:
    #total_pdf += e.get_pdf(0)
#total_pdf /= len(eigenstates)

#integral = 0
#for x, y in zip(X, total_pdf):
#    integral += x * y * dx
#print(f"mean position: {integral}")
#print(f"s: {abs(spacing / 2 - abs(integral))}")

#ax[1, 0].plot(X, total_pdf)

plt.show()