import numpy as np
from scipy.sparse.linalg import eigsh, eigs
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import PillowWriter
import scienceplots
from scipy import sparse
from skimage import measure
import torch
from torch import lobpcg
device = torch.device("cuda")

N = 120
X, Y, Z = np.mgrid[-25:25:N*1j, -25:25:N*1j, -25:25:N*1j] # in units of a0, 25 is good
dx = np.diff(X[0,:,0])[0]

def get_potential(x, y, z):
    return - (dx ** 2) / np.sqrt(x ** 2 + y ** 2 + z ** 2 + 1e-10)

V = get_potential(X, Y, Z)

diag = np.ones([N])
diags = np.array([diag, -2 * diag, diag])
D = sparse.spdiags(diags, np.array([-1, 0, 1]), N, N)
T = -1/2 * sparse.kronsum(sparse.kronsum(D, D), D)
U = sparse.diags(V.reshape(N ** 3), (0))
H = T + U

H = H.tocoo()
H = torch.sparse_coo_tensor(indices=torch.tensor([H.row, H.col]),
                            values=torch.tensor(H.data), size=H.shape).to(device)


eigenvalues, eigenvectors = lobpcg(H, k=5, largest=False)

