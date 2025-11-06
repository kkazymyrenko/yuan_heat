
# fem_matrix_builder3d_cornerflux_6q.py
#
# Build and save FEM-like matrices/vectors for a 3D steady-state
# heat equation with a localized corner heat source.
# Domain: [0,1]^3

# ============================================================

import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix, save_npz


n_elem_per_side = 3     
h = 1.0 / n_elem_per_side
k = 1.0
regul = 1e-8
suffix = "_corner6q"

print(f"Building 3D heat system with {n_elem_per_side+1}³ = {(n_elem_per_side+1)**3} nodes (6 qubits)...")


# Indexing helper

def idx(i, j, l, N):
    return i + j * N + l * N * N

# Assemble 3D stiffness matrix (7-point stencil Laplacian)

def assemble_global_stiffness_matrix(n_elem_per_side, k, h):
    n_nodes_per_side = n_elem_per_side + 1
    total_nodes = n_nodes_per_side ** 3
    K_global = np.zeros((total_nodes, total_nodes))

    def idx(i, j, l, N): return i + j*N + l*N*N
    N = n_nodes_per_side
    for i in range(N):
        for j in range(N):
            for l in range(N):
                row = idx(i, j, l, N)
                K_global[row, row] += -6
                for di, dj, dl in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    ii, jj, ll = i+di, j+dj, l+dl
                    if 0 <= ii < N and 0 <= jj < N and 0 <= ll < N:
                        col = idx(ii, jj, ll, N)
                        K_global[row, col] += 1

    K_global *= -k / (h**2)
    K_global += regul * np.eye(total_nodes)
    return K_global

# Localized corner heat source 

def create_corner_flux(n_nodes_per_side):
    N = n_nodes_per_side
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    z = np.linspace(0, 1, N)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Gaussian source at one corner (0,0,0)
    sigma = 0.25
    f = np.exp(-((X)**2 + (Y)**2 + (Z)**2) / (2 * sigma**2))
    frc = f.reshape(-1)
    frc /= np.max(frc)
    return frc


N = n_elem_per_side + 1
K_global = assemble_global_stiffness_matrix(n_elem_per_side, k, h)
frc = create_corner_flux(N)

# Solve linear system
ucl = sc.sparse.linalg.spsolve(csr_matrix(K_global), frc).real
ucln = ucl / np.linalg.norm(ucl)


# Save outputs 

save_npz(f"Hsp6{suffix}.npz", csr_matrix(K_global))
np.save(f"H6{suffix}.npy", K_global)
np.save(f"frc6{suffix}.npy", frc)
np.save(f"ucl6{suffix}.npy", ucl)
np.save(f"ucln6{suffix}.npy", ucln)

print(f" Saved files with suffix '{suffix}':")
print(f"    Hsp6{suffix}.npz")
print(f"    H6{suffix}.npy")
print(f"    frc6{suffix}.npy")
print(f"    ucl6{suffix}.npy")
print(f"    ucln6{suffix}.npy")
