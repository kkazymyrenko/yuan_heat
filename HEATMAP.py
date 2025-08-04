# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 22:32:06 2025

@author: roron
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from scipy.sparse import csr_matrix, save_npz, load_npz
pio.renderers.default = "browser"

def norm(x): return x/(np.linalg.norm(x)+1e-15)

def norm_q_fromstt(stt):
	""" Returns solution norm via Sato quantum method
	from a displacement vector """
	ropt = fpsi_fromstt(stt) / psiHpsi_fromstt(stt)
	return ropt


def psiHpsi_fromstt(stt):
	""" Returns the value of <stt| H |stt>
	where stt is given as a displacment vector """
	return np.real(np.vdot(stt, H @ stt))


def fpsi_fromstt(stt):
	""" Returns the value of <frc|stt>
	where stt is given as a displacement vector """
	#print("stt=",stt)
	#f_psi = np.real(frc @ stt)
	f_psi = np.real(np.vdot(stt, frc))
	return f_psi



frc=np.load("frc6.npy")
ucl=np.load("ucl6.npy")
K_sparse = np.load("K6.npy")
H=K_sparse
Nx=4
Ny=Nx
def reconstruct_volume(data, Nx=Nx, Ny=Ny):
	Nf = data.size // (Nx * Ny)  # Number of faces along Z
	volume = np.empty((Ny, Nx, Nf))  # Shape: (rows, columns, faces)

	for i in range(Nf):
		face = data[i * Nx * Ny:(i + 1) * Nx * Ny].reshape((Ny, Nx), order='F')
		face = np.fliplr(face)  # Align with bottom-to-top, right-to-left filling
		volume[:, :, i] = face

	return volume

#resolution=32*32
# Load classical reference solution
ucl = np.load("ucl6.npy")
volume_ref = reconstruct_volume(ucl)
ref_slice = np.flipud(np.fliplr(volume_ref[:, :, 0]))

# Compute fixed color scale from classical solution
vmin = np.min(ref_slice)
vmax = np.max(ref_slice)

# Load quantum solution (you can swap this for any uqnX)
uqn = np.real(np.load("state6_IONQ.npy"))
uqn *= norm_q_fromstt(np.real(norm(ucl)))
volume_q = reconstruct_volume(uqn)
image = np.flipud(np.fliplr(volume_q[:, :, 0]))

# Plot quantum result with colorbar based on classical reference
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(image, cmap='turbo', interpolation='lanczos', vmin=vmin, vmax=vmax)

ax.set_title("6 qubits quantum solution (Aria-1 QPU)", pad=15)
ax.axis('off')

# Shared colorbar using fixed scale
cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
cbar.set_label("Temperature")

plt.show()

# n_nodes_per_side=Nx
# x = np.linspace(0, 3, n_nodes_per_side)
# y = np.linspace(0, 3, n_nodes_per_side)
# z = np.linspace(0, 3, n_nodes_per_side)

# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
# fig = go.Figure(data=go.Scatter3d(
# 		x=X.flatten(),
# 		y=Y.flatten(),
# 		z=Z.flatten(),
# 		mode='markers',
# 		marker=dict(
# 			size=10,
# 			color=ucl,  # Color by temperature
# 			colorscale='turbo',
# 			colorbar=dict(title='Temp'),
# 			opacity=0.8
# 		),
# 		text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
# 				 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), ucl)],
# 		 hoverinfo='text'
# 	))
# 	
# fig.update_layout(
# 		scene=dict(
# 			xaxis_title='X',
# 			yaxis_title='Y',
# 			zaxis_title='Z',
# 		),
# 		title='3D Temperature Distribution obtained classically'
# 	)
# 	
# fig.show()
