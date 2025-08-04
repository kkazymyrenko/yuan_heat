# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 10:46:53 2025

@author: roron
"""


import numpy as np
import matplotlib.pyplot as plt
#import pylatexenc

import scipy as sc
import sys
import tools.method as mth
import numpy as np
import math as m
import time
import pickle
import scipy as sc
#import noisyopt as no
import qiskit as qi  # 1.1.1
from qiskit.circuit import library as li, ParameterVector
import re
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import NLocal, CCXGate, CRZGate, RXGate, RYGate, CXGate, PhaseGate
from qiskit.quantum_info import SparsePauliOp, Operator
import matplotlib.pyplot as plt
import nlopt
from qiskit import QuantumCircuit, QuantumRegister
import pickle_blosc as pkl
from qiskit import quantum_info as qit 
#from genetic import optim_genetic as ogn
from time import time as ttime
from tools.fct import *
tt = ttime()
import pandas as pd
from datetime import datetime
import os
import plotly.graph_objects as go
import plotly.io as pio
from time import perf_counter as pctime
from scipy.sparse import csr_matrix
import random as rd


nqb1=12 #Here you choose from where you start your mesh refinement
nqb2=nqb1+3
RX=1 #### rx=1 means no RX in your Ansatz, rx=2 means you have Rx in your ansatz
#Note that this could be improved but it's the way I coded it for now, sorry.

#NOTE THAT THIS IS A VERY ROUGH WAY OF APPLYING THE CASCADE, IT ONLY WORKS WITH 2 ANSATZ, THE IDEA WOULD BE
#TO HAVE A MORE GENERALIZED PROGRAM.

#Choose the number of layers. A is for the first ansatz of the circuit (the one where you know theta_opt)
#B is for the second one.
if RX==1:
	lay=20
	layers_A=30
	layers_B=20

if RX==2:
	lay=10
	layers_A=10
	layers_B=10


if RX==2:
	params_A=np.load("par_opt12x.npy")
	
	uqn1=np.load("uqn12x.npy")
	
if RX==1:
	
	params_A=np.load("par_opt12.npy")
	
	
	uqn1=np.load("uqn12.npy")
	
	
	
	

def get_Stt_lazy(par):
	""" Input : parameters. Output : statevector"""
	# print('para', ans.num_parameters , ans.num_qubits, len(par))
	return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))

def apply_permutation_with_swaps(qc, perm):
	current = list(range(len(perm)))
	for i in range(len(perm)):
		while current[i] != perm[i]:
			j = current.index(perm[i])
			qc.swap(i, j)
			current[i], current[j] = current[j], current[i]

if RX==1:
	def gen_ans_cascade(par_cascade, nqb, lay):
		"""
		Generates an ansatz with an initial H layer, followed by RX+RY and CX entanglement.
		`par_cascade` is expected to be a ParameterVector of length 2 * nqb * lay.
		"""
		qc = QuantumCircuit(nqb)
		#qc.h(range(nqb))
	
		# Parameters expected: 2 * nqb * lay (RX and RY each layer)
		if par_cascade is None:
			raise ValueError("par_cascade must be a ParameterVector or a list")
	
		param_idx = 0
		
		for k in range(lay):
			# RX layer
			
			# RY layer
			
			for q in range(nqb):
				qc.ry(par_cascade[param_idx], q)
				param_idx += 1
			# CX entanglement linear
			qc.barrier()
			for q in range(nqb - 1):
				qc.cx(q, q + 1)
			qc.barrier()
		
		# RY layer
		for q in range(nqb):
			qc.ry(par_cascade[param_idx], q)
			param_idx += 1
		
		return qc
	
if RX==2:
	
	def gen_ans_cascade(par_cascade, nqb, lay):
		"""
		Generates an ansatz with an initial H layer, followed by RX+RY and CX entanglement.
		`par_cascade` is expected to be a ParameterVector of length 2 * nqb * lay.
		"""
		qc = QuantumCircuit(nqb)
		#qc.h(range(nqb))
	
		# Parameters expected: 2 * nqb * lay (RX and RY each layer)
		if par_cascade is None:
			raise ValueError("par_cascade must be a ParameterVector or a list")
	
		param_idx = 0
		
		for k in range(lay):
			# RX layer
			for q in range(nqb):
				qc.rx(par_cascade[param_idx], q)
				param_idx += 1
			# RY layer
			
			for q in range(nqb):
				qc.ry(par_cascade[param_idx], q)
				param_idx += 1
			# CX entanglement linear
			qc.barrier()
			for q in range(nqb - 1):
				qc.cx(q, q + 1)
			qc.barrier()
		for q in range(nqb):
			qc.rx(par_cascade[param_idx], q)
			param_idx += 1
		# RY layer
		for q in range(nqb):
			qc.ry(par_cascade[param_idx], q)
			param_idx += 1
		
		return qc

ucln2=norm(np.load("ucln12.npy"))
#THERE MUST BE AN EASIER WAY TO GENERALIZE THE PERMUTATION BUT FOR NOW, THAT'S IT.
permutation=[]
permutation.append(0)
for i in range(nqb1//3):
	permutation.append(i+3)
	last=i+3
permutation.append(1)
for k in range(1,nqb1//3 +1):
	permutation.append(last+k)
	
last=last+nqb1//3
permutation.append(2)
for j in range(1,nqb1//3 +1):
	permutation.append(last+j)
perm=permutation
def Cascade(params_A, par, lay_A, lay_B,perm=perm):
	"""
	params_A: list of float, length 2*6*(lay_A+1)
	params_B: list of float, length 2*9*(lay_B+1)
	returns: final sampled statevector
	"""
	# --- Step 1: Apply ansatz A (with fixed params) to qubits 0–5
	
	ans_A= gen_ans_cascade(params_A,12, lay_A)
	if len(params_A) != RX*12*(lay_A+1):
		raise ValueError("Wrong number of parameters for ansatz A.")
	#ans_A = ans_A.assign_parameters(params_A, inplace=False)
	qc = QuantumCircuit(12)
	qc.compose(ans_A, qubits=range(12), inplace=True)
# 	qc.draw(output='mpl')
# 	plt.show()
	qs = QuantumCircuit(15)
	svv= Statevector.from_instruction(qc)
	qqq = QuantumCircuit(12)
	#apply_permutation_with_swaps(qqq, list(range(5)))
	#svv.evolve(qqq)
	svh=Statevector.from_label("000")
	qc=QuantumCircuit(3)
	qc.h(0)
	qc.h(1)
	qc.h(2)
	svh=svh.evolve(qc)
	svv=svv.tensor(svh)
	
	#svv=svv.evolve(qs)
	
	apply_permutation_with_swaps(qs, perm)
	svv=svv.evolve(qs)
	
	

	qAnB=QuantumCircuit(15)
	# --- Step 4: Apply ansatz B with theta_B and then dagger of B(0)
	ans_B= gen_ans_cascade(par,15, lay_B)
	#print(par[2])
	if len(par) != RX*15*(lay_B+1):
		raise ValueError("Wrong number of parameters for ansatz B.")
	#print(params_B[2])

	qAnB=qAnB.compose(ans_B)
	sdv=svv.evolve(qAnB)
	qAnB0=QuantumCircuit(15)

	# Zero parameters → dagger
	zeros = [0] * RX*15*(lay_B+1)
	ans_B_zero = gen_ans_cascade(zeros,15,lay_B)
	qAnB0=qAnB0.compose(ans_B_zero.inverse())
	
	sf=sdv.evolve(qAnB0)
# 	qc.draw(output='mpl')
# 	plt.show()
	sv_dict = sf.to_dict()
	info=[]
	tak=0
	if RX==1:
		
		for basis_state, amplitude in sv_dict.items():
			#tak+=1
			info.append(np.sqrt(np.real(amplitude)**2 +np.imag(amplitude)**2))
			#print(f"|{basis_state}>: {np.sqrt(np.real(amplitude)**2+np.imag(amplitude)**2)}")
			if tak==80:
				break
	if RX==2:
		for basis_state, amplitude in sv_dict.items():
			#tak+=1
			info=sf.data
			#print(f"|{basis_state}>: {np.sqrt(np.real(amplitude)**2+np.imag(amplitude)**2)}")
			if tak==80:
				break

	# --- Sample final statevector
	return info

#FIRST YOU NEED TO COMPUTE THE 15 QUBIT MATRIX, THE FLUX TERM AND VECTOR : 
#NOTE THAT IF YOU  COMPUTED IT EARLIER, YOU CAN JUST SAVE THOSE AND LOAD THEM INSTEAD OF COMPUTING THEM EACH TIME...
#NOTE THAT YOU HAVE TO DO SO FROM 15 QUBITS (1h minimum to compute the matrix + a few minutes to solve)
#SAVING IT AND LOADING IT AS SPARSE SAVES YOU A LOOOT OF TIME


h = 1.0
k=1.0
regul=0 #1e-2 #Regularization coefficient. (0= no regularization)
run_N=0
double=0 #0 if no double objective any other positive value for double objective
n_elem_per_side = 31
n_nodes_per_side = n_elem_per_side + 1
n_nodes = n_nodes_per_side ** 3
nqb=m.ceil(np.log2(n_nodes))
run_N=0
print("with",nqb,"qubits and", lay, "layers")
method = "LN_NEWUOA_BOUND"
	
normalize=True
def norm(x): return x/(np.linalg.norm(x) if normalize else 1)
pio.renderers.default = None
#Get elementary stifness Matrix (K_local)
def get_hexahedral_stiffness_matrix(k=1.0, h=1.0):
	"""
	Compute the 8x8 local stiffness matrix for a cubic element
	using 2x2x2 Gauss quadrature. Kyrylo found it.
	
	Parameters:
		k : float
			Thermal conductivity (assumed constant we can change it but it's not important for now'
		h : float
			Size of the cubic element (assumes uniform h in x, y, z)
	
	Returns:
		K 	(8, 8) stiffness matrix
	"""
	# Gauss points and weights for 2-point rule
	gauss_pts = [-1/np.sqrt(3), 1/np.sqrt(3)]
	weights = [1, 1]

	# Shape function derivatives in reference coordinates
	def dphi_dxi(i, xi, eta, zeta):
		signs = np.array([
			[-1, -1, -1],
			[ 1, -1, -1],
			[ 1,  1, -1],
			[-1,  1, -1],
			[-1, -1,  1],
			[ 1, -1,  1],
			[ 1,  1,  1],
			[-1,  1,  1]
		])
		sx, sy, sz = signs[i]
		return np.array([
			0.125 * sx * (1 + sy * eta) * (1 + sz * zeta),
			0.125 * sy * (1 + sx * xi)  * (1 + sz * zeta),
			0.125 * sz * (1 + sx * xi)  * (1 + sy * eta)
		])

	# Initialize stiffness matrix
	K = np.zeros((8, 8))

	# Jacobian matrix (constant for cube with uniform size)
	J = (h / 2) * np.identity(3)
	detJ = np.linalg.det(J)
	invJ = np.linalg.inv(J)
	#See 3DFEM png to understand where the formulas come from.

	# Loop over Gauss quadrature points
	for xi in gauss_pts:
		for eta in gauss_pts:
			for zeta in gauss_pts:
				grads = []
				for i in range(8):
					dN_ref = dphi_dxi(i, xi, eta, zeta)
					dN_phys = invJ @ dN_ref  # gradient in physical space
					grads.append(dN_phys)
				# Assemble stiffness contributions
				for i in range(8):
					for j in range(8):
						K[i, j] += k * np.dot(grads[i], grads[j]) * detJ

	return K

K_local = get_hexahedral_stiffness_matrix(k=k, h=h)



import numpy as np

def assemble_global_stiffness_matrix(n_elem_per_side, k, h,Nx):
	"""
	Assemble the global stiffness matrix for a regular 3D hexahedral mesh.

	Parameters:
	- n_elem_per_side: Number of elements per side (int)
	- k: Material parameter for local stiffness matrix (passed to get_hexahedral_stiffness_matrix)
	- h: Grid spacing
	- get_hexahedral_stiffness_matrix: function that returns 8x8 local K matrix

	Returns:
	- K_global: Assembled global stiffness matrix (numpy array)
	"""
	Nx = Ny = Nz = n_elem_per_side + 1  # Number of nodes per side

	# Helper functions
	def D(N): I = np.eye(N); I[-1, -1] = 0; return I
	def U(N): I = np.eye(N); I[0, 0] = 0; return I
	def T(N): return np.eye(N, k=1)
	def Tt(N): return np.eye(N, k=-1)
	
	def get_conn_mat(delta, N):
		if delta == 0: return D(N)
		elif delta == 1: return T(N)
		elif delta == -1: return Tt(N)
		else: return U(N)

	Dx, Dy, Dz = D(Nx), D(Ny), D(Nz)
	Ux, Uy, Uz = U(Nx), U(Ny), U(Nz)

	# Node positions in reference element (8 Hex8 nodes)
	positions = [
		(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
		(0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)
	]

	# Get local stiffness matrix
	K_local = get_hexahedral_stiffness_matrix(k=k, h=h)

	# Allocate global matrix
	total_nodes = Nx * Ny * Nz
	K_global = np.zeros((total_nodes, total_nodes))

	for i in range(8):
		for j in range(8):
			dx = positions[j][0] - positions[i][0]
			dy = positions[j][1] - positions[i][1]
			dz = positions[j][2] - positions[i][2]
			
			if abs(dx) > 1 or abs(dy) > 1 or abs(dz) > 1:
				continue

			Mx = get_conn_mat(dx, Nx)
			My = get_conn_mat(dy, Ny)
			Mz = get_conn_mat(dz, Nz)

			# Fix for upper diagonal
			if positions[j][0] != 0 and np.allclose(Mx, Dx): Mx = Ux
			if positions[j][1] != 0 and np.allclose(My, Dy): My = Uy
			if positions[j][2] != 0 and np.allclose(Mz, Dz): Mz = Uz

			kron_product = np.kron(Mz, np.kron(My, Mx))
			K_global += K_local[i, j] * kron_product
	#K_sym = 0.5 * (K_global + K_global.T)
	#K_sym = 0.5 * (K_global + K_global.T)
	#scale_factor = np.max(np.abs(K_global))
	K_global *= Nx
	print("Regularization=",regul)
	K_global += regul* np.eye(K_global.shape[0]) #this regularization is optionnal, I found an article saying it can help iterative solver like ours, especially in 3D with lots of elements.
	#The regulation is mandatory IF you don't fix temps
	#1e-1
	#K_sym= 0.5 * (K_sym + K_sym.T)

	return K_global


def create_flux_vector_fem3(flux_dict, n_elem_per_side=n_elem_per_side, h=1.0):
	"""
	Assemble the global flux vector for a structured Hex8 FEM mesh,
	supporting multiple Neumann flux boundaries with custom per-face
	flux, coverage percentage, and coverage direction.

	Parameters:
	-----------
	flux_dict : dict
		Dictionary mapping face names (e.g., 'x+', 'z-') to
		dicts with keys:
			- 'flux' : float (required) → flux value (W/m²)
			- 'coverage' : float (0–1, optional) → portion of face covered
			- 'direction' : str ('x', 'y', or 'z', optional) → axis along which coverage is applied

	n_elem_per_side : int
		Number of elements per axis (assuming cubic domain)

	h : float
		Element size (assuming uniform spacing)

	Returns:
	--------
	frc : ndarray of shape (n_nodes,)
		Assembled global RHS flux vector
	"""
	n_nodes_per_side = n_elem_per_side + 1
	n_nodes = n_nodes_per_side ** 3
	frc = np.zeros(n_nodes)

	# Gauss quadrature (2x2 on square)
	gauss_pts = [-np.sqrt(1 / 3), np.sqrt(1 / 3)]

	def shape_functions(xi, eta):
		return np.array([
			0.25 * (1 - xi) * (1 - eta),
			0.25 * (1 + xi) * (1 - eta),
			0.25 * (1 + xi) * (1 + eta),
			0.25 * (1 - xi) * (1 + eta)
		])

	def element_node_indices(i, j, k):
		nps = n_nodes_per_side
		return [
			i	 +  j*nps +  k*nps*nps,
			(i+1) +  j*nps +  k*nps*nps,
			(i+1) + (j+1)*nps +  k*nps*nps,
			i	 + (j+1)*nps +  k*nps*nps,
			i	 +  j*nps + (k+1)*nps*nps,
			(i+1) +  j*nps + (k+1)*nps*nps,
			(i+1) + (j+1)*nps + (k+1)*nps*nps,
			i	 + (j+1)*nps + (k+1)*nps*nps
		]

	# Local node indices of each face (standard Hex8)
	face_nodes = {
		'x-': [0, 3, 7, 4],
		'x+': [1, 2, 6, 5],
		'y-': [0, 1, 5, 4],
		'y+': [3, 2, 6, 7],
		'z-': [0, 1, 2, 3],
		'z+': [4, 5, 6, 7],
	}

	# For each face, define which 2 axes form its plane
	face_axes = {
		'x-': ('j', 'k'),
		'x+': ('j', 'k'),
		'y-': ('i', 'k'),
		'y+': ('i', 'k'),
		'z-': ('i', 'j'),
		'z+': ('i', 'j'),
	}

	# Iterate over each face entry in the dict
	for face, spec in flux_dict.items():
		flux_value = spec.get('flux', 1.0)
		coverage = spec.get('coverage', 1.0)
		direction = spec.get('direction', face_axes[face][0])  # default axis for coverage

		for i in range(n_elem_per_side):
			for j in range(n_elem_per_side):
				for k in range(n_elem_per_side):

					# Check if the current element touches the specified face
					if (face == 'x-' and i != 0) or (face == 'x+' and i != n_elem_per_side - 1) or \
						(face == 'y-' and j != 0) or (face == 'y+' and j != n_elem_per_side - 1) or \
						(face == 'z-' and k != 0) or (face == 'z+' and k != n_elem_per_side - 1):
						continue

					# Coverage filtering
					axis_index = {'i': i, 'j': j, 'k': k}[direction]
					if axis_index > int((coverage) * (n_elem_per_side - 1)):
						continue

					# Element info
					nodes = element_node_indices(i, j, k)
					local_face = face_nodes[face]
					J_det = h * h  # area of face

					for xi in gauss_pts:
						for eta in gauss_pts:
							N = shape_functions(xi, eta)
							for a in range(4):
								global_node = nodes[local_face[a]]
								frc[global_node] += flux_value * N[a] * J_det #/n_nodes_per_side
	return frc



def fix_temp2(K_global, frc, face, Nx, Ny, Nz, value=0.0):
	"""
	Fix temperature at specified face by modifying K and f in-place.

	Parameters:
	-----------
	K_global : ndarray
		Global stiffness matrix (modified in-place)
	frc : ndarray
		Right-hand side vector (modified in-place)
	face : str
		Face on which to fix the temperature: 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'
	Nx, Ny, Nz : int
		Number of nodes along x, y, z directions
	value : float
		Temperature value to fix (default = 0.0)
	"""
	nps = Nx * Ny * Nz  # just to confirm size matches
	fixed_nodes=[]
	if face == 'x-':
		for k in range(Nz):
			for j in range(Ny):
				nfix = 0 + j * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[:, nfix] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value
				fixed_nodes.append(nfix)
	elif face == 'x+':
		for k in range(Nz):
			for j in range(Ny):
				nfix = (Nx - 1) + j * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[:, nfix] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value
				fixed_nodes.append(nfix)

	elif face == 'y-':
		for k in range(Nz):
			for i in range(Nx):
				nfix = i + 0 * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[:, nfix] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value
				fixed_nodes.append(nfix)

	elif face == 'y+':
		for k in range(Nz):
			for i in range(Nx):
				nfix = i + (Ny - 1) * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[:, nfix] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value
				fixed_nodes.append(nfix)

	elif face == 'z-':
		for j in range(Ny):
			for i in range(Nx):
				nfix = i + j * Nx + 0 * Nx * Ny
				K_global[nfix, :] = 0
				K_global[:, nfix] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value
				fixed_nodes.append(nfix)

	elif face == 'z+':
		for j in range(Ny):
			for i in range(Nx):
				nfix = i + j * Nx + (Nz - 1) * Nx * Ny
				K_global[nfix, :] = 0
				K_global[:, nfix] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value
				fixed_nodes.append(nfix)

	else:
		raise ValueError(f"Invalid face name '{face}'. Use one of: x±, y±, z±.")
	return fixed_nodes

#There you call the functions so you either load previous saved files or you compute them again.
from scipy.sparse import csr_matrix, save_npz, load_npz
# K_local = get_hexahedral_stiffness_matrix(k=k, h=h)
Nx = Ny = Nz = n_elem_per_side + 1
K_global = load_npz("K_globalsparse.npz")# or if you want to compute it : assemble_global_stiffness_matrix(n_elem_per_side=n_elem_per_side, k=k, h=h,Nx=Nx)

# flux_dict = {'y-':{'flux': 13, 'coverage': 1.0,'direction': 'i'}
# 	 ,'z+': {'flux': 15, 'coverage': 0.5, 'direction': 'i'}
# 	#'x+': {'flux': 0.5, 'coverage': 0.6,'direction':'k'},
# 	#'z-': {'flux': 0.1, 'coverage': 0.5, 'direction': 'i'},
# }
# frc = create_flux_vector_fem3(flux_dict, n_elem_per_side=n_elem_per_side, h=1.0)
# all_fixed=fix_temp2(K_global,frc,'y+',Nx,Ny,Nz,value=4.0)
# all_fixed+=fix_temp2(K_global,frc,'x-',Nx,Ny,Nz,value=10)



frc=np.load("frc.npy")

K_sparse = load_npz("K_globalsparse.npz")#csr_matrix(K_global)
#frc=norm(frc)

ucl = np.load("ucl2.npy")#sc.sparse.linalg.spsolve(K_sparse, frc).real

Hsp=K_sparse
H=K_global
#SPOH = SparsePauliOp.from_operator(Operator(H))
#ucl = sc.sparse.linalg.spsolve(Hsp, frc).real
ucln = ucl/np.linalg.norm(ucl)


### VISUALIZING THE SOLUTION AS A CUBE IN PLOTLY 
h = 1.0  # element size
x = np.linspace(0, 3, n_nodes_per_side)
y = np.linspace(0, 3, n_nodes_per_side)
z = np.linspace(0, 3, n_nodes_per_side)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')


fig = go.Figure(data=go.Scatter3d(
	x=X.flatten(),
	y=Y.flatten(),
	z=Z.flatten(),
	mode='markers',
	marker=dict(
		size=10,
		color=ucl,  # Color by temperature
		colorscale='turbo',  # Color scale
		colorbar=dict(title='Temperature'),
		opacity=0.8
	),
	text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
			 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), ucl)],
	 hoverinfo='text'
))

fig.update_layout(
	scene=dict(
		xaxis_title='X',
		yaxis_title='Y',
		zaxis_title='Z'
	),
	title='Temperature distribution in 3D cube'
)

fig.show()

#CHECK IF THE 1ST STEP OF THE CASCADE WORKS WITH THIS NEXT PART.
# COST FUNCTION
def cost_cl(x):
	cost = -1 / 2 * np.real(frc @ x) ** 2 / np.real(x.T @ H @ x)
	return cost

def elong_ymax(stt):
	Ntot = len(stt)
	nqb = round(np.log2(Ntot))
	nx = ny = nqb // 2
	Ny = 2 ** ny
	Nx = 2 ** nx
	return sum(stt[(Ny-1)*Nx + x] for x in range(Nx)) * Ny

def norm_q(par):
	""" Returns solution norm via Sato quantum method """
	ropt = fpsi(par) / psiHpsi(par)
	return ropt

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

def psiHpsi(par):
	""" Returns the value of <psi(par)| H |psi(par)> """
	job = est.run([una_ans], [SPOH], [par], shots=shots)
	result = job.result().values[0]
	return result


def fpsi(par):
	""" Returns the value of <frc|psi(par)> """
	ans = una_ans.assign_parameters(parameters=par, inplace=False)
	psi = sampleGate(ans)
	return fpsi_fromstt(psi)



def Neg(stt,nQPU):
	qubits=[]
	for i in range(nQPU//2):
		qubits.append(i)
	S=qit.schmidt_decomposition(stt,qubits)
	somme=0
	for k in S:
		coef,b1,b2=k
		somme+=coef
	negativity=1/2 *((somme**2) -1)
	maxneg=1/2 *((2**(nQPU//2)) -1)
	return accuracy(negativity,maxneg)


svt = sec = ttime() #Seconds counted from epoch ?
optim = False
errored = False
using_nlopt = True
NCOST = 0

def cost_q_gen(fp=fpsi, php=psiHpsi, compstt=False):
	"""
	fp : function computing f psi
	php : function computing psi H psi
	compstt : whether the functions fp and php take stt as input
	"""
	def cost_q(par,*k,**kw):
		""" E = -1/2 <frc|psi>² / <psi|H|psi> """ #
		global NCOST, sec, svt
		NCOST += 1
		if not compstt:
			print("COSTQ was here",flush=True)
			ans = ans_stt.assign_parameters(parameters=par, inplace=False)
			x = sampleGate(ans)
		else:
			x = par
		cost = -1 / 2 * fp(x) ** 2 / php(x)
		if ttime() > sec + 1 and using_nlopt:
			sec = ttime()
			print(f"{NCOST}th COST : {cost*(1000 if nqb > 6 else 1):.4f}")
		if ttime() > svt + 300:
			svt = ttime()
			print(f"SAVING PARAMETERS in temp_par...")
			pkl.pickle(par, f"temp_par_{method}.pkl")
		return cost
	return cost_q

cost_q = cost_q_gen()
cost_q_fromstt = cost_q_gen(fpsi_fromstt, psiHpsi_fromstt, True)


cost_fun='energy'
using_nlopt = True
Runs=25
test_ini=[]
# 			for i in range(num_par):
# 				if i%(2*nqb)<nqb:
# 					test_ini.append(np.pi)
# 				if i%(2*nqb)>=nqb:
# 					test_ini.append(np.pi/2)
num_par=RX*nqb*(layers_B+1) #YOU'D NEED TO CHANGE THAT IF YOU CREATE ANOTHER ANSATZ
for i in range(num_par):
	test_ini.append(0)
parC = np.array(test_ini)
	
	
	
	
uqn1 = np.array(uqn1)  # Make sure it's a NumPy array

	# Reshape to 3D cube: (z, y, x)
	
shaping1=int(len(uqn1)**(1/3))+1
uqn1_cube = uqn1.reshape((shaping1,shaping1,shaping1))

# # Use Kronecker product to upscale by 2 in each dimension (2x2x2 = 8)
#np.ones((2,2,2)) makes each value expand into a cube of 8 identical values
uqn_cube_upscaled = np.kron(uqn1_cube, np.ones((2, 2, 2))) / np.sqrt(8)
	# Now reshape back to 1D if you need a flat array
uqn1b = uqn_cube_upscaled.flatten()
	
uqn= norm(Cascade(params_A,parC, layers_A,layers_B)[:n_nodes])
uqn2=uqn.copy()
	
fig = go.Figure(data=go.Scatter3d(
	x=X.flatten(),
	y=Y.flatten(),
	z=Z.flatten(),
		mode='markers',
		marker=dict(
			size=10,
			color=uqn1b,  # Color by temperature
			colorscale='turbo',
			colorbar=dict(title='Temp'),
			opacity=0.8
		),
		text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
				 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), uqn1b)],
		 hoverinfo='text'
	))

fig.update_layout(
		scene=dict(
			xaxis_title='X',
			yaxis_title='Y',
			zaxis_title='Z',
		),
		title='Warm_start of 3D Temperature Distribution obtained via Cascade method classic'
	)
fig.show()
	
fig = go.Figure(data=go.Scatter3d(
		x=X.flatten(),
		y=Y.flatten(),
		z=Z.flatten(),
		mode='markers',
		marker=dict(
			size=10,
			color=uqn,  # Color by temperature
			colorscale='turbo',
			colorbar=dict(title='Temp'),
			opacity=0.8
		),
		text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
				 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), uqn2)],
		 hoverinfo='text'
	))
	
fig.update_layout(
		scene=dict(
			xaxis_title='X',
			yaxis_title='Y',
			zaxis_title='Z',
		),
		title='Warm_start of 3D Temperature Distribution obtained via Cascade method'
	)
	
fig.show()
	

uqn= norm(Cascade(params_A,parC, layers_A,layers_B)[:n_nodes])
initial_cost=accuracy(-1 / 2 * ((fpsi_fromstt(norm(uqn[:n_nodes]))) ** 2) / (psiHpsi_fromstt(norm(uqn[:n_nodes]))),-1 / 2 * (fpsi_fromstt(ucln) ** 2) / psiHpsi_fromstt(ucln))
print("INITIAL CASCADE COST :",initial_cost,"%")
uqn1=uqn.copy()
print(np.round(uqn2[0]*np.sqrt(8),5)==np.round(uqn1[0],5))
normc = np.linalg.norm(ucl)
norm_ropt = norm_q_fromstt(uqn)

	#uqn *= norm_ropt

	# NORMALISATION
print("Norme classique Ucl :", normc)
print("Norme quantique Xquant :", norm_ropt, "\n")
print("Done with",num_par, "parameters.")
mse = MSE(uqn, ucl)
print(f"MSE error : {mse:.6f}")
	
	#uqn = reorder_solution(uqn, Nx)
	
	
fig = go.Figure(data=go.Scatter3d(
		x=X.flatten(),
		y=Y.flatten(),
		z=Z.flatten(),
		mode='markers',
		marker=dict(
			size=10,
			color=uqn,  # Color by temperature
			colorscale='turbo',
			colorbar=dict(title='Temp'),
			opacity=0.8
		),
		text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
				 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), uqn)],
		 hoverinfo='text'
	))
	
fig.update_layout(
		scene=dict(
			xaxis_title='X',
			yaxis_title='Y',
			zaxis_title='Z',
		),
		title='Warm_start of 3D Temperature Distribution obtained via Cascade method'
	)

fig.show()

for i in range(Runs):
	NCOST=0
	testQ=[]
	test=[]
	Qost=[]
	Wait=[]
	refs=[]
	par_list=[]
	neg_ref=[]
	Fidelity=[]
	ref_Fidelity=[]
	i=0



	if run_N < 10:
	
		print("testt")
		#ans_stt = gen_ansv2(nqb, lay)
		#ans_stt = Cascade(params_A6, theta_B9, layers_A, layers_B)
	
		
		print("Nombre",RX*nqb*(layers_B+1))
	
		
		
		def maxistt(par,H=H,Hsp=Hsp,frc=frc,ucl=ucl,i=i, grad=None,n_nodes=n_nodes, *k, **w):
		# 		Hsp, frc = get_S()
		# 		H=Hsp.toarray()
		# 		ucl = sc.sparse.linalg.spsolve(Hsp.tocsr(), frc).real
		# 		SPOH = SparsePauliOp.from_operator(Operator(H))  # KKK to supress, no need in current implementation
		# 		ucln = ucl/np.linalg.norm(ucl)
				global sec, NCOST, svt, nframe
				#par2=par
				NCOST += 1
	
				#stt = norm(np.real(Cascade(par,nqb,lay)[:n_nodes]))
				stt = norm(Cascade(params_A,par, layers_A,layers_B)[:n_nodes])
				#print(par[2])
				normalisation=norm_q_fromstt(stt[:n_nodes])
				par_list.append(par.copy())
				cost_fun="energy"
				if cost_fun == 'energy':
					cost = -1 / 2 * ((fpsi_fromstt(norm(stt[:n_nodes]))) ** 2) / (psiHpsi_fromstt(norm(stt[:n_nodes])))
					#print(cost)
					ref= -1 / 2 * (fpsi_fromstt(ucln) ** 2) / psiHpsi_fromstt(ucln)
					#print(ref)
					## This part needs to be updated, l'objectif serait d'avoir plutôt des raw data et manipuler ensuite ces données
					#Ca veut dire qu'on a besoin de sauvegarder la liste de refs à la fin
					#Donc faut modifier aussi le fichier qui plot et traite les données
					#Si j'ai la foi je le ferai (pour l'instant je ne l'ai pas)
					#Fidelity.append(elong_ymax(stt)/elong_ymax(ucln) *100)
					#Fidelity.append(100- MSE(ucl, stt*normalisation))
					Fidelity.append(-MSE(ucl, stt*normalisation))
					Qost.append(cost)
					Wait.append(NCOST)
					ref_Fidelity.append(100- MSE(ucl, ucl))
					refs.append(ref)
					#neg.append(Neg(stt,nqb))
					#neg_ref.append(Neg(ucln,nqb))
				if ttime() > sec + 1  and using_nlopt: #sec +1 si jamais ça bug
					sec = ttime()
					if cost_fun == 'MSE':
						def mlt(nqb):  # multiplicator
							return 10000 * (1000 if nqb > 8 else 1)
						#print(f"{NCOST}th COST : {abs(cost)*mlt(nqb):.4f}")
						print(f"{NCOST}th COST : {abs(cost/ref)*100:.4f}%")
						
					else:
							print(f"{NCOST}th COST : {accuracy(np.real(cost),ref):.4f}%")
							#print(par[2])
							#print(len(stt))
							#print(f"MSE={100-MSE(ucl, stt[:n_nodes] *normalisation) }%")
							if accuracy(cost,ref)>93:
								print(f"Checking:{np.abs(ref)-np.abs(np.real(cost))} ")
							print(f"MSE={-MSE(ucln, np.real(stt[:n_nodes])) }%")
							#print(f"Projection : {np.dot(norm(ucl),norm(stt[:n_nodes]))}")
							#print(f"MSE={MSE(norm(ucl),norm(stt[:n_nodes]))}")
				return np.real(cost)
			
		blabla=True
		if blabla==True:
			
			num_par = RX*nqb*(layers_B+1)
			optim = True  # only True if this cell is executed
			#par_ini = np.array( 2 * np.pi * np.random.random(num_par))
			#par_ini= np.full(num_par, np.pi/2)
			#par_ini=np.zeros(num_par)
			x0_cl = np.random.rand(2 ** nqb)
		
			#resCL = sc.optimize.minimize(cost_cl,x0_cl[:n_nodes] , tol=1e-5, method='L-BFGS-B')
			par_opt = [0] * num_par
			low_bnd = np.array([-np.pi*2] * num_par) #-3*np.pi instead of 0 ??
			up_bnd= np.array([np.pi*2] *num_par)
	
			if optim:
				print("Layers :", lay)
				print("Number of parameters: ", num_par)
				print("Number of DOFs: ", 2 ** nqb)
				print("Number of elements : ", n_nodes)
		
				print("	*** Beginning optimization ***	")
				start = time.perf_counter()
		# 	Hsp, frc = get_S()
		# 	H=Hsp.toarray()
		# 	ucl = sc.sparse.linalg.spsolve(Hsp.tocsr(), frc).real
		
		# 	SPOH = SparsePauliOp.from_operator(Operator(H))
			# compares to a classical minimization - more realistic than comp. to algebra
			if True:
				if True:#while abs(qval) < .1 and trust_factor > 0:#abs(resCL.fun.real)/10:
					#trust_factor -= 1
		# 			if warm_start:
		# 				par0 = list(pkl.unpickle(warm_start))
		# 				for x in range(len(par0)):
		# 					while par0[x] < -1.5*np.pi:
		# 						par0[x] += 2 * np.pi
		# 					while par0[x] > 1.5*np.pi:
		# 						par0[x] -= 2 * np.pi
					#par0 = np.array(2*np.pi * np.random.random(num_par)) #np.pi/10
					test_ini=[]
		# 			for i in range(num_par):
		# 				if i%(2*nqb)<nqb:
		# 					test_ini.append(np.pi)
		# 				if i%(2*nqb)>=nqb:
		# 					test_ini.append(np.pi/2)
					for i in range(num_par):
						test_ini.append(np.random.uniform(-0.001,0.001))
	# 					if i<nqb:
	# 						test_ini.append(np.pi)
	# 					elif i>nqb and i<2*nqb:
	# 						test_ini.append(np.pi/2)
	# 					else:
	# 						test_ini.append(2*np.pi*np.random.random())
					par0 = np.array(test_ini)
					#for k in range(nqb):
						#	par0[k] = np.pi/2 if k != 1 else 0
	
					par_opt, qval = mth.opt_res(par0, maxistt, low_bnd=low_bnd,up_bnd=up_bnd, verbose=False)
					#par_opt, qval = mth.opt_res_c7(par0, maxistt,Hsp=Hsp, frc=frc, H=H, ucl=ucl,n_nodes=n_nodes,max_retries=3,up_bnd=up_bnd, low_bnd=low_bnd, verbose=False,epsilon=5,lay=lay,ucln=ucln, nqb=nqb, norm=norm,lambda_penalty=double)
					#par_opt, qval = mth.opt_res_cma8(par0, maxistt,Hsp=Hsp, frc=frc, H=H, ucl=ucl,n_nodes=n_nodes, low_bnd=low_bnd, verbose=False,epsilon=1,lay=lay,ucln=ucln, nqb=nqb, norm=norm,lambda_penalty=5e1)
					#par_opt, qval = mth.opt_res_cmaenergy(par0, maxistt, low_bnd=low_bnd,up_bnd=up_bnd, verbose=False,lay=lay, nqb=nqb)
		
					#all_Qcost.append(Qost)
					print(len(par0) == RX*nqb*(layers_B+1))
					#all_Wait.append(Wait)
		
			else:#except Exception as e:
				print(e)
				print('(Error)',end='\n'*3)
				errored = True
				ccost,ref,NCOST= maxistt(par_opt)
				qval=ccost
				#all_Qcost.append(Qost)
				#all_Wait.append(Wait)
		else:
			print("ERROR")
			qval = 0
		end = time.perf_counter()
		exe_time = end - start
		
		#uqn = norm(np.real(Cascade(par_opt,nqb,lay)[:n_nodes]))
		uqn= norm(Cascade(params_A,par_opt, layers_A,layers_B)[:n_nodes])
		
		normc = np.linalg.norm(ucl)
		norm_ropt = norm_q_fromstt(uqn)
		
	data_folder = f"3D_{nqb2}qb_Cascade_{co}"
	
	i = 1
	while os.path.exists(os.path.join(data_folder, f"run_{i}")):
		i += 1
	run_folder = os.path.join(data_folder, f"run_{i}")
	os.makedirs(run_folder, exist_ok=True)
	
	df=pd.DataFrame({'Wait':Wait,'Qcost':Qost,'Fidelity':Fidelity,'ref_fidelity':ref_Fidelity,'ref':refs})
	filename = os.path.join(run_folder, f"results_{nqb}qb_{lay}layers.csv")
	df.to_csv(filename, index=False)
	filename2 = os.path.join(run_folder, "par_opt.npy")
	filename3 =  os.path.join(run_folder, "uqn{nqb2}.npy")
	np.save(filename2,par_opt)
	np.save(filename3,uqn)
	print(f"Données sauvegardées dans {filename}")
	NCOST=0

