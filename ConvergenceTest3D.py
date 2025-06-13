# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:34:12 2025

@author: roron
"""

import numpy as np
import matplotlib.pyplot as plt

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
from qiskit.circuit.library import NLocal, CCXGate, CRZGate, RXGate, RYGate, CXGate, PhaseGate
from qiskit.quantum_info import SparsePauliOp, Operator
import matplotlib.pyplot as plt
import nlopt
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


def norm(x): return x/(np.linalg.norm(x) if normalize else 1)
pio.renderers.default = "browser"
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

#Using the cmesh function, we create a global matrix from K_localk = 1.0
h = 1.0
k=1.0

run_N=0
lay=12
n_elem_per_side = 7
double=0 #0 if no double objective any other positive value for double objective
n_nodes_per_side = n_elem_per_side + 1
n_nodes = n_nodes_per_side ** 3
nqb=m.ceil(np.log2(n_nodes))
print("with",nqb,"qubits and", lay, "layers")
method = "LN_NEWUOA_BOUND"


#######################################################################
  # 3 nodes along each axis
K_local = get_hexahedral_stiffness_matrix(k=k, h=h)
Nx = Ny = Nz = n_elem_per_side + 1

import numpy as np

def assemble_global_stiffness_matrix(n_elem_per_side, k, h):
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
	scale_factor = np.max(np.abs(K_global))
	K_global /= scale_factor
	K_global += 1e-2* np.eye(K_global.shape[0]) #this regularization is optionnal, I found an article saying it can help iterative solver like ours, especially in 3D with lots of elements.
	#The regulation is mandatory IF you don't fix temps
	#1e-1

	return K_global

K_global = assemble_global_stiffness_matrix(n_elem_per_side=n_elem_per_side, k=k, h=h)



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
#NOTE THAT IN PYPLOT, X AND Z HAVE BEEN SWAPPED. HOW TO FIX THIS ISSUE : CHANGE THE WAY THE INDICES BEHAVE EVERYWHERE (TOO LAZY TO DO IT NOW)



flux_dict = {'y-':{'flux': 0.5, 'coverage': 1.0,'direction': 'i'}
	 ,'z+': {'flux': 0.8, 'coverage': 0.5, 'direction': 'i'}
	#'x+': {'flux': 0.5, 'coverage': 0.6,'direction':'k'},
	#'z-': {'flux': 0.1, 'coverage': 0.5, 'direction': 'i'},
}



#i for z, j for y and k for x.[]
frc = create_flux_vector_fem3(flux_dict, n_elem_per_side=n_elem_per_side, h=1.0)

def fix_temp(K_global,frc,face):
	if type(face)==str:
		if face=='x-':
			for nfix in range(0,Nz*Nx):
				print(nfix)
				K_global[nfix,:]=0
				K_global[nfix,nfix]=1
				frc[nfix]=0
		if face=='x+':
			for nfix in range(Ny*Nz*Nx -(Ny*Nx),Ny*Nz*Nx):
				

				K_global[nfix,:]=0
				K_global[nfix,nfix]=1
				frc[nfix]=0
		if face=='z-':
			for nfix in range(0,Ny*Nz):
				nfix=nfix*Ny
				K_global[nfix,:]=0
				K_global[nfix,nfix]=1
				frc[nfix]=0
		if face=='z+':
			for nfix in range(0,Ny*Nz):
				nfix=nfix*Ny -1
				K_global[nfix,:]=0
				K_global[nfix,nfix]=1
				frc[nfix]=0
		if face=='y-':
			for boup in range(0,Nz):
				for a in range(0,Ny):
					nfix=boup*Ny*Nx +a
					#print(nfix)
					K_global[nfix,:]=0
					K_global[nfix,nfix]=1
					frc[nfix]=0
		if face=='y+':
			#for boup in range(Nz*(Ny-1),Nz*Ny):
			for m in range(0,Ny):
				for a in range(0,Ny):
					nfix=Nz*(Ny-1)+a +m*Ny*Nz
					#print(nfix)
					K_global[nfix,:]=0
					K_global[nfix,nfix]=1
					frc[nfix]=0

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

	if face == 'x-':
		for k in range(Nz):
			for j in range(Ny):
				nfix = 0 + j * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value

	elif face == 'x+':
		for k in range(Nz):
			for j in range(Ny):
				nfix = (Nx - 1) + j * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value

	elif face == 'y-':
		for k in range(Nz):
			for i in range(Nx):
				nfix = i + 0 * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value

	elif face == 'y+':
		for k in range(Nz):
			for i in range(Nx):
				nfix = i + (Ny - 1) * Nx + k * Nx * Ny
				K_global[nfix, :] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value

	elif face == 'z-':
		for j in range(Ny):
			for i in range(Nx):
				nfix = i + j * Nx + 0 * Nx * Ny
				K_global[nfix, :] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value

	elif face == 'z+':
		for j in range(Ny):
			for i in range(Nx):
				nfix = i + j * Nx + (Nz - 1) * Nx * Ny
				K_global[nfix, :] = 0
				K_global[nfix, nfix] = 1
				frc[nfix] = value

	else:
		raise ValueError(f"Invalid face name '{face}'. Use one of: x±, y±, z±.")
#fix_temp(K_global,frc,face='y+')
#fix_temp2(K_global,frc,'x-',Nx,Ny,Nz,value=10.0)
normalize = True  # whether to

fix_temp2(K_global,frc,'y+',Nx,Ny,Nz,value=4)
# fix_temp2(K_global,frc,'y-',Nx,Ny,Nz,value=4.5)
# fix_temp2(K_global,frc,'z-',Nx,Ny,Nz,value=4.5)
# fix_temp2(K_global,frc,'z+',Nx,Ny,Nz,value=4.5)
fix_temp2(K_global,frc,'x-',Nx,Ny,Nz,value=10.0)
#fix_temp2(K_global,frc,'x+',Nx,Ny,Nz,value=4.5)



#fix_temp2(K_global,frc,'x-',Nx,Ny,Nz,value=9.0)
K_sparse = csr_matrix(K_global)
#frc=norm(frc)

ucl = sc.sparse.linalg.spsolve(K_sparse, frc).real

Hsp=K_sparse
H=K_global
SPOH = SparsePauliOp.from_operator(Operator(H))
ucl = sc.sparse.linalg.spsolve(Hsp, frc).real
ucln = ucl/np.linalg.norm(ucl)


#u_grid = u.reshape((n_nodes_per_side, n_nodes_per_side, n_nodes_per_side))
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


# COST FUNCTION
def cost_cl(x):
	cost = -1 / 2 * (frc @ x) ** 2 / (x.T @ H @ x)
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

def psiHpsi(par):
	""" Returns the value of <psi(par)| H |psi(par)> """
	job = est.run([una_ans], [SPOH], [par], shots=shots)
	result = job.result().values[0]
	return result

def psiHpsi_fromstt(stt):
	""" Returns the value of <stt| H |stt>
	where stt is given as a displacment vector """
	return stt @ H @ stt

def fpsi(par):
	""" Returns the value of <frc|psi(par)> """
	ans = una_ans.assign_parameters(parameters=par, inplace=False)
	psi = sampleGate(ans)
	return fpsi_fromstt(psi)

def fpsi_fromstt(stt):
	""" Returns the value of <frc|stt>
	where stt is given as a displacement vector """
	#print("stt=",stt)
	f_psi = frc @ stt
	return f_psi
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
	ans_stt = gen_ans(nqb, lay)
	print("Nombre",ans_stt.num_parameters)

	def get_Stt_lazy(par):
		""" Input : parameters. Output : statevector"""
		# print('para', ans.num_parameters , ans.num_qubits, len(par))
		return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))



	def maxistt(par,H=H,Hsp=Hsp,frc=frc,ucl=ucl,i=i, grad=None,n_nodes=n_nodes, *k, **w):
# 		Hsp, frc = get_S()
# 		H=Hsp.toarray()
# 		ucl = sc.sparse.linalg.spsolve(Hsp.tocsr(), frc).real
# 		SPOH = SparsePauliOp.from_operator(Operator(H))  # KKK to supress, no need in current implementation
# 		ucln = ucl/np.linalg.norm(ucl)
		global sec, NCOST, svt, nframe
		NCOST += 1
		stt = norm(get_Stt_lazy(par)[:n_nodes])
		normalisation=norm_q_fromstt(stt[:n_nodes])
		cost_fun="energy"
		if cost_fun == 'energy':
			cost = -1 / 2 * (fpsi_fromstt(norm(stt[:n_nodes])) ** 2) / psiHpsi_fromstt(norm(stt[:n_nodes]))
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
			par_list.append(par)
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
					print(f"{NCOST}th COST : {accuracy(cost,ref):.4f}%")
					#print(len(stt))
					#print(f"MSE={100-MSE(ucl, stt[:n_nodes] *normalisation) }%")
					if accuracy(cost,ref)>93:
						print(f"Checking:{np.abs(ref)-np.abs(cost)} ")
					print(f"MSE={-MSE(ucln, stt[:n_nodes]) }%")
					#print(f"Projection : {np.dot(norm(ucl),norm(stt[:n_nodes]))}")
					#print(f"MSE={MSE(norm(ucl),norm(stt[:n_nodes]))}")
		return cost
	num_par = ans_stt.num_parameters
	optim = True  # only True if this cell is executed
	#par_ini = np.array( 2 * np.pi * np.random.random(num_par))
	par_ini= np.full(num_par, np.pi/2)
	#par_ini=np.zeros(num_par)
	x0_cl = np.random.rand(2 ** nqb)

	resCL = sc.optimize.minimize(cost_cl,x0_cl[:n_nodes] , tol=1e-5, method='L-BFGS-B')
	par_opt = [0] * num_par
	low_bnd = np.array([-np.pi] * num_par) #-3*np.pi instead of 0 ??
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
			for i in range(num_par):
				if i%(2*nqb)<nqb:
					test_ini.append(np.pi)
				if i%(2*nqb)>=nqb:
					test_ini.append(np.pi/2)
			par0 = np.array(test_ini)
			#for k in range(nqb):
				#	par0[k] = np.pi/2 if k != 1 else 0

			#par_opt, qval = mth.opt_res(par0, maxistt, low_bnd=low_bnd, verbose=False)
			par_opt, qval = mth.opt_res_c7(par0, maxistt,Hsp=Hsp, frc=frc, H=H, ucl=ucl,n_nodes=n_nodes,max_retries=1,up_bnd=up_bnd, low_bnd=low_bnd, verbose=False,epsilon=5,lay=lay,ucln=ucln, nqb=nqb, norm=norm,lambda_penalty=double)
			#par_opt, qval = mth.opt_res_cma8(par0, maxistt,Hsp=Hsp, frc=frc, H=H, ucl=ucl,n_nodes=n_nodes, low_bnd=low_bnd, verbose=False,epsilon=1,lay=lay,ucln=ucln, nqb=nqb, norm=norm,lambda_penalty=5e1)
			#par_opt, qval = mth.opt_res_cmaenergy(par0, maxistt, low_bnd=low_bnd,up_bnd=up_bnd, verbose=False,lay=lay, nqb=nqb)

			#all_Qcost.append(Qost)
			print(len(par0) == ans_stt.num_parameters)
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

uqn = norm(get_Stt_lazy(par_opt)[:n_nodes])

normc = np.linalg.norm(ucl)
norm_ropt = norm_q_fromstt(uqn)

uqn *= norm_ropt

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
	title='3D Temperature Distribution obtained via VQA'
)

fig.show()

fig = go.Figure(data=go.Scatter3d(
	x=X.flatten(),
	y=Y.flatten(),
	z=Z.flatten(),
	mode='markers',
	marker=dict(
	size=10,
		color=uqn/norm_ropt,  # Color by temperature
		colorscale='turbo',
		colorbar=dict(title='Temp'),
		opacity=0.8
	),
	text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
		 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), uqn/norm_ropt)],
	 hoverinfo='text'
))

fig.update_layout(
	scene=dict(
		xaxis_title='X',
		yaxis_title='Y',
		zaxis_title='Z',
	),
	title='3D Temperature Distribution obtained via VQA'
)

fig.show()
fig = go.Figure(data=go.Scatter3d(
	x=X.flatten(),
	y=Y.flatten(),
	z=Z.flatten(),
	mode='markers',
	marker=dict(
		size=10,
		color=ucln,  # Color by temperature
		colorscale='turbo',
		colorbar=dict(title='Temp'),
		opacity=0.8
	),
	text=[f"x={x:.2f}<br>y={y:.2f}<br>z={z:.2f}<br>Temp={t:.2f}" 
			 for x, y, z, t in zip(X.flatten(), Y.flatten(), Z.flatten(), ucln)],
	 hoverinfo='text'
))

fig.update_layout(
	scene=dict(
		xaxis_title='X',
		yaxis_title='Y',
		zaxis_title='Z',
	),
	title='3D Temperature Distribution obtained classically'
)

fig.show()



acc=[]
maxi=[]
for i in range(len(Qost)):
	acc.append(accuracy(Qost[i],refs[i]))
for i in range(len(Qost)):
	if acc[i]==np.max(acc):
		maxi.append(i)
uqn = norm(get_Stt_lazy(par_list[maxi[0]])[:n_nodes])

normc = np.linalg.norm(ucl)
norm_ropt = norm_q_fromstt(uqn)

uqn *= norm_ropt

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
	title='3D Temperature Distribution obtained via VQA (Best one in opt)'
)

fig.show()
