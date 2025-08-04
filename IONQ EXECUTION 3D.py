# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 12:11:51 2025

@author: roron
"""


from qiskit.circuit import library as li
import os
import numpy as np
from qiskit.providers.jobstatus import JobStatus
import time
from matplotlib import pyplot as plt
import pickle_blosc as pkl
import scipy as sc
import re
from qiskit import QuantumCircuit
from qiskit_ionq import IonQProvider
from qiskit.quantum_info import Statevector
import qiskit as qi 
from qiskit import quantum_info as qit 
from time import time as ttime
tt = ttime()
import plotly.io as pio
from scipy.sparse import csr_matrix
from tools.fct import *

RX=1
#IF Cascade modify this : 
nqb1=12
nqb2=nqb1+3
lay_A=20
lay_B=20
#IF no Cascade modify this : 
nqb=12
lay=20

njob=1
shots=10000
#TO estimate the price of the job(s) : https://ionq.com/programs/research-credits/resource-estimator


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


par=np.load(f"par_opt{nqb1}x.npy")
#If not cascade : 
ans=gen_ans_cascade(par, nqb, lay)
ans.decompose()

#If cascade, use the following : 
#ans=Cascade9(params_A12,par_opt15,4,10)


def reconstruct_state_amplitudes(state_counts):
	total = sum(state_counts.values())
	max_bits = max(len(k) for k in state_counts)
	dim = 2 ** max_bits
	state_vector = np.zeros(dim, dtype=complex)

	for bitstring, count in state_counts.items():
		# Zero-pad to match max_bits
		bitstring = bitstring.zfill(max_bits)
		index = int(bitstring, 2)
		probability = count / total
		state_vector[index] = np.sqrt(probability)

	# Normalize (in case of rounding error)
	state_vector /= np.linalg.norm(state_vector)

	return state_vector.tolist()
count_list=[]
from collections import Counter

def merge_counts(count_list):
	"""
	Merge a list of count dictionaries into a single dictionary.

	Args:
		count_list (list of dict): Each dict maps quantum states (as strings) to counts (ints).

	Returns:
		dict: A dictionary with the same format, summing the counts for each state.
	"""
	total_counts = Counter()
	for counts in count_list:
		total_counts.update(counts)
	return dict(total_counts)

#IONQ
IONQ_TOKEN= "?" #If you can't create an account, send me a whatsapp message and I'll see what I can do,
#Worse case scenario I just give you my free TOKEN (it's from my account and I have 0 dollars on it so it's fine)
provider = IonQProvider(IONQ_TOKEN)

bkd = "ionq_simulator"#"ionq_qpu.aria-1" if you want the real thingy
#
backend = provider.get_backend(bkd)
isa_cir = ans  # gpm(backend=backend, optimization_level=3).run(ans)
isa_cir.name += f" {str(isa_cir.num_qubits)}Q {str(lay)}L"
print(f"Executing {isa_cir.name} on {backend.name()}.")

for i in range(njob):
	job = backend.run(isa_cir, shots=shots)#,noise_model="aria-1" if you want to emulate the noise
	while job.status() is not JobStatus.DONE:
		print(f"Job{i} status is", job.status(), "!", end='\r')
		time.sleep(5)
	count_list.append(job.get_counts())
merged = merge_counts(count_list)
state=reconstruct_state_amplitudes(merged)
np.save(f"state{nqb}_IONQ10K",state)

uqn=np.load(f"uqn{nqb}x.npy")
#np.save(f"state{nqb}xstoch",state)
#state=np.load("state6x.npy")
from scipy.sparse import csr_matrix, save_npz, load_npz
# print(-MSE(state,uqn))
# if (-MSE(state,uqn))>=95:
# 	print("Great Success !!")



#YOU may need to change a few variables whether you chose to work with the cascade or not
n_elem_per_side = int((2**nqb)**(1/3))+1

n_nodes_per_side = n_elem_per_side + 1
n_nodes = n_nodes_per_side ** 3
h = 1.0
k=1.0



normalize=True
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

K_local = get_hexahedral_stiffness_matrix(k=k, h=h)

regul=0

import numpy as np
def MSE(u,v):
	return (np.abs(np.dot(norm(u),norm(v)))**2)*100
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


K_local = get_hexahedral_stiffness_matrix(k=k, h=h)
Nx = Ny = Nz = n_elem_per_side + 1
K_global = assemble_global_stiffness_matrix(n_elem_per_side=n_elem_per_side, k=k, h=h,Nx=Nx)

flux_dict = {'y-':{'flux': 13, 'coverage': 1.0,'direction': 'i'}
	 ,'z+': {'flux': 15, 'coverage': 0.5, 'direction': 'i'}
	#'x+': {'flux': 0.5, 'coverage': 0.6,'direction':'k'},
	#'z-': {'flux': 0.1, 'coverage': 0.5, 'direction': 'i'},
}

#### HERE of course you can either compute the matrix or load it, I'd advise to only load it to gain time.

# #i for z, j for y and k for x.[]
frc = create_flux_vector_fem3(flux_dict, n_elem_per_side=n_elem_per_side, h=1.0)
all_fixed=fix_temp2(K_global,frc,'y+',Nx,Ny,Nz,value=4.0)
# # fix_temp2(K_global,frc,'y-',Nx,Ny,Nz,value=4.5)
# # fix_temp2(K_global,frc,'z-',Nx,Ny,Nz,value=4.5)
# # fix_temp2(K_global,frc,'z+',Nx,Ny,Nz,value=4.5)
# # fix_temp2(K_global,frc,'x+',Nx,Ny,Nz,value=4.5)
all_fixed+=fix_temp2(K_global,frc,'x-',Nx,Ny,Nz,value=10)

#K_sparse = load_npz("K_globalsparse.npz")# csr_matrix(K_global)
#frc=np.load("frc.npy")#norm(frc)
K_sparse=csr_matrix(K_global)
#ucl = np.load("ucl2.npy")#sc.sparse.linalg.spsolve(K_sparse, frc).real
ucl=sc.sparse.linalg.spsolve(K_sparse, frc).real
Hsp=K_sparse
H=K_sparse
#SPOH = SparsePauliOp.from_operator(Operator(H))
#ucl = #sc.sparse.linalg.spsolve(Hsp, frc).real
ucln = norm(ucl)#ucl/np.linalg.norm(ucl)



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

def psiHpsi(par):
	""" Returns the value of <psi(par)| H |psi(par)> """
	job = est.run([una_ans], [SPOH], [par], shots=shots)
	result = job.result().values[0]
	return result

def psiHpsi_fromstt(stt):
	""" Returns the value of <stt| H |stt>
	where stt is given as a displacment vector """
	return np.real(np.vdot(stt, H @ stt))

def fpsi(par):
	""" Returns the value of <frc|psi(par)> """
	ans = una_ans.assign_parameters(parameters=par, inplace=False)
	psi = sampleGate(ans)
	return fpsi_fromstt(psi)

def fpsi_fromstt(stt):
	""" Returns the value of <frc|stt>
	where stt is given as a displacement vector """
	#print("stt=",stt)
	#f_psi = np.real(frc @ stt)
	f_psi = np.real(np.vdot(stt, frc))
	return f_psi

cost = -1 / 2 * ((fpsi_fromstt(norm(state))) ** 2) / (psiHpsi_fromstt(norm(state)))
#print(cost)
ref= -1 / 2 * (fpsi_fromstt(ucln) ** 2) / psiHpsi_fromstt(ucln)
refq=-1 / 2 * (fpsi_fromstt(uqn) ** 2) / psiHpsi_fromstt(uqn)
def accuracy(a, b) -> float:
	""" Returns the relative accuracy of a wrt b, in % """
	if 0 not in [a, b]:
		return ((b / a) if abs(a) > abs(b) else (a / b)) * 100
	else:
		return 0



print("Fidelity ionq/uqn:",MSE(state,uqn),"%")
print("Fidelity ionq/ucl:",MSE(state,ucln),"%")
print(f"Accuracy ionq/ucl {accuracy(np.real(cost),ref):.4f}%")
print(f"Accuracy ionq/uqn {accuracy(np.real(cost), np.real(refq)):.4f}%")
if accuracy(np.real(cost),ref)>30:
	print("Great Success")
if accuracy(np.real(cost),np.real(refq))>50:
	print("Great Success")
#A better thing would be to save it in a CSV to then retrieve it with some code and create a table in latex
#Sorry I didn't think to do so when I coded it.
with open(f"results_ionq{nqb}xstoch.txt", "w") as f:
	f.write("Fidelity ionq/uqn: " + str(MSE(state, uqn)) + "%\n")
	f.write("Fidelity ionq/ucl: " + str(MSE(state, ucln)) + "%\n")
	f.write(f"Accuracy ionq/ucl {accuracy(np.real(cost), ref):.4f}%\n")
	f.write(f"Accuracy ionq/uqn {accuracy(np.real(cost), np.real(refq)):.4f}%\n")
	f.write(f'Shots = {shots}')




