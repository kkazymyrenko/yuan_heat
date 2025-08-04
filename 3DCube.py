import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


import plotly.graph_objects as go
import plotly.io as pio
from scipy.sparse import csr_matrix, save_npz, load_npz
pio.renderers.default = "browser"

def norm(x): return x/(np.linalg.norm(x)+1e-15)

frc=np.load("frc.npy")
ucl=np.load("ucl2.npy")
#K_sparse = np.load("K6.npy")
K_sparse=load_npz("K_globalsparse.npz")
H=K_sparse

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
def generate_hollow_cube(size=10):
	"""Create a boolean mask for a hollow cube (outer voxels only)."""
	cube = np.ones((size, size, size), dtype=bool)
	cube[1:-1, 1:-1, 1:-1] = False  # Remove inner voxels
	return cube

size = 32
hollow_mask = generate_hollow_cube(size)

uqn=np.real(np.load("uqn15x.npy"))
#uqn2=np.load(".npy")

uqn*=norm_q_fromstt(np.real(norm(uqn)))

#ucl=np.load("ucl2.npy")

Nx=32
Ny=Nx
def reconstruct_volume(data, Nx=Nx, Ny=Ny):
	Nf = data.size // (Nx * Ny)  # Number of faces along Z
	volume = np.empty((Ny, Nx, Nf))  # Shape: (rows, columns, faces)

	for i in range(Nf):
		face = data[i * Nx * Ny:(i + 1) * Nx * Ny].reshape((Ny, Nx), order='F')
		face = np.fliplr(face)  # Align with bottom-to-top, right-to-left filling
		volume[:, :, i] = face

	return volume

volume=reconstruct_volume(uqn)


# Synthetic temperature data (replace with your actual values)
z_coords, y_coords, x_coords = np.indices((size, size, size))
temperature_data =volume# np.linspace(0, 100, size)[z_coords]  # replace HERE
temperature_data[~hollow_mask] = np.nan  # Remove inner voxels

def plot_cube(data,save=''):
	""" Plot Voxelized Cube with Temperature Colors """
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')

	# Normalize temperature for colormap
	norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(temperature_data))
	cmap = plt.cm.turbo  # Choose colormap (e.g., 'plasma', 'inferno')

	# Plot only outer voxels with temperature colors
	ax.voxels(
		hollow_mask,
		facecolors=cmap(norm(data)),
		edgecolor='k',  # Black edges
		linewidth=0.1  # Thin edges
	)

	# Add colorbar
	sm = ScalarMappable(cmap=cmap, norm=norm)
	sm.set_array([])
	cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
	cbar.set_label('Temperature (°C)')
	# Labels and title
	#ax.set_xlabel('X Axis')
	#ax.set_ylabel('Y Axis')
	#ax.set_zlabel('Z Axis')
	#ax.set_title('3D Heatmap : 15 qubits noise-free simulation results using the Cascade warm-start')

	# Save as SVG (After Interactive Rotation)
	plt.tight_layout()

	# Uncomment to save:
	if save:
		plt.savefig(f'{save}.svg', format='svg', bbox_inches='tight')

	plt.show()

plot_cube(temperature_data,"cube")