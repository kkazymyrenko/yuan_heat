from matplotlib import pyplot as plt
import numpy as np
import pickle_blosc as pkl
import glob
import scipy as sc
from qiskit.circuit import library as li
from shutil import move as shtlmv
import re
from qiskit.circuit.library import NLocal, CCXGate, CRZGate, RXGate, RYGate, CXGate, TGate, HGate, PhaseGate, CRYGate, CRXGate
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator, Sampler
from plotly import graph_objects as go, express as px
from pathlib import Path

est = Estimator()
sampler = Sampler()
shots = None  # number of shots for circuits sampling

theta = Parameter("θ")
def min_max(l):
	if isinstance(l, list) or isinstance(l, np.ndarray):
		t = [min_max(v) for v in l]
		return min([m[0] for m in t]), max([m[1] for m in t])
	else:
		return l, l





# STIFFNESS MATRIX
def move_jobs():
	for fpath in glob.iglob('./api_job_[*.pkl'):
		fname = fpath.split('\\')[-1]
		shtlmv(fpath, "./jobs/" + fname)


def get_K(Nx, Ny, dx, dy, alpha):
	""" Returns stiffness matrix K with bottom-to-top, column-by-column numbering """
	N = Nx * Ny
	K = sc.sparse.lil_matrix((N, N))

	for i in range(Nx):  # Itérer sur les colonnes
		for j in range(Ny):  # Itérer sur les lignes du bas vers le haut
			idx = j + i * Ny  # Indexation correcte

			#print(f"Grid point ({i}, {j}) -> Index {idx}")  # Check qu'on fait bien de bas en haut

			if i < Nx - 1:
				right = idx + Ny  # Passage à la colonne suivante
				if right < N:  
					K[idx, idx] += alpha / dx ** 2
					K[idx, right] -= alpha / dx ** 2
					K[right, idx] -= alpha / dx ** 2
					K[right, right] += alpha / dx ** 2

			if j < Ny - 1:  
				top = idx + 1  # Passage vers le haut
				if top < N:  
					K[idx, idx] += alpha / dy ** 2
					K[idx, top] -= alpha / dy ** 2
					K[top, idx] -= alpha / dy ** 2
					K[top, top] += alpha / dy ** 2
	#print(K.toarray())
	return K
def get_K2(Nx, Ny, dx, dy, alpha):
	""" Returns stiffness matrix K """
	N = Nx * Ny
	K = sc.sparse.lil_matrix((N, N))
	for i in range(Nx):
		for j in range(Ny):
			idx = i * Ny + j

			if i < Nx - 1:
				right = idx + Ny
				K[idx, idx] += alpha / dx ** 2
				K[idx, right] -= alpha / dx ** 2
				K[right, idx] -= alpha / dx ** 2
				K[right, right] += alpha / dx ** 2

			if j < Ny - 1:
				top = idx + 1
				K[idx, idx] += alpha / dy ** 2
				K[idx, top] -= alpha / dy ** 2
				K[top, idx] -= alpha / dy ** 2
				K[top, top] += alpha / dy ** 2
	return K



def flux(nx, ny, prc, flux_bc):
	""" Returns force vector frc """
	nqb = nx + ny  # nombre de qubits
	f = np.zeros(2 ** nqb)  # ddl number
	Nx = 2 ** nx
	Ny = 2 ** ny
	miny = round((prc) * Ny)  # pourcentage de la zone a droite où le flux est present
	
	#print("min=",miny)
	#maxy=round((1 - prc) * Ny)
	#print(maxy) 
	#print(Nx)
	for i in range(0,miny):
		#print(miny*Nx)
		#print(i)
		
		#print(Nx*miny + i)
		f[(i)*Nx+ Nx -1] = flux_bc #flux à droite
# 	for k in range(0,Ny):
# 		f[k*Nx]=flux_bc #Pour rajouter un flux à gauche (pour retirer, juste mettre /1000000)
	# no need for classical normalization
	return f



def flux2(nx, ny, prc, flux_bc):
	""" Returns force vector frc """
	nqb = nx + ny  # nombre de qubits
	f = np.zeros(2 ** nqb)  # ddl number
	Nx = 2 ** nx
	Ny = 2 ** ny
	miny = round((1 - prc) * Ny)  # pourcentage de la zone a droite où le flux est present
	for i in range(miny, Ny):
		f[(Nx - 1) * Ny + i] = flux_bc
	for k in range(0,Ny):
		f[k]=flux_bc #Pour rajouter un flux à gauche de 0 au max
	# no need for classical normalization
	#print(f.reshape((Nx, Ny)))
	return f

def get_K3(Nx, Ny, dx, dy, alpha):
	""" Returns stiffness matrix K with serpentine numbering """
	N = Nx * Ny
	K = sc.sparse.lil_matrix((N, N))
	
	# Création du mapping serpentin
	
	# Construction de la matrice de rigidité
	for i in range(Ny):
		for j in range(Nx):
			if i%2==0:
				idx=j+i*Ny
			else:
				idx=(i+1)*Ny -(j+1)
			#print(f"({i},{j}) -> {idx}", end="\t")
			#print(f"Index {idx}: ", end="")
			#print(f"Grid point ({i}, {j}) -> Index {idx}")
			
			bonus=0
			#print(1,idx)
			for k in range(2,Ny,2):
				if idx>=(k*Ny) and idx<((k+2)*Ny):
					
					bonus=2*k*Ny 
					#print("Bonus=",bonus)
					break
				else:
					bonus=0
				
			
			if i < Nx - 1: 
				
				if i%2==0:
					top = 2*Ny - 1 - idx + bonus
					#print(top-2*bonus)
					 # Passage à la ligne d'en bas
					#top=idx+ 2*Ny -1  -2*(idx%Ny)
					#print(f"Top -> {top}, ", end="")
					if top < N:  
						K[idx, idx] += alpha / dy ** 2
						K[idx, top] -= alpha / dy ** 2
						K[top, idx] -= alpha / dy ** 2
						K[top, top] += alpha / dy ** 2
					
				else: 
					top=Ny*4 -1 -idx + bonus
					#top=idx +2*Ny -1 -2*(Ny -1 - (idx%Ny))
					#print(f"Top -> {top}, ", end="")
					if top < N:  
						K[idx, idx] += alpha / dy ** 2
						K[idx, top] -= alpha / dy ** 2
						K[top, idx] -= alpha / dy ** 2
						K[top, top] += alpha / dy ** 2
				
				

			if j < Ny - 1:  
				if i%2==0:
					right = idx + 1  # Passage vers la droite
					#print(f"Right -> {right}, ", end="")
					if right < N:  
						K[idx, idx] += alpha / dx ** 2
						K[idx, right] -= alpha / dx ** 2
						K[right, idx] -= alpha / dx ** 2
						K[right, right] += alpha / dx ** 2
				else:
					right = idx - 1  # Passage vers la droite
					#print(f"Right -> {right}, ", end="")
					if right < N:  
						K[idx, idx] += alpha / dx ** 2
						K[idx, right] -= alpha / dx ** 2
						K[right, idx] -= alpha / dx ** 2
						K[right, right] += alpha / dx ** 2
	
				
				
	#print(K.toarray())
	return K




def flux3(nx, ny, prc, flux_bc):
	""" Retourne un vecteur de forces avec une colonne complète à gauche et une colonne incomplète à droite """
	nqb = nx + ny  # nombre de qubits
	f = np.zeros(2 ** nqb)  # ddl number
	Nx = 2 ** (nqb//2)
	Ny = 2 ** (nqb//2)
	minx = round((prc) * Nx)  # pourcentage de la zone à droite où le flux est présent
	#print(minx)

	# Flux dans la colonne gauche (index 0)
# 	for j in range(Ny):
# 		if j%2==0:
# 			f[j*Ny]=flux_bc
# 		else:
# 			f[(j+1)*Ny-1]=flux_bc
	# Flux dans la colonne droite (index Nx-1), jusqu'à la ligne correspondant à prc
	for j in range(minx):
		#print(j)
		if j%2==0:
			f[(j+1)*Ny-1]=flux_bc
			
		else:
			f[j*Ny]=flux_bc
	
	return f


def get_Hf3(nqb):
	""" returns the penalized matrix as an array and a sparse, and frc """

	# SPACE PARAMETERS
	nx = nqb // 2  # number of qubits on x axis
	ny = nqb - nx  # number of qubits on y axis
	Nx = 2 ** nx  # Number of points in x-direction
	Ny = 2 ** ny  # Number of points in y-direction
	L = 1.0  # Length of the square domain
	dx = L / (Nx - 1.)
	dy = L / (Ny - 1.)
	alpha = 1.0  # Thermal diffusivity #When is it used ???
	pen = 0.1 * 9 / 4  # to get the same result as for 4 qbits simulation of Arthur ##???

	Ksp = get_K3(Nx, Ny, dx, dy, alpha)
	Ksp += pen * np.eye(Nx * Ny)  # regularization scalable
	H = Ksp.toarray()  # KKK to supress, memory consumption

	flux_bc = (10 / 9) *Nx  # KKK flux de chaleur, the value is chosen to get the same resuls for 4 qbits
	prc = 0.5  # ratio of left edge where the flux is applied, it was 0.5 before.
	frc = flux3(nx, ny, prc, flux_bc)
	return H, Ksp, frc


def get_Hf(nqb):
	""" returns the penalized matrix as an array and a sparse, and frc """

	# SPACE PARAMETERS
	nx = nqb // 2  # number of qubits on x axis
	ny = nqb - nx  # number of qubits on y axis
	Nx = 2 ** nx  # Number of points in x-direction
	Ny = 2 ** ny  # Number of points in y-direction
	L = 1.0  # Length of the square domain
	dx = L / (Nx - 1.)
	dy = L / (Ny - 1.)
	alpha = 1.0  # Thermal diffusivity #When is it used ???
	pen = 0.1 * 9 / 4  # to get the same result as for 4 qbits simulation of Arthur ##???

	Ksp = get_K(Nx, Ny, dx, dy, alpha)
	Ksp += pen * np.eye(Nx * Ny)  # regularization scalable
	H = Ksp.toarray()  # KKK to supress, memory consumption

	flux_bc = (10 / 9 ) *Nx  # KKK flux de chaleur, the value is chosen to get the same resuls for 4 qbits
	prc = 0.5  # ratio of left edge where the flux is applied, it was 0.5 before.
	frc = flux(nx, ny, prc, flux_bc)
	return H, Ksp, frc


# stt operations

def accuracy(a, b) -> float:
	""" Returns the relative accuracy of a wrt b, in % """
	if 0 not in [a, b]:
		return ((b / a) if abs(a) > abs(b) else (a / b)) * 100
	else:
		return 0


def shorten(txt):
	""" Shortens the txt so that its no more than 17 characters long """
	if len(txt) < 17:
		return txt
	else:
		return txt[:7] + '[...]' + txt[-5:]


def majsleep(t):
	""" updates the sleep time """
	return min(10, max(t + 1, round(t * 1.2)))


def diagdev(aucl, x):
	""" diagonal deviation of two reshaped arrays """
	dg = np.fliplr(x).diagonal()
	udg = np.fliplr(aucl).diagonal()
	return round(MSE(dg, udg) * 1e8)


def diagdevs(aucl, x):
	""" diagonal deviation of two stt """
	nqb = round(np.log(len(x)) / np.log(2))
	nx = nqb // 2  # number of qubits on x axis
	ny = nqb - nx  # number of qubits on y axis
	Nx = 2 ** nx  # Number of points in x-direction
	Ny = 2 ** ny  # Number of points in y-direction
	xx = x.reshape(Ny, Nx)[::-1].T
	xaucl = aucl.reshape(Ny, Nx)[::-1].T
	dg = np.fliplr(xx).diagonal()
	udg = np.fliplr(xaucl).diagonal()
	return (MSE(dg, udg) * 1e8)



def VEC(u, v):
	""" Vectorial Product Error """
	return - u @ v


# def MSE(u, v):
# 	""" Mean Square Error """
# 	return abs(sum((u.flatten() - v.flatten()) ** 2) / len(u.flatten()))

# def MSE(u, v):
# 	""" Relative mean square error in percent """
# 	return 100 * np.mean((np.abs(u - v) / np.maximum(np.abs(u), 1e-12)) ** 2)
# def MSE(u, v):
# 	"""
# 	Projection-based loss (1 - cosine similarity)^2.
# 	Minimizes when u and v point in the same direction.
# 	Assumes u and v are normalized.
# 	"""
# 	u=u/np.linalg.norm(u)
# 	v=v/np.linalg.norm(v)
# 	cos_sim = np.dot(u, v)  # In [-1, 1]
# 	return (1 - cos_sim) ** 2
def norm(x): return x/(np.linalg.norm(x)+1e-15)

def MSE(u,v):
	return -(np.dot(norm(u),norm(v))**2)*100

def grd_fm(v, d='x'):
	""" gradient computation """
	nqb = round(np.log(len(v)) / np.log(2))
	nx = nqb // 2  # number of qubits on x axis
	ny = nqb - nx  # number of qubits on y axis
	Nx = 2 ** nx  # Number of points in x-direction
	Ny = 2 ** ny  # Number of points in y-direction
	vr = v.reshape(Ny, Nx)[::-1].T
	if d == 'x':
		return vr[1:, :] - vr[:-1, :]
	else:
		return -vr[:, 1:] + vr[:, :-1]


def GRD(u, v):
	""" Sobolev cost function
	u : classical
	v : quantum """
	nn = round(np.sqrt(len(u)))
	ugx = grd_fm(u, 'x').reshape(nn * (nn - 1))
	vgx = grd_fm(v, 'x').reshape(nn * (nn - 1))
	ugy = grd_fm(u, 'y').reshape(nn * (nn - 1))
	vgy = grd_fm(v, 'y').reshape(nn * (nn - 1))
	ct = - (u @ v) / np.linalg.norm(u)
	cgx = - 100 * (ugx @ vgx) / np.linalg.norm(ugx)
	cgy = - 100 * (ugy @ vgy) / np.linalg.norm(ugy)
	sqy = (vgy @ vgy) / np.linalg.norm(vgy)
	sqx = (vgy @ vgx) / np.linalg.norm(vgx)
	# print(f"ct{ct}\ncgx{cgx}\ncgy{cgy}")
	return ct + cgx + cgy  # + sqy + sqx


def qnorm(stt, frc, H):
	""" computes the actual q norm with the correct frc and H """
	return stt @ frc / (stt.T @ H @ stt)


def qE(stt, frc, H):
	""" computes the actual q energy with the correct frc and H """
	return -(stt @ frc) ** 2 / (stt.T @ H @ stt) / 2


def get_diag(ucl):
	Nx = round(np.sqrt(len(ucl)))
	ucl_res = ucl.reshape((Nx, Nx)).T[::-1]
	return np.fliplr(ucl_res).diagonal()


def roll_avg(rucl):
	""" computes the rolling average of reshaped ucl """
	ucl = np.array(rucl)

	nqb = round(np.log(len(ucl.flatten())) / np.log(2))

	Ny, Nx = rucl.shape

	ucl3 = np.empty(shape=(Nx, Ny), dtype=float)
	if nqb >= 8:
		mask = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]

		def wgt(i):
			if abs(mask[i][0]) == 2:
				if abs(mask[i][1]) == 2:
					return 1
				if abs(mask[i][1]) == 1:
					return 2
				if abs(mask[i][1]) == 0:
					return 2
			if abs(mask[i][0]) == 1:
				if abs(mask[i][1]) == 2:
					return 2
				if abs(mask[i][1]) == 1:
					return 6
				if abs(mask[i][1]) == 0:
					return 12
			if abs(mask[i][0]) == 0:
				if abs(mask[i][1]) == 2:
					return 4
				if abs(mask[i][1]) == 1:
					return 12
				if abs(mask[i][1]) == 0:
					return 25
			raise Exception("Broken Mask")
	else:

		mask = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]

		def wgt(i):
			return 2 ** len([False for x in mask[i] if x == 0])
	weights = [wgt(i) for i in range(len(mask))]

	lic = []
	wgg = []
	for x in range(Ny):
		for y in range(Nx):
			lic = []
			liq = []
			wgg = []
			for z in range(len(mask)):
				yy, xx = mask[z]
				if 0 <= y + yy < Ny and 0 <= x + xx < Nx:
					lic.append(ucl[y + yy, x + xx])
					wgg.append(weights[z])
			ucl3[y, x] = np.average(lic, weights=wgg)
	return ucl3


def qD(stt, ucl):
	""" computes the actual diag approx with the correct frc and H """
	return MSE(get_diag(stt), get_diag(ucl))

#Ici on peut changer les gates de l'ansatz



def gen_ans(nqb, lay):
	""" generates an unassigned ansatz from nqb, lay """
	#li.RealAmplitudes(num_qubits=nqb, entanglement='sca', reps=lay)
	#li.n_local(num_qubits=nqb, rotation_blocks=[li.RYGate(theta), li.CRZGate(theta)], entanglement_blocks=CXGate(), entanglement="linear", reps=lay)
	#li.PhaseGate(theta)
	#entanglement='sca'
	theta = Parameter("θ")
	return li.n_local(num_qubits=nqb, rotation_blocks=[li.RXGate(theta),li.RYGate(theta)], entanglement_blocks= li.CXGate(), entanglement ="linear", reps=lay)





def sampleGate(gate):
	""" Creates a circuit with 1 gate and sample created statevector """
	job = sampler.run([gate.measure_all(inplace=False)], shots=shots)
	probas = job.result().quasi_dists[0]
	# state is all positive
	return np.fromiter(((np.sqrt(probas[i]) if i in probas else 0) for i in range(2 ** gate.num_qubits)), dtype=float)


def get_stt(par, nqb, lay):
	""" Input : parameters. Output : statevector"""
	ans = gen_ans(nqb, lay)
	# print('para', ans.num_parameters , ans.num_qubits, len(par))
	return np.array(sampleGate(ans.assign_parameters(par, inplace=False)))


# optimization related functions

def nevals(par0):
	""" number of cost function evaluation allowed """
	return min(int(2e5), 150 * len(par0))


# file manipulation functions

def extract_info(file_name, force=False):
	if 'stt' in file_name and not force:
		raise Exception("This file contains a statevector but no parameters.")
	nqb = int(file_name.split('qb')[0])
	lay = int(re.split('qb_|l', file_name)[1])
	return nqb, lay


def import_stt(nqb, lay, noise=-1, use_dep=True):
	"""
	Imports statevectors from all possible files according to:
	nqb : number of qubits;
	lay : number of layers;
	noise: noise level
	(-1 : sv, 0, stochastic, 1 : noise model, 3 : aria-1, -2 : everything)
	use_dep : whether to use pre-decompressed files
	"""
	if use_dep:
		try:
			simli = pkl.unpickle(f'./dep/{nqb}q_{lay}lay_depressed.pkl')
		except FileNotFoundError:
			print("Dep. version not found, importing file by file.")
			depressor(nqb, lay)
			simli = pkl.unpickle(f'./dep/{nqb}q_{lay}lay_depressed.pkl')
		return [x for x in simli if x['n'] == noise] if noise >= -1 else simli
	nimport = 0
	simli = []
	for fpath in glob.iglob('./*.pkl'):
		fname = fpath.split('\\')[-1]
		try:
			n, l = extract_info(fname, force=True)
		except Exception as e:
			if 'job' not in fname:
				print("Exception :", e)
			n, l = -1, -1
		# print(fname, n, l)
		if n == nqb and l == lay:
			# print(fname)
			# ensure this file has to be taken into account
			# and retrieve t_noise, temporary noise
			if ' V ' in fname:
				t_noise = 0
				if noise >= -1 and noise != 0:
					continue
			elif ' N ' in fname:
				t_noise = 1
				if noise >= -1 and noise != 1:
					continue
			elif ' M ' in fname:
				t_noise = 2
				if noise >= -1 and noise != 2:
					continue
			elif ' A ' in fname:
				t_noise = 3
				if noise >= -1 and noise != 3:
					continue
			elif ' F ' in fname:
				t_noise = 4
				if noise >= -1 and noise != 4:
					continue
			else:  # not (' V ' in fname or ' N ' in fname or ' M ' in fname or ' A ' in fname or ' F ' in fname):
				t_noise = -1
				if noise >= 0:
					continue
			nimport += 1
			print(f"{fname} is being imported... ({nimport})")

			# get statevector
			if 'stt' in fname:
				stt = pkl.unpickle(fname)
			else:
				par = pkl.unpickle(fname)
				stt = get_stt(par, nqb, lay)
			assert len(stt) == round(2 ** nqb)

			# get weight of that simulation
			if t_noise == -1:  # all runs have the same weight for sv simulations
				weight = 1  # merge aren't really useful for such simulations
			else:
				ns = fname.split(' ')[-3]
				if 'k' in ns:
					weight = int(ns[:-1]) * 1000
				else:
					try:
						weight = int(ns)
					except ValueError:
						weight = 10000  # default shot number
			# print(fname,'with weight',weight)

			dic = {'stt': np.array(stt), 'fn': fname, 'w': weight, 'n': t_noise}
			simli.append(dic)
	return simli


def merge(nqb, lay, noise=-1, rtw=False):
	"""
	Performs an averaged merge over all stt of corresponding
	nqb : number of qubits;
	lay : number of layers;
	noise: noise level
	(-1 : sv, 0, stochastic, 1 : noise model, 3 : aria-1)
	rtw : return total weights
	"""
	simli = import_stt(nqb, lay, noise)

	# weighted sum of statevectors
	wstt = np.zeros(int(2 ** nqb), dtype=float)
	for dic in simli:
		wstt += np.array(dic['stt']) * dic['w']
	total = sum(dic['w'] for dic in simli) if simli else 1
	wstt /= total
	if rtw:
		return wstt, total
	else:
		return wstt


def depressor(nqb, lay):
	""" puts all data corresponding to nqb, lay into the same file
	so that decompression is fast ! """
	pkl.pickle(import_stt(nqb, lay, -2, use_dep=False), f"./dep/{nqb}q_{lay}lay_depressed.pkl")


def viz_par(nqb=10, lay=2, fn="temp_par.pkl"):
	""" vizualizes parameters """
	if type(fn) is str:
		par = pkl.unpickle(fn)
	else:
		par = fn
	z = plt.imshow((np.array(par) / np.pi).reshape(lay + 1, nqb).T, cmap='turbo')
	plt.colorbar(z)
	plt.show()


def vyz_par(nqb, lay, pli):
	""" vizualizes parameters with plotly """
	fig = go.Figure().set_subplots(len(pli), 1, horizontal_spacing=0.1)
	cmin, cmax = min_max(pli)
	for (i, par) in enumerate(pli):
		fig.add_trace(
			go.Heatmap(
				z=(np.array(par) / np.pi).reshape(lay + 1, nqb).T,
				colorscale='Turbo',
				colorbar=dict(title='Parameters (in π units)'),
				zmin=cmin / np.pi,
				zmax=cmax / np.pi),
			row=i + 1,
			col=1)
	fig.show()


def export_filename(title, extension):
	""" returns a fresh filename for exporting to a specific format """
	i = 0
	while Path(title + "_" + str(i) + extension).is_file():
		i += 1
	return title + "_" + str(i) + extension


def export_to_paraview(tli, title="barrage3d"):
	""" exports a list (n, (x,y,z), t) to a csv file """
	txt = 'x coord, y coord, z coord, scalar\n'
	try:
		for (_, (x, y, z), t) in tli:
			txt += f'{x}, {y}, {z}, {t}\n'
	except ValueError:
		for ((x, y, z), t) in tli:
			txt += f'{x}, {y}, {z}, {t}\n'
	with open(export_filename(title, ".csv"), "w") as file:
		file.write(txt)


def export_to_aster(tli, title="barrage3d/data/exported/barrage3d"):
	""" Exports the data to a code-aster readable file """

	creachamp_beg = """mesh = LIRE_MAILLAGE(UNITE=20)

depl1 = CREA_CHAMP(AFFE=(
"""
	creachamp_end = """),

\t\t\tMAILLAGE=mesh,
\t\t\tOPERATION='AFFE',
\t\t\tTYPE_CHAM='NOEU_DEPL_R')


IMPR_RESU(FORMAT='MED',
\t\tRESU=_F(CHAM_GD=depl1),
\t\tUNITE=80)
"""

	creachamp_mid = ""

	for nodenum in range(1, len(tli) + 1):
		creachamp_mid += __strnodeaster(tli[nodenum - 1][0] + 1, tli[nodenum - 1][-1])

	with open(export_filename(title, ".comm"), "w") as f:
		f.write(creachamp_beg + creachamp_mid + creachamp_end)


def __strnodeaster(nodenum, temp):
	""" Returns the line corresponding to the nodenum-th node, moving by dx and dy, in code-aster """
	return f"\t_F (NOEUD=('N{nodenum}',) , NOM_CMP=('TEMP',) , VALE=({temp})),\n"


def project_0(uqn, ucl, width=4):
	""" projects the first (1/width)% of the data 'uqn' onto 'ucl'
	ok for now, but needs to be changed to something taking the global
	percentage into account  (how much is projected) whihc requires
	a T_ref measurement
	"""
	for x in range(len(uqn)):
		if x < len(uqn) // width:
			uqn[x] = ucl[x]
	return uqn


def pklout(data, txt):
	if not txt.endswith('.pkl'):
		txt += '.pkl'
	if not txt.startswith(('pkl/', './pkl/',)):
		txt = './pkl/' + txt
	pkl.pickle(data, 'txt')


def pklin(txt):
	if not txt.endswith('.pkl'):
		txt += '.pkl'
	if not txt.startswith(('pkl/', './pkl/',)):
		txt = './pkl/' + txt
	return pkl.unpickle(txt)


def fname3d(cost_fun, nqb, lay, method, fld=''):
	""" Returns a valid figure name for a Barrage3d figure """
	if cost_fun == 'MSE':
		return f'./fig/barrage3d/{fld}{nqb}qb_{lay}lay_{method}'
	return f'./fig/barrage3d/{fld}{nqb}qb_{lay}lay_{method}_{cost_fun[:3]}'


def get_mingroup(file_name, dfv=0., dfp=None):
	""" returns the last (float * (float list)) tuple from a file.
	If no such file exists, returns (dfv, dfp) """
	try:
		with open(file_name, "r") as fil:
			for line in fil:
				pass
			last_line = line
			try:
				min_val, min_grp = line.split(' -> ')
			except ValueError as e:
				print(f'Check how the file {file_name} is formatted !')
				raise e
		lis = [float(par) for par in min_grp.strip()[1:-1].split(', ')]
		return float(min_val), lis
	except FileNotFoundError:
		return dfv, ([] if dfp is None else dfp)
