import numpy as np
import math as m
import time
import pickle
import scipy as sp
import noisyopt as no
import qiskit as qi # 1.1.1
from qiskit.circuit import library as li, ParameterVector
from qiskit.quantum_info import SparsePauliOp, Operator
from qiskit.primitives import Estimator, Sampler
import matplotlib.pyplot as plt

# this file contains right scaling for FE problem
# if we change only Nx and Ny we converge to the continuous solution with the same regularisation
# the variable of input flux_bc and penalization (heat loss) 



est = Estimator()
sampler = Sampler()


## STIFFNESS MATRIX 
def get_K(Nx, Ny, dx, dy, alpha):
    """ Returns stiffness matrix K """
    N = Nx * Ny
    K = sp.sparse.lil_matrix((N, N))
    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j
            
            if i < Nx-1:
                right = idx + Ny
                K[idx, idx] += alpha / dx**2
                K[idx, right] -= alpha / dx**2
                K[right, idx] -= alpha / dx**2
                K[right, right] += alpha / dx**2

            if j < Ny-1:
                top = idx + 1
                K[idx, idx] += alpha / dy**2
                K[idx, top] -= alpha / dy**2
                K[top, idx] -= alpha / dy**2
                K[top, top] += alpha / dy**2
    return K

def flux(nx, ny, prc, flux_bc):
    """ Returns force vector f """
    nqb = nx + ny  # nombre de qubits
    f = np.zeros(2 ** nqb)  # ddl number
    Nx = 2 ** nx
    Ny = 2 ** ny
    miny = round((1-prc) * Ny) # pourcentage de la zone a droite où le flux est present
    for i in range(miny, Ny):
        f[(Nx - 1) * Ny + i] = flux_bc
    f /= np.linalg.norm(f) # KKK no need for classical calculations
    return f

## COST FUNCTION
def cost_cl(x, H, f):
    cost = -1 / 2 * (f @ x) ** 2 / (x.T @ H @ x)
    return cost

def cost_func(par, SPOH, f, una_ans):
    """ f = -1/2 <f|psi>² / <psi|H|psi> """
    cost = -1/2 * fpsi(f, una_ans, par)**2 / psiHpsi(SPOH, una_ans, par)
    return cost

def psiHpsi(SPOH, una_ans, par):
    """ Returns the value of <psi(par)| H |psi(par)> """
    obs = SPOH  # SparsePauliOp(H)
    cir = una_ans
    job = est.run([cir], [obs], [par], shots=shots)
    result = job.result().values[0] 
    return result

def fpsi(f, una_ans, par):
    """ Returns the value of <f|psi(par)> """
    ans = una_ans.assign_parameters(parameters=par, inplace=False)
    psi = sampleGate(ans)
    fpsi = f @ psi
    return fpsi

def sampleGate(gate):
    """ Creates a circuit with 1 gate and sample created statevector """
    cir = qi.QuantumCircuit(nqb)
    cir.append(gate, list(range(nqb)))
    cir.measure_all()
    job = sampler.run([cir], shots=shots)
    probas = job.result().quasi_dists[0]
    state = np.zeros(2 ** nqb)  # probas : dict -> state : array
    for i in range(2 ** nqb):
        state[i] = np.sqrt(probas[i]) if i in probas else 0
    return state

def norm_q(par, H, f, una_ans):
    """ Returns solution norm via Sato quantum method """
    SPOH = SparsePauliOp.from_operator(Operator(H))
    ropt = fpsi(f, una_ans, par) / psiHpsi(SPOH, una_ans, par)
    return ropt

def build_callback(SPOH, f, callback_dict, una_ans):
    """Returns: Callable: Callback function object"""

    def callback(current_vector):
        """Callback function storing previous solution vector, computing the intermediate cost value, and displaying number
        of completed iterations and average time per iteration. Store values in mutable dictionnary.
        Parameters: current_vector (ndarray): Current vector of parameters returned by optimizer"""
        callback_dict["iters"] += 1
        callback_dict["prev_vector"] = current_vector

        current_cost = cost_func(current_vector.real, SPOH, f, una_ans)

        callback_dict["cost_history"].append(current_cost)
        print(f"Iteration : {callback_dict['iters']}, Current cost: {current_cost.real}")#,end="\r",flush=True)
        # print("x :", current_vector)
    return callback


## QUANTUM PARAMETERS
nx = 2                            # number of qubits on x axis
ny = 2                            # number of qubits on y axis
nqb = nx + ny                     # number of qubits for circuits
lay = m.floor(2 ** nqb / nqb - 1) # ansatz layers number
lay = 2                           # nqb,lay : 4,3 ; 5,4 ; 6,6
shots = None                       # compass doesn't work if < 1e6. number of shots for circuits sampling
# nqb : 4 , 5 , 6 ,  6,   7 ,  8 ,  9 ,  10
# ddl : 16, 32, 64, 64,  128, 256, 512, 1024
# lay :  3,  4,  5,   6,   6,   7,   8,   9
# par : 16, 20, 36,  48,   49,  64,  81, 100,
# par = (lay + 1) * nqb



## SPACE PARAMETERS
L = 1.0  # Length of the square domain
Nx = 2 ** nx  # Number of points in x-direction
Ny = 2 ** ny  # Number of points in y-direction
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
dx = L / (Nx-1.)
dy = L / (Ny-1.)


## STIFFNESS MATRIX
alpha = 1.0  # Thermal diffusivity
Ksp = get_K(Nx, Ny, dx, dy, alpha)
pen = 0.1*9/4 # to get the same result as for 4 qbits simulation of Arthur
Ksp += pen * np.eye(Nx*Ny) # regularization scalable
K = Ksp.toarray() # KKK to supress, memory consumption
SPOH = SparsePauliOp.from_operator(Operator(K)) #KKK to supress, no need in current implementation


## BOUNDARY CONDITIONS
therm_bc = 0  # thermostat 
flux_bc = 3/np.sqrt(2)/4/dx  #KKK flux de chaleur, the value is chosen to get the same resuls for 4 qbits
prc = 0.5 # ratio of left edge where the flux is applied
f = flux(nx, ny, prc, flux_bc)


## ANSATZ
una_ans = li.RealAmplitudes(num_qubits=nqb, entanglement='sca', reps=lay)
# una_ans = ansatzpers(nqb, nb_par=16)   # marche moins bien


## CALLBACK
callback_dict = {"prev_vector": None, "iters": 0, "cost_history": []}
callback = build_callback(SPOH, f, callback_dict, una_ans)


## BOUNDARY CONDITIONS
num_par = una_ans.num_parameters 
BC = [[-2*np.pi, 2*np.pi] for _ in range(num_par)] 
# Needs to be 2pi because Ry gates are parametrized by theta/2, 
# therefore they make complete rotation with angle 4pi.


## INITIALIZATION
par_warm = np.array(2 * np.pi * np.random.random(num_par))






## OPTIMIZATION
optim = True
if optim : 
    print("Layers :", lay)
    print("Number of parameters: ", num_par)
    print("Number of DOFs: ", 2**nqb)

    print("   *** Beginning optimization ***   ")
    start = time.perf_counter()
    if shots == None :
         res = sp.optimize.minimize(cost_func, par_warm, args=(SPOH, f, una_ans), tol=1e-7, method='L-BFGS-B', callback=callback, bounds=BC)
    else : 
        res = no.minimizeCompass(cost_func, par_warm, args=(SPOH, f, una_ans), bounds=BC, scaling=None, redfactor=2.0, deltainit=1.0, deltatol=1e-3, 
            feps=1e-3, errorcontrol=True, funcNinit=40, funcmultfactor=1.0, paired=False, alpha=0.1, disp=False, callback=callback)
    end = time.perf_counter()
    exe_time = end - start

    par_ini = np.array(2 * np.pi * np.random.random(2 ** nqb))
    resCL = sp.optimize.minimize(cost_cl, par_ini, args=(SPOH.to_matrix(), f), tol=1e-5, method='L-BFGS-B')

    print("Message :", res.message)
    print("Success :", res.success)
    print("nfev :", res.nfev)
    print("x :", res.x)
    print(f"Optimization time: {exe_time//60:.0f}m {exe_time%60:.2f}s")
    print("Quantum minimum :", res.fun)
    print("Classical minimum :", resCL.fun.real)

    res = res.x

else : pass


def get_Xq(res_x, una_ans):
    """ Input : ansatz & parameters. Output : statevector"""
    ans_sol = una_ans.assign_parameters(parameters=res_x, inplace=False)
    cir_sol = qi.QuantumCircuit(nqb)
    cir_sol.append(ans_sol, list(range(nqb)))
    cir_sol.measure_all()
    job_sol = sampler.run([cir_sol])
    res_sol = job_sol.result()
    res_sol= res_sol.quasi_dists   # probabilities
    Xquant = np.zeros(2 ** nqb)   # res_sol: dict -> Xquant: list
    for i in range(2 ** nqb):
        Xquant[i] = np.sqrt(res_sol[0][i]) if i in res_sol[0] else 0
    return Xquant

## CALCUL CHAMPS
xquant = get_Xq(res, una_ans)
ucl = sp.sparse.linalg.spsolve(Ksp.tocsr(), f).real 
## PRINT SOME ADDITIONAL DATA
print("Classical minimum with linalg:", cost_cl(ucl, K, f).real)
par_ini =  np.random.random(2 ** nqb)/100
resCL = sp.optimize.minimize(cost_cl, par_ini, args=(K, f), tol=1e-7, method='L-BFGS-B')
print("Classical minimum with BFGS:", resCL.fun.real)


## NORMALISATION
normq = np.linalg.norm(xquant)
normc = np.linalg.norm(ucl)
print("Norme classique Xquant :", normq)
print("Norme classique Ucl :", normc)
# Xquant = np.multiply(xquant, 1.05)
# res_cl_inv = np.multiply(ucl, 1/normc)

norm_ropt = norm_q(res, K, f, una_ans)
print("Norme quantique Xquant :", norm_ropt, "\n")
xquant = np.multiply(xquant, norm_ropt)


## ERREUR 
# erreur = np.linalg.norm(abs(ucl.real - xquant)) / np.linalg.norm(ucl.real)
# print(f"Norm error : {erreur:.6f}") 

def MSE(u, v):
    return sum((u - v) ** 2) / len(u)
mse = MSE(xquant, ucl)
print(f"MSE error : {mse:.6f}") 

u = xquant.reshape((Nx, Ny)).T
ucl = ucl.reshape((Nx, Ny)).T[::-1]

#print(u[::-1, ::])

vmin = min(u.min(), ucl.min())
vmax = max(u.max(), ucl.max())  



## AFFICHAGE 
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
iso = False   # isovalues lines

# Field optimization
if iso : 
    cax_heat = ax[0][0].contourf(x, y, u[::, ::], levels=20, cmap='plasma') 
else : 
    cax_heat = ax[0][0].imshow(u[::-1, ::], cmap='plasma', interpolation='nearest') #, vmin=vmin, vmax=vmax) 
ax[0][0].set_title(label='Champ de température (Optimisation quantique)')
ax[0][0].set_ylabel("Axe y")
# ax[0][0].set_xlabel("Axe x")
fig.colorbar(cax_heat, ax=ax[0][0], label='Température')

# Heat flux optimization
# bad normalization here
grad_x, grad_y = np.gradient(np.multiply(u[::, ::], -1))

ax[0][1].quiver(grad_y, grad_x, scale=1)
ax[0][1].set_title('Gradient de température (Optimisation quantique)')
# ax[0][1].set_xlabel("Axe x")
ax[0][1].set_ylabel("Axe y")

# Field inversion
if iso : 
    cax_heat = ax[1][0].contourf(x, y, ucl[::-1, ::], levels=20, cmap='plasma')
else :
    cax_heat = ax[1][0].imshow(ucl[::, ::], cmap='plasma', interpolation='nearest') #, vmin=vmin, vmax=vmax)
ax[1][0].set_title(label='Champ de température (Inversion classique)')
ax[1][0].set_xlabel("Axe x")
ax[1][0].set_ylabel("Axe y")
fig.colorbar(cax_heat, ax=ax[1][0], label='Température')

# Heat flux inversion
grad_x, grad_y = np.gradient(np.multiply(ucl[::-1], -1))
ax[1][1].quiver(grad_y, grad_x, scale=1) 
ax[1][1].set_title('Gradient de température (Inversion classique)')
ax[1][1].set_xlabel("Axe x")
ax[1][1].set_ylabel("Axe y")

plt.show()
