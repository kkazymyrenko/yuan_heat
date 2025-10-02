# remeshing_6_to_9.py
# Refine 6 to 9 qubits using RX–RY ansatz, warmstart from coldstart6, Newton-CG optimization

import os
import time
import numpy as np
import scipy as sc
from scipy.sparse import load_npz
import qiskit as qi
from qiskit.circuit.library import NLocal, RXGate, RYGate, CXGate
from qiskit.quantum_info import Statevector
import pandas as pd

# Error metrics
def norm(x):
    n = np.linalg.norm(x)
    return x if n == 0 else (x / n)

def accuracy(a, b) -> float:
    if 0 not in [a, b]:
        return ((b / a) if abs(a) > abs(b) else (a / b)) * 100
    else:
        return 0

def MSE(u, v):
    return float(np.mean((u - v) ** 2))


# Config
nqb1 = 6                 # start with 6 qubits
nqb2 = nqb1 + 3          # refine to 9 qubits
layers_A = 10            # depth for 6-qubit ansatz
layers_B = 10            # depth for 9-qubit ansatz
tol = 1e-6
maxiter = 200
results_csv = "remeshing_6_to_9_runs.csv"


# FEM systems
def load_fem_set(tag):
    Hsp = load_npz(f"Hsp{tag}.npz")
    H   = np.load(f"H{tag}.npy")
    frc = np.load(f"frc{tag}.npy")
    ucl = np.load(f"ucl{tag}.npy")
    ucln = np.load(f"ucln{tag}.npy")
    return H, Hsp, frc, ucl, ucln

H6, Hsp6, frc6, ucl6, ucl6n = load_fem_set("6")
H9, Hsp9, frc9, ucl9, ucl9n = load_fem_set("9")

ref_cost9 = -0.5 * float((frc9 @ ucl9n) ** 2) / float(ucl9n @ (H9 @ ucl9n) + 1e-15)


# Ansatz: RX + RY with full entanglement
def ansatz_rxry_full(nqb, reps):
    theta_x = qi.circuit.Parameter("θx")
    theta_y = qi.circuit.Parameter("θy")
    return NLocal(
        num_qubits=nqb,
        rotation_blocks=[RXGate(theta_x), RYGate(theta_y)],
        entanglement_blocks=CXGate(),
        entanglement="full",
        reps=reps
    )

ansA = ansatz_rxry_full(nqb1, layers_A)
ansB = ansatz_rxry_full(nqb2, layers_B)
num_par_B = ansB.num_parameters


# Warm start (from coldstart6)
have_par6 = os.path.exists("par_opt6.npy")
have_stt6 = os.path.exists("uqn6.npy")
par6 = np.load("par_opt6.npy") if have_par6 else None
stt6 = np.load("uqn6.npy") if have_stt6 else None

if not (have_par6 or have_stt6):
    raise FileNotFoundError("Warm start not found: need par_opt6.npy and/or uqn6.npy")


# Build init circuit (9q)
def build_init_circuit_9():
    qc = qi.QuantumCircuit(nqb2)
    if have_par6:
        qc.append(ansA.assign_parameters(par6, inplace=False), list(range(nqb1)))
    else:
        qc.initialize(stt6.astype(complex), list(range(nqb1)))
    
    for i in range(nqb1, nqb2):
        qc.h(i)
    return qc

U_init = build_init_circuit_9()


# State preparation
def get_state_B(parB):
    qc = qi.QuantumCircuit(nqb2)
    qc.compose(U_init, inplace=True)
    qc.append(ansB.assign_parameters(parB, inplace=False), list(range(nqb2)))
    return Statevector(qc).data.real


# Cost, Gradient, Hessian
def cost_func(parB):
    psi = get_state_B(parB)
    num = float(frc9 @ psi)
    den = float(psi @ (H9 @ psi)) + 1e-15
    return -0.5 * (num**2) / den

def shift_rule_grad(parB):
    grad = np.zeros_like(parB)
    eps = np.pi / 2
    for i in range(len(parB)):
        p_plus = parB.copy();  p_plus[i] += eps
        p_minus = parB.copy(); p_minus[i] -= eps
        grad[i] = (cost_func(p_plus) - cost_func(p_minus)) / (2 * eps)
    return grad

def shift_rule_hess_PD(parB):
    n = len(parB)
    hess = np.zeros((n, n))
    eps = np.pi / 2
    f0 = cost_func(parB)
    for i in range(n):
        p_plus = parB.copy();  p_plus[i] += eps
        p_minus = parB.copy(); p_minus[i] -= eps
        fp = cost_func(p_plus)
        fm = cost_func(p_minus)
        hess[i, i] = (fp - 2*f0 + fm) / (eps**2)

    def is_pd(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    reg = 1e-8
    max_reg = 1e-1
    while not is_pd(hess) and reg <= max_reg:
        hess += reg * np.eye(n)
        reg *= 10.0
    return hess


# Runs
runs = []
best = {"acc": -np.inf, "par": None, "stt": None, "res": None}

for r in range(10):
    print(f"\n=== Remeshing 6→9 | Run {r+1}/10 ===")
    par0 = np.random.uniform(-0.01, 0.01, num_par_B)

    iters = {"count": 0}
    def cb(_x): iters["count"] += 1

    t0 = time.perf_counter()
    res = sc.optimize.minimize(
        cost_func, par0,
        method="Newton-CG",
        jac=shift_rule_grad,
        hess=shift_rule_hess_PD,
        tol=tol,
        options={"maxiter": maxiter, "disp": True},
        callback=cb
    )
    elapsed = time.perf_counter() - t0

    par_opt = res.x
    psi_opt = norm(get_state_B(par_opt))

    # Metrics
    q_cost = -0.5 * float((frc9 @ psi_opt) ** 2) / float(psi_opt @ (H9 @ psi_opt) + 1e-15)
    acc = accuracy(q_cost, ref_cost9)
    mse_val = MSE(psi_opt, ucl9n)

    runs.append({
        "run": r+1,
        "success": bool(res.success),
        "final_cost": float(q_cost),
        "accuracy_pct": float(acc),
        "mse_metric": float(mse_val),
        "time_sec": float(elapsed),
        "iterations": int(iters["count"]),
        "nfev": int(getattr(res, "nfev", np.nan)),
        "nit": int(getattr(res, "nit", iters["count"])),
    })

    if acc > best["acc"]:
        best.update({"acc": acc, "par": par_opt, "stt": psi_opt, "res": res})


# Save best results
np.save("par_opt9.npy", best["par"])
np.save("uqn9.npy", best["stt"])
print("\nSaved best: par_opt9.npy, uqn9.npy")


# Summary
df = pd.DataFrame(runs)
df.to_csv(results_csv, index=False)
print(f"Saved runs table: {results_csv}")

summary = {
    "mean_accuracy_pct": float(df["accuracy_pct"].mean()),
    "mean_mse_metric": float(df["mse_metric"].mean()),
    "mean_time_sec": float(df["time_sec"].mean()),
    "mean_iterations": float(df["iterations"].mean()),
    "max_accuracy_pct": float(df["accuracy_pct"].max()),
    "min_accuracy_pct": float(df["accuracy_pct"].min()),
}
summary_df = pd.DataFrame([summary])
print("\n=== Summary (10 runs) ===")
print(summary_df.to_string(index=False))
