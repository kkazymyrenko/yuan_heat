# coldstart_6qubit.py
# Run 6-qubit cold start, test 3 optimizers, save best run for cascade


import time
import numpy as np
import scipy as sc
import qiskit as qi
from qiskit.circuit.library import NLocal, RXGate, RYGate, CXGate
from qiskit.quantum_info import Statevector
from scipy.sparse import load_npz
import pandas as pd

# Config
nqb = 6
lay = 10
init_mode = "medium"   # choose: "small [-0.1, 0.1]", "medium [-pi/2, pi/2]", "wide [-pi, pi]"
rng_seed = None
maxiter = 500
time_limit_sec = 2 * 60 * 60   # 2 hours
n_runs = 10        # 10 runs
results_csv = "coldstart6_runs.csv"
summary_csv = "coldstart6_summary.csv"


# FEM setup
Hsp = load_npz("Hsp6.npz")
H = np.load("H6.npy")
frc = np.load("frc6.npy")
ucl = np.load("ucl6.npy")
ucln = np.load("ucln6.npy")


# Error metrics

def norm(x):
    return x / (np.linalg.norm(x) + 1e-15)

def accuracy(a, b) -> float:
    if 0 not in [a, b]:
        return ((b / a) if abs(a) > abs(b) else (a / b)) * 100
    else:
        return 0

def MSE(u, v):
    return np.mean((u - v)**2)


# Ansatz: RX + RY with full entanglement
def ansatz_rxry_full(nqb, lay):
    theta_x = qi.circuit.Parameter("θx")
    theta_y = qi.circuit.Parameter("θy")
    return NLocal(
        num_qubits=nqb,
        rotation_blocks=[RXGate(theta_x), RYGate(theta_y)],
        entanglement_blocks=CXGate(),
        entanglement="full",
        reps=lay
    )

una_ans = ansatz_rxry_full(nqb, lay)
num_par = una_ans.num_parameters

# State preparation
def get_state(par):
    ans = una_ans.assign_parameters(par, inplace=False)
    qc = qi.QuantumCircuit(nqb)
    qc.append(ans, list(range(nqb)))
    return Statevector(qc).data.real

# Cost Functions (Physics + Penalty)
def rayleigh_cost(par):
    psi = get_state(par)
    num = float(frc @ psi)
    den = float(psi @ (H @ psi))
    return -0.5 * (num**2) / max(den, 1e-15)

def mse_penalty(par):
    psi = norm(get_state(par))
    return MSE(psi, ucln)

def combined_cost(par, lam):
    return rayleigh_cost(par) + lam * mse_penalty(par)

def shift_rule_grad(par, lam):
    grad = np.zeros_like(par, dtype=float)
    eps = np.pi / 2
    for i in range(len(par)):
        p_plus = par.copy();  p_plus[i] += eps
        p_minus = par.copy(); p_minus[i] -= eps
        grad[i] = (combined_cost(p_plus, lam) - combined_cost(p_minus, lam)) / (2 * eps)
    return grad


# Initialization helper
def sample_init(num_par, mode="medium", rng=None):
    if rng is None:
        rng = np.random
    if mode == "small":
        return 0.1 * (rng.rand(num_par) - 0.5) * 2
    elif mode == "medium":
        return (np.pi/2) * (rng.rand(num_par) - 0.5) * 2
    elif mode == "wide":
        return np.pi * (rng.rand(num_par) - 0.5) * 2
    else:
        raise ValueError("init_mode must be 'small','medium','wide'")

if rng_seed is not None:
    np.random.seed(rng_seed)


# Run experiments
methods = {
    "L-BFGS-B": {"jac": True,  "options": {"maxiter": maxiter, "disp": False}},
    "Powell":   {"jac": False, "options": {"maxiter": maxiter, "disp": False}},
    "COBYLA":   {"jac": False, "options": {"maxiter": maxiter, "disp": False}},
}

runs_all = []
best_acc = -1
best_par = None
best_stt = None

for method, opts in methods.items():
    print(f"\n=== Optimizer: {method} ===")
    for run in range(n_runs):
        par0 = sample_init(num_par, mode=init_mode)

        # Adaptive lambda
        g_phys = np.linalg.norm(shift_rule_grad(par0, 0.0))
        g_pen  = np.linalg.norm(shift_rule_grad(par0, 1.0))
        lam    = g_phys / (g_pen + 1e-8)
        lam    = max(lam, 1e-2)
        print(f"Run {run+1}: Adaptive lambda = {lam:.3e}")

        t0 = time.perf_counter()
        try:
            if opts["jac"]:
                res = sc.optimize.minimize(
                    combined_cost, par0, args=(lam,),
                    method=method,
                    jac=lambda p, l=lam: shift_rule_grad(p, l),
                    options=opts["options"], tol=1e-6
                )
            else:
                res = sc.optimize.minimize(
                    combined_cost, par0, args=(lam,),
                    method=method,
                    options=opts["options"], tol=1e-6
                )
        except Exception as e:
            print(f"Run {run+1} failed: {e}")
            continue

        elapsed = time.perf_counter() - t0
        if elapsed > time_limit_sec:
            print(f"Run {run+1}: exceeded time limit")
            continue

        par_opt = res.x
        stt = norm(get_state(par_opt))

        q_cost = rayleigh_cost(par_opt)
        mse_val = mse_penalty(par_opt)
        ref_cost = -0.5 * (frc @ ucln)**2 / (ucln @ (H @ ucln))
        acc = accuracy(q_cost, ref_cost)
        nit_val = getattr(res, "nit", None)
        iterations = int(nit_val) if isinstance(nit_val, (int, np.integer)) else np.nan

        run_data = {
            "optimizer": method,
            "run": run+1,
            "lambda": float(lam),
            "success": bool(res.success),
            "final_cost": float(q_cost),
            "accuracy_pct": float(acc),
            "mse_metric": float(mse_val),
            "time_sec": float(elapsed),
            "iterations": iterations,
        }
        runs_all.append(run_data)

        print(f"Run {run+1}: acc={acc:.2f}%, MSE={mse_val:.3e}, "
              f"cost={q_cost:.6f}, iters={run_data['iterations']}")

        # Track best across all methods
        if acc > best_acc:
            best_acc = acc
            best_par = par_opt
            best_stt = stt


# Save best results for cascade
if best_par is not None and best_stt is not None:
    np.save("par_opt6.npy", best_par)
    np.save("uqn6.npy", best_stt)
    print(f"\nBest accuracy across all optimizers/runs: {best_acc:.2f}%")
    print("Saved par_opt6.npy and uqn6.npy for cascade.")
else:
    print("No successful runs, nothing saved.")


# Summary

df = pd.DataFrame(runs_all)
df.to_csv(results_csv, index=False)

summary = df.groupby("optimizer").agg(
    mean_accuracy_pct=("accuracy_pct", "mean"),
    max_accuracy_pct=("accuracy_pct", "max"),
    min_accuracy_pct=("accuracy_pct", "min"),
    mean_iterations=("iterations", "mean"),
    mean_mse_metric=("mse_metric", "mean"),
).reset_index()

print("\n=== Summary (3 optimizers with adaptive lambda) ===")
print(summary.to_string(index=False))
summary.to_csv(summary_csv, index=False)
