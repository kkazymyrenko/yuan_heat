import nlopt
from numpy import pi, random, nan
from sys import stderr
from tools.fct import nevals
from tools.fct import *
from random import random as rdd
from time import perf_counter as pctime
normalize = True  # whether to
def norm(x): return x/(np.linalg.norm(x) if normalize else 1)
method = 'LN_NEWUOA_BOUND'  # nlopt method
#method = "L-BFGS-B"
algos = {
	'LN_SBPLX': nlopt.LN_SBPLX,
	'LN_COBYLA': nlopt.LN_COBYLA,
	'LN_NEWUOA': nlopt.LN_NEWUOA,
	'LN_NEWUOA_BOUND': nlopt.LN_NEWUOA_BOUND,
	'LN_BOBYQA': nlopt.LN_BOBYQA,
	'LN_NELDERMEAD': nlopt.LN_NELDERMEAD,
	'LN_PRAXIS': nlopt.LN_PRAXIS,
	'LN_AUGLAG': nlopt.LN_AUGLAG,
	'LN_AUGLAG_EQ': nlopt.LN_AUGLAG_EQ,
	'LD_SLSQP': nlopt.LD_SLSQP,
	'LD_MMA': nlopt.LD_MMA,
	'LD_CCSAQ': nlopt.LD_CCSAQ,
	'LD_LBFGS': nlopt.LD_LBFGS,
	'LD_TNEWTON': nlopt.LD_TNEWTON,
	'LD_TNEWTON_PRECOND': nlopt.LD_TNEWTON_PRECOND,
	'LD_TNEWTON_RESTART': nlopt.LD_TNEWTON_RESTART,
	'LD_TNEWTON_PRECOND_RESTART': nlopt.LD_TNEWTON_PRECOND_RESTART,
	'LD_VAR1': nlopt.LD_VAR1,
	'LD_VAR2': nlopt.LD_VAR2,
	'GN_DIRECT': nlopt.GN_DIRECT,
	'GN_DIRECT_L': nlopt.GN_DIRECT_L,
	'GN_DIRECT_L_RAND': nlopt.GN_DIRECT_L_RAND,
	'GN_DIRECT_NOSCAL': nlopt.GN_DIRECT_NOSCAL,
	'GN_DIRECT_L_NOSCAL': nlopt.GN_DIRECT_L_NOSCAL,
	'GN_DIRECT_L_RAND_NOSCAL': nlopt.GN_DIRECT_L_RAND_NOSCAL,
	'GN_ORIG_DIRECT': nlopt.GN_ORIG_DIRECT,
	'GN_ORIG_DIRECT_L': nlopt.GN_ORIG_DIRECT_L,
	'GN_CRS2_LM': nlopt.GN_CRS2_LM,
	'GN_AGS': nlopt.GN_AGS,
	'GN_ISRES': nlopt.GN_ISRES,
	'GN_ESCH': nlopt.GN_ESCH,
	'G_MLSL': nlopt.G_MLSL,
	'G_MLSL_LDS': nlopt.G_MLSL_LDS,
	'GD_STOGO': nlopt.GD_STOGO,
	'GD_STOGO_RAND': nlopt.GD_STOGO_RAND
}
shunned = {  # useless algorithms
	'LN_PRAXIS': nlopt.LN_PRAXIS,
	'LD_SLSQP': nlopt.LD_SLSQP,
	'LD_MMA': nlopt.LD_MMA,
	'LD_CCSAQ': nlopt.LD_CCSAQ,
	'LD_LBFGS': nlopt.LD_LBFGS,
	'LD_TNEWTON': nlopt.LD_TNEWTON,
	'LD_TNEWTON_PRECOND': nlopt.LD_TNEWTON_PRECOND,
	'LD_TNEWTON_RESTART': nlopt.LD_TNEWTON_RESTART,
	'LD_TNEWTON_PRECOND_RESTART': nlopt.LD_TNEWTON_PRECOND_RESTART,
	'LD_VAR1': nlopt.LD_VAR1,
	'LD_VAR2': nlopt.LD_VAR2,
	'GN_ISRES': nlopt.GN_ISRES,
	'GN_DIRECT': nlopt.GN_DIRECT,
	'GN_DIRECT_L': nlopt.GN_DIRECT_L,
	'GN_DIRECT_L_RAND': nlopt.GN_DIRECT_L_RAND,
	'GN_DIRECT_NOSCAL': nlopt.GN_DIRECT_NOSCAL,
	'GN_DIRECT_L_NOSCAL': nlopt.GN_DIRECT_L_NOSCAL,
	'GN_DIRECT_L_RAND_NOSCAL': nlopt.GN_DIRECT_L_RAND_NOSCAL,
	'GN_ORIG_DIRECT': nlopt.GN_ORIG_DIRECT,
	'GN_ORIG_DIRECT_L': nlopt.GN_ORIG_DIRECT_L
}
if method in shunned:
	print(f"Warning : method {method} shunned.", file=stderr)


# method = list(algos.keys())[run_N]  # to test several methods


def ini_opt(par0, min_obj, low_bnd=None, verbose=False, maxtime=7000):
	""" initializes a nlopt optimizer """
	obj = nlopt.opt(algos[method], len(par0))
	# defines which nlopt algorithm to use
	obj.set_local_optimizer(nlopt.opt(nlopt.LN_NEWUOA, len(par0)))
	if low_bnd is not None:
		obj.set_lower_bounds(low_bnd)
		obj.set_upper_bounds(-low_bnd)
	if 'LN_NEWUOA' in method:
		fa = 5e-10
	elif method[:3] == 'LN_':
		fa = 5e-12
	else:
		fa = 5e-12
	obj.set_ftol_abs(fa)  # absolute tolerance on result
	if 'LN_NEWUOA' in method:
		fr = 5e-10
	elif method[:3] == 'LN_':
		fr = 5e-12
	else:
		fr = 5e-9
	obj.set_ftol_rel(fr)  # relative tolerance on result
	# obj.set_stopval(-167.2)
	if rdd() > .5: obj.set_initial_step(pi / 3)
	# obj.set_maxeval(10000)
	obj.set_min_objective(min_obj)
	obj.set_maxtime(maxtime)  # max time in seconds
	stoppp = nevals(par0)
	if verbose: print(f'Starting optimization via {method} - ({stoppp}calls)!')
	return obj


init_time = pctime()

def get_Stt_lazy(par):
	""" Input : parameters. Output : statevector"""
	# print('para', ans.num_parameters , ans.num_qubits, len(par))
	return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))
def opt_res(par0, min_obj, low_bnd=None, verbose=False):
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0
	res = nan
	relaunch = True
	while relaunch:
		relaunch = False
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)
		try:
			xopt = opt.optimize(loc_par0)
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0
		res = opt.last_optimum_value()
		nl_res = opt.last_optimize_result()
		if nl_res == nlopt.SUCCESS:
			print("Generic success return value.")
		elif nl_res == nlopt.STOPVAL_REACHED:
			print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
		elif nl_res == nlopt.FTOL_REACHED:
			print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
		elif nl_res == nlopt.XTOL_REACHED:
			print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
		elif nl_res == nlopt.MAXEVAL_REACHED:
			print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
		elif nl_res == nlopt.MAXTIME_REACHED:
			print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
		elif nl_res < 0:
			print('Errored stoppage :', nl_res)
			relaunch = True
			loc_par0 += random.uniform(-1, 1, size=len(loc_par0)) * 0.05
		else:
			print("Unknown stoppage case.")
		print()
	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res

def combined_objective(x, grad, min_obj, epsilon, ucln, nqb, norm):
	# Calculate the main objective (energy)
	energy = min_obj(x)

	# Calculate the MSE constraint term
	mse_value = MSE(ucln*(4**nqb), norm(x)*(4**nqb))

	# Combine both objectives: minimize energy and keep MSE <= epsilon
	# You can weigh the MSE term if necessary (e.g., by multiplying by a factor)
	combined_value = energy + max(0, mse_value - epsilon)  # You add the penalty if the MSE exceeds epsilon
	
	# If grad is requested, calculate it (if needed)
	if grad is not None:
		# Gradients would also need to be computed for both terms, this part depends on your setup
		grad_energy = ...  # Compute gradient of energy
		grad_mse = ...  # Compute gradient of MSE
		grad[:] = grad_energy + grad_mse  # Combine them

	return combined_value

def opt_res_c(par0, min_obj, low_bnd=None, verbose=False, epsilon=0.1):
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0
	res = nan
	relaunch = True
	
	while relaunch:
		relaunch = False
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)

		# Set the main energy objective (you already have the main objective here)
		opt.set_min_objective(lambda x, grad: min_obj(x))

		# Add the MSE constraint: MSE must be <= epsilon
		opt.add_inequality_constraint(lambda x, grad: mse_constraint(x) - epsilon, 1e-8)

		# Main optimization loop
		try:
			while True:  # Keep optimizing until the MSE constraint is satisfied
				xopt = opt.optimize(loc_par0)
				res = opt.last_optimum_value()
				nl_res = opt.last_optimize_result()

				# Check if MSE is within the allowed epsilon, if not continue
				mse_value = mse_constraint(xopt)
				if mse_value > epsilon:
					print(f"MSE constraint violated: {mse_value} > {epsilon}, continuing optimization.")
					continue  # Skip checking the other stop conditions and keep optimizing
				
				# If we pass the MSE check, exit the loop
				if nl_res == nlopt.SUCCESS:
					print("Generic success return value.")
					break
				elif nl_res == nlopt.STOPVAL_REACHED:
					print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
					break
				elif nl_res == nlopt.FTOL_REACHED:
					print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
					break
				elif nl_res == nlopt.XTOL_REACHED:
					print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
					break
				elif nl_res == nlopt.MAXEVAL_REACHED:
					print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
					break
				elif nl_res == nlopt.MAXTIME_REACHED:
					print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
					break
				elif nl_res < 0:
					print('Errored stoppage:', nl_res)
					relaunch = True
					loc_par0 += random.uniform(-1, 1, size=len(loc_par0)) * 0.05
					break
				else:
					print("Unknown stoppage case.")
					break
				
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0
		
		print()

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res

def opt_res_c2(par0, min_obj, low_bnd=None, verbose=False, epsilon=0.1, lambda_penalty=1e3, ucln=None, nqb=None, norm=None):
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0
	res = nan
	relaunch = True
	
	while relaunch:
		relaunch = False
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)

		# Définir l'objectif combiné
		def combined_objective(x, grad):
			min_obj_val = min_obj(x)
			mse_val = MSE(ucln * (4**nqb), norm(min_obj(x)) * (4**nqb))
			penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2
			return min_obj_val + penalty

		# Définir l'objectif principal comme l'objectif combiné
		opt.set_min_objective(combined_objective)

		try:
			xopt = opt.optimize(loc_par0)
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0
		
		res = opt.last_optimum_value()
		nl_res = opt.last_optimize_result()

		if nl_res == nlopt.SUCCESS:
			print("Generic success return value.")
		elif nl_res == nlopt.STOPVAL_REACHED:
			print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
		elif nl_res == nlopt.FTOL_REACHED:
			print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
		elif nl_res == nlopt.XTOL_REACHED:
			print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
		elif nl_res == nlopt.MAXEVAL_REACHED:
			print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
		elif nl_res == nlopt.MAXTIME_REACHED:
			print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
		elif nl_res < 0:
			print('Errored stoppage:', nl_res)
			relaunch = True
			loc_par0 += random.uniform(-1, 1, size=len(loc_par0)) * 0.05
		else:
			print("Unknown stoppage case.")
		
		print()

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res


def opt_res_c3(par0, min_obj, low_bnd=None, verbose=False, epsilon=0.1, lambda_penalty=1e2, max_retries=5,lay=None, ucln=None, nqb=None, norm=None):
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0
	res = nan
	relaunch = True
	retry_count = 0
	
	while retry_count < max_retries:
		relaunch = False  # Reset relaunch flag at the start of each attempt
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)

		# Fonction combinée : objectif principal + pénalité MSE
		def combined_objective(x, grad):
			# Calcul de l'objectif principal
			min_obj_val = min_obj(x)
			
			# Calcul de la contrainte MSE
			mse_val = MSE(ucln * (4**nqb), norm(get_Stt_lazy(x)) * (4**nqb))

			# Si MSE dépasse epsilon, ajouter une pénalité, mais de manière progressive
			penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2

			# L'objectif combiné est l'objectif principal + pénalité
			return min_obj_val + penalty
		ans_stt = gen_ans(nqb, lay)
	
		def get_Stt_lazy(par):
			""" Input : parameters. Output : statevector"""
			# print('para', ans.num_parameters , ans.num_qubits, len(par))
			return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))
		
		# Définir la fonction objectif combinée
		opt.set_min_objective(combined_objective)

		try:
			xopt = opt.optimize(loc_par0)
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0
		
		res = opt.last_optimum_value()
		nl_res = opt.last_optimize_result()

		# Calculate the final MSE value
		mse_val_final = MSE(ucln * (4**nqb), norm(get_Stt_lazy(xopt)) * (4**nqb))
		print(ucln)
		print(norm(get_Stt_lazy(xopt)))
		print(mse_val_final)
		# Check the MSE condition
		if mse_val_final <= 10: # à modifier, je pense qu'une valeur entre 1 et 5 pourrait être possible
			print(f"MSE is {mse_val_final}, which is less than or equal to 10. Stopping optimization.")
			break  # Stop the optimization without relaunching

		# Check optimization results and handle different cases
		if nl_res == nlopt.SUCCESS:
			print("Generic success return value.")
		elif nl_res == nlopt.STOPVAL_REACHED:
			print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
		elif nl_res == nlopt.FTOL_REACHED:
			print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
			# If optimization was stopped due to tolerance, we can relaunch with new parameters
			print("Retrying optimization with adjusted parameters...")
			retry_count += 1
			loc_par0 += random.uniform(-1, 1, size=len(loc_par0)) * 0.05  # Perturbation des paramètres initiaux
			relaunch = True  # Set relaunch flag to true if we need to retry
		elif nl_res == nlopt.XTOL_REACHED:
			print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
		elif nl_res == nlopt.MAXEVAL_REACHED:
			print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
		elif nl_res == nlopt.MAXTIME_REACHED:
			print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
		elif nl_res < 0:
			print('Errored stoppage:', nl_res)
			retry_count += 1
			loc_par0 += random.uniform(-1, 1, size=len(loc_par0)) * 0.05
			relaunch = True
		else:
			print("Unknown stoppage case.")
		
		print()

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res





def opt_res_c4(par0, min_obj, low_bnd=None, verbose=False, epsilon=0.01, lambda_penalty=2e2, max_retries=1, lay=None, ucln=None, nqb=None, norm=None):
	#Penser à modifier epsilon, on peut ptet tenter epsilon 0.01 ou encore la penalty
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0  # Initialize xopt with initial parameters
	res = nan
	relaunch = True
	retry_count = 0
	H, Hsp, frc = get_Hf(nqb)
	ucl = sc.sparse.linalg.spsolve(Hsp.tocsr(), frc).real
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
		f_psi = frc @ stt
		return f_psi
	while retry_count < max_retries:
		relaunch = False  # Reset relaunch flag at the start of each attempt
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)

		# Fonction combinée : objectif principal + pénalité MSE
		def combined_objective(x, grad):
			# Calcul de l'objectif principal
			min_obj_val = min_obj(x)
			#MSE(ucln * (5**nqb), norm(get_Stt_lazy(x)) * (5**nqb))
			# Calcul de la contrainte MSE
			normalisation=norm_q_fromstt(get_Stt_lazy(x))
			mse_val = MSE(ucl, ((norm(get_Stt_lazy(x)))*normalisation)) #modifier ça si changement

			# Si MSE dépasse epsilon, ajouter une pénalité, mais de manière progressive
			penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2

			# L'objectif combiné est l'objectif principal + pénalité
			return min_obj_val + penalty
		
		ans_stt = gen_ans(nqb, lay)
	
		def get_Stt_lazy(par):
			""" Input : parameters. Output : statevector"""
			return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))
		
		# Définir la fonction objectif combinée
		opt.set_min_objective(combined_objective)

		try:
			xopt = opt.optimize(loc_par0)
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0  # If exception occurs, restart from the original parameters
		
		res = opt.last_optimum_value()
		nl_res = opt.last_optimize_result()
		normalisation=norm_q_fromstt(get_Stt_lazy(xopt))

		# Calculate the final MSE value
		#print("normalisation=",normalisation)
		#print((get_Stt_lazy(xopt))*normalisation)
		#print(ucl)
		mse_val_final = MSE(ucl, norm((get_Stt_lazy(xopt)))*normalisation) #Modifier ça si changement
		print(f"MSE value: {mse_val_final}")

		# Check the MSE condition
		if mse_val_final <= 0.001:  # Modify this threshold as needed (0.05 to 0.1 could be reasonable)
			print(f"MSE is {mse_val_final}, which is less than or equal to 0.001 . Stopping optimization.")
			break  # Stop the optimization without relaunching

		# Check optimization results and handle different cases
		if nl_res == nlopt.SUCCESS:
			print("Generic success return value.")
		elif nl_res == nlopt.STOPVAL_REACHED:
			print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
		elif nl_res == nlopt.FTOL_REACHED:
			print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
			# Relaunch with the previous best parameters (xopt) rather than random perturbation
			print("Retrying optimization with adjusted parameters...")
			retry_count += 1
			loc_par0 = xopt  # Use the latest optimized parameters (xopt) for the next iteration
			relaunch = True  # Set relaunch flag to true if we need to retry
		elif nl_res == nlopt.XTOL_REACHED:
			print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
		elif nl_res == nlopt.MAXEVAL_REACHED:
			print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
		elif nl_res == nlopt.MAXTIME_REACHED:
			print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
		elif nl_res < 0:
			print('Errored stoppage:', nl_res)
			retry_count += 1
			loc_par0 = xopt  # Use the latest optimized parameters (xopt) for the next iteration
			relaunch = True
		else:
			print("Unknown stoppage case.")
		
		print()

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res

from scipy.optimize import minimize

def opt_res_lbfgsb_c4(par0, min_obj, low_bnd=None, verbose=False,
					  epsilon=0.01, lambda_penalty=2e2, max_retries=1,
					  lay=None, ucln=None, nqb=None, norm=None):

	global init_time
	init_time = pctime()
	loc_par0 = np.array(par0)
	retry_count = 0
	xopt = loc_par0
	res = np.nan
	

	# Construct system
	H, Hsp, frc = get_Hf(nqb)
	ucl = sc.sparse.linalg.spsolve(Hsp.tocsr(), frc).real
	ans_stt = gen_ans(nqb, lay)

	def get_Stt_lazy(par):
		return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))

	def psiHpsi_fromstt(stt):
		return stt @ H @ stt

	def fpsi_fromstt(stt):
		return frc @ stt

	def norm_q_fromstt(stt):
		return fpsi_fromstt(stt) / psiHpsi_fromstt(stt)

	def combined_objective(x):
		stt = get_Stt_lazy(x)
		norm_factor = norm_q_fromstt(stt)
		mse_val = MSE(ucl, stt[:len(ucl)] * norm_factor)
		penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2
		obj_val = min_obj(x)
		if verbose:
			print(f"Obj: {obj_val:.5f}, MSE: {mse_val:.5e}, Penalty: {penalty:.5f}")
		return obj_val + penalty

	bounds = None
	if low_bnd is not None:
		bounds = [(low, -low) for low in low_bnd]

	while retry_count < max_retries:
		result = minimize(
			combined_objective,
			loc_par0,
			method="L-BFGS-B",
			bounds=bounds,
			options={'disp': verbose, 'maxiter': 1000}
		)

		xopt = result.x
		res = result.fun
		stt_final = get_Stt_lazy(xopt)
		norm_final = norm_q_fromstt(stt_final)
		mse_val_final = MSE(ucl, stt_final[:len(ucl)] * norm_final)

		print(f"MSE value: {mse_val_final:.4e}")

		if mse_val_final <= 0.001:
			print(f"MSE is {mse_val_final:.4e}, which is within tolerance. Optimization succeeded.")
			break

		print(f"Retrying... ({retry_count + 1}/{max_retries})")
		loc_par0 = xopt
		retry_count += 1

	total_duration = pctime() - init_time
	print(f"Final objective: {res:.6f} in {total_duration:.2f} seconds.")
	return xopt, res

def opt_res_hybrid(par0, min_obj, low_bnd=None, verbose=False,
				   epsilon=0.01, lambda_penalty=2e2,
				   lay=None, ucln=None, nqb=None, norm=None,
				   start_with="NEWUOA", max_switches=3, sigma0=0.4):

	from scipy.optimize import minimize
	def MSE(u, v):
		""" Relative mean square error in percent """
		return 100 * np.mean((np.abs(u - v) / np.maximum(np.abs(u), 1e-12)) ** 2)

	global init_time
	init_time = pctime()
	loc_par0 = np.array(par0)
	xopt = loc_par0
	res = np.nan
	switch_count = 0
	current_algo = start_with.upper()

	H, Hsp, frc = get_Hf(nqb)
	ucl = sc.sparse.linalg.spsolve(Hsp.tocsr(), frc).real
	ans_stt = gen_ans(nqb, lay)

	def get_Stt_lazy(par):
		return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))
	normalize = True  # whether to
	def norm(x): return x/(np.linalg.norm(x) if normalize else 1)


	def psiHpsi_fromstt(stt):
		return stt @ H @ stt

	def fpsi_fromstt(stt):
		return frc @ stt

	def norm_q_fromstt(stt):
		return fpsi_fromstt(stt) / psiHpsi_fromstt(stt)

	def combined_objective(x, grad=None):
		stt = get_Stt_lazy(x)
		norm_factor = norm_q_fromstt(stt)
		mse_val = MSE(ucl, (stt[:len(ucl)]) * norm_factor)
		penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2
		obj_val = min_obj(x)
		return obj_val + penalty

	bounds = None
	if low_bnd is not None:
		bounds = [(low, -low) for low in low_bnd]

	while switch_count < max_switches:
		print(f"\n Attempt {switch_count+1} using {current_algo}...")

		if current_algo == "L-BFGS-B":
			result = minimize(
				combined_objective,
				xopt,
				method="L-BFGS-B",
				bounds=bounds,
				options={'disp': False, 'maxiter': 500}
			)
			xopt = result.x
			res = result.fun
			message = result.message if hasattr(result, "message") else "No message"

		elif current_algo == "NEWUOA":
			opt = ini_opt(xopt, combined_objective, low_bnd=low_bnd, verbose=verbose)
			try:
				xopt = opt.optimize(xopt)
			except Exception as e:
				print("NEWUOA Exception:", e)
				pass
			res = opt.last_optimum_value()
			message = f"NLopt code: {opt.last_optimize_result()}"

		else:
			raise ValueError(f"Unknown optimizer: {current_algo}")

		# Check MSE
		stt_final = get_Stt_lazy(xopt)
		norm_final = norm_q_fromstt(stt_final)
		mse_val_final = MSE(ucl, (stt_final) * norm_final)

		print(f" Result: obj={res:.6f}, MSE={mse_val_final:.4e}")
		print("MSE(ucl, ucl) =", MSE(ucl, ucl))
		print(
			f"MSE={100-MSE(ucl, stt_final*norm_final) }%")
		print(f" Message: {message}")
		
		
	


		if mse_val_final <= 0.001:
			print(f" MSE constraint satisfied. Done.")
			break

		# Switch optimizer
		current_algo = "L-BFGS-B" if current_algo == "NEWUOA" else "NEWUOA"
		switch_count += 1

	total_duration = pctime() - init_time
	print(f"\n Final result: {res:.6f} in {total_duration:.2f} seconds.")
	

	return xopt, res

def cmesh(n):  #Number of Nodes in Each Direction For Meshing Function

	n_nodes = int(n**3)
	n_elem = int((n-1)**3)
	
	#This code generates the connectivity table for the mesh

	conn = np.zeros(((n-1)**3, 8), dtype = int)

	#First row of connectivity table
	conn[0,0] = 1
	conn[0,1] = 2
	conn[0,2] = n + 2
	conn[0,3] = n + 1
	conn[0,4] = n**2 + 1
	conn[0,5] = n**2 + 2
	conn[0,6] = n**2 + n + 2
	conn[0,7] = n**2 + n + 1


	counter  = 0 #counter for row of connectivity table
	for ii in range(0, n-1): #outer jumps in z + n**2
		if ii < n-1:
			conn[counter] = conn[counter-(n-1)**2] + n**2
			conn[0,0] = 1
			conn[0,1] = 2
			conn[0,2] = n+2
			conn[0,3] = n+1
			conn[0,4] = n**2 + 1
			conn[0,5] = n**2 + 2
			conn[0,6] = n**2 + n + 2
			conn[0,7] = n**2 + n + 1
			counter = counter + 1
			jj = 0
		else:
			jj = 0
		for jj in range(0, n-1): #outer jumps in x + 1
			if jj < n - 2:
				conn[counter] = conn[counter-1] + 1
				counter = counter + 1
				kk = 0
			else:
				kk = 0
				for kk in range(0, n-2): #inner jumps in y + 2
					conn[counter] = conn[counter-1] + 2
					counter = counter + 1
					for jj in range(0, n-2): #outer jumps in x + 1
						conn[counter] = conn[counter-1] + 1
						counter = counter + 1
		

	#Adjust the indexes to 0-base
	conn = conn - 1

	for r in range(0, len(conn)):
		for col in range(0, 4):
			conn[r,col] = int(conn[r,col]) #Make the entries integers

	#print conn
		
	#Now I need to generate the nodes corresponding to the connectivity table

	nodes = np.zeros((n**3,3))
	nodes2 = np.zeros((n**3,3))

	space = np.linspace(0, 1, n)

	step = space[1] - space[0]

	cc1 = 0

	for zz in range(0, n):
		nodes[cc1,2] = space[zz]
		nodes[cc1,1] = 0
		nodes[cc1,0] = 0
		#cc1 = cc1 + 1
		nodes[0] = 0
		for yy in range(0, n):
			nodes[cc1,2] = space[zz]
			nodes[cc1,1] = space[yy]
			#cc1 = cc1 + 1
			for xx in range(0, n):
				nodes[cc1,2] = space[zz]
				nodes[cc1,1] = space[yy]
				nodes[cc1,0] = space[xx]
				cc1 = cc1 + 1


	return nodes, conn
Nx = 4
[nodes, conn] = cmesh(Nx)
flux_coverage=0.5
#Mapping Functions

# The shape functions
def phi0(z1,z2,z3):
	phi0 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)*(1.0 - z3)
	return phi0

def phi1(z1,z2,z3):
	phi1 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)*(1.0 - z3)
	return phi1

def phi2(z1,z2,z3):
	phi2 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)*(1.0 - z3)
	return phi2

def phi3(z1,z2,z3):
	phi3 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)*(1.0 - z3)
	return phi3

def phi4(z1,z2,z3):
	phi4 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)*(1.0 + z3)
	return phi4

def phi5(z1,z2,z3):
	phi5 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)*(1.0 + z3)
	return phi5

def phi6(z1,z2,z3):
	phi6 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)*(1.0 + z3)
	return phi6

def phi7(z1,z2,z3):
	phi7 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)*(1.0 + z3)
	return phi7

#Derivatives of the shape function with respect to z1

def dphi0dz1(z1,z2,z3):
	dphi0dz1 = (1.0/8.0)*(-1.0)*(1.0 - z2)*(1.0 - z3)
	return dphi0dz1

def dphi1dz1(z1,z2,z3):
	dphi1dz1 = (1.0/8.0)*(1.0 - z2)*(1.0 - z3)
	return dphi1dz1

def dphi2dz1(z1,z2,z3):
	dphi2dz1 = (1.0/8.0)*(1.0 + z2)*(1.0 - z3)
	return dphi2dz1

def dphi3dz1(z1,z2,z3):
	dphi3dz1 = (1.0/8.0)*(-1.0)*(1.0 + z2)*(1.0 - z3)
	return dphi3dz1

def dphi4dz1(z1,z2,z3):
	dphi4dz1 = (1.0/8.0)*(-1.0)*(1.0 - z2)*(1.0 + z3)
	return dphi4dz1

def dphi5dz1(z1,z2,z3):
	dphi5dz1 = (1.0/8.0)*(1.0 - z2)*(1.0 + z3)
	return dphi5dz1

def dphi6dz1(z1,z2,z3):
	dphi6dz1 = (1.0/8.0)*(1.0 + z2)*(1.0 + z3)
	return dphi6dz1

def dphi7dz1(z1,z2,z3):
	dphi7dz1 = (1.0/8.0)*(-1.0)*(1.0 + z2)*(1.0 + z3)
	return dphi7dz1


#Derivatives of the shape function with respect to z2

def dphi0dz2(z1,z2,z3):
	dphi0dz2 = (1.0/8.0)*(1.0 - z1)*(-1.0)*(1.0 - z3)
	return dphi0dz2

def dphi1dz2(z1,z2,z3):
	dphi1dz2 = (1.0/8.0)*(1.0 + z1)*(-1.0)*(1.0 - z3)
	return dphi1dz2

def dphi2dz2(z1,z2,z3):
	dphi2dz2 = (1.0/8.0)*(1.0 + z1)*(1.0 - z3)
	return dphi2dz2

def dphi3dz2(z1,z2,z3):
	dphi3dz2 = (1.0/8.0)*(1.0 - z1)*(1.0 - z3)
	return dphi3dz2

def dphi4dz2(z1,z2,z3):
	dphi4dz2 = (1.0/8.0)*(1.0 - z1)*(-1.0)*(1.0 + z3)
	return dphi4dz2

def dphi5dz2(z1,z2,z3):
	dphi5dz2 = (1.0/8.0)*(1.0 + z1)*(-1.0)*(1.0 + z3)
	return dphi5dz2

def dphi6dz2(z1,z2,z3):
	dphi6dz2 = (1.0/8.0)*(1.0 + z1)*(1.0 + z3)
	return dphi6dz2

def dphi7dz2(z1,z2,z3):
	dphi7dz2 = (1.0/8.0)*(1.0 - z1)*(1.0 + z3)
	return dphi7dz2


#Derivatives of the shape function with respect to z3

def dphi0dz3(z1,z2,z3):
	dphi0dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)*(-1.0)
	return dphi0dz3

def dphi1dz3(z1,z2,z3):
	dphi1dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)*(-1.0)
	return dphi1dz3

def dphi2dz3(z1,z2,z3):
	dphi2dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)*(-1.0)
	return dphi2dz3

def dphi3dz3(z1,z2,z3):
	dphi3dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)*(-1.0)
	return dphi3dz3

def dphi4dz3(z1,z2,z3):
	dphi4dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)
	return dphi4dz3

def dphi5dz3(z1,z2,z3):
	dphi5dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)
	return dphi5dz3

def dphi6dz3(z1,z2,z3):
	dphi6dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)
	return dphi6dz3

def dphi7dz3(z1,z2,z3):
	dphi7dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)
	return dphi7dz3
def cmesh_serpentine(n):
	n_nodes = n**3
	n_elem = (n-1)**3
	
	# Create nodes
	nodes = np.zeros((n_nodes, 3))
	space = np.linspace(0, 1, n)
	
	cc1 = 0
	for zz in range(n):
		for yy in range(n):
			if yy % 2 == 0:
				# even row: left to right
				for xx in range(n):
					nodes[cc1, 0] = space[xx]
					nodes[cc1, 1] = space[yy]
					nodes[cc1, 2] = space[zz]
					cc1 += 1
			else:
				# odd row: right to left
				for xx in reversed(range(n)):
					nodes[cc1, 0] = space[xx]
					nodes[cc1, 1] = space[yy]
					nodes[cc1, 2] = space[zz]
					cc1 += 1

	# Now create connectivity
	conn = np.zeros((n_elem, 8), dtype=int)
	node_indices = np.arange(n_nodes).reshape((n, n, n))  # (x, y, z)

	# We need to reorder indices in node_indices according to serpentine
	new_indices = np.zeros_like(node_indices)
	cc = 0
	for k in range(n):
		for j in range(n):
			if j % 2 == 0:
				for i in range(n):
					new_indices[i, j, k] = cc
					cc += 1
			else:
				for i in reversed(range(n)):
					new_indices[i, j, k] = cc
					cc += 1

	# Then define connectivity
	elem = 0
	for k in range(n-1):
		for j in range(n-1):
			for i in range(n-1):
				conn[elem, 0] = new_indices[i  , j  , k  ]
				conn[elem, 1] = new_indices[i+1, j  , k  ]
				conn[elem, 2] = new_indices[i+1, j+1, k  ]
				conn[elem, 3] = new_indices[i  , j+1, k  ]
				conn[elem, 4] = new_indices[i  , j  , k+1]
				conn[elem, 5] = new_indices[i+1, j  , k+1]
				conn[elem, 6] = new_indices[i+1, j+1, k+1]
				conn[elem, 7] = new_indices[i  , j+1, k+1]
				elem += 1

	return nodes, conn
def get_S(Kt=10.0, zsource=0.0, qflux=1000.0, Pstar=1e7):
	import numpy as np
	from numpy.linalg import inv
	import scipy.sparse as sps
	[nodes,conn]=cmesh_serpentine(Nx) #enlever _serpentine si on veut pas serpentine.
	# Gauss quadrature points and weights #I have no clue where it comes from
	w5 = np.array([0.568888888888889, 0.478628670499366, 0.478628670499366, 0.236926885056189, 0.236926885056189])
	gauss5 = np.array([0.000000000000000, 0.538469310105683, -0.538469310105683, 0.906179845938664, -0.906179845938664])

	# Preallocations
	NP = len(nodes)
	Ne = len(conn)
	S = sps.lil_matrix((NP, NP))
	R = np.zeros(NP)

	# Basis function definitions and derivatives...
	# You can factor those out or import them if modularized.

	# Helper functions: phi0 to phi7, dphi0dz1 to dphi7dz3
	# ⏩ (Not shown here, assumed already defined in your script)

	# Local arrays
	x1 = np.zeros(8)
	x2 = np.zeros(8)
	x3 = np.zeros(8)

	# === Integral 1: stiffness matrix ===
	for e in range(Ne):
		x1[:] = nodes[conn[e, :], 0]
		x2[:] = nodes[conn[e, :], 1]
		x3[:] = nodes[conn[e, :], 2]

		fS1 = np.zeros((8, 8))
		for A in range(8):
			for B in range(8):
				val = 0.0
				for ii in range(5):
					for jj in range(5):
						for kk in range(5):
							z1, z2, z3 = gauss5[ii], gauss5[jj], gauss5[kk]
							w = w5[ii] * w5[jj] * w5[kk]

							# Jacobian matrix F and its inverse
							F = np.zeros((3, 3))
							for i in range(8):
								dphi_i = np.array([
									eval(f'dphi{i}dz1')(z1, z2, z3),
									eval(f'dphi{i}dz2')(z1, z2, z3),
									eval(f'dphi{i}dz3')(z1, z2, z3)
								])
								F[0] += x1[i] * dphi_i
								F[1] += x2[i] * dphi_i
								F[2] += x3[i] * dphi_i

							FinvT = inv(F).T
							gradA = FinvT @ np.array([
								eval(f'dphi{A}dz1')(z1, z2, z3),
								eval(f'dphi{A}dz2')(z1, z2, z3),
								eval(f'dphi{A}dz3')(z1, z2, z3)
							])
							gradB = FinvT @ np.array([
								eval(f'dphi{B}dz1')(z1, z2, z3),
								eval(f'dphi{B}dz2')(z1, z2, z3),
								eval(f'dphi{B}dz3')(z1, z2, z3)
							])

							val += w * (gradA @ gradB) * np.linalg.det(F) * Kt
				fS1[A, B] = val

		for i in range(8):
			for j in range(8):
				S[conn[e, i], conn[e, j]] += fS1[i, j]

	# === Integral 3: volumetric source ===
	if zsource != 0.0:
		for e in range(Ne):
			x1[:] = nodes[conn[e, :], 0]
			x2[:] = nodes[conn[e, :], 1]
			x3[:] = nodes[conn[e, :], 2]

			fR = np.zeros(8)
			for A in range(8):
				val = 0.0
				for ii in range(5):
					for jj in range(5):
						for kk in range(5):
							z1, z2, z3 = gauss5[ii], gauss5[jj], gauss5[kk]
							w = w5[ii] * w5[jj] * w5[kk]

							F = np.zeros((3, 3))
							for i in range(8):
								dphi_i = np.array([
									eval(f'dphi{i}dz1')(z1, z2, z3),
									eval(f'dphi{i}dz2')(z1, z2, z3),
									eval(f'dphi{i}dz3')(z1, z2, z3)
								])
								F[0] += x1[i] * dphi_i
								F[1] += x2[i] * dphi_i
								F[2] += x3[i] * dphi_i

							phiA = eval(f'phi{A}')(z1, z2, z3)
							val += w * phiA * np.linalg.det(F) * zsource
				fR[A] = val

			for i in range(8):
				R[conn[e, i]] += fR[i]

	# === Integral 4: partial Neumann on x = 0, top 30% (patchy flux) ===
	x_face_tol = 1e-6
	face_nodes = [0, 3, 7, 4]
	flux_coverage = 0.3  # Top 30% in z
	for e in range(Ne):
		global_nodes = conn[e, face_nodes]
		x_coords = nodes[global_nodes, 0]
		z_avg = np.mean(nodes[global_nodes, 2])
	
		if np.all(x_coords < x_face_tol) and z_avg > 1.0 - flux_coverage:
			# Random flux per face (cloudy sunlight effect)
			local_qflux = qflux * np.random.uniform(0.5, 5) #Completely random.
			#local_qflux = qflux * (0.5 + 0.5 * np.sin(np.pi * z_avg))
	
			x_face = nodes[global_nodes]
			for ii in range(5):
				for jj in range(5):
					z2, z3 = gauss5[ii], gauss5[jj]
					w = w5[ii] * w5[jj]
	
					phi_vals = [
						phi0(-1, z2, z3),
						phi3(-1, z2, z3),
						phi7(-1, z2, z3),
						phi4(-1, z2, z3)
					]
	
					p0, p1, p2 = x_face[0], x_face[1], x_face[2]
					t1 = p1 - p0
					t2 = p2 - p0
					dS = np.linalg.norm(np.cross(t1, t2))
	
					for A_local, node in enumerate(global_nodes):
						R[node] += w * phi_vals[A_local] * local_qflux * dS
						#R[node] += w * phi_vals[A_local] * qflux * dS
	# === Integral 6: Dirichlet on x = 1 with penalty ===
	xup = 1.0
	Nx_tol = 1e-6
	face_nodes = [1, 2, 6, 5]
	for e in range(Ne):
		global_nodes = conn[e, face_nodes]
		x_coords = nodes[global_nodes, 0]

		if np.all(np.abs(x_coords - xup) < Nx_tol):
			x_face = nodes[global_nodes]
			for A in range(4):
				for B in range(4):
					val = 0.0
					for ii in range(5):
						for jj in range(5):
							z2, z3 = gauss5[ii], gauss5[jj]
							w = w5[ii] * w5[jj]

							phi_vals = [
								phi1(1, z2, z3),
								phi2(1, z2, z3),
								phi6(1, z2, z3),
								phi5(1, z2, z3)
							]

							p0, p1, p2 = x_face[0], x_face[1], x_face[2]
							t1 = p1 - p0
							t2 = p2 - p0
							dS = np.linalg.norm(np.cross(t1, t2))

							val += w * phi_vals[A] * phi_vals[B] * dS * Pstar

					a = global_nodes[A]
					b = global_nodes[B]
					S[a, b] += val

	return S.tocsr(), R
def get_S2(Kt=10.0, zsource=0.0, qflux=2000.0, qflux2=5000.0, Pstar=1e7):
	import numpy as np
	from numpy.linalg import inv
	import scipy.sparse as sps
	[nodes, conn] = cmesh(Nx)
	
	w5 = np.array([0.568888888888889, 0.478628670499366, 0.478628670499366, 0.236926885056189, 0.236926885056189])
	gauss5 = np.array([0.000000000000000, 0.538469310105683, -0.538469310105683, 0.906179845938664, -0.906179845938664])

	NP = len(nodes)
	Ne = len(conn)
	S = sps.lil_matrix((NP, NP))
	R = np.zeros(NP)

	x1 = np.zeros(8)
	x2 = np.zeros(8)
	x3 = np.zeros(8)

	for e in range(Ne):
		x1[:] = nodes[conn[e, :], 0]
		x2[:] = nodes[conn[e, :], 1]
		x3[:] = nodes[conn[e, :], 2]

		fS1 = np.zeros((8, 8))
		for A in range(8):
			for B in range(8):
				val = 0.0
				for ii in range(5):
					for jj in range(5):
						for kk in range(5):
							z1, z2, z3 = gauss5[ii], gauss5[jj], gauss5[kk]
							w = w5[ii] * w5[jj] * w5[kk]

							F = np.zeros((3, 3))
							for i in range(8):
								dphi_i = np.array([
									eval(f'dphi{i}dz1')(z1, z2, z3),
									eval(f'dphi{i}dz2')(z1, z2, z3),
									eval(f'dphi{i}dz3')(z1, z2, z3)
								])
								F[0] += x1[i] * dphi_i
								F[1] += x2[i] * dphi_i
								F[2] += x3[i] * dphi_i

							FinvT = inv(F).T
							gradA = FinvT @ np.array([
								eval(f'dphi{A}dz1')(z1, z2, z3),
								eval(f'dphi{A}dz2')(z1, z2, z3),
								eval(f'dphi{A}dz3')(z1, z2, z3)
							])
							gradB = FinvT @ np.array([
								eval(f'dphi{B}dz1')(z1, z2, z3),
								eval(f'dphi{B}dz2')(z1, z2, z3),
								eval(f'dphi{B}dz3')(z1, z2, z3)
							])

							val += w * (gradA @ gradB) * np.linalg.det(F) * Kt
				fS1[A, B] = val

		for i in range(8):
			for j in range(8):
				S[conn[e, i], conn[e, j]] += fS1[i, j]

	if zsource != 0.0:
		for e in range(Ne):
			x1[:] = nodes[conn[e, :], 0]
			x2[:] = nodes[conn[e, :], 1]
			x3[:] = nodes[conn[e, :], 2]

			fR = np.zeros(8)
			for A in range(8):
				val = 0.0
				for ii in range(5):
					for jj in range(5):
						for kk in range(5):
							z1, z2, z3 = gauss5[ii], gauss5[jj], gauss5[kk]
							w = w5[ii] * w5[jj] * w5[kk]

							F = np.zeros((3, 3))
							for i in range(8):
								dphi_i = np.array([
									eval(f'dphi{i}dz1')(z1, z2, z3),
									eval(f'dphi{i}dz2')(z1, z2, z3),
									eval(f'dphi{i}dz3')(z1, z2, z3)
								])
								F[0] += x1[i] * dphi_i
								F[1] += x2[i] * dphi_i
								F[2] += x3[i] * dphi_i

							phiA = eval(f'phi{A}')(z1, z2, z3)
							val += w * phiA * np.linalg.det(F) * zsource
				fR[A] = val

			for i in range(8):
				R[conn[e, i]] += fR[i]

	# Neumann on x=0, top 30%
	x_face_tol = 1e-6
	flux_coverage = 0.3
	face_nodes_x0 = [0, 3, 7, 4]
	for e in range(Ne):
		global_nodes = conn[e, face_nodes_x0]
		x_coords = nodes[global_nodes, 0]
		z_avg = np.mean(nodes[global_nodes, 2])
		if np.all(x_coords < x_face_tol) and z_avg > 1.0 - flux_coverage:
			x_face = nodes[global_nodes]
			for ii in range(5):
				for jj in range(5):
					z2, z3 = gauss5[ii], gauss5[jj]
					w = w5[ii] * w5[jj]
					phi_vals = [
						phi0(-1, z2, z3),
						phi3(-1, z2, z3),
						phi7(-1, z2, z3),
						phi4(-1, z2, z3)
					]
					p0, p1, p2 = x_face[0], x_face[1], x_face[2]
					t1 = p1 - p0
					t2 = p2 - p0
					dS = np.linalg.norm(np.cross(t1, t2))
					for A_local, node in enumerate(global_nodes):
						R[node] += w * phi_vals[A_local] * qflux * dS

	# Neumann on y=0, top 50%
	y_face_tol = 1e-6
	flux_coverage_y = 0.5
	face_nodes_y0 = [0, 1, 5, 4]
	for e in range(Ne):
		global_nodes = conn[e, face_nodes_y0]
		y_coords = nodes[global_nodes, 1]
		z_avg = np.mean(nodes[global_nodes, 2])
		if np.all(y_coords < y_face_tol) and z_avg > 1.0 - flux_coverage_y:
			y_face = nodes[global_nodes]
			for ii in range(5):
				for jj in range(5):
					z1, z3 = gauss5[ii], gauss5[jj]
					w = w5[ii] * w5[jj]
					phi_vals = [
						phi0(z1, -1, z3),
						phi1(z1, -1, z3),
						phi5(z1, -1, z3),
						phi4(z1, -1, z3)
					]
					p0, p1, p2 = y_face[0], y_face[1], y_face[2]
					t1 = p1 - p0
					t2 = p2 - p0
					dS = np.linalg.norm(np.cross(t1, t2))
					for A_local, node in enumerate(global_nodes):
						R[node] += w * phi_vals[A_local] * qflux2 * dS

	# Dirichlet on x=1
	xup = 1.0
	Nx_tol = 1e-6
	face_nodes = [1, 2, 6, 5]
	for e in range(Ne):
		global_nodes = conn[e, face_nodes]
		x_coords = nodes[global_nodes, 0]
		if np.all(np.abs(x_coords - xup) < Nx_tol):
			x_face = nodes[global_nodes]
			for A in range(4):
				for B in range(4):
					val = 0.0
					for ii in range(5):
						for jj in range(5):
							z2, z3 = gauss5[ii], gauss5[jj]
							w = w5[ii] * w5[jj]
							phi_vals = [
								phi1(1, z2, z3),
								phi2(1, z2, z3),
								phi6(1, z2, z3),
								phi5(1, z2, z3)
							]
							p0, p1, p2 = x_face[0], x_face[1], x_face[2]
							t1 = p1 - p0
							t2 = p2 - p0
							dS = np.linalg.norm(np.cross(t1, t2))
							val += w * phi_vals[A] * phi_vals[B] * dS * Pstar
					a = global_nodes[A]
					b = global_nodes[B]
					S[a, b] += val

	return S.tocsr(), R


# Hsp, frc = get_S()
# H = Hsp.toarray()
# ucl = sc.sparse.linalg.spsolve(Hsp, frc).real
# ucln = ucl/np.linalg.norm(ucl)



def opt_res_c5(par0, min_obj, low_bnd=None, verbose=False, epsilon=0.01, lambda_penalty=2e2, max_retries=5, lay=None, ucln=None, nqb=None, norm=None):
	#Penser à modifier epsilon, on peut ptet tenter epsilon 0.01 ou encore la penalty
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0  # Initialize xopt with initial parameters
	res = nan
	relaunch = True
	retry_count = 0
	Hsp, frc = get_S()
	H = Hsp.toarray()
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
		f_psi = frc @ stt
		return f_psi
	
	while retry_count < max_retries:
		relaunch = False  # Reset relaunch flag at the start of each attempt
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)

		# Fonction combinée : objectif principal + pénalité MSE
		def combined_objective(x, grad):
			# Calcul de l'objectif principal
			min_obj_val = min_obj(x)
			#MSE(ucln * (5**nqb), norm(get_Stt_lazy(x)) * (5**nqb))
			# Calcul de la contrainte MSE
			normalisation=norm_q_fromstt(get_Stt_lazy(x))
			mse_val = MSE(ucl, (get_Stt_lazy(x))*normalisation) #modifier ça si changement

			# Si MSE dépasse epsilon, ajouter une pénalité, mais de manière progressive
			penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2

			# L'objectif combiné est l'objectif principal + pénalité
			return min_obj_val + penalty
		
		ans_stt = gen_ans(nqb, lay)
	
		def get_Stt_lazy(par):
			""" Input : parameters. Output : statevector"""
			return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))
		
		# Définir la fonction objectif combinée
		opt.set_min_objective(combined_objective)

		try:
			xopt = opt.optimize(loc_par0)
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0  # If exception occurs, restart from the original parameters
		
		res = opt.last_optimum_value()
		nl_res = opt.last_optimize_result()
		normalisation=norm_q_fromstt(get_Stt_lazy(xopt))

		# Calculate the final MSE value
		#print("normalisation=",normalisation)
		#print((get_Stt_lazy(xopt))*normalisation)
		#print(ucl)
		mse_val_final = MSE(ucl, (get_Stt_lazy(xopt))*normalisation) #Modifier ça si changement
		print(f"MSE value: {mse_val_final}")

		# Check the MSE condition
		if abs(mse_val_final) <= 0.001:  # Modify this threshold as needed (0.05 to 0.1 could be reasonable)
			print(f"MSE is {mse_val_final}, which is less than or equal to 0.001 . Stopping optimization.")
			break  # Stop the optimization without relaunching

		# Check optimization results and handle different cases
		if nl_res == nlopt.SUCCESS:
			print("Generic success return value.")
		elif nl_res == nlopt.STOPVAL_REACHED:
			print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
		elif nl_res == nlopt.FTOL_REACHED:
			print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
			# Relaunch with the previous best parameters (xopt) rather than random perturbation
			print("Retrying optimization with adjusted parameters...")
			retry_count += 1
			loc_par0 = xopt  # Use the latest optimized parameters (xopt) for the next iteration
			relaunch = True  # Set relaunch flag to true if we need to retry
		elif nl_res == nlopt.XTOL_REACHED:
			print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
		elif nl_res == nlopt.MAXEVAL_REACHED:
			print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
		elif nl_res == nlopt.MAXTIME_REACHED:
			print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
		elif nl_res < 0:
			print('Errored stoppage:', nl_res)
			retry_count += 1
			loc_par0 = xopt  # Use the latest optimized parameters (xopt) for the next iteration
			relaunch = True
		else:
			print("Unknown stoppage case.")
		
		print()

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res


def opt_res_c7(par0, min_obj,Hsp,frc,H,ucl,n_nodes,lambda_penalty,up_bnd=None, low_bnd=None, verbose=False, epsilon=1, max_retries=4, lay=None, ucln=None, nqb=None, norm=None):
	#Penser à modifier epsilon, on peut ptet tenter epsilon 0.01 ou encore la penalty
	global init_time
	init_time = pctime()
	loc_par0 = par0
	xopt = loc_par0  # Initialize xopt with initial parameters
	res = nan
	relaunch = True
	retry_count = 0
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
		f_psi = frc @ stt
		return f_psi
	
	while retry_count < max_retries:
		relaunch = False  # Reset relaunch flag at the start of each attempt
		opt = ini_opt(loc_par0, min_obj, low_bnd=low_bnd, verbose=verbose)
		opt.set_upper_bounds(up_bnd)

		# Fonction combinée : objectif principal + pénalité MSE
		def combined_objective(x, grad):
			# Calcul de l'objectif principal
			min_obj_val = min_obj(x)
			#MSE(ucln * (5**nqb), norm(get_Stt_lazy(x)) * (5**nqb))
			# Calcul de la contrainte MSE
			STT=get_Stt_lazy(x)
			#print(len(STT))
			STT=norm(STT[:n_nodes])
			normalisation=norm_q_fromstt(STT)
			mse_val = MSE(ucl, (STT)*normalisation) #modifier ça si changement

			# Si MSE dépasse epsilon, ajouter une pénalité, mais de manière progressive
			penalty = lambda_penalty * max(0, (100-mse_val) - epsilon) ** 2

			# L'objectif combiné est l'objectif principal + pénalité
			return min_obj_val + penalty
		
		ans_stt = gen_ans(nqb, lay)
	
		def get_Stt_lazy(par):
			""" Input : parameters. Output : statevector"""
			return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))
		
		# Définir la fonction objectif combinée
		opt.set_min_objective(combined_objective)
		

		try:
			print("Initial combined objective:", combined_objective(loc_par0, None))
			
			xopt = opt.optimize(loc_par0)
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			traceback.print_exc()
			xopt = loc_par0  # If exception occurs, restart from the original parameters
		
		res = opt.last_optimum_value()
		nl_res = opt.last_optimize_result()
		normalisation=norm_q_fromstt(norm(get_Stt_lazy(xopt)[:n_nodes]))

		# Calculate the final MSE value
		#print("normalisation=",normalisation)
		#print((get_Stt_lazy(xopt))*normalisation)
		#print(ucl)
		mse_val_final = MSE(ucl, (norm(get_Stt_lazy(xopt)[:n_nodes]))*normalisation) #Modifier ça si changement
		print(f"MSE value: {mse_val_final}")

		# Check the MSE condition
		if abs(mse_val_final) > 95:  # Modify this threshold as needed (0.05 to 0.1 could be reasonable)
			print(f"MSE is {mse_val_final}, which is less than or equal to 0.001 . Stopping optimization.")
			break  # Stop the optimization without relaunching

		# Check optimization results and handle different cases
		if nl_res == nlopt.SUCCESS:
			print("Generic success return value.")
		elif nl_res == nlopt.STOPVAL_REACHED:
			print(f"Stopped as an objective value of at least stopval={opt.get_Stopval()} was found.")
		elif nl_res == nlopt.FTOL_REACHED:
			print(f"Optimization stopped because ftol_rel={opt.get_ftol_rel()} or ftol_abs={opt.get_ftol_abs()} was reached.")
			# Relaunch with the previous best parameters (xopt) rather than random perturbation
			print("Retrying optimization with adjusted parameters...")
			retry_count += 1
			loc_par0 = xopt  # Use the latest optimized parameters (xopt) for the next iteration
			relaunch = True  # Set relaunch flag to true if we need to retry
		elif nl_res == nlopt.XTOL_REACHED:
			print(f"Optimization stopped because xtol_rel={opt.get_xtol_rel()} or xtol_abs={opt.get_xtol_abs()} was reached.")
		elif nl_res == nlopt.MAXEVAL_REACHED:
			print(f"Optimization stopped because maxeval={opt.get_maxeval()} was reached.")
		elif nl_res == nlopt.MAXTIME_REACHED:
			print(f"Optimization stopped because maxtime={opt.get_maxtime()} was reached.")
		elif nl_res < 0:
			print('Errored stoppage:', nl_res)
			retry_count += 1
			loc_par0 = xopt  # Use the latest optimized parameters (xopt) for the next iteration
			relaunch = True
		else:
			print("Unknown stoppage case.")
		
		print()

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res

import cma

def opt_res_cma7(par0, min_obj, Hsp, frc, H, ucl, n_nodes,low_bnd=None, verbose=False, epsilon=0.01,lambda_penalty=2e2, max_retries=4,lay=None, ucln=None, nqb=None, norm=None,sigma0=0.1):

	global init_time
	init_time = pctime()
	loc_par0 = np.array(par0)
	xopt = loc_par0
	res = nan
	retry_count = 0

	# Generate the ansatz
	ans_stt = gen_ans(nqb, lay)

	def get_Stt_lazy(par):
		return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))

	def psiHpsi_fromstt(stt):
		return stt @ H @ stt

	def fpsi_fromstt(stt):
		return frc @ stt

	def norm_q_fromstt(stt):
		return fpsi_fromstt(stt) / psiHpsi_fromstt(stt)

	def combined_objective(x):
		stt = get_Stt_lazy(x)[:n_nodes]
		normalisation = norm_q_fromstt(stt)
		mse_val = MSE(ucl, stt * normalisation)
		penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2
		min_obj_val = min_obj(x)
		return min_obj_val + penalty

	while retry_count < max_retries:
		try:
			options = {
				'maxiter': 1000,
				'tolx': 1e-6,
				'tolfun': 1e-6,
				'verbose': int(verbose),
			}
			if low_bnd is not None:
				options['bounds'] = [low_bnd, -np.array(low_bnd)]
		except Exception as e:
			print("EXCEPTION ENCOUNTERED", e)
			xopt = loc_par0
			retry_count += 1

		es = cma.CMAEvolutionStrategy(loc_par0, sigma0, options)
		xopt, res = es.optimize(combined_objective).result[:2]

		# Check MSE constraint
		stt = get_Stt_lazy(xopt)[:n_nodes]
		mse_val_final = MSE(ucl, stt * norm_q_fromstt(stt))
		print(f"MSE value: {mse_val_final}")

		if abs(mse_val_final) <= 0.001:
			 print(f"MSE is {mse_val_final}, which is within tolerance. Stopping optimization.")
			 break
		else:
			 print(f"MSE {mse_val_final} above threshold, retrying...")
			 loc_par0 = xopt + np.random.uniform(-0.001, 0.001, size=len(par0))
			 retry_count += 1

		

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res

def opt_res_cma8(par0, min_obj, Hsp, frc, H, ucl, n_nodes,low_bnd=None, verbose=False, epsilon=0.01,lambda_penalty=5e1, max_generations=300,lay=None, ucln=None, nqb=None, norm=None,sigma0=0.42):

	global init_time
	init_time = pctime()
	loc_par0 = np.array(par0)
	dim = len(loc_par0)

	# Generate the ansatz
	ans_stt = gen_ans(nqb, lay)

	def get_Stt_lazy(par):
		return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))

	def psiHpsi_fromstt(stt):
		return stt @ H @ stt

	def fpsi_fromstt(stt):
		return frc @ stt

	def norm_q_fromstt(stt):
		return fpsi_fromstt(stt) / psiHpsi_fromstt(stt)

	def combined_objective(x):
		stt = get_Stt_lazy(x)[:n_nodes]
		normalisation = norm_q_fromstt(stt)
		mse_val = MSE(ucl, stt * normalisation)
		penalty = lambda_penalty * max(0, mse_val - epsilon) ** 2
		min_obj_val = min_obj(x)
		return min_obj_val + penalty

	# CMA-ES options
	options = {
		'verbose': -9 if not verbose else 1,
		'tolx': 1e-8,
		'tolfun': 1e-8,
		'popsize': 8 + int(3 * np.log(dim)),
		'maxiter': max_generations
	}
	if low_bnd is not None:
		options['bounds'] = [low_bnd, -np.array(low_bnd)]

	# Launch optimizer
	es = cma.CMAEvolutionStrategy(loc_par0, sigma0, options)
	xopt = loc_par0
	res = nan

	for gen in range(max_generations):
		X = es.ask()
		fitnesses = [combined_objective(x) for x in X]
		es.tell(X, fitnesses)
		xopt = es.result.xbest
		res = es.result.fbest

		# Evaluate constraint
		stt = get_Stt_lazy(xopt)[:n_nodes]
		mse_val = MSE(ucl, stt * norm_q_fromstt(stt))

		if verbose:
			print(f"[Gen {gen}] Best obj: {res:.6f}, MSE: {mse_val:.4e}")

		if mse_val <= epsilon:
			print(f"MSE constraint satisfied: {mse_val:.4e} ≤ {epsilon}.")
			break

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res


def opt_res_cmaenergy(par0, min_obj, low_bnd=None,up_bnd=None, verbose=False,
							max_generations=300, sigma0=0.42, nqb=None, lay=None):

	global init_time
	init_time = pctime()
	loc_par0 = np.array(par0)
	dim = len(loc_par0)

	# Generate the ansatz
	ans_stt = gen_ans(nqb, lay)

	def get_Stt_lazy(par):
		return np.array(sampleGate(ans_stt.assign_parameters(par, inplace=False)))

	# CMA-ES options
	options = {
		'verbose': -9 if not verbose else 1,
		'tolx': 1e-8,
		'tolfun': 1e-8,
		'popsize': 8 + int(3 * np.log(dim)),
		'maxiter': max_generations
	}
	if low_bnd is not None:
		options['bounds'] = [low_bnd, up_bnd]

	# Launch optimizer
	es = cma.CMAEvolutionStrategy(loc_par0, sigma0, options)
	initial_fitness = min_obj(loc_par0)
	es.inject([loc_par0])
	xopt = loc_par0
	res = nan

	for gen in range(max_generations):
		X = es.ask()
		if gen == 0:
			X[0] = loc_par0
		fitnesses = [min_obj(x) for x in X]
		es.tell(X, fitnesses)
		xopt = es.result.xbest
		res = es.result.fbest

		if verbose:
			print(f"[Gen {gen}] Best energy: {res:.6f}")

		# Optional: early stopping if energy stops improving (e.g., no change for N generations)

	total_duration = pctime() - init_time
	print(f"{res}\nIn {total_duration:.3f} seconds.")
	return xopt, res