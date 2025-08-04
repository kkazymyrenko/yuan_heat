# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 10:27:48 2025

@author: roron
"""

import pandas as pd
import os
from tools.fct import accuracy
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio



pio.renderers.default = "browser"
#This part depends on how you called your files and where you put them.
parent_dir = "3D_15qb"



if "Rx" in parent_dir or "2" in parent_dir:
	RxRy=True
else:
	RxRy=False
print(RxRy)

if RxRy==False:
	filename= "results_15qb_20layers.csv"#"results_6qb_4layers.csv"
if RxRy==True : 
	filename= "results_15qb_20layers.csv"#"results_6qb_2layers.csv"

def load_all_runs(parent_dir, filename=filename):
	runs = {}
	for run_name in sorted(os.listdir(parent_dir)):
		run_path = os.path.join(parent_dir, run_name)
		if os.path.isdir(run_path):
			file_path = os.path.join(run_path, filename)
			if os.path.exists(file_path):
				df = pd.read_csv(file_path)
				runs[run_name] = df
	return runs

#The functions are pretty much obvious in how they work and their output

def plot_observable(runs, observable="Qcost", ref_col="ref", percentiles=(25, 75)):

	# Get union of all Wait values
	all_waits = sorted(set().union(*[df["Wait"] for df in runs.values()]))
	df_common = pd.DataFrame({"Wait": all_waits})
	
	interpolated = []
	for df in runs.values():
		df_interp = pd.merge(df_common, df[["Wait", observable]], on="Wait", how="left")
		# Fill missing values by propagating last known value
		df_interp[observable] = df_interp[observable].ffill()
		# If there are still NaNs at the start, fill them with first known value
		df_interp[observable] = df_interp[observable].bfill()
		interpolated.append(df_interp[observable])

	data_matrix = np.array(interpolated, dtype=np.float64)
	mean_vals = np.mean(data_matrix, axis=0)
	p_low, p_high = np.percentile(data_matrix, percentiles, axis=0)

	# Plot
	fig = go.Figure()

	# Raw runs
	for run_name, df in runs.items():
		fig.add_trace(go.Scatter(x=df["Wait"], y=df[observable],
								 mode='lines', name=run_name,
								 line=dict(width=1), opacity=0.4))

	# Reference line
	# Reference line (extend to all_waits)
	first_run = next(iter(runs.values()))
	if ref_col in first_run.columns:
		ref_interp = pd.merge(df_common, first_run[["Wait", ref_col]], on="Wait", how="left")
		ref_interp[ref_col] = ref_interp[ref_col].ffill().bfill()
		fig.add_trace(go.Scatter(x=all_waits, y=ref_interp[ref_col],
								 mode='lines', name="Reference",
								 line=dict(color='black', dash='dash')))


	# Mean
	fig.add_trace(go.Scatter(x=all_waits, y=mean_vals, name="Mean",
							 line=dict(color="blue", width=3)))

	# Percentile area
	fig.add_trace(go.Scatter(
		x=all_waits + all_waits[::-1],
		y=np.concatenate([p_low, p_high[::-1]]),
		fill='toself',
		fillcolor='rgba(50,205,50,0.5)',
		line=dict(color='rgba(255,255,255,0)'),
		hoverinfo="skip",
		name=f"{percentiles[0]}-{percentiles[1]} Percentile"
	))

	fig.update_layout(title=f"{observable} over Wait, {parent_dir}",
					  xaxis_title="Wait",
					  yaxis_title=observable,
					  template="plotly_white")
	fig.show()

def plot_accuracy(runs, observable="Qcost", ref_col="ref", accuracy_fn=None, percentiles=(25, 75)):
	# Step 1: Gather all possible Wait values
	all_waits = sorted(set().union(*[df["Wait"] for df in runs.values()]))
	df_common = pd.DataFrame({"Wait": all_waits})

	acc_curves = []
	best_run_acc = None
	best_run_name = ""
	best_final_acc = -np.inf

	for name, df in runs.items():
		acc = [accuracy_fn(p, r) for p, r in zip(df[observable], df[ref_col])]
		df_acc = pd.DataFrame({"Wait": df["Wait"], "acc": acc})
		df_interp = pd.merge(df_common, df_acc, on="Wait", how="left")
		df_interp["acc"] = df_interp["acc"].ffill().bfill()
		acc_values = df_interp["acc"].values
		acc_curves.append(acc_values)

		if not np.isnan(acc_values[-1]) and acc_values[-1] > best_final_acc:
			best_final_acc = acc_values[-1]
			best_run_acc = acc_values
			best_run_name = name

	acc_curves = np.array(acc_curves, dtype=np.float64)
	mean_acc = np.mean(acc_curves, axis=0)
	p_low, p_high = np.percentile(acc_curves, percentiles, axis=0)

	fig = go.Figure()

	# All runs (gray lines)
# 	for acc in acc_curves:
# 		fig.add_trace(go.Scatter(x=all_waits, y=acc,
# 								 mode='lines',
# 								 line=dict(width=1, color='gray'),
# 								 opacity=0.3,
# 								 showlegend=False))

	# Mean
	fig.add_trace(go.Scatter(x=all_waits, y=mean_acc,
							 name="Mean Accuracy",
							 line=dict(color="blue", width=3)))

	# Best run in red
	if best_run_acc is not None:
		fig.add_trace(go.Scatter(x=all_waits, y=best_run_acc,
								 name=f"Best Run: {best_run_name}",
								 line=dict(color="red", width=3)))

	# Percentile shading
	fig.add_trace(go.Scatter(
		x=all_waits + all_waits[::-1],
		y=np.concatenate([p_low, p_high[::-1]]),
		fill='toself',
		fillcolor='rgba(50,205,50,0.3)',
		line=dict(color='rgba(255,255,255,0)'),
		hoverinfo="skip",
		name=f"{percentiles[0]}-{percentiles[1]} Percentile"
	))

	fig.update_layout(title=f"Accuracy of {observable} over Wait, {parent_dir}",
					  xaxis_title="Wait",
					  yaxis_title="Accuracy",
					  template="plotly_white")
	fig.show()




def summarize_runs(runs, observable="Qcost", ref_col="ref", accuracy_fn=None):
	summary = []

	for name, df in runs.items():
		# Remove NaNs in Wait/observable/ref
		df = df.dropna(subset=["Wait", observable, ref_col])

		final_val = df[observable].values[-1]
		final_wait = df["Wait"].values[-1]
		final_ref = df[ref_col].values[-1]
		final_acc = accuracy_fn(final_val, final_ref) if accuracy_fn else None

		accs = [accuracy_fn(p, r) for p, r in zip(df[observable], df[ref_col])] if accuracy_fn else [None]*len(df)
		accs = np.array(accs)
		max_acc = np.max(accs) if accuracy_fn else None
		min_acc = np.min(accs) if accuracy_fn else None

		summary.append({
			"name": name,
			"final_obs": final_val,
			"final_acc": final_acc,
			"final_wait": final_wait,
			"max_acc": max_acc,
			"min_acc": min_acc,
		})

	summary_df = pd.DataFrame(summary)

	idx_max_final_acc = summary_df['final_acc'].idxmax()
	idx_min_final_acc = summary_df['final_acc'].idxmin()

	max_wait = summary_df['final_wait'].max()
	min_wait = summary_df['final_wait'].min()

	print("\n=== Summary Statistics ===")
	print(f"Mean Final Observable: {summary_df['final_obs'].mean():.4f} ± {summary_df['final_obs'].std():.4f}")
	print(f"Mean Final Accuracy:	{summary_df['final_acc'].mean():.4f} ± {summary_df['final_acc'].std():.4f}")
	print(f"Mean Final Wait Time:   {summary_df['final_wait'].mean():.2f} ± {summary_df['final_wait'].std():.2f}")

	print(f"Max Final Accuracy:  {summary_df.loc[idx_max_final_acc, 'final_acc']:.4f} "
		  f"from run '{summary_df.loc[idx_max_final_acc, 'name']}' "
		  f"at Wait = {summary_df.loc[idx_max_final_acc, 'final_wait']}")

	print(f"Min Final Accuracy:  {summary_df.loc[idx_min_final_acc, 'final_acc']:.4f} "
		  f"from run '{summary_df.loc[idx_min_final_acc, 'name']}' "
		  f"at Wait = {summary_df.loc[idx_min_final_acc, 'final_wait']}")

	print(f"Max Final Wait Time across runs: {max_wait}")
	print(f"Min Final Wait Time across runs: {min_wait}")

	return summary_df


runs = load_all_runs(parent_dir)
#summary_df = summarize_runs(runs, observable="Qcost", ref_col="ref", accuracy_fn=accuracy)
summary_df = summarize_runs(runs, observable="Qcost", ref_col="ref", accuracy_fn=accuracy)

#plot_observable(runs, observable="Qcost", ref_col="ref", percentiles=(25, 75))
#plot_accuracy(runs, observable="Qcost", ref_col="ref", accuracy_fn=accuracy, percentiles=(10, 90))




