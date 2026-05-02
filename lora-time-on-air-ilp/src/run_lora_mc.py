import sys, os, json, time, math, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mod_path = '/mnt/data/lora_ilp_experiment_outputs/lora_ilp_experiment.py'
spec = importlib.util.spec_from_file_location('loraexp', mod_path)
lora = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lora)

outdir = Path('/mnt/data/lora_mc_evaluation')
outdir.mkdir(exist_ok=True)

Ns_reps = {50:30, 100:30, 200:30, 500:10}
rows=[]
representatives=[]
seed_base = 20260502
start_all=time.perf_counter()
for N, reps in Ns_reps.items():
    for r in range(reps):
        seed = seed_base + N*1000 + r*37
        inst = lora.make_instance_with_retry(n=N, seed_start=seed, max_tries=300)
        greedy = lora.greedy_assignment(inst, arrival_seed=seed+17)
        # if greedy fails, regenerate a little; for fair paired comparison keep only complete greedy instances
        tries=0
        while not greedy['feasible_all'] and tries<50:
            seed += 1
            inst = lora.make_instance_with_retry(n=N, seed_start=seed, max_tries=300)
            greedy = lora.greedy_assignment(inst, arrival_seed=seed+17)
            tries+=1
        ilp = lora.solve_ilp(inst, time_limit_seconds=120.0)
        if not (greedy['feasible_all'] and ilp['success']):
            rows.append({'N':N,'rep':r,'seed':seed,'success':False,'greedy_complete':greedy['feasible_all'],'ilp_success':ilp['success']})
            continue
        gt = greedy['total_toa']
        it = ilp['objective']
        imp = 100*(gt-it)/gt if gt>0 else np.nan
        changed = int(np.sum(greedy['assigned_idx'] != ilp['assigned_idx']))
        rows.append({
            'N':N,'rep':r,'seed':seed,'success':True,
            'greedy_total_toa_s':gt,'ilp_total_toa_s':it,
            'improvement_pct':imp,
            'ilp_solve_time_s':ilp['solve_time_seconds'],
            'ilp_variables':ilp['num_variables'],'ilp_constraints':ilp['num_constraints'],
            'changed_nodes':changed,'changed_frac':changed/N
        })
        # store representatives for N=100
        if N==100:
            representatives.append((imp, seed, inst, greedy, ilp))

elapsed_all=time.perf_counter()-start_all
results=pd.DataFrame(rows)
results.to_csv(outdir/'monte_carlo_results.csv', index=False)
valid=results[results['success']].copy()

# Stats by N and overall
summary_rows=[]
for label, group in [('all', valid)] + [(str(N), valid[valid['N']==N]) for N in sorted(Ns_reps)]:
    if len(group)==0: continue
    vals=group['improvement_pct'].to_numpy()
    st=group['ilp_solve_time_s'].to_numpy()
    mean=vals.mean(); sd=vals.std(ddof=1); se=sd/math.sqrt(len(vals)); ci=1.96*se
    med=np.median(vals); q1=np.percentile(vals,25); q3=np.percentile(vals,75)
    summary_rows.append({
        'N':label,'trials':len(vals),
        'improvement_mean_pct':mean,
        'improvement_sd_pct':sd,
        'improvement_95ci_halfwidth_pct':ci,
        'improvement_median_pct':med,
        'improvement_q1_pct':q1,
        'improvement_q3_pct':q3,
        'solve_time_mean_s':st.mean(),
        'solve_time_sd_s':st.std(ddof=1) if len(st)>1 else 0,
        'changed_frac_mean':group['changed_frac'].mean()
    })
summary=pd.DataFrame(summary_rows)
summary.to_csv(outdir/'monte_carlo_summary.csv', index=False)

# Representative N=100 instance: closest to N=100 mean improvement
n100_mean=summary[summary['N']=='100']['improvement_mean_pct'].iloc[0]
rep = min(representatives, key=lambda tup: abs(tup[0]-n100_mean))
rep_imp, rep_seed, rep_inst, rep_greedy, rep_ilp = rep
rep_df = lora.assignment_dataframe(rep_inst, rep_greedy, rep_ilp)
rep_df.to_csv(outdir/'representative_N100_assignments.csv', index=False)

# Transition matrix for representative
sf_list=rep_inst['sf_list']; S=len(sf_list)
mat=np.zeros((S,S), dtype=int)
for i in range(rep_inst['n']):
    g=int(rep_greedy['assigned_idx'][i]); o=int(rep_ilp['assigned_idx'][i])
    if g>=0 and o>=0: mat[g,o]+=1
pd.DataFrame(mat, index=[f'SF{s}' for s in sf_list], columns=[f'SF{s}' for s in sf_list]).to_csv(outdir/'representative_transition_matrix.csv')

# Create 2x2 compact figure
fig, axes = plt.subplots(2,2, figsize=(7.2,5.7))
ax=axes[0,0]
data=[valid[valid['N']==N]['improvement_pct'].to_numpy() for N in sorted(Ns_reps)]
ax.boxplot(data, tick_labels=[str(N) for N in sorted(Ns_reps)], showmeans=True, meanline=True)
ax.set_xlabel('Number of nodes')
ax.set_ylabel('ToA reduction (%)')
ax.set_title('(a) Improvement distribution')
ax.grid(True, axis='y', alpha=0.25)

ax=axes[0,1]
im=ax.imshow(mat, aspect='auto')
ax.set_xticks(np.arange(S)); ax.set_yticks(np.arange(S))
ax.set_xticklabels([f'SF{s}' for s in sf_list], fontsize=8)
ax.set_yticklabels([f'SF{s}' for s in sf_list], fontsize=8)
ax.set_xlabel('ILP SF')
ax.set_ylabel('Greedy SF')
ax.set_title(f'(b) Transition heatmap (N=100)')
for r in range(S):
    for c in range(S):
        ax.text(c,r,str(int(mat[r,c])),ha='center',va='center',fontsize=7)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax=axes[1,0]
assigned_sf=np.array([sf_list[j] for j in rep_ilp['assigned_idx']])
ax.scatter(rep_inst['distances_km'], assigned_sf, s=11, alpha=0.75)
ax.set_xlabel('Distance (km)')
ax.set_ylabel('ILP assigned SF')
ax.set_yticks(sf_list)
ax.set_title('(c) Distance vs assigned SF')
ax.grid(True, alpha=0.25)

ax=axes[1,1]
run_summary=valid.groupby('N').agg(solve_mean=('ilp_solve_time_s','mean'), solve_sd=('ilp_solve_time_s','std')).reset_index()
ax.errorbar(run_summary['N'], run_summary['solve_mean'], yerr=run_summary['solve_sd'], marker='o', capsize=3)
ax.set_xlabel('Number of nodes')
ax.set_ylabel('Solve time (s)')
ax.set_title('(d) Runtime scaling')
ax.grid(True, alpha=0.25)

fig.suptitle('Monte Carlo evaluation of regulatory-aware ILP', y=0.995, fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97])
fig_path=outdir/'fig_mc_compact_evaluation.png'
fig.savefig(fig_path, dpi=300)
plt.close(fig)

# Create a small table figure? Maybe not. Summary via LaTeX text.
# System info
sysinfo = lora.get_system_info()
sysinfo['mc_elapsed_wall_time_s']=elapsed_all
sysinfo['total_valid_trials']=len(valid)
sysinfo['trial_design']=Ns_reps
with open(outdir/'mc_system_info.json','w') as f: json.dump(sysinfo,f,indent=2)

# TeX snippet
overall=summary[summary['N']=='all'].iloc[0].to_dict()
tex = f"""
% Monte Carlo summary generated automatically.
The Monte Carlo study used {int(overall['trials'])} feasible paired instances over $N\in\{{50,100,200,500\}}$ nodes. Across these instances, the ILP reduced total planned uplink ToA by {overall['improvement_mean_pct']:.1f}\% on average, with standard deviation {overall['improvement_sd_pct']:.1f}\% and an approximate 95\% confidence interval of {overall['improvement_mean_pct']-overall['improvement_95ci_halfwidth_pct']:.1f}\%--{overall['improvement_mean_pct']+overall['improvement_95ci_halfwidth_pct']:.1f}\%. The median reduction was {overall['improvement_median_pct']:.1f}\%.
"""
(outdir/'mc_summary_sentence.tex').write_text(tex)

print('elapsed', elapsed_all)
print(summary.to_string(index=False))
print('representative improvement', rep_imp, 'seed', rep_seed)
print('fig', fig_path)
