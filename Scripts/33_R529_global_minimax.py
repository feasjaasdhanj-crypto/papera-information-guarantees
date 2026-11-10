"""R529 global minimax portfolio allocation; binary MILP, four non-equivalent fixed cost scenarios.
Solve min_{x_i in {0,1},sum x_i=K} max_s (sum_i x_i norm_training_loss_{i,s})/K.
The 'per-item max' heuristic is a separate, potentially suboptimal allocation rule.
"""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
p=Path(sys.argv[1]);data=pd.read_csv(p/'R529_five_settings_1008_tasks.csv')
heuristic=pd.read_csv(p/'R529_3budgets_8policies_5settings.csv')
contracts=['01_balanced_0_1','02_third_state_cost3','03_reject_cost035','05_third_state_false_alarm_cost3']
all_scenarios=['01_balanced_0_1','02_third_state_cost3','03_reject_cost035','04_third_state_prior060','05_third_state_false_alarm_cost3']
rows=[];summary=[]
for (ds,rep),g0 in data.groupby(['dataset','repeat_index'],sort=True):
 g=g0.reset_index(drop=True);n=len(g)
 norms=[]
 for task in contracts:
  v=g['train_signed_'+task].to_numpy(float)
  rng=np.ptp(v)
  norms.append((v-v.min())/rng if rng>1e-12 else np.zeros(n))
 M=np.asarray(norms)
 for budget in [.25,.5,.75]:
  K=max(1,round(n*budget))
  # [M/K,-1] @ [binary x,continuous t] <= 0; sum(x)==K
  A=np.vstack([np.column_stack([M.T.sum(axis=0)*0.,np.zeros(n)])]) if False else None
  inequalities=np.concatenate([M/K,-np.ones((M.shape[0],1))],axis=1)
  equality=np.append(np.ones(n),0)[None,:]
  obj=np.append(np.zeros(n),1.)
  fit=milp(c=obj,integrality=np.append(np.ones(n),0),bounds=Bounds(np.zeros(n+1),np.ones(n+1)),
   constraints=[LinearConstraint(inequalities,-np.inf,0.),LinearConstraint(equality,K,K)],
   options={'time_limit':15.,'mip_rel_gap':1e-9})
  if fit.status!=0 or fit.x is None:raise RuntimeError(('MILP failed',ds,rep,budget,fit.status,fit.message))
  x=(fit.x[:n]>0.5);assert sum(x)==K
  chosen=g.loc[x]
  hand=heuristic[(heuristic.dataset==ds)&(heuristic.split==rep)&(heuristic.budget_fraction==budget)&(heuristic.policy=='robust_family_minimax')].iloc[0]
  hand_ids=[int(k) for k in hand.selected_signature.split(';')]
  hand_max=np.max(M[:,hand_ids].mean(axis=1))
  opt_max=np.max(M[:,x].mean(axis=1))
  assert opt_max<=hand_max+2e-7
  row=dict(dataset=ds,split=rep,budget_fraction=budget,N=n,K=K,
   train_minimax_optimal=opt_max,train_peritem_worst_heuristic=hand_max,
   train_optimality_gain=hand_max-opt_max,
   integer_solver_gap=float(getattr(fit,'mip_gap',0.)),
   selected_signature=';'.join(map(str,np.where(x)[0])),test_delta=chosen.test_Delta.mean())
  for task in all_scenarios:
   v=chosen['deploy_test_signed_'+task]
   row['test_deployed_signed_'+task]=v.mean()
   row['test_deployed_positive_'+task]=np.maximum(v,0).mean()
  rows.append(row)
  full=heuristic[(heuristic.dataset==ds)&(heuristic.split==rep)&(heuristic.budget_fraction==budget)&(heuristic.policy=='full_Delta')].iloc[0]
  for task in all_scenarios:
   summary.append(dict(dataset=ds,split=rep,budget_fraction=budget,task=task,
    global_minimax_minus_full=row['test_deployed_signed_'+task]-full['test_deployed_signed_'+task],
    global_minimax_minus_greedy=row['test_deployed_signed_'+task]-hand['test_deployed_signed_'+task],
    global_minimax_minus_task_opt=row['test_deployed_signed_'+task]-heuristic[(heuristic.dataset==ds)&(heuristic.split==rep)&(heuristic.budget_fraction==budget)&(heuristic.policy==task)].iloc[0]['test_deployed_signed_'+task]))
res=pd.DataFrame(rows);res.to_csv(p/'R529_global_minimax_168_portfolios.csv',index=False)
comp=pd.DataFrame(summary);comp.to_csv(p/'R529_global_minimax_840_outcomes.csv',index=False)
print('SOLVED',len(res),'portfolios', 'max_solver_gap',res.integer_solver_gap.max())
print('TRAIN GLOBAL IMPROVEMENT',int((res.train_optimality_gain>1e-8).sum()),'of',len(res),'mean',res.train_optimality_gain.mean(),'max',res.train_optimality_gain.max())
print('MEANS (negative means better than reference)')
print(comp.groupby('budget_fraction')[['global_minimax_minus_full','global_minimax_minus_greedy','global_minimax_minus_task_opt']].mean().round(6).to_string())
print('SOURCE-LEVEL count better global than full')
print(comp.groupby(['dataset','budget_fraction','task'])['global_minimax_minus_full'].mean().groupby('budget_fraction').agg(lambda x:int((x<0).sum())).to_string())
report={'n_solved':len(res),'n_output_scenario_comparisons':len(comp),'n_true_non_equivalent_scenarios':len(contracts),
 'num_training_strict_global_better_than_itemwise':int((res.train_optimality_gain>1e-8).sum()),
 'max_solver_mip_gap':float(res.integer_solver_gap.max()),
 'global_minimax_better_than_full_source_scenarios_at_each_budget':{str(frac):int((grp.groupby(['dataset','task']).global_minimax_minus_full.mean()<0).sum()) for frac,grp in comp.groupby('budget_fraction')}}
(p/'R529_global_minimax_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
