"""R531 real physical sensor portfolio test: fixed 8 physical channels at all times.
The eight base sensors are #1..8, with eight one-for-one replacement options #9..16.
Budget K=2,4,6 means EXACTLY K physical substitutions; not 20x8 independent
procurement decisions. We assess 20 3-gas scientific tasks on 1-sensor marginal
measurements per task. This is not a joint 8-sensor classifier.
"""
import sys,itertools,json,time
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(sys.argv[1]);tab=pd.read_csv(OUT/'R531_gas_800_tasks.csv');UNIQUE=['balanced','miss3','reject035','falsealarm3']
result=[];comp=[];exhaustive=[]
for split,g in tab.groupby('split',sort=False):
 assert len(g)==160
 agg=g.groupby('sensor_A').agg({**{'train_Delta':'mean','train_r2':'mean','test_Delta':'mean'},**{'train_signed_'+x:'mean' for x in UNIQUE},**{'deploy_signed_'+x:'mean' for x in UNIQUE}}).reset_index()
 assert list(agg.sensor_A)==list(range(1,9))
 v=np.column_stack([agg['train_signed_'+x].values for x in UNIQUE]);norm=np.zeros_like(v)
 for j in range(4):
  rng=np.ptp(v[:,j]);norm[:,j]=(v[:,j]-v[:,j].min())/rng if rng>1e-12 else 0
 scores={'full_Delta':agg.train_Delta.values,'local_r2':agg.train_r2.values,'robust_individual':norm.max(axis=1)}
 scores.update({'match_'+x:agg['train_signed_'+x].values for x in UNIQUE})
 for K in (2,4,6):
  choices={name:np.lexsort((np.arange(8),np.round(score,12)))[:K] for name,score in scores.items()}
  # exhaustive combinatorial exact minimax over C(8,K) feasible physical substitutions
  allcombs=list(itertools.combinations(range(8),K))
  objs=[norm[list(ids)].mean(axis=0).max() for ids in allcombs]
  best=min(range(len(allcombs)),key=lambda i:(round(objs[i],12),allcombs[i]))
  choices['global_minimax']=np.asarray(allcombs[best])
  exhaustive.append({'split':split,'K_physical_swaps':K,'candidate_portfolios':len(allcombs),
     'global_worst_normalized_train':float(objs[best]),'individual_worst_normalized_train':float(norm[choices['robust_individual']].mean(axis=0).max()),
     'strict_gain_train':float(norm[choices['robust_individual']].mean(axis=0).max()-objs[best])})
  for pol,ids in choices.items():
   s=agg.iloc[ids]
   r={'split':split,'physical_sensor_count_fixed':8,'K_physical_swaps':K,'policy':pol,'swapped_sensor_pairs':';'.join(f'{i}->{i+8}' for i in (s.sensor_A.astype(int))),
      'mean_train_Delta':float(s.train_Delta.mean()),'mean_test_Delta':float(s.test_Delta.mean()),
      'mean_train_r2':float(s.train_r2.mean())}
   for c in UNIQUE:
    # average change per physical slot out of 8 (others unchanged), across 20 decision tasks.
    r['train_portfolio_change_'+c]=float(s['train_signed_'+c].sum()/8)
    r['test_portfolio_change_'+c]=float(s['deploy_signed_'+c].sum()/8)
   result.append(r)
  bypol={r['policy']:r for r in result if r['split']==split and r['K_physical_swaps']==K}
  for c in UNIQUE:
   base=bypol['full_Delta']['test_portfolio_change_'+c]
   for pol in ['match_'+c,'local_r2','robust_individual','global_minimax']:
    x=bypol[pol]['test_portfolio_change_'+c]-base
    comp.append({'split':split,'physical_swaps':K,'contract':c,'policy':pol,
          'matched_minus_full':x,'better':int(x< -1e-10),'equal':int(abs(x)<=1e-10),'worse':int(x>1e-10)})
rr=pd.DataFrame(result);cc=pd.DataFrame(comp);ee=pd.DataFrame(exhaustive)
assert len(rr)==5*3*8 and len(cc)==5*3*4*4 and len(ee)==15
rr.to_csv(OUT/'R531_physical_120_portfolios.csv',index=False)
cc.to_csv(OUT/'R531_physical_240_comparisons.csv',index=False)
ee.to_csv(OUT/'R531_physical_15_exhaustive_certificates.csv',index=False)
main=cc[(cc.split=='official_chronological_1to9_vs10')&(cc.policy.str.startswith('match_'))]
print('PHYSICAL PRIMARY')
print(main[['physical_swaps','contract','matched_minus_full','better','worse']].to_string(index=False))
print('\nPHYSICAL ALL TIME SPLITS')
print(cc[cc.policy.str.startswith('match_')].groupby('split').agg(better=('better','sum'),worse=('worse','sum'),equal=('equal','sum'),mean=('matched_minus_full','mean')).to_string())
print('GLOBAL improvements',ee.strict_gain_train.gt(1e-10).sum(),'out of',len(ee),'exhaustive portfolios',ee.candidate_portfolios.sum())
