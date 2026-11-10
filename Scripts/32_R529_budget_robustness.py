"""R529: multiple fixed equal-unit approval budgets and a predeclared multi-contract robust decision policy.
This uses no test outcomes to choose budgets, policies or hyperparameters. Test outcomes are diagnostic only.
"""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
P=Path(sys.argv[1]);p=P/'R529_five_settings_1008_tasks.csv'; d=pd.read_csv(p)
contracts=['01_balanced_0_1','02_third_state_cost3','03_reject_cost035','04_third_state_prior060','05_third_state_false_alarm_cost3']
unique_contracts=['01_balanced_0_1','02_third_state_cost3','03_reject_cost035','05_third_state_false_alarm_cost3']
budgets=[.25,.50,.75]
records=[]
for (ds,rep),g0 in d.groupby(['dataset','repeat_index'],sort=True):
 g=g0.copy().reset_index(drop=True)
 norms=[]
 for contract in unique_contracts:
  v=g['train_signed_'+contract];a=v.max()-v.min()
  norms.append((v-v.min())/a if a>1e-12 else pd.Series(0.,index=g.index))
 g['robust_minimax_norm']=pd.concat(norms,axis=1).max(axis=1)
 methods={'full_Delta':'train_Delta','local_r2':'train_r2','robust_family_minimax':'robust_minimax_norm'}
 methods.update({x:'train_signed_'+x for x in contracts})
 for frac in budgets:
  K=max(1,round(len(g)*frac))
  for policy,score in methods.items():
   g['_sortscore']=g[score].round(12)
   a=g.sort_values(['_sortscore','states','streamA','feature'],kind='stable').head(K)
   r={'dataset':ds,'split':rep,'budget_fraction':frac,'K':K,'N':len(g),'policy':policy,
   'selected_signature':';'.join(str(i) for i in sorted(a.index)),
   'test_delta':a.test_Delta.mean(),'train_delta':a.train_Delta.mean()}
   for task in contracts:
    v=a['deploy_test_signed_'+task]
    r['test_deployed_signed_'+task]=v.mean()
    r['test_deployed_positive_'+task]=np.maximum(v,0.).mean()
   records.append(r)
res=pd.DataFrame(records);res.to_csv(P/'R529_3budgets_8policies_5settings.csv',index=False)
rows=[]
for (ds,rep,frac),g in res.groupby(['dataset','split','budget_fraction'],sort=True):
 g=g.set_index('policy')
 for task in contracts:
  col='test_deployed_signed_'+task
  rows.append({'dataset':ds,'split':rep,'budget_fraction':frac,'K':int(g.K.iloc[0]),'task':task,
  'matched_minus_full':g.loc[task,col]-g.loc['full_Delta',col],
  'robust_minus_full':g.loc['robust_family_minimax',col]-g.loc['full_Delta',col],
  'matched_minus_robust':g.loc[task,col]-g.loc['robust_family_minimax',col],
  'matched_vs_full_changed':int(g.loc[task,'selected_signature']!=g.loc['full_Delta','selected_signature']),
  'robust_vs_full_changed':int(g.loc['robust_family_minimax','selected_signature']!=g.loc['full_Delta','selected_signature'])})
res2=pd.DataFrame(rows);res2.to_csv(P/'R529_budget_840_target_comparisons.csv',index=False)
summary=res2.groupby(['dataset','budget_fraction','task']).agg(splits=('split','count'),matched_improvement=('matched_minus_full','mean'),robust_improvement=('robust_minus_full','mean'),matched_vs_full_changed=('matched_vs_full_changed','sum')).reset_index()
summary.to_csv(P/'R529_budget_source_summary.csv',index=False)
report={}
for frac,x in res2.groupby('budget_fraction'):
 means=x.groupby(['dataset','task'])[['matched_minus_full','robust_minus_full']].mean()
 report[str(frac)]={'n_paired_comparisons':len(x),'n_source_setting_pairs_better_matched':int((means.matched_minus_full<0).sum()),'total_source_setting_pairs':len(means),
 'n_source_setting_pairs_better_robust':int((means.robust_minus_full<0).sum()),
 'avg_matched_minus_full':float(x.matched_minus_full.mean()),'avg_robust_minus_full':float(x.robust_minus_full.mean())}
 (P/'R529_budget_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('BUDGET RESULT',json.dumps(report,ensure_ascii=False,indent=2))
print('ALL SOURCE AND BUDGET MEANS')
print(res2.groupby(['dataset','budget_fraction'])[['matched_minus_full','robust_minus_full']].mean().round(6).to_string())
# 50% budget replay equivalence with the original single-K selection, including fixed trained decision outcomes
old=pd.read_csv(P/'R529_56split_7policies_5decision_outcomes.csv')
joined=res[res.budget_fraction==.5].merge(old,on=['dataset','split','policy'],suffixes=('','_old'),validate='one_to_one')
assert len(joined)==len(old)
for task in contracts:
 assert np.allclose(joined['test_deployed_signed_'+task],joined['deploy_test_signed_'+task],atol=1e-12)
assert (joined.selected_signature==joined.selected_signature_old).all()
print('50% REPRODUCED',len(joined),'rows exact selection signatures and fixed-decision risk')
