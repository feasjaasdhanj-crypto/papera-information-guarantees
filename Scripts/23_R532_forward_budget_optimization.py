"""R532 reproducible time-forward decisions, derived from unchanged R531 task-level train/validation/test files.
For each t=7..10 train=1..(t-2), validation=t-1, test=t. No test outcomes enter policy decisions.
'At most K' means 0..K replacements with deterministic negative training/validation score.
All portfolio losses are calculated on fixed panel denominator (160 task units or 8 sensor slots).
Validation sensor scores are an empirical calibration requiring labels in preceding batch, not future labels.
"""
from pathlib import Path
import numpy as np,pandas as pd,json,hashlib,sys
OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=True,parents=True)
parent=OUT/'parent'/'03_实证结果'
F=pd.read_csv(parent/'R531_prequential_640_task_channels.csv'); P=pd.read_csv(parent/'R531_prequential_96_policy_checks.csv')
assert len(F)==640 and len(P)==96
C=['balanced','miss3','reject035','falsealarm3']
STRATS=['no_swap','full_fixed','target_fixed','validation_fixed','target_optional','validation_optional','blend_optional','agreement_optional','buffer05_optional','buffer10_optional','buffer20_optional','robust_worst_optional','extrapolate_optional','oracle_optional']
rows=[];sels=[]
for t in (7,8,9,10):
 spl=f'train1to{t-2}_val{t-1}_test{t}'
 g0=F[F.split==spl].reset_index(drop=True)
 assert len(g0)==160
 for level in ('task','physical'):
  if level=='task':
   g=g0;N=160
  else:
   g=g0.groupby('sensor_A',sort=True).mean(numeric_only=True).reset_index();N=8
  for fraction in (.25,.5,.75):
   K=round(N*fraction)
   for contract in C:
    tr=g[f'train_signed_{contract}'].to_numpy();v=g[f'val_signed_{contract}'].to_numpy();te=g[f'test_signed_{contract}'].to_numpy();full=g.train_Delta.to_numpy()
    scores={
      'full_fixed':full,
      'target_fixed':tr,
      'validation_fixed':v,
      'target_optional':tr,
      'validation_optional':v,
      'blend_optional':.5*tr+.5*v,
      'agreement_optional':np.where((tr<0)&(v<0),.5*(tr+v),np.inf),
      'buffer05_optional':v+.5*np.abs(v-tr),
      'buffer10_optional':v+np.abs(v-tr),
      'buffer20_optional':v+2*np.abs(v-tr),
      'robust_worst_optional':np.maximum(tr,v),
      'extrapolate_optional':2*v-tr,
      'oracle_optional':te,
    }
    allpol={'no_swap':np.array([],dtype=int)}
    for policy,score in scores.items():
     # round only for deterministic ordering, not for treatment qualification
     order=np.lexsort((np.arange(N),np.round(score,12)))
     if policy.endswith('_fixed'):
      choose=order[:K]
     else:
      choose=order[score[order]< -1e-12][:K]
     allpol[policy]=np.asarray(choose,dtype=int)
    # Check initial R531 aggregation if same design
    target_fixed_risk=float(te[allpol['target_fixed']].sum()/N)
    row=P[(P.split==spl)&(P.level==level)&(P.budget==fraction)&(P.contract==contract)]
    assert len(row)==1
    base_fixed_risk=float(te[allpol['full_fixed']].sum()/N)
    matched_fixed_risk=float(te[allpol['target_fixed']].sum()/N)
    if level=='physical':
     assert np.isclose(matched_fixed_risk-base_fixed_risk,float(row.test_target_minus_full.iloc[0]),atol=1e-12)
    else:
     assert np.isclose((matched_fixed_risk-base_fixed_risk)*N/K,float(row.test_target_minus_full.iloc[0]),atol=1e-12)
    for policy,ids in allpol.items():
     row={'split':spl,'t':t,'train_to':t-2,'validation_t':t-1,'test_t':t,'level':level,'contract':contract,
          'budget':fraction,'K_max':K,'N_units':N,'policy':policy,'K_used':len(ids),
          'test_risk_change_panel':float(te[ids].sum()/N),
          'train_risk_change_panel':float(tr[ids].sum()/N),
          'validation_risk_change_panel':float(v[ids].sum()/N),
          'mean_selected_test_delta':float(te[ids].mean()) if len(ids) else 0.,
          'selected_ids':';'.join(map(str,sorted(ids)))}
     rows.append(row)
     for idx in ids:
      sels.append({'split':spl,'level':level,'contract':contract,'budget':fraction,'policy':policy,'idx':int(idx),
                   'train_delta':float(tr[idx]),'val_delta':float(v[idx]),'test_delta':float(te[idx])})
R=pd.DataFrame(rows);sel=pd.DataFrame(sels)
assert len(R)==4*2*3*4*len(STRATS)
assert set(R.policy)==set(STRATS)
R.to_csv(OUT/'R532_forward_1344_policy_outcomes.csv',index=False)
sel.to_csv(OUT/'R532_selected_decisions.csv',index=False)
# compare to fixed baseline and best available as 0 / no swap
V=R.pivot_table(index=['split','t','level','contract','budget','K_max','N_units'],columns='policy',values=['test_risk_change_panel','K_used']).reset_index()
V.columns=['_'.join(str(x) for x in k if x!='') for k in V.columns.to_flat_index()]
for pol in STRATS:
 if pol=='oracle_optional':continue
 V['gain_over_full_'+pol]=V['test_risk_change_panel_full_fixed']-V['test_risk_change_panel_'+pol]
 V['gain_over_target_'+pol]=V['test_risk_change_panel_target_fixed']-V['test_risk_change_panel_'+pol]
 V['improves_vs_no_swap_'+pol]=V['test_risk_change_panel_'+pol]<-1e-10
V.to_csv(OUT/'R532_96_scenarios_comparison.csv',index=False)
summ=[]
for level in ('task','physical'):
 for tt,gv in [('all_4',V[V.level==level]),('early_7to9',V[(V.level==level)&(V.t<10)]),('final_10',V[(V.level==level)&(V.t==10)])]:
  for pol in STRATS:
   q=gv['test_risk_change_panel_'+pol].to_numpy()
   base=gv['test_risk_change_panel_full_fixed'].to_numpy()
   targ=gv['test_risk_change_panel_target_fixed'].to_numpy()
   summ.append({'level':level,'evaluation':tt,'policy':pol,'n_scenarios':len(gv),
       'mean_panel_risk_change':float(q.mean()),'mean_K_used':float(gv['K_used_'+pol].mean()),
       'better_than_no_swap':int((q<-1e-10).sum()),'worse_than_no_swap':int((q>1e-10).sum()),
       'better_than_full_fixed':int((q<base-1e-10).sum()),'worse_than_full_fixed':int((q>base+1e-10).sum()),
       'better_than_target_fixed':int((q<targ-1e-10).sum()),'worse_than_target_fixed':int((q>targ+1e-10).sum())})
S=pd.DataFrame(summ);S.to_csv(OUT/'R532_policy_summary.csv',index=False)
# Prospectively choose exactly one algorithm family based only on t7..9 aggregated over contracts and budget
choice=[]
for level in ('task','physical'):
 previous=S[(S.level==level)&(S.evaluation=='early_7to9')&(~S.policy.isin(['oracle_optional']))]
 winner=previous.sort_values(['mean_panel_risk_change','policy']).iloc[0]['policy']
 for tt in ['early_7to9','final_10']:
  r=S[(S.level==level)&(S.evaluation==tt)&(S.policy==winner)].iloc[0].to_dict()
  choice.append({'level':level,'previous_batch_policy_select':winner,'evaluated_on':tt,**r})
pd.DataFrame(choice).to_csv(OUT/'R532_early_only_policy_selection.csv',index=False)
with open(OUT/'R532_reconstruction_check.json','w') as f:
 json.dump({'input_file':'R531_prequential_640_task_channels.csv','input_sha256':hashlib.sha256((parent/'R531_prequential_640_task_channels.csv').read_bytes()).hexdigest(),
           'n_prequential_cases':len(V),'n_method_records':len(R),'n_selected_records':len(sel),
           't_values':[7,8,9,10], 'no_test_selection_leak':True},f,ensure_ascii=False,indent=2)
print('\nPOLICY SUMMARY (all folds):')
print(S[S.evaluation=='all_4'].sort_values(['level','mean_panel_risk_change'])[['level','policy','mean_panel_risk_change','better_than_no_swap','worse_than_no_swap','mean_K_used']].to_string(index=False))
print('\nEARLY FROZEN CHOICE:');print(pd.DataFrame(choice).to_string(index=False))
