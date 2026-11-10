"""Use frozen per-fold validation subject-risk matrices to test sample-label budgets, not independent replication."""
from pathlib import Path
import numpy as np,pandas as pd
O=Path('/mnt/data/pa_r542_analysis');src=Path('/mnt/data/pa_r540_work')
rng=np.random.default_rng(5420503);rows=[]
for processing,folder in [('calibrated','R540_EEG_RESULTS'),('global_train_only','R540_EEG_RAW_RESULTS')]:
 archive=pd.read_csv(src/folder/'R540_EEG_secondary_5fold_1280portfolios.csv')
 information=pd.read_csv(src/folder/'R540_EEG_secondary_5fold_information.csv')
 for fold in range(5):
  with np.load(O/f'R542_validation_{processing}_fold{fold}.npz') as f:val={k:f[k] for k in f.files}
  info=information.query('fold==@fold').set_index('index');ar=archive.query('fold==@fold').set_index('mask');K=val['balanced'].shape[1]
  assert K in [21,22]
  for budget in [2,4,6]:
   feasible=np.array([mask for mask in range(256) if mask.bit_count()==budget])
   score={m:np.array([sum(info.loc[j,'full_delta' if m=='full' else 'local_r2'] for j in range(8) if mask>>j&1) for mask in feasible]) for m in ('full','local')}
   top={k:feasible[np.lexsort((feasible,score[k]))[:10]] for k in score}
   for n in (5,10,K):
    nrepeat=30 if n<K else 1
    for rep in range(nrepeat):
     chosen=rng.choice(K,n,replace=False) if n<K else np.arange(K)
     for contract in val:
      V=val[contract][:,chosen].mean(axis=1)
      config={'target_all_val':feasible[np.lexsort((feasible,V[feasible]))][0]}
      for met in ('full','local'):
       cand=top[met];config[met+'_top10_val']=cand[np.lexsort((cand,V[cand]))][0]
      for pol,mask in config.items():
       rows.append({'processing':processing,'fold':fold,'budget':budget,'contract':contract,'n_validation_subjects':n,'replicate':rep,'policy':pol,'mask':int(mask),'heldout_risk':float(ar.loc[mask,'test_'+contract]),'n_test_subjects':22 if fold==0 else 21})
P=pd.DataFrame(rows);P.to_csv(O/'R542_validation_label_budget_selection.csv',index=False)
W=P.pivot(index=['processing','fold','budget','contract','n_validation_subjects','replicate','n_test_subjects'],columns='policy',values='heldout_risk').reset_index()
for ref in ('full_top10_val','local_top10_val'):W['diff_vs_'+ref]=W.target_all_val-W[ref]
W.to_csv(O/'R542_validation_label_budget_paired.csv',index=False)
TT=[]
for (proc,budget,n,ref),g in W.melt(id_vars=['processing','fold','budget','contract','n_validation_subjects','replicate','n_test_subjects'],value_vars=['diff_vs_full_top10_val','diff_vs_local_top10_val'],var_name='reference',value_name='diff').query("contract=='false_alarm_T2_x3'").groupby(['processing','budget','n_validation_subjects','reference']):
 TT.append({'processing':proc,'budget':budget,'n_validation_subjects':n,'reference':ref,'average_target_minus_reference':np.average(g['diff'],weights=g.n_test_subjects),'split_fraction_target_better':(g['diff']<0).mean(),'n_repeated_selections_not_independent':len(g)})
T=pd.DataFrame(TT);T.to_csv(O/'R542_validation_label_budget_summary.csv',index=False)
print('Calibrated high false alarm, standard 21/22 validation subjects for comparison')
print(T.query("processing=='calibrated'").to_string(index=False))
print('Aggregate across three budgets, same subset labels in each method')
print(T.groupby(['processing','n_validation_subjects','reference']).average_target_minus_reference.mean().to_string())
