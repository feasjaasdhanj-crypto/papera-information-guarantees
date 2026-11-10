from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
O=Path('/mnt/data/pa_r542_analysis');A=pd.read_csv(O/'R542_equal_supervision_policy_subject_risks.csv')
wide=A.pivot(index=['processing','subject','budget','contract'],columns='policy',values='risk').reset_index()
assert len(wide)==2*106*3*4
rng=np.random.default_rng(5422026);B=25000
rows=[];pvals=[]
refs=['full_info_only','local_info_only','full_local_val_pick','full_top5_val','full_top10_val','local_top5_val','local_top10_val']
for (proc,contract),d in wide.groupby(['processing','contract']):
 for ref in refs:
  # Each person is one unit; repeat K values averaged within person first
  v=(d.target_all_val-d[ref]).groupby(d.subject).mean().values
  assert len(v)==106
  b=v[rng.integers(0,len(v),size=(B,len(v)))].mean(axis=1)
  p_less=ttest_1samp(v,0,alternative='less').pvalue
  p_greater=ttest_1samp(v,0,alternative='greater').pvalue
  rows.append({'processing':proc,'contract':contract,'reference':ref,'N_subjects':len(v),'n_budgets_averaged':3,
   'mean_target_minus_ref':v.mean(),'CI_2p5':np.quantile(b,.025),'CI_97p5':np.quantile(b,.975),
   'n_subjects_target_better':int(np.sum(v<-1e-10)),'n_subjects_target_worse':int(np.sum(v>1e-10)),
   'one_sided_t_p_improvement':p_less,'one_sided_t_p_worsening':p_greater,'normalized_mean_diff':v.mean()/(3 if 'x3' in contract else 1)})
T=pd.DataFrame(rows)
# Multiplicity correction among 4 decision contracts within each comparator, preprocessing.
for (proc,ref),idx in T.groupby(['processing','reference']).groups.items():
 idx=list(idx)
 T.loc[idx,'p_Holm_improvement_four_contracts']=multipletests(T.loc[idx,'one_sided_t_p_improvement'],method='holm')[1]
 T.loc[idx,'p_Holm_worsening_four_contracts']=multipletests(T.loc[idx,'one_sided_t_p_worsening'],method='holm')[1]
T.to_csv(O/'R542_subject_level_bootstrap_Holm_56.csv',index=False)
# fold LOO and exact signed contrast of 5 folded average
fr=[]
for (proc,contract),df in wide.groupby(['processing','contract']):
 for ref in refs:
  dd=df.assign(diff=df.target_all_val-df[ref]);x=dd.groupby('subject')['diff'].mean();fold=pd.read_csv('/mnt/data/pa_r540_work/R540_EEG_RESULTS/R540_EEG_secondary_5fold_subject_map.csv').set_index('subject').loc[x.index]['fold'];fmeans=x.groupby(fold.values).mean();
  for f in range(5):
   left=x[fold.values!=f]
   fr.append({'processing':proc,'contract':contract,'reference':ref,'held_out_fold':f,'leave_one_fold_out_mean':left.mean(),'n_left_subjects':len(left),'five_fold_mean':x.mean(),'folds_improve':int((fmeans<0).sum()),'folds_worse':int((fmeans>0).sum())})
F=pd.DataFrame(fr);F.to_csv(O/'R542_fold_robustness.csv',index=False)
print('CORE, calibrated:')
print(T.query("processing=='calibrated' and reference in ['full_info_only','full_local_val_pick','full_top5_val','full_top10_val','local_top10_val']")[['contract','reference','mean_target_minus_ref','CI_2p5','CI_97p5','p_Holm_improvement_four_contracts','p_Holm_worsening_four_contracts','n_subjects_target_better','n_subjects_target_worse']].to_string(index=False))
print('FOLD robustness false_alarm calibrated')
print(F.query("processing=='calibrated' and contract=='false_alarm_T2_x3'").groupby('reference').agg(min_LOFO=('leave_one_fold_out_mean','min'),max_LOFO=('leave_one_fold_out_mean','max'),folds_improve=('folds_improve','first')).to_string())
print('count Holm sig improvement/worsening calibrated',len(T.query("processing=='calibrated' and p_Holm_improvement_four_contracts<0.05")),len(T.query("processing=='calibrated' and p_Holm_worsening_four_contracts<0.05")))
