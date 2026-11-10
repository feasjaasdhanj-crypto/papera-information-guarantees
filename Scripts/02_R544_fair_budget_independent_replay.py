#!/usr/bin/env python3
"""Paper A independent reconstruction from R543 packaged R542/R540 source archives.
Every policy sees identical train and validation subjects, ranks all same-budget masks in training,
shortlists same k candidates, and uses identical validation labels to select once.
Final evaluation: per-heldout-subject balanced class risk, no test-based selection.
"""
from pathlib import Path
import io,zipfile,gzip,hashlib,json,time
import numpy as np,pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

BASE=Path('/mnt/data');OUT=BASE/'pa_r544_work'; OUT.mkdir(exist_ok=True)
T=time.time(); ROOT=BASE/'PaperA_R543_当前完整科研交接_公平候选预算与复现断点_20261103.zip'
with zipfile.ZipFile(ROOT) as parent:
 r542=zipfile.ZipFile(io.BytesIO(parent.read('00_PARENT/PaperA_R542_COMPLETE_ORIGINAL.zip')))
 r540=zipfile.ZipFile(io.BytesIO(r542.read('00_PARENT/R540_COMPLETE_ORIGINAL.zip')))
 eeg=zipfile.ZipFile(io.BytesIO(r540.read('01_INPUT/RETURN_TO_CHATGPT_EEG_R539.zip')))
 names=sorted(n for n in eeg.namelist() if n.endswith('.csv.gz'))
 D=pd.concat([pd.read_csv(io.BytesIO(eeg.read(n)),compression='gzip') for n in names],ignore_index=True)
 assert len(D)==9536 and D.subject.nunique()==106
 def getdf(n):return pd.read_csv(io.BytesIO(r540.read(n)))
 info={
  'calibrated':getdf('02_EVIDENCE_CALIBRATED/R540_EEG_secondary_5fold_information.csv'),
  'global_train_only':getdf('03_EVIDENCE_RAW_SENSITIVITY/R540_EEG_secondary_5fold_information.csv')}
 stored={
  'calibrated':getdf('02_EVIDENCE_CALIBRATED/R540_EEG_secondary_5fold_1280portfolios.csv'),
  'global_train_only':getdf('03_EVIDENCE_RAW_SENSITIVITY/R540_EEG_secondary_5fold_1280portfolios.csv')}
 fold_map=getdf('02_EVIDENCE_CALIBRATED/R540_EEG_secondary_5fold_subject_map.csv')
 archive_sha=hashlib.sha256(ROOT.read_bytes()).hexdigest()
E=['FC5','FC3','FC1','C5','C3','C1','CP5','CP3'];R=['FC6','FC4','FC2','C6','C4','C2','CP6','CP4'];B=['theta','alpha','beta']
cols=[e+'_'+b for e in E+R for b in B];Y=D.state.map({'T0':0,'T1':1,'T2':2}).to_numpy();S=D.subject.to_numpy();IDs={c:i for i,c in enumerate(cols)}
loss={
 'balanced':np.array([[0,1,1],[1,0,1],[1,1,0.]],float),
 'miss_T2_x3':np.array([[0,1,1],[1,0,1],[3,3,0.]],float),
 'false_alarm_T2_x3':np.array([[0,1,3],[1,0,3],[1,1,0.]],float),
 'reject035':np.array([[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]],float)}
folds=[fold_map.loc[fold_map.fold==f,'subject'].to_numpy() for f in range(5)]
assert set(S)==set(fold_map.subject) and len(set(fold_map.subject))==106
contracts=list(loss)
all_policies=[];all_person=[];all_masks=[];validation_checks=[]
# save all person risks only for full chosen masks; enough to reproduce each policy
for process in ('calibrated','global_train_only'):
 X=D[cols].copy()
 if process=='calibrated':
  gr=X.groupby(D.subject)
  X=(X-gr.transform('mean'))/gr.transform('std').clip(lower=.05)
 X=X.to_numpy(dtype=float)
 for fold in range(5):
  testsubs=folds[fold]; valsubs=folds[(fold+1)%5]; trainsubs=sorted(set(S)-set(testsubs)-set(valsubs))
  tr=np.isin(S,trainsubs); va=np.isin(S,valsubs); te=np.isin(S,testsubs)
  mu=X[tr].mean(axis=0);sd=X[tr].std(axis=0);sd[sd<.01]=1; xx=(X-mu)/sd
  expected=stored[process].query('fold == @fold').set_index('mask')
  metric=info[process].query('fold == @fold').set_index('index')
  mask_train={};mask_person={};mask_v={};
  maxcheck=0.
  for mask in range(256):
   select=[R[j] if (mask>>j)&1 else E[j] for j in range(8)]
   ci=[IDs[e+'_'+b] for e in select for b in B]
   model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[1/3]*3).fit(xx[tr][:,ci],Y[tr])
   subject_d={}
   for sp,ii,subgroup in [('train',tr,trainsubs),('validation',va,valsubs),('test',te,testsubs)]:
    probs=model.predict_proba(xx[ii][:,ci]);acty=Y[ii];acts=S[ii];
    # each subject contributes one balanced average over states
    subject_d[sp]={}
    for key,lmat in loss.items():
     actions=np.argmin(probs@lmat,axis=1)
     values=[]
     for sub in subgroup:
      cond=acts==sub
      v=np.mean([lmat[acty[cond & (acty==cls)],actions[cond & (acty==cls)]].mean() for cls in range(3)])
      values.append(v)
     subject_d[sp][key]=np.asarray(values)
   train={key:np.mean(subject_d['train'][key]) for key in contracts}
   v={key:np.mean(subject_d['validation'][key]) for key in contracts}
   t={key:np.mean(subject_d['test'][key]) for key in contracts}
   mask_train[mask]=train;mask_person[mask]=subject_d['test'];mask_v[mask]=v
   for key in contracts:
    maxcheck=max(maxcheck,abs(v[key]-float(expected.loc[mask,'val_'+key])),abs(t[key]-float(expected.loc[mask,'test_'+key])))
   all_masks.append({'process':process,'fold':fold,'mask':mask,'budget':mask.bit_count(),'train_'+contracts[0]:train[contracts[0]],'train_'+contracts[1]:train[contracts[1]],'train_'+contracts[2]:train[contracts[2]],'train_'+contracts[3]:train[contracts[3]],**{'val_'+c:v[c] for c in contracts}})
  validation_checks.append({'process':process,'fold':fold,'archive_max_abs_difference':maxcheck})
  assert maxcheck <1e-10, (process,fold,maxcheck)
  for budget in (2,4,6):
   available=[m for m in range(256) if m.bit_count()==budget]
   full_ranks=sorted(available,key=lambda m:(sum(metric.loc[j,'full_delta'] for j in range(8) if (m>>j)&1),m))
   local_ranks=sorted(available,key=lambda m:(sum(metric.loc[j,'local_r2'] for j in range(8) if (m>>j)&1),m))
   for contract in contracts:
    target_ranks=sorted(available,key=lambda m:(mask_train[m][contract],m))
    for k in (5,10):
     for pol,ranks in [('target',target_ranks),('full',full_ranks),('local',local_ranks)]:
      shortlist=ranks[:k]
      selected=min(shortlist,key=lambda m:(mask_v[m][contract],m))
      all_policies.append({'processing':process,'fold':fold,'budget':budget,'contract':contract,'k':k,'policy':pol,'selected_mask':selected,'validation_risk':mask_v[selected][contract],'test_risk':np.mean(mask_person[selected][contract]),'train_risk':mask_train[selected][contract],'train_rank':ranks.index(selected)+1,'n_training_subjects':len(trainsubs),'n_validation_subjects':len(valsubs),'n_testing_subjects':len(testsubs)})
      for sub,rr in zip(testsubs,mask_person[selected][contract]):
       all_person.append({'processing':process,'fold':fold,'subject':sub,'budget':budget,'contract':contract,'k':k,'policy':pol,'selected_mask':selected,'risk':rr})
  print('DONE',process,'fold',fold,'archive_abs_err',maxcheck,'seconds',round(time.time()-T,1),flush=True)

P=pd.DataFrame(all_policies);Q=pd.DataFrame(all_person);M=pd.DataFrame(all_masks);V=pd.DataFrame(validation_checks)
P.to_csv(OUT/'R544_equal_candidate_policy_choices.csv',index=False)
Q.to_csv(OUT/'R544_equal_candidate_heldout_subject_risks.csv',index=False)
M.to_csv(OUT/'R544_allmask_training_risks.csv',index=False)
V.to_csv(OUT/'R544_archive_replay_QA.csv',index=False)

wide=Q.pivot_table(index=['processing','subject','budget','contract','k'],columns='policy',values='risk').reset_index()
assert wide.groupby(['processing','budget','contract','k']).subject.nunique().eq(106).all()
res=[];bootstrap_rng=np.random.default_rng(544106)
for (process,contract,k),g in wide.groupby(['processing','contract','k'],sort=True):
 grouped=g.groupby('subject')[['target','full','local']].mean().sort_index()
 assert len(grouped)==106
 for ref in ('full','local'):
  vals=(grouped['target']-grouped[ref]).to_numpy()
  boot=vals[bootstrap_rng.integers(len(vals),size=(20000,len(vals)))].mean(axis=1)
  p=float(ttest_1samp(vals,0,alternative='less').pvalue)
  res.append({'processing':process,'contract':contract,'k':k,'reference':ref,'mean_target_minus_ref':vals.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'one_sided_t_p':p,'subjects_better':int((vals<0).sum()),'subjects_worse':int((vals>0).sum()),'subjects_tie':int((vals==0).sum()),'n_subjects':106})
A=pd.DataFrame(res)
for (proc,ref),df in A.groupby(['processing','reference']):
 ix=df.index
 A.loc[ix,'Holm_p_8tasks']=multipletests(df.one_sided_t_p,method='holm')[1]
A.to_csv(OUT/'R544_fair_candidate_subject_bootstrap_Holm.csv',index=False)
C=P.pivot_table(index=['processing','fold','contract','budget','k'],columns='policy',values='test_risk').reset_index()
for ref in ('full','local'): C['target_minus_'+ref]=C.target-C[ref]
C.to_csv(OUT/'R544_fair_candidate_5fold_comparisons.csv',index=False)
print('SUMMARY false alarms:')
print(A[A.contract=='false_alarm_T2_x3'].to_string(index=False))
print('SOURCE_R543_REFERENCE',archive_sha,'runtime',round(time.time()-T,1))
