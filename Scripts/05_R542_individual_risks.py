"""Reproduce locked R540 5-fold models for selected R542 validation-tuned masks.
Preprocessing, train subject folds, loss matrices and classifier are identical to R540.
Provides one held-out risk per person/method/budget/decision, without choosing on test labels.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
T=time.time();O=Path('/mnt/data/pa_r542_analysis')
P=pd.read_csv(O/'R542_equal_supervision_policy_choices.csv')
D=pd.read_pickle('/mnt/data/pa_r540_work/EEG_features_extracted.pkl')
E=['FC5','FC3','FC1','C5','C3','C1','CP5','CP3'];R=['FC6','FC4','FC2','C6','C4','C2','CP6','CP4'];B=['theta','alpha','beta']
cols=[e+'_'+b for e in E+R for b in B]
Y=D.state.map({'T0':0,'T1':1,'T2':2}).to_numpy();S=D.subject.to_numpy();ids={c:i for i,c in enumerate(cols)}
sub=np.array(sorted(np.unique(S))); rng=np.random.default_rng(540539);folds=np.array_split(rng.permutation(sub),5)
loss={
 'balanced':np.array([[0,1,1],[1,0,1],[1,1,0]],float),
 'miss_T2_x3':np.array([[0,1,1],[1,0,1],[3,3,0]],float),
 'false_alarm_T2_x3':np.array([[0,1,3],[1,0,3],[1,1,0]],float),
 'reject035':np.array([[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]],float)}
rows=[];checks=[]
for proc in ('calibrated','global_train_only'):
 X=D[cols].copy()
 if proc=='calibrated':
  X=(X-X.groupby(D.subject).transform('mean'))/X.groupby(D.subject).transform('std').clip(lower=.05)
 X=X.to_numpy()
 for fold in range(5):
  testsub=folds[fold];valsub=folds[(fold+1)%5];trainsub=sorted(set(sub)-set(testsub)-set(valsub))
  tr=np.isin(S,trainsub);te=np.isin(S,testsub)
  m=X[tr].mean(axis=0);sd=X[tr].std(axis=0);sd[sd<.01]=1; xx=(X-m)/sd
  specs=P[(P.processing==proc)&(P.fold==fold)]
  group=specs[['mask','contract']].drop_duplicates(); cc={int(mask):set(group[group['mask']==mask]['contract']) for mask in sorted(group['mask'].unique())}
  subsort=sorted(testsub)
  for mask,lkset in cc.items():
   sel=[R[j] if mask>>j&1 else E[j] for j in range(8)]
   ci=[ids[e+'_'+b] for e in sel for b in B]
   model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[1/3]*3).fit(xx[tr][:,ci],Y[tr])
   pp=model.predict_proba(xx[te][:,ci]);y=Y[te];ss=S[te]
   for lk in lkset:
    l=loss[lk]; action=np.argmin(pp@l,axis=1)
    risk=np.array([np.mean([l[y[(ss==s)&(y==c)],action[(ss==s)&(y==c)]].mean() for c in range(3)]) for s in subsort])
    for s,r in zip(subsort,risk):rows.append({'processing':proc,'fold':fold,'subject':s,'mask':mask,'contract':lk,'risk':float(r)})
   # verify all mask heldout risk is same as archived mean
   archival=pd.read_csv('/mnt/data/pa_r540_work/'+('R540_EEG_RESULTS' if proc=='calibrated' else 'R540_EEG_RAW_RESULTS')+'/R540_EEG_secondary_5fold_1280portfolios.csv')
   row=archival[(archival.fold==fold)&(archival['mask']==mask)].iloc[0]
   for lk in lkset:
    computed=np.mean([r['risk'] for r in rows if r['processing']==proc and r['fold']==fold and r['mask']==mask and r['contract']==lk]); expected=float(row['test_'+lk]);
    checks.append(abs(computed-expected))
  print('RECONSTRUCT',proc,fold,'models',len(cc),'elapsed',round(time.time()-T,1),flush=True)
A=pd.DataFrame(rows);assert max(checks)<1e-10,(max(checks),np.percentile(checks,90))
A.to_csv(O/'R542_all_selected_policies_subject_risks.csv',index=False)
# Expand to policy level, each of same subject's outcomes repeated over budgets to allow direct within-subject averaging.
long=P.merge(A,on=['processing','fold','mask','contract'],validate='many_to_many')
assert long.groupby(['processing','budget','contract','policy']).subject.nunique().eq(106).all()
long.to_csv(O/'R542_equal_supervision_policy_subject_risks.csv',index=False)
print('DONE rows',len(long),'max archive risk delta',max(checks),'seconds',round(time.time()-T,1))
