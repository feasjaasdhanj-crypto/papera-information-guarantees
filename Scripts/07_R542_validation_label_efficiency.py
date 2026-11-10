"""Exploratory validation-label budget: same subject-level validation information for methods.
Trained LDA models and test risks locked as in R540; only validation ranking
is recomputed from subsets of 5, 10, or all 21/22 validation participants.
Repeated subset selections are NOT independent data sources or tests.
"""
from pathlib import Path
import numpy as np,pandas as pd,time,os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
T=time.time();O=Path('/mnt/data/pa_r542_analysis');src=Path('/mnt/data/pa_r540_work')
D=pd.read_pickle(src/'EEG_features_extracted.pkl')
E=['FC5','FC3','FC1','C5','C3','C1','CP5','CP3'];R=['FC6','FC4','FC2','C6','C4','C2','CP6','CP4'];B=['theta','alpha','beta']
cols=[e+'_'+b for e in E+R for b in B];idx={c:i for i,c in enumerate(cols)}
y=D.state.map({'T0':0,'T1':1,'T2':2}).to_numpy();s=D.subject.to_numpy();allsub=np.array(sorted(np.unique(s)));rng0=np.random.default_rng(540539);folds=np.array_split(rng0.permutation(allsub),5)
loss={'balanced':np.array([[0,1,1],[1,0,1],[1,1,0]],float),'miss_T2_x3':np.array([[0,1,1],[1,0,1],[3,3,0]],float),'false_alarm_T2_x3':np.array([[0,1,3],[1,0,3],[1,1,0]],float),'reject035':np.array([[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]],float)}
fullrecords=[];val_by_fold={};data_by_fold={}
for processing,folder in [('calibrated','R540_EEG_RESULTS'),('global_train_only','R540_EEG_RAW_RESULTS')]:
 x0=D[cols].copy()
 if processing=='calibrated': x0=(x0-x0.groupby(D.subject).transform('mean'))/x0.groupby(D.subject).transform('std').clip(lower=.05)
 x0=x0.to_numpy()
 archive=pd.read_csv(src/folder/'R540_EEG_secondary_5fold_1280portfolios.csv')
 information=pd.read_csv(src/folder/'R540_EEG_secondary_5fold_information.csv')
 for fold in range(5):
  if processing!=os.getenv('PROCESS') or fold!=int(os.getenv('FOLD','-1')): continue
  te_sub=folds[fold]; val_sub=sorted(folds[(fold+1)%5]);tr_sub=sorted(set(allsub)-set(te_sub)-set(val_sub))
  tr=np.isin(s,tr_sub);va=np.isin(s,val_sub)
  mu=x0[tr].mean(axis=0);std=x0[tr].std(axis=0);std[std<.01]=1;x=(x0-mu)/std
  S=s[va];Y=y[va]
  v={lk:np.empty((256,len(val_sub))) for lk in loss}
  for mask in range(256):
   electrodes=[R[j] if mask>>j&1 else E[j] for j in range(8)]
   ci=[idx[e+'_'+b] for e in electrodes for b in B]
   model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[1/3]*3).fit(x[tr][:,ci],y[tr])
   prob=model.predict_proba(x[va][:,ci])
   for lk,L in loss.items():
    act=np.argmin(prob@L,axis=1)
    v[lk][mask,:]=np.array([np.mean([L[Y[(S==sub)&(Y==c)],act[(S==sub)&(Y==c)]].mean() for c in range(3)]) for sub in val_sub])
   check=np.array([v[lk][mask,:].mean()-archive.query('fold==@fold and mask==@mask').iloc[0]['val_'+lk] for lk in loss]);assert np.max(np.abs(check))<1e-10
  np.savez_compressed(O/f'R542_validation_{processing}_fold{fold}.npz',**v)
  val_by_fold[(processing,fold)]=v
  data_by_fold[(processing,fold)]={'archive':archive.query('fold==@fold').set_index('mask'),'info':information.query('fold==@fold').set_index('index').sort_index(),'valsub':val_sub,'test_n':len(te_sub)}
  print('VALIDATION REBUILT',processing,fold,'subjects',len(val_sub),'time',round(time.time()-T,1),flush=True)
