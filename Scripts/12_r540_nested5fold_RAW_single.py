"""R540 secondary 5-fold subject OOF for fixed EEG physical-channel design.
Each test subject appears once. Validation fold is next disjoint fold; training is other three folds.
All parameters and method choices set before 5-fold outcomes. Exploratory analysis, not a preregistered study.
"""
import io,hashlib,json,itertools,time,os
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
T=time.time();OUT=Path('/mnt/data/pa_r540_work/R540_EEG_RAW_RESULTS');OUT.mkdir(exist_ok=True)
D=pd.read_pickle('/mnt/data/pa_r540_work/EEG_features_extracted.pkl')
E=['FC5','FC3','FC1','C5','C3','C1','CP5','CP3'];R=['FC6','FC4','FC2','C6','C4','C2','CP6','CP4'];BANDS=['theta','alpha','beta']
cols=[e+'_'+b for e in E+R for b in BANDS];Y=D.state.map({'T0':0,'T1':1,'T2':2}).to_numpy();S=D.subject.to_numpy()
X=D[cols].copy() # raw only; no subject-specific transductive normalization
X=X.to_numpy(); ids={c:i for i,c in enumerate(cols)}
loss={
 'balanced':np.array([[0,1,1],[1,0,1],[1,1,0]],float),
 'miss_T2_x3':np.array([[0,1,1],[1,0,1],[3,3,0]],float),
 'false_alarm_T2_x3':np.array([[0,1,3],[1,0,3],[1,1,0]],float),
 'reject035':np.array([[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]],float)}

# Construct fixed 5 disjoint test folds, each next fold is disjoint validation, remaining 3 folds training
subs=np.array(sorted(np.unique(S))); rng=np.random.default_rng(540539); folds=np.array_split(rng.permutation(subs),5)
plan=pd.DataFrame([{'subject':s,'fold':k} for k,group in enumerate(folds) for s in group]).sort_values('subject');plan.to_csv(OUT/'R540_EEG_secondary_5fold_subject_map.csv',index=False)

def deficiency(P,Q):
 n,m=P.shape;k=m*m;N=k+n*m+1;Eidx=N-1
 Aeq=[];beq=[];Aub=[];b=[]
 for a in range(m):
  row=np.zeros(N);row[a*m:a*m+m]=1;Aeq.append(row);beq.append(1)
 for i in range(n):
  for j in range(m):
   row=np.zeros(N)
   for a in range(m):row[a*m+j]=P[i,a]
   row[k+i*m+j]=-1;Aub.append(row);b.append(Q[i,j])
   row=np.zeros(N)
   for a in range(m):row[a*m+j]=-P[i,a]
   row[k+i*m+j]=-1;Aub.append(row);b.append(-Q[i,j])
  row=np.zeros(N);row[k+i*m:k+i*m+m]=1;row[Eidx]=-2;Aub.append(row);b.append(0)
 c=np.zeros(N);c[Eidx]=1
 res=linprog(c,A_ub=np.array(Aub),b_ub=np.array(b),A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=(0,None),method='highs')
 if not res.success:raise ValueError(res.message)
 return res.fun

def calc_info(tr,train_sub):
 ans=[]
 for e in E+R:
  # subject normalized beta; train-only quantile cuts
  col=X[:,ids[e+'_beta']];cuts=np.quantile(col[tr],[1/3,2/3]); yy=np.searchsorted(cuts,col,side='right')
  row=[]
  for state in range(3):
   freq=[np.bincount(yy[(S==s)&(Y==state)],minlength=3)/np.count_nonzero((S==s)&(Y==state)) for s in train_sub]
   row.append(np.mean(freq,axis=0))
  ans.append(np.asarray(row))
 v=[]
 for i in range(8):
  P,Q=ans[i],ans[i+8]
  d=max(deficiency(P,Q),deficiency(Q,P))
  r=max(max(deficiency(P[list(ii)],Q[list(ii)]),deficiency(Q[list(ii)],P[list(ii)])) for ii in ((0,1),(0,2),(1,2)))
  assert 0<=r<=d+1e-7
  v.append({'index':i,'local_r2':r,'full_delta':d})
 return pd.DataFrame(v)

riskout=[];choices=[];information=[];portfolios=[]
for fold in range(5):
 if fold != int(os.environ['R540_FOLD']): continue
 test_sub=folds[fold];val_sub=folds[(fold+1)%5];train_sub=sorted(set(subs)-set(test_sub)-set(val_sub))
 tr=np.isin(S,train_sub);va=np.isin(S,val_sub);te=np.isin(S,test_sub)
 assert len(set(train_sub)&set(test_sub))==len(set(train_sub)&set(val_sub))==0
 m=X[tr].mean(axis=0);sd=X[tr].std(axis=0);sd[sd<.01]=1;xx=(X-m)/sd
 info=calc_info(tr,train_sub)
 for r in info.to_dict('records'):information.append({'fold':fold,**r})
 reV={};reT={};
 for mask in range(256):
  sel=[R[j] if mask>>j&1 else E[j] for j in range(8)]
  ci=[ids[e+'_'+b] for e in sel for b in BANDS]
  model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[1/3]*3).fit(xx[tr][:,ci],Y[tr])
  pV=model.predict_proba(xx[va][:,ci]);pT=model.predict_proba(xx[te][:,ci]);
  yv=Y[va];yt=Y[te]; sv=S[va];st=S[te]
  vdict={};tdict={}
  for lk,l in loss.items():
   av=np.argmin(pV@l,axis=1);at=np.argmin(pT@l,axis=1)
   vdict[lk]=np.array([np.mean([l[yv[(sv==s)&(yv==c)],av[(sv==s)&(yv==c)]].mean() for c in range(3)]) for s in val_sub])
   tdict[lk]=np.array([np.mean([l[yt[(st==s)&(yt==c)],at[(st==s)&(yt==c)]].mean() for c in range(3)]) for s in test_sub])
  reV[mask]=vdict;reT[mask]=tdict
  portfolios.append({'fold':fold,'mask':mask,'n_replacements':mask.bit_count(),**{'val_'+kk:v.mean() for kk,v in vdict.items()},**{'test_'+kk:v.mean() for kk,v in tdict.items()}})
 for budget in [2,4,6]:
  local=sum(1<<int(i) for i in info.sort_values(['local_r2','index']).head(budget)['index']);full=sum(1<<int(i) for i in info.sort_values(['full_delta','index']).head(budget)['index'])
  avail=[j for j in range(256) if j.bit_count()==budget];upto=[j for j in range(256) if j.bit_count()<=budget]
  for lk in loss:
   policy={'local_r2':local,'full_delta':full,'target_match_exact':min(avail,key=lambda j:(reV[j][lk].mean(),j)),
    'target_match_up_to':min(upto,key=lambda j:(reV[j][lk].mean(),j.bit_count(),j)), 'no_replacement':0}
   for pol,mask in policy.items():
    a=reT[mask][lk];b=reT[0][lk]
    choices.append({'fold':fold,'budget':budget,'contract':lk,'policy':pol,'mask':mask,'actual_replacements':mask.bit_count(),
     'validation_risk':reV[mask][lk].mean(),'test_risk':a.mean(),'test_minus_keep':(a-b).mean(),
     'n_train':len(train_sub),'n_validation':len(val_sub),'n_test':len(test_sub)})
    for s,rr,bb in zip(test_sub,a,b):
     riskout.append({'fold':fold,'subject':s,'budget':budget,'contract':lk,'policy':pol,'risk':rr,'minus_keep':rr-bb})
 print('FINISHED_FOLD',fold,'subjects',len(train_sub),len(val_sub),len(test_sub),'elapsed',round(time.time()-T,1),flush=True)


for name,rows in [('information',information),('portfolios',portfolios),('choices',choices),('riskout',riskout)]:
 pd.DataFrame(rows).to_csv(OUT/f'R540_partial_fold{os.environ["R540_FOLD"]}_{name}.csv',index=False)
print("DONE_FOLD",os.environ["R540_FOLD"],round(time.time()-T,2))
