"""Paper A R526: 56 new reproducible subject-disjoint resplits, natural 3-class Bayes 0-1 loss and information audit.
Pure Python internal verification; no new original datasets. RDS decoded from R522 complete parent.
"""
import zipfile,io,lzma,struct,sys,itertools,json,time
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.optimize import linprog
source=Path('/mnt/data/PaperA_R522_完整科研交接_四来源重复划分_20261102.zip')
base='00_必须保留_完整科研交接包/R521_已取得的轻量化分析数据/'

def read_rds(b):
 st=io.BytesIO(lzma.decompress(b));assert st.read(2)==b'X\n'
 def ri():return struct.unpack('>i',st.read(4))[0]
 v,w,m,n=[ri() for _ in range(4)];st.read(n);flags=ri();assert flags&255==19
 ncol=ri();assert ncol==6;cols=[]
 for _ in range(ncol):
  f=ri();kind=f&255;n=ri()
  if kind==16:
   ss=[]
   for i in range(n):
    cc=ri();assert cc&255==9;size=ri();ss.append(None if size<0 else st.read(size).decode('utf8'))
   cols.append(ss)
  elif kind==14:cols.append(np.frombuffer(st.read(8*n),dtype='>f8').astype(float))
  else:raise ValueError(kind)
 return pd.DataFrame(dict(zip(['unit','state','stream','mag_mean','mag_sd','x_sd'],cols)))

def D(P,Q):
 n,m=P.shape;N=m*m+n*m+1;k=m*m;E=N-1
 c=np.zeros(N);c[-1]=1
 eq=[];beq=[];ub=[];b=[]
 for a in range(m):
  row=np.zeros(N);row[a*m:a*m+m]=1;eq.append(row);beq.append(1)
 for i in range(n):
  for j in range(m):
   r=np.zeros(N)
   for a in range(m):r[a*m+j]=P[i,a]
   r[k+i*m+j]=-1;ub.append(r);b.append(Q[i,j])
   r=np.zeros(N)
   for a in range(m):r[a*m+j]=-P[i,a]
   r[k+i*m+j]=-1;ub.append(r);b.append(-Q[i,j])
  r=np.zeros(N);r[k+i*m:k+i*m+m]=1;r[E]=-2;ub.append(r);b.append(0)
 z=linprog(c,A_ub=ub,b_ub=b,A_eq=eq,b_eq=beq,bounds=(0,None),method='highs')
 if not z.success:raise ValueError(z.message)
 return z.fun

def audit(P,Q):
 d=max(D(P,Q),D(Q,P))
 r2=max(max(D(P[list(j)],Q[list(j)]),D(Q[list(j)],P[list(j)])) for j in [(0,1),(0,2),(1,2)])
 return d,r2

def cclass(P):return 1-np.max(P,axis=0).sum()/len(P)

def configs(ds):
 if ds=='WISDM':return ([('phone_accel','watch_accel'),('phone_gyro','watch_gyro')],[('A','C','D'),('A','D','E'),('A','B','C')])
 if ds=='DSADS':return ([('torso','right_arm'),('torso','right_leg'),('left_arm','right_leg')],[('a01','a02','a09'),('a02','a09','a12')])
 if ds=='MEX':return ([('thigh','wrist')],[('ex01','ex02','ex03'),('ex04','ex05','ex06'),('ex02','ex06','ex07')])
 if ds=='MHEALTH':return ([('chest','arm'),('chest','ankle'),('arm','ankle')],[('1','4','10'),('2','6','11'),('4','9','12')])

def cache_dataset(df):
 out={}
 for (unit,stream,state),g in df.groupby(['unit','stream','state'],sort=False):
  out[(unit,stream,state)]={feat:v[np.isfinite(v)] for feat in ['mag_mean','mag_sd','x_sd'] for v in [g[feat].to_numpy(float)]}
 return out

def fit_channels(cache,units,states,pair,feat,cuts):
 P=[];Q=[]
 for state in states:
  found=[]
  for unit in units:
   a=cache.get((unit,pair[0],state),{}).get(feat,[])
   b=cache.get((unit,pair[1],state),{}).get(feat,[])
   if len(a)>=2 and len(b)>=2:
    found.append((np.bincount(np.searchsorted(cuts[0],a,side='right'),minlength=3)/len(a),np.bincount(np.searchsorted(cuts[1],b,side='right'),minlength=3)/len(b)))
  if len(found)<2:raise ValueError(('missing paired state',state,len(found)))
  P.append(np.mean([v[0] for v in found],axis=0))
  Q.append(np.mean([v[1] for v in found],axis=0))
 return np.asarray(P),np.asarray(Q)

def one_task(cache,train,test,states,pair,feat):
 cuts=[]
 for s in pair:
  v=[cache[(u,s,st)][feat] for u in train for st in states if (u,s,st) in cache and len(cache[(u,s,st)][feat])]
  if not v:raise ValueError('no train windows')
  cc=np.quantile(np.concatenate(v),[1/3,2/3],method='linear')
  if not cc[0]<cc[1]:raise ValueError('degenerate quantiles')
  cuts.append(cc)
 p,q=fit_channels(cache,train,states,pair,feat,cuts)
 pte,qte=fit_channels(cache,test,states,pair,feat,cuts)
 delta,r2=audit(p,q)
 delta_te,r2_te=audit(pte,qte)
 trR0=cclass(p);trR1=cclass(q)
 testR0=cclass(pte);testR1=cclass(qte)
 return dict(train_Delta=delta,train_r2=r2,test_Delta=delta_te,test_r2=r2_te,
   train_bayes_P=trR0,train_bayes_Q=trR1,test_bayes_P=testR0,test_bayes_Q=testR1,
   test_bayes_signed=testR1-testR0,test_bayes_increase=max(0.,testR1-testR0),test_bayes_absolute=abs(testR1-testR0))

summary=[];taskrows=[];skipped=[]
start=time.time()
with zipfile.ZipFile(source) as z:
 for di,ds in enumerate(['WISDM','DSADS','MEX','MHEALTH']):
  df=read_rds(z.read(base+ds+'_ANALYSIS_READY.rds'))
  cache=cache_dataset(df)
  subjects=np.array(sorted(df.unit.unique()))
  n=len(subjects); reps={'WISDM':20,'DSADS':20,'MEX':8,'MHEALTH':8}[ds]
  pairs,statesets=configs(ds)
  for r in range(1,reps+1):
   rng=np.random.default_rng(526000+di*1000+r)
   order=subjects[rng.permutation(n)]
   nt=int(.55*n); nv=max(2,int(.20*n));train=order[:nt];val=order[nt:nt+nv];test=order[nt+nv:]
   tasks=[]
   for states in statesets:
    for pair in pairs:
     for feat in ['mag_mean','mag_sd','x_sd']:
      try:
       x=one_task(cache,train,test,states,pair,feat)
       tasks.append(dict(states='-'.join(states),streamA=pair[0],streamB=pair[1],feature=feat,**x))
      except Exception as e:
       skipped.append(dict(source=ds,rep=r,task='-'.join(states)+':'+pair[0]+':'+feat,error=str(e)))
   if len(tasks)<4:continue
   t=pd.DataFrame(tasks)
   t['_fd']=t.train_Delta.round(10);t['_r2']=t.train_r2.round(10)
   K=round(.5*len(t))
   full=t.sort_values(['_fd','states','streamA','feature']).index[:K]
   loc=t.sort_values(['_r2','states','streamA','feature']).index[:K]
   for metric in ['test_Delta','test_bayes_absolute','test_bayes_increase']:
    f=float(t.loc[full,metric].mean());l=float(t.loc[loc,metric].mean())
    summary.append(dict(dataset=ds,repeat_index=r,n_participants=n,train_units=nt,validation_units=nv,test_units=len(test),n_tasks=len(t),approval_K=K,metric=metric,full_mean=f,local_mean=l,full_minus_local=f-l,rank_swaps=len(set(full)-set(loc))))
   for ix in t.index:
    zrow=t.loc[ix].to_dict();zrow.pop('_fd');zrow.pop('_r2')
    taskrows.append(dict(dataset=ds,repeat_index=r,participant_test_list=';'.join(test),full_approve=(ix in full),local_approve=(ix in loc),**zrow))
  print(ds,'split batches',reps,'elapsed',round(time.time()-start,1),'s',flush=True)
res=pd.DataFrame(summary);detail=pd.DataFrame(taskrows)
res.to_csv('/mnt/data/R526_repartition_natural_bayes_vs_delta.csv',index=False)
detail.to_csv('/mnt/data/R526_task_level_natural_bayes_vs_delta.csv',index=False)
pd.DataFrame(skipped).to_csv('/mnt/data/R526_skipped_tasks.csv',index=False)
print('SUMMARY')
for (ds,metric),gg in res.groupby(['dataset','metric'],sort=False):
 v=gg.full_minus_local
 print(ds,metric,len(v),'better/tie/worse',(int((v< -1e-9).sum()),int((abs(v)<=1e-9).sum()),int((v>1e-9).sum())),'mean difference',round(float(v.mean()),6))
print('Completed',len(res),'decision rows',len(detail),'task rows',len(skipped),'skips',round(time.time()-start,1),'s')
