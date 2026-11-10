"""Paper A R530: external UCI HAR raw-signal replication, official held-out subjects primary.
Two independently measured modalities: total_acc (accelerometer) vs body_gyro (gyroscope).
Primary direction total_acc -> body_gyro. Subject-equal channel matrices, training-only cuts;
fixed trained actions evaluated on disjoint held-out people. Exact full Delta and local r2
are LP-computed (not estimated from classification risks). Four nonequivalent decision contracts.
Run: python R530_UCI_HAR_external.py <user_uci_zip> <output_folder>
"""
import sys,io,itertools,json,hashlib,time,zipfile
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
from scipy.sparse import csr_matrix
T0=time.time()
PATH=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
LOSS={
 '01_balanced_0_1':dict(prior=[1/3]*3,L=[[0,1,1],[1,0,1],[1,1,0]]),
 '02_third_state_cost3':dict(prior=[1/3]*3,L=[[0,1,1],[1,0,1],[3,3,0]]),
 '03_reject_cost035':dict(prior=[1/3]*3,L=[[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]]),
 '04_third_state_prior060':dict(prior=[.2,.2,.6],L=[[0,1,1],[1,0,1],[1,1,0]]),
 '05_third_state_false_alarm_cost3':dict(prior=[1/3]*3,L=[[0,1,3],[1,0,3],[1,1,0]])
}
UNIQUE=['01_balanced_0_1','02_third_state_cost3','03_reject_cost035','05_third_state_false_alarm_cost3']
PAIR=('accel','gyro');FEATURES=['mag_mean','mag_sd','x_sd'];STATES=list(itertools.combinations(range(1,7),3))

def risk(ch,prior,L,train=None):
 v=((np.asarray(prior)[:,None]*ch).T)@np.asarray(L)
 if train is None: return float(v.min(axis=1).sum())
 actions=np.argmin(((np.asarray(prior)[:,None]*train).T)@np.asarray(L),axis=1)
 return float(v[np.arange(3),actions].sum())

def deficiency(P,Q):
 n,m=P.shape;k=m*m;N=k+n*m+1;E=N-1
 eq=[];beq=[];ub=[];b=[]
 for a in range(m):
  row=np.zeros(N);row[a*m:a*m+m]=1;eq.append(row);beq.append(1)
 for i in range(n):
  for j in range(m):
   row=np.zeros(N)
   for a in range(m):row[a*m+j]=P[i,a]
   row[k+i*m+j]=-1;ub.append(row);b.append(Q[i,j])
   row=np.zeros(N)
   for a in range(m):row[a*m+j]=-P[i,a]
   row[k+i*m+j]=-1;ub.append(row);b.append(-Q[i,j])
  row=np.zeros(N);row[k+i*m:k+i*m+m]=1;row[E]=-2;ub.append(row);b.append(0)
 c=np.zeros(N);c[E]=1
 r=linprog(c,A_ub=np.asarray(ub),b_ub=np.asarray(b),A_eq=np.asarray(eq),b_eq=np.asarray(beq),bounds=(0,None),method='highs')
 if not r.success: raise RuntimeError(r.message)
 return r.fun

def audit(P,Q):
 delta=max(deficiency(P,Q),deficiency(Q,P))
 r2=max(max(deficiency(P[list(x)],Q[list(x)]),deficiency(Q[list(x)],P[list(x)])) for x in [(0,1),(0,2),(1,2)])
 assert delta+1e-8>=r2 and 0<=delta<=1+1e-8 and 0<=r2<=1+1e-8
 return delta,r2

def load():
 buf=PATH.read_bytes();outer_hash=hashlib.sha256(buf).hexdigest()
 with zipfile.ZipFile(io.BytesIO(buf)) as z:
  assert z.testzip() is None
  inner=z.read('UCI HAR Dataset.zip')
 with zipfile.ZipFile(io.BytesIO(inner)) as z:
  assert z.testzip() is None
  labels=[];subjects=[];sets=[];feat_frames=[]
  for part in ('train','test'):
   prefix='UCI HAR Dataset/'+part+'/'
   y=np.loadtxt(io.BytesIO(z.read(prefix+'y_'+part+'.txt')),dtype=int).reshape(-1)
   subj=np.loadtxt(io.BytesIO(z.read(prefix+'subject_'+part+'.txt')),dtype=int).reshape(-1)
   n=len(y);assert n==len(subj)
   features={}
   for stream,series in [('accel','total_acc'),('gyro','body_gyro')]:
    raw=[]
    for ax in 'xyz':
     member=prefix+'Inertial Signals/'+series+'_'+ax+'_'+part+'.txt'
     mat=np.loadtxt(io.BytesIO(z.read(member)),dtype=np.float64)
     assert mat.shape==(n,128), (member,mat.shape)
     raw.append(mat)
    x,yax,zax=raw
    mag=np.sqrt(x*x+yax*yax+zax*zax)
    features[(stream,'mag_mean')]=mag.mean(axis=1)
    features[(stream,'mag_sd')]=mag.std(axis=1,ddof=0)
    features[(stream,'x_sd')]=x.std(axis=1,ddof=0)
   feat_frames.append(features)
   labels.append(y);subjects.append(subj);sets.append(np.array([part]*n))
 Y=np.concatenate(labels);U=np.concatenate(subjects);SET=np.concatenate(sets)
 F={k:np.concatenate([d[k] for d in feat_frames]) for k in feat_frames[0]}
 assert len(Y)==10299 and len(np.unique(U))==30 and len(set(U[SET=='train']).intersection(U[SET=='test']))==0
 assert len(U[SET=='train'])==7352 and len(U[SET=='test'])==2947
 return U,Y,SET,F,outer_hash

def precompute(U,Y,F):
 # each unit-state-series-feature yields all windows from same participant and activity
 c={};freq={}
 for u in np.unique(U):
  for state in range(1,7):
   idx=(U==u)&(Y==state)
   freq[(int(u),state)]=int(idx.sum())
   for k,arr in F.items(): c[(int(u),state,*k)]=arr[idx]
 return c,freq

def channel(c,units,states,stream,feat,cuts):
 ans=[]
 for state in states:
  rows=[]
  for u in units:
   a=c.get((int(u),state,stream,feat),np.array([]))
   if len(a)<2:raise RuntimeError(('missing participant-state samples',u,state,stream,feat))
   rows.append(np.bincount(np.searchsorted(cuts,a,side='right'),minlength=3)/len(a))
  ans.append(np.mean(rows,axis=0))
 result=np.asarray(ans)
 assert np.allclose(result.sum(axis=1),1,atol=1e-12)
 return result

def calc_split(c,train,test,split):
 tasks=[]
 for states in STATES:
  for feat in FEATURES:
   cuts={}
   for stream in PAIR:
    xs=np.concatenate([c[(int(u),state,stream,feat)] for u in train for state in states])
    a=np.quantile(xs,[1/3,2/3],method='linear')
    if a[0]>=a[1]:raise RuntimeError('degenerate cuts')
    cuts[stream]=a
   P=channel(c,train,states,'accel',feat,cuts['accel'])
   Q=channel(c,train,states,'gyro',feat,cuts['gyro'])
   Pe=channel(c,test,states,'accel',feat,cuts['accel'])
   Qe=channel(c,test,states,'gyro',feat,cuts['gyro'])
   delta,r2=audit(P,Q); dtest,rtest=audit(Pe,Qe)
   row={'source':'UCI_HAR','split':split,'states':'-'.join(map(str,states)),'feature':feat,
    'streamA':'total_acc','streamB':'body_gyro','train_Delta':delta,'train_r2':r2,
    'test_Delta':dtest,'test_r2':rtest}
   for k,dd in LOSS.items():
    pri=np.asarray(dd['prior']);L=np.asarray(dd['L'])
    row['train_P_'+k]=risk(P,pri,L);row['train_Q_'+k]=risk(Q,pri,L)
    row['train_signed_'+k]=row['train_Q_'+k]-row['train_P_'+k]
    row['test_P_'+k]=risk(Pe,pri,L);row['test_Q_'+k]=risk(Qe,pri,L)
    row['test_signed_'+k]=row['test_Q_'+k]-row['test_P_'+k]
    row['deploy_P_'+k]=risk(Pe,pri,L,train=P)
    row['deploy_Q_'+k]=risk(Qe,pri,L,train=Q)
    row['deploy_signed_'+k]=row['deploy_Q_'+k]-row['deploy_P_'+k]
    assert row['deploy_P_'+k]>=row['test_P_'+k]-1e-10 and row['deploy_Q_'+k]>=row['test_Q_'+k]-1e-10
   tasks.append(row)
 return pd.DataFrame(tasks)

def solve_minimax(v,K):
 # v normalized [n,4]. Min t with sum_i v[i,k]*x_i/K <=t, sum_i x_i=K
 n,p=v.shape;A=np.zeros((p,n+1));A[:,:n]=v.T/K;A[:,-1]=-1
 c=np.zeros(n+1);c[-1]=1
 constraints=[LinearConstraint(np.r_[np.ones(n),0].reshape(1,-1),K,K),LinearConstraint(A,-np.inf,0)]
 sol=milp(c,integrality=np.r_[np.ones(n),0],bounds=Bounds(np.zeros(n+1),np.r_[np.ones(n),1]),constraints=constraints,options={'mip_rel_gap':1e-9})
 if sol.status!=0:raise RuntimeError(('MILP',sol.message))
 idx=np.flatnonzero(sol.x[:n]>.5)
 if len(idx)!=K:raise RuntimeError('K mismatch')
 return idx,float(sol.fun),float(getattr(sol,'mip_gap',np.nan))

def select(df):
 rows=[];minimax=[];comparisons=[]
 for split,g0 in df.groupby('split',sort=False):
  g=g0.reset_index(drop=True);n=len(g)
  norm=[]
  for contract in UNIQUE:
   a=g['train_signed_'+contract].to_numpy()
   rg=a.max()-a.min();norm.append((a-a.min())/rg if rg>1e-12 else np.zeros(n))
  norm=np.array(norm).T
  hmax=norm.max(axis=1)
  scores={'full_Delta':g.train_Delta.to_numpy(),'local_r2':g.train_r2.to_numpy(),
         'robust_individual':hmax}
  scores.update({x:g['train_signed_'+x].to_numpy() for x in LOSS})
  for frac in [.25,.5,.75]:
   K=max(1,round(n*frac))
   chosen={}
   for policy,vec in scores.items():
    # deterministic secondary ordering states / feature (already sorted by enumeration)
    ix=np.lexsort((np.arange(n),np.round(vec,12)))[:K]
    chosen[policy]=ix
   ix,obj,gap=solve_minimax(norm,K)
   chosen['robust_global_minimax']=ix
   ind_val=float(norm[chosen['robust_individual']].mean(axis=0).max())
   if obj>ind_val+1e-8:raise RuntimeError('minimax worse than heuristic')
   minimax.append(dict(source='UCI_HAR',split=split,budget_fraction=frac,K=K,
      minimax_train=obj,individual_train=ind_val,train_improvement=ind_val-obj,
      gap=gap,selected_signature=';'.join(map(str,sorted(ix)))))
   for policy,ids in chosen.items():
    dd={'source':'UCI_HAR','split':split,'budget_fraction':frac,'K':K,
     'policy':policy,'selected_signature':';'.join(map(str,sorted(ids))),
     'test_Delta':float(g.iloc[ids].test_Delta.mean()),
     'train_Delta':float(g.iloc[ids].train_Delta.mean())}
    for k in LOSS:
     dd['train_signed_'+k]=float(g.iloc[ids]['train_signed_'+k].mean())
     dd['test_signed_'+k]=float(g.iloc[ids]['test_signed_'+k].mean())
     dd['deploy_signed_'+k]=float(g.iloc[ids]['deploy_signed_'+k].mean())
     dd['deploy_positive_'+k]=float(np.maximum(g.iloc[ids]['deploy_signed_'+k],0).mean())
    rows.append(dd)
  if len(rows)%27==0:print('  POLICY',split,len(rows),'elapsed',round(time.time()-T0,1),flush=True)
 res=pd.DataFrame(rows);rm=pd.DataFrame(minimax)
 for (s,b),g in res.groupby(['split','budget_fraction']):
  g=g.set_index('policy')
  for k in UNIQUE:
   for policy in [k,'robust_individual','robust_global_minimax','local_r2']:
    comparisons.append(dict(split=s,budget_fraction=b,contract=k,policy=policy,
     matched_minus_full=float(g.loc[policy,'deploy_signed_'+k]-g.loc['full_Delta','deploy_signed_'+k]),
     selection_changed=int(g.loc[policy,'selected_signature']!=g.loc['full_Delta','selected_signature'])))
 return res,rm,pd.DataFrame(comparisons)

def main():
 U,Y,part,F,hash0=load();c,freq=precompute(U,Y,F)
 train=sorted(set(U[part=='train']));test=sorted(set(U[part=='test']))
 print('INPUT 10299/30 official 7352/2947 21/9, sha256',hash0,flush=True)
 print('Min per subject state windows',min(freq.values()),flush=True)
 splits=[('official_21_9',train,test)]
 # repeat resplits secondary descriptive analysis; distinct underlying study/overlap explicitly tracked
 everyone=np.array(sorted(set(U)))
 for r in range(1,13):
  rng=np.random.default_rng(53000+r)
  perm=everyone[rng.permutation(len(everyone))]
  splits.append((f'resplit_{r:02d}',list(perm[:21]),list(perm[21:])))
 meta=[];frames=[]
 for label,tr,te in splits:
  assert len(set(tr)&set(te))==0 and len(tr)==21 and len(te)==9
  base=calc_split(c,tr,te,label)
  frames.append(base)
  meta.append({'split':label,'train_subjects':','.join(map(str,tr)),'test_subjects':','.join(map(str,te)),
   'train_n':len(tr),'test_n':len(te),'records':len(base)})
  print('FINISHED',label,len(base),'elapsed',round(time.time()-T0,1),flush=True)
 all_tasks=pd.concat(frames,ignore_index=True)
 assert len(all_tasks)==13*60
 for xx in ['train_P_','train_Q_','test_P_','test_Q_','deploy_P_','deploy_Q_','train_signed_','test_signed_','deploy_signed_']:
  assert np.allclose(all_tasks[xx+'02_third_state_cost3'],(5/3)*all_tasks[xx+'04_third_state_prior060'],atol=1e-12)
 results,mm,cmp=select(all_tasks)
 assert len(results)==13*3*9 and len(mm)==13*3 and len(cmp)==13*3*4*4
 all_tasks.to_csv(OUT/'R530_UCI_HAR_780_真实通道任务.csv',index=False)
 results.to_csv(OUT/'R530_UCI_HAR_351_策略结果.csv',index=False)
 mm.to_csv(OUT/'R530_UCI_HAR_39_整数优化.csv',index=False)
 cmp.to_csv(OUT/'R530_UCI_HAR_624_测试比较.csv',index=False)
 pd.DataFrame(meta).to_csv(OUT/'R530_参与者划分.csv',index=False)
 aggreg=cmp.groupby(['split','budget_fraction','policy'])['matched_minus_full'].mean().reset_index()
 aggs=cmp.groupby(['budget_fraction','contract','policy']).agg(mean_diff=('matched_minus_full','mean'),median_diff=('matched_minus_full','median'),wins=('matched_minus_full',lambda x:int((x<-1e-10).sum())),ties=('matched_minus_full',lambda x:int((abs(x)<=1e-10).sum())),losses=('matched_minus_full',lambda x:int((x>1e-10).sum())),n=('matched_minus_full','size')).reset_index()
 aggs.to_csv(OUT/'R530_13划分_策略情景汇总.csv',index=False)
 official=cmp[cmp.split=='official_21_9'].copy();official.to_csv(OUT/'R530_官方测试组_主要结果.csv',index=False)
 summary={'source_name':'UCI HAR Using Smartphones','source_independence':'new independent recording study, not additional cohorts per split',
  'source_original_zip_sha256':hash0,'total_windows':int(len(U)),'subjects':30,'official_train_subjects':21,'official_test_subjects':9,
  'official_train_windows':int((part=='train').sum()),'official_test_windows':int((part=='test').sum()),
  'classes':6,'hardware_modalities':['physical accelerometer total_acc','physical gyroscope body_gyro'],'direction':'accelerometer to gyroscope only',
  'tasks_per_split':60,'split_count':len(splits),'total_task_rows':len(all_tasks),'approval_budget':[.25,.5,.75],
  'policy_count':int(results.policy.nunique()),'strategy_rows':len(results),'minimax_models':len(mm),
  'four_non_equivalent_contracts':UNIQUE,'decision_equivalence_verified':True,
  'official_results':official.to_dict(orient='records'),
  'pooled_resplit_means':aggs.to_dict(orient='records'),'elapsed_s':round(time.time()-T0,2),
  'caution':'Only one new recording study; 12 resplits are correlated reuse; no actual purchase cost or prospective interventional effect.'}
 (OUT/'R530_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 print('OFFICIAL RESULTS\n',official.pivot(index='contract',columns=['budget_fraction','policy'],values='matched_minus_full').round(5).to_string(),flush=True)
 print('RESPLIT PERFORM\n',aggs[aggs.policy.isin(['robust_global_minimax','01_balanced_0_1'])].round(5).to_string(index=False),flush=True)
 print('ALL FINISHED',round(time.time()-T0,1),'sec',flush=True)
if __name__=='__main__':main()
