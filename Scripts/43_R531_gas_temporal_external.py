"""Paper A R531, independent chemical-sensor temporal holdout validation.
Input: original UCI Gas Sensor Array Drift at Different Concentrations ZIP.
Uses eight precommitted disjoint real sensor pairs (1,9)...(8,16), *one original
physical sensor* per channel, first response feature ΔR, three bins based only on
train rows. Compares ordered replacement sensor A->B for 20 possible 3-gas tasks;
no synthetic or copied channels. Each task has a 3x3 class-conditional channel.
Primary: train batches 1..9, holdout 10; four forward temporal splits 1..(t-1)->t,
t=6..9. Same physical experiment across splits, not separate sources.
"""
import sys,time,json,zipfile,hashlib,itertools,collections
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog,milp,Bounds,LinearConstraint
T0=time.time(); IN=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
LOSS={
 'balanced':dict(prior=[1/3]*3,L=[[0,1,1],[1,0,1],[1,1,0]]),
 'miss3':dict(prior=[1/3]*3,L=[[0,1,1],[1,0,1],[3,3,0]]),
 'reject035':dict(prior=[1/3]*3,L=[[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]]),
 'thirdprior060':dict(prior=[.2,.2,.6],L=[[0,1,1],[1,0,1],[1,1,0]]),
 'falsealarm3':dict(prior=[1/3]*3,L=[[0,1,3],[1,0,3],[1,1,0]])}
UNIQUE=['balanced','miss3','reject035','falsealarm3']; PAIRS=[(i,i+8) for i in range(1,9)];TRIPLES=list(itertools.combinations(range(1,7),3))

def load():
 with zipfile.ZipFile(IN) as z:
  assert z.testzip() is None
  assert set(z.namelist())=={f'batch{i}.dat' for i in range(1,11)}
  nrows=0;batches=[];classes=[];concentrations=[];features=[]
  for batch in range(1,11):
   for l in z.read(f'batch{batch}.dat').decode('utf8').splitlines():
    cells=l.strip().split();hdr=cells.pop(0).split(';');
    if len(hdr)!=2 or len(cells)!=128:raise ValueError('bad input row '+str(batch))
    cls=int(hdr[0]);ppm=float(hdr[1]);
    parsed=np.empty(128)
    for i,cell in enumerate(cells):
     num,v=cell.split(':')
     if int(num)!=i+1:raise ValueError((batch,nrows,i,num))
     parsed[i]=float(v)
    nrows+=1;batches.append(batch);classes.append(cls);concentrations.append(ppm);features.append(parsed)
 X=np.asarray(features);batch=np.asarray(batches);y=np.asarray(classes);ppm=np.asarray(concentrations)
 assert X.shape==(13910,128) and np.isfinite(X).all() and np.isfinite(ppm).all()
 assert set(y)==set(range(1,7))
 return X,batch,y,ppm

def risk(ch,prior,L,trained=None):
 p=np.asarray(prior);c=np.asarray(L)
 w=(p[:,None]*ch).T@c
 if trained is None:return float(w.min(axis=1).sum())
 acts=((p[:,None]*trained).T@c).argmin(axis=1)
 return float(w[np.arange(w.shape[0]),acts].sum())

def deficiency(P,Q):
 # Original R530 LP implementation, unchanged in optimization convention.
 n,m=P.shape;k=m*m;N=k+n*m+1;E=N-1;eq=[];beq=[];ub=[];b=[]
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
 sol=linprog(c,A_ub=np.asarray(ub),b_ub=np.asarray(b),A_eq=np.asarray(eq),b_eq=np.asarray(beq),bounds=(0,None),method='highs')
 if not sol.success:raise RuntimeError(sol.message)
 return float(sol.fun)

def audit(P,Q):
 D=max(deficiency(P,Q),deficiency(Q,P))
 r=max(max(deficiency(P[list(x)],Q[list(x)]),deficiency(Q[list(x)],P[list(x)])) for x in ((0,1),(0,2),(1,2)))
 if D+1e-8<r: raise RuntimeError('local greater than full')
 return D,r

def channel(vals,labels,triplet,cuts):
 return np.asarray([np.bincount(np.searchsorted(cuts,vals[labels==j],side='right'),minlength=3)/int((labels==j).sum()) for j in triplet])

def splits():
 yield 'official_chronological_1to9_vs10', list(range(1,10)),[10]
 for t in range(6,10):yield f'forward_1to{t-1}_vs{t}',list(range(1,t)),[t]

def collect(X,batch,y,ppm):
 rows=[];summary=[]
 for split,tr,te in splits():
  tra=np.isin(batch,tr);test=np.isin(batch,te)
  aY=y[tra];eY=y[test]
  for gas in range(1,7):
   if min((aY==gas).sum(),(eY==gas).sum())<5: raise RuntimeError((split,gas,'not enough gas'))
  summary.append({'split':split,'train_batches':','.join(map(str,tr)),'test_batches':','.join(map(str,te)),
                  'train_n':int(tra.sum()),'test_n':int(test.sum()),'test_class_counts':str(dict(collections.Counter(eY))),
                  'disjoint_batches':int(not set(tr).intersection(te))})
  for triple in TRIPLES:
   aT=np.isin(aY,triple);eT=np.isin(eY,triple)
   for aa,bb in PAIRS:
    ai=(aa-1)*8;bi=(bb-1)*8
    Av=X[tra,ai];Bv=X[tra,bi];Ae=X[test,ai];Be=X[test,bi]
    ac=np.quantile(Av[aT],[1/3,2/3]);bc=np.quantile(Bv[aT],[1/3,2/3])
    if ac[0]==ac[1] or bc[0]==bc[1]:raise RuntimeError(('degenerate',split,triple,aa,bb))
    P=channel(Av,aY,triple,ac);Q=channel(Bv,aY,triple,bc)
    Pe=channel(Ae,eY,triple,ac);Qe=channel(Be,eY,triple,bc)
    D,r=audit(P,Q);De,re=audit(Pe,Qe)
    row={'source':'UCI_GAS270','split':split,'gases':'-'.join(map(str,triple)),
         'sensor_A':aa,'sensor_B':bb,'feature':'deltaR',
         'train_Delta':D,'train_r2':r,'test_Delta':De,'test_r2':re,
         'train_n_gas1':int((aY==triple[0]).sum()),'train_n_gas2':int((aY==triple[1]).sum()),
         'train_n_gas3':int((aY==triple[2]).sum()),
         'test_n_gas1':int((eY==triple[0]).sum()),'test_n_gas2':int((eY==triple[1]).sum()),
         'test_n_gas3':int((eY==triple[2]).sum())}
    for key,config in LOSS.items():
     pri=config['prior'];L=config['L']
     row['train_signed_'+key]=risk(Q,pri,L)-risk(P,pri,L)
     row['test_signed_'+key]=risk(Qe,pri,L)-risk(Pe,pri,L)
     row['deploy_signed_'+key]=risk(Qe,pri,L,trained=Q)-risk(Pe,pri,L,trained=P)
     row['deploy_P_'+key]=risk(Pe,pri,L,trained=P)
     row['deploy_Q_'+key]=risk(Qe,pri,L,trained=Q)
    assert abs(row['train_signed_miss3']-5/3*row['train_signed_thirdprior060'])<1e-10
    assert abs(row['deploy_signed_miss3']-5/3*row['deploy_signed_thirdprior060'])<1e-10
    rows.append(row)
  print('TASK DONE',split,len(rows),'seconds',round(time.time()-T0,1),flush=True)
 return pd.DataFrame(rows),pd.DataFrame(summary)

def solve_global(v,K):
 n,p=v.shape;mat=np.zeros((p,n+1));mat[:,:n]=v.T/K;mat[:,-1]=-1
 obj=np.zeros(n+1);obj[-1]=1
 lc=[LinearConstraint(np.r_[np.ones(n),0].reshape(1,-1),K,K),LinearConstraint(mat,-np.inf,0)]
 sol=milp(obj,integrality=np.r_[np.ones(n),0],bounds=Bounds(np.zeros(n+1),np.r_[np.ones(n),1]),constraints=lc,options={'mip_rel_gap':1e-8})
 if sol.status!=0:raise RuntimeError(('MILP',sol.message))
 ids=np.flatnonzero(sol.x[:n]>.5)
 assert len(ids)==K
 return ids,float(sol.fun),float(sol.mip_gap)

def policies(df):
 rows=[];comp=[];opt=[]
 for split,g0 in df.groupby('split',sort=False):
  g=g0.reset_index(drop=True);n=len(g)
  z=np.column_stack([g['train_signed_'+key].values for key in UNIQUE])
  norm=np.zeros_like(z)
  for j in range(z.shape[1]):
   v=z[:,j];spread=np.ptp(v)
   if spread>1e-12:norm[:,j]=(v-v.min())/spread
  scores={'full_Delta':g.train_Delta.values,'local_r2':g.train_r2.values,'robust_individual':norm.max(axis=1)}
  for key in UNIQUE:scores['match_'+key]=g['train_signed_'+key].values
  for frac in (.25,.5,.75):
   K=round(n*frac)
   choices={policy:np.lexsort((np.arange(n),np.round(x,12)))[:K] for policy,x in scores.items()}
   glob,obj,gap=solve_global(norm,K)
   choices['global_minimax']=glob
   robustval=norm[choices['robust_individual']].mean(axis=0).max()
   if obj>robustval+1e-7:raise RuntimeError('MILP not better than heuristic')
   opt.append({'source':'UCI_GAS270','split':split,'budget':frac,'K':K,'global_obj_train':obj,
               'heuristic_obj_train':float(robustval),'train_advantage':float(robustval-obj), 'mip_gap':gap})
   for policy,idx in choices.items():
    s=g.iloc[idx]
    row={'source':'UCI_GAS270','split':split,'budget':frac,'K':K,'policy':policy,
         'selection_signature':';'.join(map(str,sorted(idx))),'train_Delta':float(s.train_Delta.mean()),
         'test_Delta':float(s.test_Delta.mean()),'train_r2':float(s.train_r2.mean())}
    for key in UNIQUE:
     row['train_signed_'+key]=float(s['train_signed_'+key].mean())
     row['deploy_signed_'+key]=float(s['deploy_signed_'+key].mean())
     row['deploy_positive_'+key]=float(np.maximum(s['deploy_signed_'+key],0).mean())
    rows.append(row)
   gs=[r for r in rows if r['split']==split and r['budget']==frac]
   indexed={r['policy']:r for r in gs}
   for key in UNIQUE:
    baseline=indexed['full_Delta']['deploy_signed_'+key]
    for pol in ['match_'+key,'local_r2','robust_individual','global_minimax']:
     x=indexed[pol]['deploy_signed_'+key]-baseline
     comp.append({'split':split,'budget':frac,'contract':key,'policy':pol,
                  'matched_minus_full':x,'improves':int(x< -1e-10),
                  'ties':int(abs(x)<=1e-10),'worse':int(x>1e-10),
                  'same_selection':int(indexed[pol]['selection_signature']==indexed['full_Delta']['selection_signature'])})
  print('POLICY DONE',split,'seconds',round(time.time()-T0,1),flush=True)
 return pd.DataFrame(rows),pd.DataFrame(comp),pd.DataFrame(opt)

def main():
 X,b,y,ppm=load()
 tasks,folds=collect(X,b,y,ppm)
 pol,comp,opt=policies(tasks)
 assert len(tasks)==5*20*8 and len(pol)==5*3*8 and len(comp)==5*3*4*4 and len(opt)==5*3
 names={'R531_gas_800_tasks.csv':tasks,'R531_120_policies.csv':pol,
  'R531_240_comparisons.csv':comp,'R531_15_milp.csv':opt,'R531_temporal_splits.csv':folds}
 for name,df in names.items():df.to_csv(OUT/name,index=False)
 report={'input_sha256':hashlib.sha256(IN.read_bytes()).hexdigest(),'dataset_rows':len(y),'physical_sensors':16,
         'features_per_sensor':8,'time_batches':10,'gas_classes':6,'concentration_min_ppmv':float(ppm.min()),
         'concentration_max_ppmv':float(ppm.max()),'splits':len(folds),'n_triplets':20,'physical_sensor_pairs':8,
         'feature':'first untransformed response deltaR per sensor','task_rows':len(tasks),'policy_rows':len(pol),
         'comparisons':len(comp),'milp_optimizations':len(opt),'lp_calls':len(tasks)*16,
         'primary_matched_comparisons':comp[(comp.split=='official_chronological_1to9_vs10') & (comp.policy.str.startswith('match_'))].to_dict('records'),
         'all_matched_comparisons':comp[comp.policy.str.startswith('match_')].groupby('split').agg(n=('improves','size'),better=('improves','sum'),tie=('ties','sum'),worse=('worse','sum')).reset_index().to_dict('records'),
         'elapsed_seconds':round(time.time()-T0,2)}
 (OUT/'R531_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 print('SUMMARY',json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
