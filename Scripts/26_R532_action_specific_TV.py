"""R532 policy-conditioned TV bound.
For fixed actions a_P(y), a_Q(y), the effective per-row range is
range_y L(x,a_P(y)), not range over all possible legal actions. This is
<= the conventional TV range and may be zero when fixed actions are constant.
Future batch channels are used ONLY for retrospective bound verification and
NOT to select the portfolios or claim prospective certification.
"""
import itertools,sys,zipfile,json
from pathlib import Path
import numpy as np,pandas as pd
O=Path(sys.argv[1]);I=Path(sys.argv[2]);
Y=[];T=[];X=[]
with zipfile.ZipFile(I) as z:
 for i in range(1,11):
  for l in z.read(f'batch{i}.dat').decode().splitlines():
   parts=l.split();y,p=parts.pop(0).split(';');Y.append(int(y));T.append(i)
   X.append([float(v.split(':')[1]) for v in parts])
X=np.asarray(X);T=np.asarray(T);Y=np.asarray(Y)
contracts={
 'balanced':([1/3]*3,[[0,1,1],[1,0,1],[1,1,0]]),
 'miss3':([1/3]*3,[[0,1,1],[1,0,1],[3,3,0]]),
 'reject035':([1/3]*3,[[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]]),
 'falsealarm3':([1/3]*3,[[0,1,3],[1,0,3],[1,1,0]])}
TASK=pd.read_csv(O/'parent/03_实证结果/R531_gas_800_tasks.csv');CERT=pd.read_csv(O/'parent/03_实证结果/R531_800_drift_TV_task_bounds.csv')
rows=[]
def chan(mask,sensor,trip,cuts):
 vals=X[mask,(sensor-1)*8];lab=Y[mask]
 return np.stack([np.bincount(np.searchsorted(cuts,vals[lab==g],side='right'),minlength=3)/(lab==g).sum() for g in trip])
for target in range(6,11):
 tr=T<target;te=T==target
 label='official_chronological_1to9_vs10' if target==10 else f'forward_1to{target-1}_vs{target}'
 for gas in itertools.combinations(range(1,7),3):
  subset=tr&np.isin(Y,gas)
  for a in range(1,9):
   b=a+8;cutsA=np.quantile(X[subset,(a-1)*8],[1/3,2/3]);cutsB=np.quantile(X[subset,(b-1)*8],[1/3,2/3])
   P=chan(tr,a,gas,cutsA);Q=chan(tr,b,gas,cutsB);PE=chan(te,a,gas,cutsA);QE=chan(te,b,gas,cutsB)
   tvp=.5*np.abs(P-PE).sum(axis=1);tvq=.5*np.abs(Q-QE).sum(axis=1)
   r={'split':label,'gases':'-'.join(map(str,gas)),'sensor_A':a,'sensor_B':b}
   for c,(prior,L) in contracts.items():
    prior=np.asarray(prior);L=np.asarray(L)
    actP=((prior[:,None]*P).T@L).argmin(axis=1);actQ=((prior[:,None]*Q).T@L).argmin(axis=1)
    rangeP=np.ptp(L[:,actP],axis=1);rangeQ=np.ptp(L[:,actQ],axis=1)
    r['epsilon_action_'+c]=float(np.dot(prior,rangeP*tvp+rangeQ*tvq))
    full_range=np.ptp(L,axis=1)
    r['epsilon_generic_'+c]=float(np.dot(prior,full_range*(tvp+tvq)))
   rows.append(r)
 print('ACTION TV',label,flush=True)
R=pd.DataFrame(rows);assert len(R)==800
R=TASK[['split','gases','sensor_A','sensor_B']].merge(R,on=['split','gases','sensor_A','sensor_B'],how='left',validate='one_to_one')
for c in contracts:
 assert np.allclose(R['epsilon_generic_'+c],CERT['epsilon_'+c],atol=1e-12)
 assert (R['epsilon_action_'+c]<=(R['epsilon_generic_'+c]+1e-12)).all()
 assert ((TASK['deploy_signed_'+c]-TASK['train_signed_'+c]).abs()<=R['epsilon_action_'+c]+1e-10).all()
R.to_csv(O/'R532_action_conditioned_800_TV.csv',index=False)
result=[]
for split, task in TASK.groupby('split',sort=False):
 e=R[R.split==split].reset_index(drop=True)
 task=task.reset_index(drop=True)
 for level in ('task','physical'):
  g=task if level=='task' else task.groupby('sensor_A').mean(numeric_only=True).reset_index()
  h=e if level=='task' else e.groupby('sensor_A').mean(numeric_only=True).reset_index()
  N=160 if level=='task' else 8
  for K in ([40,80,120] if level=='task' else [2,4,6]):
   full=set(np.lexsort((np.arange(N),np.round(g.train_Delta.values,12)))[:K])
   for c in contracts:
    target=set(np.lexsort((np.arange(N),np.round(g['train_signed_'+c].values,12)))[:K])
    symmetric=list(full.symmetric_difference(target))
    den=K if level=='task' else 8
    train_margin=float((g.iloc[list(target)]['train_signed_'+c].sum()-g.iloc[list(full)]['train_signed_'+c].sum())/den)
    test_margin=float((g.iloc[list(target)]['deploy_signed_'+c].sum()-g.iloc[list(full)]['deploy_signed_'+c].sum())/den)
    generic=float(h.iloc[symmetric]['epsilon_generic_'+c].sum()/den)
    action=float(h.iloc[symmetric]['epsilon_action_'+c].sum()/den)
    assert abs(test_margin-train_margin)<=action+1e-10
    result.append({'split':split,'level':level,'K':K,'contract':c,'train_margin':train_margin,'test_margin':test_margin,
       'generic_TV_bound':generic,'action_TV_bound':action,'ratio_new_old':action/generic if generic>1e-10 else 0.,
       'generic_certificate_target':int(train_margin+generic<-1e-10),
       'action_certificate_target':int(train_margin+action<-1e-10),
       'action_certificate_full':int(train_margin-action>1e-10),
       'n_symmetric':len(symmetric)})
S=pd.DataFrame(result);assert len(S)==120
S.to_csv(O/'R532_action_conditioned_120_certificates.csv',index=False)
print(S.groupby(['level','split']).agg(generic_better=('generic_certificate_target','sum'),action_better=('action_certificate_target','sum'),action_full=('action_certificate_full','sum'),median_sharpness=('ratio_new_old','median'),mean_sharpness=('ratio_new_old','mean')).to_string())
print('TOT',S[['generic_certificate_target','action_certificate_target','action_certificate_full']].sum().to_dict(),'median ratio',S.ratio_new_old.median())
