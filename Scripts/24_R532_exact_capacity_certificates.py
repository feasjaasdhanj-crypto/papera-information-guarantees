"""Exhaustive eight-sensor certificates: for at-most-K and uniform price,
select negative adjusted marginal scores in increasing order. 720 cases.
"""
from pathlib import Path
import sys,itertools,numpy as np,pandas as pd
O=Path(sys.argv[1]);F=pd.read_csv(O/'parent/03_实证结果/R531_prequential_640_task_channels.csv')
rows=[]
for t in (7,8,9,10):
 split=f'train1to{t-2}_val{t-1}_test{t}'
 g=F[F.split==split].groupby('sensor_A').mean(numeric_only=True).reset_index()
 assert len(g)==8
 for c in ['balanced','miss3','reject035','falsealarm3']:
  tr=g['train_signed_'+c].to_numpy();val=g['val_signed_'+c].to_numpy()
  for K in (2,4,6):
   subsets=[s for r in range(K+1) for s in itertools.combinations(range(8),r)]
   for price in [0.,.0025,.005,.01,.02]:
    for pol,sc in [('target',tr),('validation',val),('blend',.5*tr+.5*val)]:
     order=np.lexsort((np.arange(8),np.round(sc,12)))
     choose=order[sc[order]<-price-1e-12][:K]
     obj=float((sc[choose]+price).sum())
     true=min(float((sc[list(ss)]+price).sum()) for ss in subsets)
     assert np.isclose(obj,true,atol=1e-10)
     rows.append({'split':split,'contract':c,'K':K,'lambda':price,'policy':pol,
                  'min_predicted_risk_plus_cost':true,'chosen_score':obj,'n_candidates':len(subsets),
                  'n_used':len(choose),'verified':1})
D=pd.DataFrame(rows);assert len(D)==720
D.to_csv(O/'R532_720_exact_physical_budget_certificates.csv',index=False)
print('EXHAUSTIVE VERIFIED',len(D),'subsets checked',D.n_candidates.sum())
