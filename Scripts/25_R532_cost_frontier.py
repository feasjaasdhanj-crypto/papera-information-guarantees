"""R532 upper-budget sensor replacement with modeled per-unit transaction cost.
Follows forward batches t7..10; training vs latest validation vs blend only.
Price lambda is a decision-risk-unit penalty per installed task/sensor replacement,
not an observed market price or a claimed monetary saving.
"""
from pathlib import Path
import pandas as pd,numpy as np,sys,json
O=Path(sys.argv[1]);F=pd.read_csv(O/'parent/03_实证结果/R531_prequential_640_task_channels.csv')
C=['balanced','miss3','reject035','falsealarm3'];PRICES=[0.,.0025,.005,.01,.02]
policies=('target','validation','blend');rows=[]
for t in (7,8,9,10):
 split=f'train1to{t-2}_val{t-1}_test{t}';df=F[F.split==split]
 assert len(df)==160
 for level in ('task','physical'):
  df2=df.reset_index(drop=True) if level=='task' else df.groupby('sensor_A').mean(numeric_only=True).reset_index()
  N=len(df2)
  for K in (round(N*f) for f in (.25,.5,.75)):
   for c in C:
    tr=df2['train_signed_'+c].to_numpy();va=df2['val_signed_'+c].to_numpy();te=df2['test_signed_'+c].to_numpy()
    for lam in PRICES:
     for policy in policies:
      sc={'target':tr,'validation':va,'blend':.5*tr+.5*va}[policy]
      order=np.lexsort((np.arange(N),np.round(sc,12)))
      selected=order[sc[order] < -lam-1e-12][:K]
      realized_risk=float(te[selected].sum()/N)
      realized_price=float(lam*len(selected)/N)
      rows.append({'split':split,'test_t':t,'level':level,'contract':c,'K_cap':K,'N_units':N,
                   'lambda_assumed':lam,'policy':policy,'selected':len(selected),
                   'test_risk_change':realized_risk,'modeled_procurement_penalty':realized_price,
                   'test_total_net_change':realized_risk+realized_price})
R=pd.DataFrame(rows);assert len(R)==4*2*3*4*5*3
R.to_csv(O/'R532_cost_frontier_1440.csv',index=False)
S=R.groupby(['level','lambda_assumed','policy']).agg(mean_net=('test_total_net_change','mean'),mean_risk=('test_risk_change','mean'),mean_units=('selected','mean'),
  net_benefit_cases=('test_total_net_change',lambda v:int((v<-1e-10).sum())),net_harm_cases=('test_total_net_change',lambda v:int((v>1e-10).sum()))).reset_index()
S.to_csv(O/'R532_cost_frontier_summary.csv',index=False)
print(S.to_string(index=False))
