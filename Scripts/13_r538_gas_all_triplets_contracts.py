"""All 20 prespecified 3-gas tasks, not cherry-picked strict reversal subset.
Train-only selection among eight physical replacement options.
"""
import numpy as np,pandas as pd
from pathlib import Path
O=Path('/mnt/data/pa_r538/evidence');D=pd.read_csv('/mnt/data/pa_r538/source/R531_gas_800_tasks.csv')
LOSS=['balanced','miss3','reject035','falsealarm3'];rows=[]
for (split,gases),g in D.groupby(['split','gases'],sort=False):
 l=g.loc[g.train_r2.idxmin()];f=g.loc[g.train_Delta.idxmin()]
 strict=int(l.name!=f.name and f.train_r2-l.train_r2>1e-9 and l.train_Delta-f.train_Delta>1e-9)
 for loss in LOSS:
  t=g.loc[g['train_signed_'+loss].idxmin()]
  lr=float(l['deploy_signed_'+loss]);fr=float(f['deploy_signed_'+loss]);tr=float(t['deploy_signed_'+loss]);
  rows.append(dict(split=split,gases=gases,loss=loss,strict_reversal=strict,
    local_A=int(l.sensor_A),local_B=int(l.sensor_B),full_A=int(f.sensor_A),full_B=int(f.sensor_B),target_A=int(t.sensor_A),target_B=int(t.sensor_B),
    local_test=lr,full_test=fr,target_test=tr,
    target_win_vs_both=int(tr<min(lr,fr)-1e-9),target_worse_than_both=int(tr>max(lr,fr)+1e-9),
    target_prevents_harm=int(tr<-1e-9 and lr>1e-9 and fr>1e-9),
    target_deploy_positive=int(tr>1e-9),
    full_minus_local=fr-lr,target_minus_full=tr-fr))
R=pd.DataFrame(rows);R.to_csv(O/'R538_gas_all_20_triplets_physical_selection.csv',index=False)
summary=R.groupby(['split','loss']).agg(tasks=('gases','size'),strict=('strict_reversal','sum'),target_win_both=('target_win_vs_both','sum'),target_worse_both=('target_worse_than_both','sum'),target_prevents_harm=('target_prevents_harm','sum'),target_deploy_harmful=('target_deploy_positive','sum'),target_minus_full=('target_minus_full','mean')).reset_index();summary.to_csv(O/'R538_gas_all_20_triplets_summary.csv',index=False)
print(summary[summary.split=='official_chronological_1to9_vs10'].to_string(index=False))
print('time-forward count target prevents harm\n',summary.groupby('loss').agg(n_tests=('tasks','sum'),target_win_both=('target_win_both','sum'),target_prevents_harm=('target_prevents_harm','sum'),target_deploy_harmful=('target_deploy_harmful','sum')).to_string())
# concentration scenarios for all 20, 4 losses primary, frozen training choices
C=pd.read_csv('/mnt/data/pa_r532_work/R532_1920_gas_contract_concentration.csv')
sub=R[R.split=='official_chronological_1to9_vs10'];rows=[]
for _,r in sub.iterrows():
 for mode in ['raw','anchor','common_equal']:
  vals={}
  for pol in ['local','full','target']:
   a,b=int(r[pol+'_A']),int(r[pol+'_B'])
   d=C[(C.gases==r.gases)&(C.contract==r.loss)&(C['mode']==mode)&(C.sensor_A==a)&(C.sensor_B==b)]
   assert len(d)==1;vals[pol]=float(d.heldout_R531_action.iloc[0])
  rows.append(dict(gases=r.gases,loss=r.loss,mode=mode,strict_reversal=r.strict_reversal,**vals,
    target_prevents_harm=int(vals['target']<0 and vals['local']>0 and vals['full']>0),
    target_beneficial=int(vals['target']<0),target_win_both=int(vals['target']<min(vals['local'],vals['full'])-1e-9)))
T=pd.DataFrame(rows);T.to_csv(O/'R538_gas_all_20_concentration_three_policy.csv',index=False)
res=T.groupby(['loss','mode']).agg(n=('gases','size'),target_prevents_harm=('target_prevents_harm','sum'),target_beneficial=('target_beneficial','sum'),target_win_both=('target_win_both','sum')).reset_index();res.to_csv(O/'R538_gas_all_20_concentration_summary.csv',index=False)
print('concentration ALL cases\n',res.to_string(index=False))
