"""Apply original TRAIN selected {local,full,target} policies to R532 concentration scenarios.
No re-ranking with test/standardized data. Fixed original trained actions from R531.
"""
import pandas as pd
from pathlib import Path
O=Path('/mnt/data/pa_r538/evidence');allpol=pd.read_csv(O/'R538_four_action_choices_all_strict_reversals.csv')
allpol=allpol[(allpol.domain=='GAS_DRIFT')&(allpol.split=='official_chronological_1to9_vs10')]
control=pd.read_csv('/mnt/data/pa_r532_work/R532_1920_gas_contract_concentration.csv')
rows=[]
for _,r in allpol.iterrows():
 label=str(r.chosen_item)
 a,b=[int(s) for s in label.split('to')]
 for mode in ['raw','anchor','common_equal']:
  d=control[(control.gases==r.gases)&(control.contract==r.contract)&(control['mode']==mode)&(control.sensor_A==a)&(control.sensor_B==b)]
  assert len(d)==1,(r.gases,r.contract,mode,label)
  val=float(d.heldout_R531_action.iloc[0]);
  rows.append(dict(gases=r.gases,contract=r.contract,mode=mode,policy=r.policy,selected_sensor_pair=label,risk_change=val))
res=pd.DataFrame(rows);res.to_csv(O/'R538_gas_target_local_full_concentration_audit.csv',index=False)
pivot=res.pivot_table(index=['gases','contract','mode'],columns='policy',values='risk_change').reset_index()
pivot['target_vs_local']=pivot.target-pivot.local;pivot['target_vs_full']=pivot.target-pivot.full;pivot['target_beneficial']=pivot.target<0
pivot.to_csv(O/'R538_gas_three_policy_concentration_pivot.csv',index=False)
print(pivot[pivot.contract.isin(['balanced','falsealarm3'])][['gases','contract','mode','local','full','target','target_beneficial']].to_string(index=False))
