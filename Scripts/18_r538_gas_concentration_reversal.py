"""R538 concentration sensitivity for training-frozen local-vs-full sensor choice.
Mode-specific heldout_R531_action keeps the R531 trained actions; comparisons descriptive.
"""
import pandas as pd
from pathlib import Path
O=Path('/mnt/data/pa_r538/evidence');D=pd.read_csv('/mnt/data/pa_r532_work/R532_1920_gas_contract_concentration.csv')
R=pd.read_csv(O/'R538_gas_20x5_triplets_train_selected_four_losses.csv')
R=R[(R.split=='official_chronological_1to9_vs10')&(R.strict_reversal==1)].copy()
rows=[]
for _,r in R.iterrows():
 for mode in ['raw','anchor','common_equal']:
  gg=D[(D.gases==r.gases)&(D.contract==r.contract)&(D['mode']==mode)]
  local=gg[(gg.sensor_A==r.local_sensor_A)&(gg.sensor_B==r.local_sensor_B)]
  full=gg[(gg.sensor_A==r.full_sensor_A)&(gg.sensor_B==r.full_sensor_B)]
  assert len(local)==len(full)==1,(r.gases,r.contract,mode)
  lr=float(local.heldout_R531_action.iloc[0]);fr=float(full.heldout_R531_action.iloc[0]);
  rows.append(dict(gases=r.gases,contract=r.contract,mode=mode,
       local_sensor=f'{int(r.local_sensor_A)}→{int(r.local_sensor_B)}',full_sensor=f'{int(r.full_sensor_A)}→{int(r.full_sensor_B)}',
       local_change=lr,full_change=fr,full_minus_local=fr-lr,
       winner='full' if fr<lr-1e-9 else 'local' if lr<fr-1e-9 else 'tie',
       full_benefit_vs_keep=(fr<-1e-9),local_benefit_vs_keep=(lr<-1e-9)))
z=pd.DataFrame(rows);z.to_csv(O/'R538_gas_official_reversal_concentration_control.csv',index=False)
print(z[(z.contract=='balanced')][['gases','mode','local_change','full_change','winner']].to_string(index=False))
print('\nby mode all 12 (non-independent)\n',z.groupby(['mode','winner']).size().to_string())
print('\nloss switches\n',z.pivot_table(index=['gases','contract'],columns='mode',values='winner',aggfunc='first').to_string())
