"""Cross-discipline 16-physical-sensor replacement ranking reversal, train-only selection,
chronological forward holdout. All 20 triplets included, multiple folds descriptive only.
"""
import pandas as pd, numpy as np
from pathlib import Path
OUT=Path('/mnt/data/pa_r538/evidence'); data=pd.read_csv('/mnt/data/pa_r538/source/R531_gas_800_tasks.csv')
contracts=['balanced','miss3','reject035','falsealarm3'];rows=[];cand=[]
for (split,gases),g in data.groupby(['split','gases'],sort=False):
 loc=g.loc[g.train_r2.idxmin()]; full=g.loc[g.train_Delta.idxmin()]
 lm=float(full.train_r2-loc.train_r2);fm=float(loc.train_Delta-full.train_Delta)
 strict=loc.name!=full.name and lm>1e-9 and fm>1e-9
 for ct in contracts:
  lr=float(loc['deploy_signed_'+ct]);fr=float(full['deploy_signed_'+ct]);
  rows.append(dict(split=split,gases=gases,n_candidates=len(g),strict_reversal=int(strict),
    local_sensor_A=int(loc.sensor_A),local_sensor_B=int(loc.sensor_B),
    full_sensor_A=int(full.sensor_A),full_sensor_B=int(full.sensor_B),
    train_local_r2=float(loc.train_r2),train_local_Delta=float(loc.train_Delta),
    train_full_r2=float(full.train_r2),train_full_Delta=float(full.train_Delta),
    local_margin=lm,full_margin=fm,contract=ct,
    local_deploy_change=lr,full_deploy_change=fr,full_minus_local=fr-lr,
    winner='full' if fr<lr-1e-10 else ('local' if lr<fr-1e-10 else 'tie'),
    local_harm_full_benefit=int(lr>1e-10 and fr<-1e-10),
    local_benefit_full_harm=int(lr<-1e-10 and fr>1e-10)))
rows=pd.DataFrame(rows);rows.to_csv(OUT/'R538_gas_20x5_triplets_train_selected_four_losses.csv',index=False)
strict=rows[rows.strict_reversal==1]; sm=strict[strict.contract=='balanced']
agg=strict.groupby(['split','contract']).agg(n=('gases','size'),full_wins=('winner',lambda s:(s=='full').sum()),local_wins=('winner',lambda s:(s=='local').sum()),sign_local_harm_full_help=('local_harm_full_benefit','sum'),sign_local_help_full_harm=('local_benefit_full_harm','sum'),mean_full_minus_local=('full_minus_local','mean')).reset_index()
agg.to_csv(OUT/'R538_gas_strict_reversal_summary.csv',index=False)
print('all groups',len(rows)//4,'strict group count',len(sm));print('by temporal split\n',sm.groupby('split').agg(groups=('gases','size'),fullwins=('winner',lambda s:(s=='full').sum()),localwins=('winner',lambda s:(s=='local').sum()),sign_a=('local_harm_full_benefit','sum'),sign_b=('local_benefit_full_harm','sum')).to_string())
print('official strict individual\n',strict[(strict.split=='official_chronological_1to9_vs10') & (strict.contract=='balanced')][['gases','local_sensor_A','local_sensor_B','full_sensor_A','full_sensor_B','local_margin','full_margin','local_deploy_change','full_deploy_change','winner']].to_string(index=False))
# setting-switch count for strict groups
switch=strict.pivot(index=['split','gases'],columns='contract',values='winner').reset_index()
switch['switches_loss_winner']=switch[contracts].apply(lambda r:len(set(r)-{'tie'})>1,axis=1)
switch.to_csv(OUT/'R538_gas_decision_contract_switches.csv',index=False)
print('Loss-dependent switches',int(switch.switches_loss_winner.sum()),'/',len(switch),'official',int(switch[switch.split=='official_chronological_1to9_vs10'].switches_loss_winner.sum()))
print('official all losses\n',agg[agg.split=='official_chronological_1to9_vs10'].to_string(index=False))
