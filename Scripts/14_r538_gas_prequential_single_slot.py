"""Paper A R538: physical one-replacement scientific decision under 4 prospective forward splits.
Training only ranks 8 paired sensor swaps. Validation only calibrates opt-in and picks
from four fixed options. All future test actions were frozen at training batch t-2.
All counts are descriptive within one original chemical measurement study.
"""
import pandas as pd, numpy as np
from pathlib import Path
O=Path('/mnt/data/pa_r538/evidence')
P=pd.read_csv('/mnt/data/pa_r537/R537_RELEASE/04_evidence/R537_INPUT_R531_640.csv')
H=pd.read_csv('/mnt/data/pa_r538/source/R531_gas_800_tasks.csv')
fold_map={'train1to5_val6_test7':'forward_1to5_vs6','train1to6_val7_test8':'forward_1to6_vs7','train1to7_val8_test9':'forward_1to7_vs8','train1to8_val9_test10':'forward_1to8_vs9'}
q=[]
for fold,hname in fold_map.items():
 a=P[P.split==fold].copy();b=H[H.split==hname][['gases','sensor_A','sensor_B','train_r2','train_Delta']].copy()
 a=a.merge(b,on=['gases','sensor_A','sensor_B'],how='inner',validate='one_to_one',suffixes=('','_history'))
 assert len(a)==160
 assert np.allclose(a.train_Delta,a.train_Delta_history,atol=1e-12)
 q.append(a.drop(columns=['train_Delta_history']))
P=pd.concat(q,ignore_index=True)
rows=[]; choices=[]
for (split,gases),g in P.groupby(['split','gases'],sort=False):
 local=g.loc[g.train_r2.idxmin()];full=g.loc[g.train_Delta.idxmin()]
 strict=int(local.name!=full.name and full.train_r2-local.train_r2>1e-9 and local.train_Delta-full.train_Delta>1e-9)
 for ct in ['balanced','miss3','reject035','falsealarm3']:
  tar=g.loc[g['train_signed_'+ct].idxmin()]
  train={k:float(r['train_signed_'+ct]) for k,r in [('local',local),('full',full),('target',tar)]}
  val={k:float(r['val_signed_'+ct]) for k,r in [('local',local),('full',full),('target',tar)]}
  test={k:float(r['test_signed_'+ct]) for k,r in [('local',local),('full',full),('target',tar)]}
  best=min(['keep','local','full','target'],key=lambda k: 0 if k=='keep' else val[k])
  optional='target' if val['target']<0 else 'keep'
  conservative='target' if train['target']<0 and val['target']<0 else 'keep'
  # Even lower-cost than choosing via val: only deploy if *both* training and validation
  # favor this swap over keeping the original and separately over the full-info choice.
  robust='target' if min(-train['target'],-val['target'],train['full']-train['target'],val['full']-val['target'])>0 else 'keep'
  policy={'local':'local','full':'full','target':'target','keep':'keep',
   'val_optional_target':optional,'train_and_val_optional':conservative,
   'val_best_four':best,'val_target_dominance':robust}
  for name,item in policy.items():
   out=dict(split=split,gas_triplet=gases,contract=ct,strict_reversal=strict,policy=name,chosen_action=item,
    actual_sensor_from=int(({'local':local,'full':full,'target':tar}[item]).sensor_A) if item!='keep' else 0,
    actual_sensor_to=int(({'local':local,'full':full,'target':tar}[item]).sensor_B) if item!='keep' else 0,
    train_change=train[item] if item!='keep' else 0.,val_change=val[item] if item!='keep' else 0.,
    future_change=test[item] if item!='keep' else 0.,replaced=int(item!='keep'))
   rows.append(out)
R=pd.DataFrame(rows);R.to_csv(O/'R538_prequential_four_folds_one_physical_slot.csv',index=False)
summary=R.groupby(['split','contract','policy']).agg(n=('gas_triplet','size'),future_mean=('future_change','mean'),future_beneficial=('future_change',lambda x:int((x<-1e-9).sum())),future_harmful=('future_change',lambda x:int((x>1e-9).sum())),mean_replacements=('replaced','mean')).reset_index()
summary.to_csv(O/'R538_prequential_four_folds_one_slot_summary.csv',index=False)
print('n fully prospective decisions',len(R),'20 triplets ×4 losses ×4 folds ×8 policies')
print('overall descriptive means by fixed policy\n',R.groupby(['contract','policy']).agg(mean_future=('future_change','mean'),mean_replacements=('replaced','mean'),harmful=('future_change',lambda s:int((s>1e-9).sum()))).round(6).to_string())
print('balanced by split\n',summary[summary.contract=='balanced'].pivot(index='split',columns='policy',values='future_mean').round(5).to_string())
print('count strict group by fold',R[R.policy=='local'].groupby('split').strict_reversal.sum().to_dict())
