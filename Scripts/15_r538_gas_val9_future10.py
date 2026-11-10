"""Gas sensor stricter chronological split: train batches 1..8, validation batch 9,
final test batch 10 with the SAME train 1..8 frozen quantile cutpoints and actions.
This revisits data that were examined in prior rounds; exploratory not preregistered.
"""
from pathlib import Path
import importlib.util,sys,json,hashlib
import numpy as np,pandas as pd
ROOT=Path('/mnt/data'); OUT=ROOT/'pa_r538/evidence';OUT.mkdir(parents=True,exist_ok=True)
src=ROOT/'pa_r538/source/R531_gas_temporal_external.py'
sys.argv=['R531_gas_temporal_external.py',str(ROOT/'gas+sensor+array+drift+dataset+at+different+concentrations.zip'),str(OUT)]
spec=importlib.util.spec_from_file_location('r531',str(src));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
X,batch,y,ppm=m.load()
m.splits=lambda:iter([('train1to8_val9',list(range(1,9)),[9]),('train1to8_test10',list(range(1,9)),[10])])
full,meta=m.collect(X,batch,y,ppm)
assert len(full)==320
arch=pd.read_csv(ROOT/'pa_r538/source/R531_gas_800_tasks.csv')
old=arch[arch.split=='forward_1to8_vs9'].drop(columns=['source','split']).sort_values(['gases','sensor_A']).reset_index(drop=True)
new=full[full.split=='train1to8_val9'].drop(columns=['source','split']).sort_values(['gases','sensor_A']).reset_index(drop=True)
assert list(new.columns)==list(old.columns)
num=new.select_dtypes('number').columns
err=np.max(abs(new[num].to_numpy()-old[num].to_numpy()));assert err<1e-9,err
records=[];DETAIL=[]
for triad in m.TRIPLES:
 gas='-'.join(map(str,triad));v=full[(full.split=='train1to8_val9')&(full.gases==gas)].reset_index(drop=True)
 t=full[(full.split=='train1to8_test10')&(full.gases==gas)].reset_index(drop=True)
 assert all(v.sensor_A==t.sensor_A) and all(v.sensor_B==t.sensor_B)
 l=int(v.train_r2.idxmin());f=int(v.train_Delta.idxmin())
 for loss in m.UNIQUE:
  tar=int(v['train_signed_'+loss].idxmin());s={'local':l,'full':f,'target':tar}
  val={k:float(v.loc[ix,'deploy_signed_'+loss]) for k,ix in s.items()}
  tes={k:float(t.loc[ix,'deploy_signed_'+loss]) for k,ix in s.items()}
  # choose a scheme using validation ONLY; no test10 access
  option=min(['keep','local','full','target'],key=lambda z: 0. if z=='keep' else val[z])
  val_opt_tar='target' if val['target']<0 else 'keep'
  train_opt_tar='target' if float(v.loc[tar,'train_signed_'+loss])<0 else 'keep'
  val_best_risk=0 if option=='keep' else val[option]
  test_best_risk=0 if option=='keep' else tes[option]
  test_opt_tar=0 if val_opt_tar=='keep' else tes['target']
  test_train_opt_tar=0 if train_opt_tar=='keep' else tes['target']
  entry=dict(gases=gas,contract=loss,selected_local_A=int(v.loc[l,'sensor_A']),selected_full_A=int(v.loc[f,'sensor_A']),selected_target_A=int(v.loc[tar,'sensor_A']),
     strict_local_full=int(l!=f and v.loc[f,'train_r2']>v.loc[l,'train_r2']+1e-9 and v.loc[l,'train_Delta']>v.loc[f,'train_Delta']+1e-9),
     validation_local=val['local'],validation_full=val['full'],validation_target=val['target'],
     future_test_local=tes['local'],future_test_full=tes['full'],future_test_target=tes['target'],
     validation_chosen_policy=option,future_test_val_best=test_best_risk,
     validation_target_optional=val_opt_tar,future_test_val_target_optional=test_opt_tar,
     train_target_optional=train_opt_tar,future_test_train_target_optional=test_train_opt_tar,
     future_test_keep=0.0,
     val_best_policy_risk=val_best_risk)
  records.append(entry)
  for item in ['local','full','target','val_best','val_optional_target','train_optional_target','keep']:
   valv=0 if item=='keep' else val_best_risk if item=='val_best' else (val['target'] if item=='val_optional_target' and val_opt_tar=='target' else 0 if item=='val_optional_target' else (val['target'] if item=='train_optional_target' and train_opt_tar=='target' else 0 if item=='train_optional_target' else val.get(item,np.nan)))
   tst={'local':tes['local'],'full':tes['full'],'target':tes['target'],'val_best':test_best_risk,'val_optional_target':test_opt_tar,'train_optional_target':test_train_opt_tar,'keep':0}[item]
   DETAIL.append(dict(gases=gas,contract=loss,policy=item,validation_change=float(valv),future_test_change=float(tst)))
R=pd.DataFrame(records);R.to_csv(OUT/'R538_gas_train1to8_val9_final10_80_decisions.csv',index=False)
D=pd.DataFrame(DETAIL);D.to_csv(OUT/'R538_gas_train1to8_val9_final10_policy_long.csv',index=False)
summary=D.groupby(['contract','policy']).agg(n=('gases','size'),mean_future=('future_test_change','mean'),wins_vs_keep=('future_test_change',lambda a:int((a<-1e-9).sum())),harmful_vs_keep=('future_test_change',lambda a:int((a>1e-9).sum()))).reset_index()
summary.to_csv(OUT/'R538_gas_train1to8_val9_final10_policy_summary.csv',index=False)
print('Sanity: val9 EXACT vs archival',err, 'gas source SHA256',hashlib.sha256((ROOT/'gas+sensor+array+drift+dataset+at+different+concentrations.zip').read_bytes()).hexdigest())
print(summary.pivot(index='policy',columns='contract',values='mean_future').round(6).to_string())
print('wins/harmful by policy\n',summary[summary.contract=='balanced'].to_string(index=False))
print('chosen policy counts\n',R.validation_chosen_policy.value_counts())
print('strict reversals count',R[R.contract=='balanced'].strict_local_full.sum(),' (out of 20 tasks)')
print('future validation-best better than all three selected? same action among 4; descriptive')
