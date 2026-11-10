from pathlib import Path
import pandas as pd, numpy as np
O=Path('/mnt/data/pa_r542_analysis');src=Path('/mnt/data/pa_r540_work')
pol=[]
for process,sub in [('calibrated','R540_EEG_RESULTS'),('global_train_only','R540_EEG_RAW_RESULTS')]:
 d=src/sub
 p=pd.read_csv(d/'R540_EEG_secondary_5fold_1280portfolios.csv')
 inf=pd.read_csv(d/'R540_EEG_secondary_5fold_information.csv')
 for fold, g in p.groupby('fold'):
  sub_info=inf[inf.fold==fold].set_index('index').sort_index()
  assert len(g)==256
  for budget in (2,4,6):
   masks=g[g.n_replacements==budget].copy()
   masks['info_full_score']=[sum(sub_info.loc[j,'full_delta'] for j in range(8) if int(mask)>>j&1) for mask in masks['mask']]
   masks['info_local_score']=[sum(sub_info.loc[j,'local_r2'] for j in range(8) if int(mask)>>j&1) for mask in masks['mask']]
   for contract in ('balanced','miss_T2_x3','false_alarm_T2_x3','reject035'):
    # matched labels available to every policy, validation-only selection inside training-derived shortlist
    allrank=masks.sort_values(['val_'+contract,'mask'])
    setups={'target_all_val':int(allrank.iloc[0]['mask'])}
    for metric in ('full','local'):
     order=masks.sort_values(['info_'+metric+'_score','mask'])
     setups[metric+'_info_only']=int(order.iloc[0]['mask'])
     for k in (5,10):
      setups[metric+'_top'+str(k)+'_val']=int(order.head(k).sort_values(['val_'+contract,'mask']).iloc[0]['mask'])
    # validation can select between full/local information masks but not the other 26/68 masks
    info_masks=masks[masks['mask'].isin([setups['full_info_only'],setups['local_info_only']])]
    setups['full_local_val_pick']=int(info_masks.sort_values(['val_'+contract,'mask']).iloc[0]['mask'])
    for name,mask in setups.items():
     r=masks[masks['mask']==mask].iloc[0]
     pol.append({'processing':process,'fold':int(fold),'budget':budget,'contract':contract,'policy':name,'mask':mask,'val_risk':r['val_'+contract],'test_risk':r['test_'+contract]})
D=pd.DataFrame(pol);D.to_csv(O/'R542_equal_supervision_policy_choices.csv',index=False)
# compare 5-fold OOF average each 1 test subject once, weighted by fold subjects
foldmap=pd.read_csv(src/'R540_EEG_RESULTS/R540_EEG_secondary_5fold_subject_map.csv')
counts=foldmap.groupby('fold').subject.nunique().to_dict()
D['weight']=D.fold.map(counts)
res=[]
for (proc,budget,contract),df in D.groupby(['processing','budget','contract']):
 r=df.pivot(index='fold',columns='policy',values='test_risk')
 assert len(r)==5
 for benchmark in ('full_info_only','local_info_only','full_local_val_pick','full_top5_val','full_top10_val','local_top5_val','local_top10_val'):
  delta=r.target_all_val-r[benchmark]
  mean=np.average(delta,weights=[counts[f] for f in delta.index])
  res.append({'processing':proc,'budget':budget,'contract':contract,'reference':benchmark,'mean_target_minus_ref':mean,'folds_target_better':int((delta<0).sum()),'folds_target_worse':int((delta>0).sum()),'n_test_subjects':106})
R=pd.DataFrame(res);R.to_csv(O/'R542_equal_supervision_mean_5fold.csv',index=False)
print('per-contract mean across 3 budgets / 106 subjects; negative target better')
print(R.query("processing=='calibrated'").groupby(['contract','reference']).mean_target_minus_ref.mean().unstack().round(5).to_string())
print('noncalibrated')
print(R.query("processing=='global_train_only'").groupby(['contract','reference']).mean_target_minus_ref.mean().unstack().round(5).to_string())
print('outputs',len(D),len(R))
