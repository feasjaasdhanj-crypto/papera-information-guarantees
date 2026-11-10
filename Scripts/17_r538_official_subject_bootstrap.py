"""Paper A R538 official UCI HAR sample-unit audit for discordant TRAIN-selected features.
Input: user-supplied original ZIP and archived R530 raw experiment table.
No held-out data used for selecting the features; bootstrap conditional on 9 test persons.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np,pandas as pd
PATH=Path('/mnt/data'); OUT=PATH/'pa_r538/evidence'; OUT.mkdir(parents=True,exist_ok=True)
p=PATH/'pa_r538/source/R530_UCI_HAR_external.py'
sys.argv=['R530_UCI_HAR_external.py',str(PATH/'human+activity+recognition+using+smartphones.zip'),str(OUT)]
spec=importlib.util.spec_from_file_location('r530',str(p));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
U,Y,part,F,hash0=m.load();cache,freq=m.precompute(U,Y,F)
train=sorted(set(U[part=='train'])); test=sorted(set(U[part=='test']))
assert len(train)==21 and len(test)==9
archived=pd.read_csv(PATH/'pa_r533_work/R530_R530_UCI_HAR_780_真实通道任务.csv')
discordant=pd.read_csv(OUT/'R538_predefined_train_only_selection_4_contracts.csv')
reversals=discordant[(discordant.domain=='UCI_HAR')&(discordant.split=='official_21_9')]
reversal_groups=reversals[['group_id','states','local_feature','full_feature']].drop_duplicates()
rows=[];subject_rows=[];rng=np.random.default_rng(20261103); B=20000
bootstrap_idx=rng.integers(0,len(test),size=(B,len(test)))
for _,g in reversal_groups.iterrows():
 states=tuple(int(v) for v in g.states.split('-'))
 feat_set={g.local_feature,g.full_feature}
 # reconstruct participant-conditional risk for each train-chosen feature/contract
 data={}
 for feat in feat_set:
  cuts={}
  for stream in m.PAIR:
   xx=np.concatenate([cache[(u,st,stream,feat)] for u in train for st in states])
   cuts[stream]=np.quantile(xx,[1/3,2/3])
  ch={stream:m.channel(cache,train,states,stream,feat,cuts[stream]) for stream in m.PAIR}
  for ct in m.UNIQUE:
   d=m.LOSS[ct];pr=np.asarray(d['prior']);L=np.asarray(d['L'])
   actions={stream:np.argmin(((pr[:,None]*ch[stream]).T)@L,axis=1) for stream in m.PAIR}
   vals=[]
   for u in test:
    risk={}
    for stream in m.PAIR:
     testchan=m.channel(cache,[u],states,stream,feat,cuts[stream])
     errors=((pr[:,None]*testchan).T)@L
     risk[stream]=float(errors[np.arange(3),actions[stream]].sum())
    vals.append(risk['gyro']-risk['accel'])
   data[(feat,ct)]=np.asarray(vals)
   # stringent check vs archived R530 participant-aggregated held-out risk
   oldrow=archived[(archived.split=='official_21_9')&(archived.states==g.states)&(archived.feature==feat)]
   assert len(oldrow)==1
   assert abs(np.mean(vals)-oldrow.iloc[0]['deploy_signed_'+ct])<2e-12,(g.states,feat,ct,np.mean(vals),oldrow.iloc[0]['deploy_signed_'+ct])
 for ct in m.UNIQUE:
  diff=data[(g.full_feature,ct)]-data[(g.local_feature,ct)]
  draws=diff[bootstrap_idx].mean(axis=1)
  loc_data=data[(g.local_feature,ct)];full_data=data[(g.full_feature,ct)]
  locdraw=loc_data[bootstrap_idx].mean(axis=1);fulldraw=full_data[bootstrap_idx].mean(axis=1)
  stats=dict(states=g.states,contract=ct,full_feature=g.full_feature,local_feature=g.local_feature,
    n_test_subjects=len(test),local_minus_no_replacement=float(loc_data.mean()),full_minus_no_replacement=float(full_data.mean()),
    full_abs_ci_lower=float(np.quantile(fulldraw,.025)),full_abs_ci_upper=float(np.quantile(fulldraw,.975)),
    local_abs_ci_lower=float(np.quantile(locdraw,.025)),local_abs_ci_upper=float(np.quantile(locdraw,.975)),
    full_minus_local=float(diff.mean()),ci_lower=float(np.quantile(draws,.025)),ci_upper=float(np.quantile(draws,.975)),
    bootstrap_prob_full_better=float((draws<0).mean()),bootstrap_prob_full_deploy_beneficial=float((fulldraw<0).mean()),
    leave_one_subject_out_min=float(min(np.delete(diff,i).mean() for i in range(len(test)))),
    leave_one_subject_out_max=float(max(np.delete(diff,i).mean() for i in range(len(test)))))
  rows.append(stats)
  for subject,value in zip(test,diff):subject_rows.append(dict(states=g.states,contract=ct,subject=int(subject),full_minus_local=float(value),local_minus_no_replacement=float(loc_data[list(test).index(subject)]),full_minus_no_replacement=float(full_data[list(test).index(subject)])))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'R538_official_three_tasks_subject_bootstrap.csv',index=False)
pd.DataFrame(subject_rows).to_csv(OUT/'R538_official_three_tasks_per_subject.csv',index=False)
print('UCI SHA256',hash0,'test persons',test,'n verify',len(rows));print(summary[['states','contract','full_minus_local','ci_lower','ci_upper','bootstrap_prob_full_better']].to_string(index=False))
