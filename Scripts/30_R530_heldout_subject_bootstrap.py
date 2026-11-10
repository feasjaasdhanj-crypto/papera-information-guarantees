"""UCI HAR official test: subject-level conditional nonparametric bootstrap.
Selection, bin cutpoints, and bin-to-action maps all trained using official 21 subjects.
For fixed actions and balanced class priors, expected heldout loss is a linear average of
nine participant-level conditional losses, so sample-with-replacement is appropriate.
Not a sampling model for the train participant cohort; 12 secondary repartitions correlated.
"""
import sys,json,importlib.util
from pathlib import Path
import numpy as np,pandas as pd
zip_file=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.argv=['R530_UCI_HAR_external.py',str(zip_file),str(out)]
spec=importlib.util.spec_from_file_location('r530',str(out/'R530_UCI_HAR_external.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
U,Y,part,F,hash0=m.load();cache,freq=m.precompute(U,Y,F)
train=sorted(set(U[part=='train']));test=sorted(set(U[part=='test']))
assert len(test)==9
old=pd.read_csv(out/'R530_UCI_HAR_351_策略结果.csv')
old=old[old.split=='official_21_9'].copy()
perperson={}
for states in m.STATES:
 for feat in m.FEATURES:
  task='-'.join(map(str,states))+'_'+feat
  cuts={}
  for stream in m.PAIR:
   xx=np.concatenate([cache[(u,st,stream,feat)] for u in train for st in states])
   cuts[stream]=np.quantile(xx,[1/3,2/3])
  ch={stream:m.channel(cache,train,states,stream,feat,cuts[stream]) for stream in m.PAIR}
  for ct,d in m.LOSS.items():
   pr=np.asarray(d['prior']);L=np.asarray(d['L'])
   actions={stream:np.argmin(((pr[:,None]*ch[stream]).T)@L,axis=1) for stream in m.PAIR}
   values=[]
   for u in test:
    risk={}
    for stream in m.PAIR:
     testchan=m.channel(cache,[u],states,stream,feat,cuts[stream])
     errors=((pr[:,None]*testchan).T)@L
     risk[stream]=float(errors[np.arange(3),actions[stream]].sum())
    values.append(risk['gyro']-risk['accel'])
   perperson[(task,ct)]=np.asarray(values)
rows=[];N=10000;rng=np.random.default_rng(5302026)
bsidx=rng.integers(0,len(test),size=(N,len(test)))
for budget,db in old.groupby('budget_fraction'):
  db=db.set_index('policy')
  full=[int(x) for x in db.loc['full_Delta','selected_signature'].split(';')]
  for ct in m.UNIQUE:
   for policy in [ct,'robust_individual','robust_global_minimax','local_r2']:
    selected=[int(x) for x in db.loc[policy,'selected_signature'].split(';')]
    def mean_person(ids):
     return np.mean([perperson[('-'.join(map(str,m.STATES[i//3]))+'_'+m.FEATURES[i%3],ct)] for i in ids],axis=0)
    subj_diff=mean_person(selected)-mean_person(full)
    assert abs(subj_diff.mean()-(db.loc[policy,'deploy_signed_'+ct]-db.loc['full_Delta','deploy_signed_'+ct]))<1e-12,(budget,ct,policy)
    boots=subj_diff[bsidx].mean(axis=1)
    leaveone=np.array([np.delete(subj_diff,i).mean() for i in range(len(test))])
    rows.append(dict(source='UCI_HAR',budget_fraction=budget,contract=ct,policy=policy,
      official_test_mean=float(subj_diff.mean()),test_subjects=9,
      subject_bootstrap_95_lower=float(np.quantile(boots,.025)),
      subject_bootstrap_95_upper=float(np.quantile(boots,.975)),
      bootstrap_probability_better=float((boots<0).mean()),
      leave_one_subject_out_min=float(leaveone.min()),leave_one_subject_out_max=float(leaveone.max()),
      heldout_subject_values=';'.join(f'{i}:{x:.9f}' for i,x in zip(test,subj_diff))))
summary=pd.DataFrame(rows)
summary.to_csv(out/'R530_官方九受试者_抽样不确定性.csv',index=False)
report={'bootstrap_subjects':len(test),'iterations':N,'seed':5302026,
 'comparisons':len(rows),'exact_prior_loss_equivalence':True,'reconstructed_primary_vs_prior_results':True,
 'matched_bootstrap_ci_below_zero':int(((summary.policy==summary.contract)&(summary.subject_bootstrap_95_upper<0)).sum()),
 'total_matched':int((summary.policy==summary.contract).sum()),
 'method':'conditional fixed learned rules, resample nine disjoint heldout participants (not windows)',
 'warning':'Conditional on the training subjects and selected tasks; bootstrap intervals not independent external study replication; not preregistered study.'}
(out/'R530_bootstrap_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(summary[(summary.budget_fraction==.5)&(summary.policy==summary.contract)][['contract','official_test_mean','subject_bootstrap_95_lower','subject_bootstrap_95_upper','bootstrap_probability_better']].to_string(index=False))
print(report)
