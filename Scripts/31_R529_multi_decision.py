"""Paper A R529: independently recompute per-split 3-state channels from archived four study RDS.
Train-only selection; four predeclared downstream science-decision contracts. Requires Python numpy pandas.
RDS copied from R526 directly inside parent archive; no unarchived data or external network access.
"""
import io,zipfile,lzma,struct,json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd

PARENT=Path(sys.argv[1]); OUT=Path(sys.argv[2]); OUT.mkdir(parents=True,exist_ok=True)

def get_parent_data():
 with zipfile.ZipFile(PARENT) as a:
  b=a.read('00_必须保留_完整科研交接包/ORIGINAL_R527_COMPLETE.zip')
  with zipfile.ZipFile(io.BytesIO(b)) as c:
   d=c.read('00_必须保留_完整科研交接包/ORIGINAL_R526_COMPLETE.zip')
   with zipfile.ZipFile(io.BytesIO(d)) as e:
    base='03_直接保存的四来源轻量化数据/'
    payload={ds:e.read(base+ds+'_ANALYSIS_READY.rds') for ds in ['WISDM','DSADS','MEX','MHEALTH']}
    r526=e.read('01_新的结果与研究结论/R526_task_level_natural_bayes_vs_delta.csv')
 return payload,pd.read_csv(io.BytesIO(r526))

def read_rds(b):
 st=io.BytesIO(lzma.decompress(b)); assert st.read(2)==b'X\n'
 def ri(): return struct.unpack('>i',st.read(4))[0]
 v,w,m,n=[ri() for _ in range(4)];st.read(n);flags=ri();assert flags&255==19
 ncol=ri();assert ncol==6;cols=[]
 for _ in range(ncol):
  f=ri();kind=f&255;n=ri()
  if kind==16:
   ss=[]
   for i in range(n):
    cc=ri();assert cc&255==9;size=ri();ss.append(None if size<0 else st.read(size).decode('utf8'))
   cols.append(ss)
  elif kind==14:cols.append(np.frombuffer(st.read(8*n),dtype='>f8').astype(float))
  else:raise ValueError(kind)
 return pd.DataFrame(dict(zip(['unit','state','stream','mag_mean','mag_sd','x_sd'],cols)))

def cache_dataset(df):
 out={}
 for (unit,stream,state),g in df.groupby(['unit','stream','state'],sort=False):
  out[(unit,stream,state)]={feat:g[feat].to_numpy(float)[np.isfinite(g[feat].to_numpy(float))] for feat in ['mag_mean','mag_sd','x_sd']}
 return out

def fit(cache,units,states,pair,feat,cuts):
 P=[];Q=[]
 for state in states:
  found=[]
  for unit in units:
   a=cache.get((unit,pair[0],state),{}).get(feat,[])
   b=cache.get((unit,pair[1],state),{}).get(feat,[])
   if len(a)>=2 and len(b)>=2:
    found.append((np.bincount(np.searchsorted(cuts[0],a,side='right'),minlength=3)/len(a),np.bincount(np.searchsorted(cuts[1],b,side='right'),minlength=3)/len(b)))
  if len(found)<2:raise ValueError(('missing',state,len(found)))
  P.append(np.mean([x[0] for x in found],axis=0));Q.append(np.mean([x[1] for x in found],axis=0))
 return np.asarray(P),np.asarray(Q)

def risks(P,priors,L):
 return float(np.min(((priors[:,None]*P).T)@L,axis=1).sum())

def deployed_risk(train_ch,test_ch,priors,L):
    # The actions in each discretized observation bin are fitted only on training participants.
    train_losses=((priors[:,None]*train_ch).T)@L
    actions=np.argmin(train_losses,axis=1)
    test_losses=((priors[:,None]*test_ch).T)@L
    return float(test_losses[np.arange(len(actions)),actions].sum())

LOSS={
 '01_balanced_0_1':dict(prior=[1/3]*3,L=[[0,1,1],[1,0,1],[1,1,0]],description='balanced three-way 0-1 classification'),
 '02_third_state_cost3':dict(prior=[1/3]*3,L=[[0,1,1],[1,0,1],[3,3,0]],description='third true state missed incurs 3x other mistakes'),
 '03_reject_cost035':dict(prior=[1/3]*3,L=[[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]],description='three labels plus reject with fixed 0.35 penalty'),
 '04_third_state_prior060':dict(prior=[.2,.2,.6],L=[[0,1,1],[1,0,1],[1,1,0]],description='third true state prior 0.60, others 0.20 (exactly equivalent to scenario 02 modulo multiplier)'),
 '05_third_state_false_alarm_cost3':dict(prior=[1/3]*3,L=[[0,1,3],[1,0,3],[1,1,0]],description='false alarm: predicting the third state when true state is 1 or 2 costs 3x'),
}

def main():
 payload,ref=get_parent_data()
 rows=[];skip=[];recover=[]
 for di,ds in enumerate(['WISDM','DSADS','MEX','MHEALTH']):
  df=read_rds(payload[ds]);cache=cache_dataset(df);subjects=np.array(sorted(df.unit.unique()));n=len(subjects)
  reps={'WISDM':20,'DSADS':20,'MEX':8,'MHEALTH':8}[ds]
  print('dataset',ds,'participants',n,'records',len(df),flush=True)
  for rep in range(1,reps+1):
   rng=np.random.default_rng(526000+di*1000+rep)
   order=subjects[rng.permutation(n)]
   nt=int(.55*n);nv=max(2,int(.20*n));train=order[:nt];test=order[nt+nv:]
   rr=ref[(ref.dataset==ds)&(ref.repeat_index==rep)]
   for ix,r in rr.iterrows():
    states=r.states.split('-');pair=(r.streamA,r.streamB);feat=r.feature
    cuts=[]
    for st in pair:
     v=[cache[(u,st,s)][feat] for u in train for s in states if (u,st,s) in cache and len(cache[(u,st,s)][feat])]
     if not v:raise RuntimeError((ds,rep,st,states))
     cut=np.quantile(np.concatenate(v),[1/3,2/3],method='linear')
     assert cut[0]<cut[1];cuts.append(cut)
    Pt,Qt=fit(cache,train,states,pair,feat,cuts)
    Pe,Qe=fit(cache,test,states,pair,feat,cuts)
    d=dict(dataset=ds,repeat_index=rep,states=r.states,streamA=r.streamA,streamB=r.streamB,feature=feat,
           train_Delta=r.train_Delta,test_Delta=r.test_Delta,train_r2=r.train_r2,test_r2=r.test_r2)
    for key,meta in LOSS.items():
     pri=np.array(meta['prior']);loss=np.array(meta['L'])
     d['train_P_'+key]=risks(Pt,pri,loss)
     d['train_Q_'+key]=risks(Qt,pri,loss)
     d['test_P_'+key]=risks(Pe,pri,loss)
     d['test_Q_'+key]=risks(Qe,pri,loss)
     d['deploy_test_P_'+key]=deployed_risk(Pt,Pe,pri,loss)
     d['deploy_test_Q_'+key]=deployed_risk(Qt,Qe,pri,loss)
     d['deploy_test_signed_'+key]=d['deploy_test_Q_'+key]-d['deploy_test_P_'+key]
     assert d['deploy_test_P_'+key]>=d['test_P_'+key]-1e-12 and d['deploy_test_Q_'+key]>=d['test_Q_'+key]-1e-12
     d['train_signed_'+key]=d['train_Q_'+key]-d['train_P_'+key]
     d['test_signed_'+key]=d['test_Q_'+key]-d['test_P_'+key]
    recover.append([d['train_P_01_balanced_0_1']-r.train_bayes_P,
     d['train_Q_01_balanced_0_1']-r.train_bayes_Q,
     d['test_P_01_balanced_0_1']-r.test_bayes_P,
     d['test_Q_01_balanced_0_1']-r.test_bayes_Q])
    rows.append(d)
  print('  recovered',sum(x['dataset']==ds for x in rows),'tasks',flush=True)
 base=pd.DataFrame(rows);err=float(np.max(np.abs(recover)));print('BAYES_RECONSTRUCTION_MAX_ERROR',err,flush=True)
 assert len(rows)==len(ref)==1008 and err<1e-12
 for prefix in ['train_P','train_Q','test_P','test_Q','deploy_test_P','deploy_test_Q']:
  col1=base[prefix+'_02_third_state_cost3']; col2=base[prefix+'_04_third_state_prior060']
  assert np.allclose(col1,(5/3)*col2,atol=1e-12)
 print('PRIOR_LOSS_EQUIVALENCE VERIFIED all train/test/deploy risk channels',flush=True)
 base.to_csv(OUT/'R529_five_settings_1008_tasks.csv',index=False)
 strategies=['full_Delta','local_r2']+list(LOSS)
 selected=[];regrets=[]
 for (ds,rep),g in base.groupby(['dataset','repeat_index']):
  g=g.reset_index(drop=True);K=round(len(g)*.5)
  for policy in strategies:
   col={'full_Delta':'train_Delta','local_r2':'train_r2'}.get(policy,'train_signed_'+policy)
   g['_sortscore']=g[col].round(12)
   t=g.sort_values(['_sortscore','states','streamA','feature'],kind='stable').head(K)
   result=dict(dataset=ds,split=rep,policy=policy,train_K=K,total_n=len(g),selected_signature=';'.join(str(i) for i in sorted(t.index)))
   for task in LOSS:
    result['test_signed_'+task]=t['test_signed_'+task].mean()
    result['deploy_test_signed_'+task]=t['deploy_test_signed_'+task].mean()
    result['deploy_test_harm_'+task]=np.maximum(t['deploy_test_signed_'+task],0).mean()
    result['test_harm_'+task]=np.maximum(t['test_signed_'+task],0).mean()
    result['train_signed_'+task]=t['train_signed_'+task].mean()
   result['test_Delta']=t.test_Delta.mean()
   result['train_Delta']=t.train_Delta.mean()
   selected.append(result)
 res=pd.DataFrame(selected)
 res.to_csv(OUT/'R529_56split_7policies_5decision_outcomes.csv',index=False)
 summaries=[]
 for (ds,policy),g in res.groupby(['dataset','policy']):
  row=dict(dataset=ds,policy=policy,split_count=len(g),test_Delta=float(g.test_Delta.mean()))
  for task in LOSS:
   row['test_signed_'+task]=float(g['test_signed_'+task].mean())
   row['deploy_test_signed_'+task]=float(g['deploy_test_signed_'+task].mean())
   row['test_harm_'+task]=float(g['test_harm_'+task].mean())
  summaries.append(row)
 s=pd.DataFrame(summaries);s.to_csv(OUT/'R529_source_policy_means.csv',index=False)
 contract_switch=[]
 for (ds,rep),g in res.groupby(['dataset','split']):
  g=g.set_index('policy')
  target=list(LOSS)
  for i,x in enumerate(target):
   for y in target[i+1:]:
    contract_switch.append(dict(dataset=ds,split=rep,contract_A=x,contract_B=y,
     different_selection=int(g.loc[x,'selected_signature']!=g.loc[y,'selected_signature']),
     A_on_A=g.loc[x,'test_signed_'+x],B_on_A=g.loc[y,'test_signed_'+x],
     A_on_B=g.loc[x,'test_signed_'+y],B_on_B=g.loc[y,'test_signed_'+y],
     preference_A_test=float(g.loc[x,'test_signed_'+x]<g.loc[y,'test_signed_'+x]-1e-12),
     preference_B_test=float(g.loc[y,'test_signed_'+y]<g.loc[x,'test_signed_'+y]-1e-12)))
 cs=pd.DataFrame(contract_switch);cs.to_csv(OUT/'R529_scenario_switches.csv',index=False)
 comparison=[]
 for (ds,rep),g in res.groupby(['dataset','split']):
  g=g.set_index('policy')
  for task in LOSS:
   comparison.append(dict(dataset=ds,split=rep,task=task,
     target_vs_full_test_signed=g.loc[task,'test_signed_'+task]-g.loc['full_Delta','test_signed_'+task],
     target_vs_full_deploy_signed=g.loc[task,'deploy_test_signed_'+task]-g.loc['full_Delta','deploy_test_signed_'+task],
     target_vs_local_test_signed=g.loc[task,'test_signed_'+task]-g.loc['local_r2','test_signed_'+task],
     target_vs_full_test_delta=g.loc[task,'test_Delta']-g.loc['full_Delta','test_Delta'],
     target_vs_full_test_harm=g.loc[task,'test_harm_'+task]-g.loc['full_Delta','test_harm_'+task],
     changed_target_vs_full=int(g.loc[task,'selected_signature']!=g.loc['full_Delta','selected_signature'])))
 cmp=pd.DataFrame(comparison);cmp.to_csv(OUT/'R529_target_vs_full_280_pairs.csv',index=False)
 report={'reconstructed_0_1_max_error':err,'orig_sources':4,'subject_disjoint_splits':56,'task_records':len(base),
 'policy_records':len(res),'policy_types':len(strategies),'scenario_comparisons':len(cmp),'contract_pairs':len(cs),
 'decision_definitions':LOSS,'n_original_sources_added':0}
 (OUT/'R529_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print('COMPARISONS',cmp.groupby(['dataset','task'])[['target_vs_full_test_signed','target_vs_full_deploy_signed','target_vs_full_test_harm','changed_target_vs_full']].mean().round(5).to_string(),flush=True)
 print('SWITCH',cs.groupby(['contract_A','contract_B'])['different_selection'].agg(['sum','mean']).round(3).to_string(),flush=True)

if __name__=='__main__':main()
