"""R532 conditional-concentration sensitivity: evaluate R531 sensor decisions
under gas-specific exact concentration overlap rather than varying mixture of ppm.
'anchor' is a single high-support shared ppm per gas; 'common_equal' uses equal weights
for shared ppm strata with >=10 observations in BOTH train and future test. Both
strata definitions depend on the *observed* final test batch and are thus strictly
retrospective sensitivity analyses, not prospective policy validation.
"""
import sys,zipfile,itertools,collections,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(sys.argv[1]);IN=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
X=[];b=[];y=[];ppm=[]
with zipfile.ZipFile(IN) as z:
 assert z.testzip() is None
 for bt in range(1,11):
  for l in z.read(f'batch{bt}.dat').decode().splitlines():
   els=l.split();a,c=els.pop(0).split(';'); b.append(bt);y.append(int(a));ppm.append(float(c))
   X.append([float(x.split(':')[1]) for x in els])
X=np.asarray(X);b=np.array(b);y=np.array(y);ppm=np.array(ppm)
assert X.shape==(13910,128)
tr=b<10; te=b==10; gcounts=[];common={};anchor={}
for gas in range(1,7):
 ct=collections.Counter(ppm[tr&(y==gas)]);ce=collections.Counter(ppm[te&(y==gas)])
 allowed=sorted(k for k in set(ct)&set(ce) if ct[k]>=10 and ce[k]>=10)
 assert len(allowed)>=1
 anc=max(allowed,key=lambda x:(min(ct[x],ce[x]),ct[x],-x))
 anchor[gas]=anc;common[gas]=allowed
 for k in sorted(set(ct)|set(ce)):
  gcounts.append({'gas':gas,'ppm':k,'train_n':ct[k],'test_n':ce[k],'in_common_equal':int(k in allowed),'is_anchor':int(k==anc)})
counts=pd.DataFrame(gcounts);counts.to_csv(OUT/'R532_concentration_support.csv',index=False)
print('ANCHORS',anchor,'COMMON',common,flush=True)
contracts={
 'balanced':([1/3]*3,[[0,1,1],[1,0,1],[1,1,0]]),
 'miss3':([1/3]*3,[[0,1,1],[1,0,1],[3,3,0]]),
 'reject035':([1/3]*3,[[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]]),
 'falsealarm3':([1/3]*3,[[0,1,3],[1,0,3],[1,1,0]])}

def weighted_channel(sensor,mask,gas_order,cuts,mode):
 channel=[]
 for gas in gas_order:
  strata=([anchor[gas]] if mode=='anchor' else common[gas]) if mode!='raw' else [None]
  out=[]
  for p in strata:
   sub=mask & (y==gas) if p is None else mask & (y==gas)&(ppm==p)
   vv=X[sub,(sensor-1)*8]
   assert len(vv)>=10
   out.append(np.bincount(np.searchsorted(cuts,vv,side='right'),minlength=3)/len(vv))
  channel.append(np.asarray(out).mean(axis=0))
 return np.asarray(channel)

def test_signed(P,Q,PT,QT,key):
 prior,L=contracts[key]; prior=np.asarray(prior);L=np.asarray(L)
 def risk(ch,trained):
  actions=((prior[:,None]*trained).T@L).argmin(axis=1)
  return float(((prior[:,None]*ch).T@L)[np.arange(3),actions].sum())
 return risk(QT,Q)-risk(PT,P)

T=pd.read_csv(OUT/'parent/03_实证结果/R531_gas_800_tasks.csv')
T=T[T.split=='official_chronological_1to9_vs10'].reset_index(drop=True)
assert len(T)==160
rows=[]
for triple in itertools.combinations(range(1,7),3):
 include=np.isin(y,triple)&tr
 for aa in range(1,9):
  bb=aa+8
  ca=np.quantile(X[include,(aa-1)*8],[1/3,2/3]);cb=np.quantile(X[include,(bb-1)*8],[1/3,2/3])
  ch={}
  for mode in ('raw','anchor','common_equal'):
   for period,mask in (('train',tr),('test',te)):
    ch[mode,period,'P']=weighted_channel(aa,mask,triple,ca,mode)
    ch[mode,period,'Q']=weighted_channel(bb,mask,triple,cb,mode)
  base={'gases':'-'.join(map(str,triple)),'sensor_A':aa,'sensor_B':bb}
  for contract in contracts:
   for mode in ('raw','anchor','common_equal'):
    r={**base,'contract':contract,'mode':mode,
       'train_signed':test_signed(ch[mode,'train','P'],ch[mode,'train','Q'],ch[mode,'train','P'],ch[mode,'train','Q'],contract),
       'heldout_fixed_action':test_signed(ch[mode,'train','P'],ch[mode,'train','Q'],ch[mode,'test','P'],ch[mode,'test','Q'],contract),
       'heldout_R531_action':test_signed(ch['raw','train','P'],ch['raw','train','Q'],ch[mode,'test','P'],ch[mode,'test','Q'],contract)}
    rows.append(r)
R=pd.DataFrame(rows);assert len(R)==160*4*3
R.to_csv(OUT/'R532_1920_gas_contract_concentration.csv',index=False)
for contract in contracts:
 sel=(R.contract==contract)&(R['mode']=='raw'); d=R[sel].reset_index(drop=True)
 err=float(np.max(np.abs(d.heldout_R531_action-T[f'deploy_signed_{contract}'])))
 print('RAW RECON',contract,err)
 assert err<1e-12
# Compare original R531 selection unchanged, with two adjusted test channels.
S=[]
for mode in ('raw','anchor','common_equal'):
 for level in ('task','physical'):
  g=T.copy() if level=='task' else T.groupby('sensor_A').mean(numeric_only=True).reset_index()
  N=160 if level=='task' else 8
  for K in ([40,80,120] if level=='task' else [2,4,6]):
   full=np.lexsort((np.arange(N),np.round(g.train_Delta,12)))[:K]
   for contract in contracts:
    z=R[(R['mode']==mode)&(R.contract==contract)].reset_index(drop=True)
    if level=='physical':z=z.groupby('sensor_A').mean(numeric_only=True).reset_index()
    targ=np.lexsort((np.arange(N),np.round(g['train_signed_'+contract],12)))[:K]
    for policy,inds in [('full_fixed',full),('target_fixed',targ)]:
     S.append({'mode':mode,'level':level,'contract':contract,'K':K,'policy':policy,
         'heldout_risk_per_panel':float(z.heldout_R531_action.iloc[inds].sum()/N),
         'K_used':K,'source':'R531 original trained policy on reweighted test'})
SS=pd.DataFrame(S);SS.to_csv(OUT/'R532_original_policies_standardized_tests.csv',index=False)
V=SS.pivot_table(index=['mode','level','contract','K'],columns='policy',values='heldout_risk_per_panel').reset_index()
V['target_minus_full']=V.target_fixed-V.full_fixed;V['target_better']=(V.target_minus_full<-1e-10)
V.to_csv(OUT/'R532_standardized_comparison_72.csv',index=False)
print(V.groupby(['mode','level']).agg(target_better=('target_better','sum'),mean_delta=('target_minus_full','mean'),mean_full=('full_fixed','mean'),mean_target=('target_fixed','mean')).to_string())
with open(OUT/'R532_concentration_validation.json','w') as f:
 json.dump({'data_sha256':hashlib.sha256(IN.read_bytes()).hexdigest(),'anchor':anchor,'common':common,'note':'strata are constructed by inspecting final test concentrations; exploratory diagnostic only','raw_reconstruction':'verified <1e-12'},f,ensure_ascii=False,indent=2)
