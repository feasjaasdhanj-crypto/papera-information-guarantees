"""Paper A R540 exploratory independent-subject EEG study: eight fixed physical electrode swap pairs.
Only R539 uploaded features used, no EDF download. All choice rules fixed before full 256-mask test analysis.
Primary: unlabeled subject-session normalization, 8 electrodes, theta/alpha/beta LDA, uniform prior.
One channel for deficiency is prechosen beta feature and train-only tertiles per electrode.
Validation can select goal portfolio, heldout subject data only for policy evaluation.
"""
import io,json,zipfile,hashlib,math,time,itertools,os
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linprog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
T=time.time();OUT=Path('/mnt/data/pa_r540_work/R540_EEG_RESULTS');OUT.mkdir(parents=True,exist_ok=True)
UP=Path('/mnt/data/RETURN_TO_CHATGPT_EEG_R539.zip');Z=zipfile.ZipFile(UP)
D=pd.read_pickle('/mnt/data/pa_r540_work/EEG_features_extracted.pkl')
sp=pd.read_csv(io.BytesIO(Z.read('RETURN_TO_CHATGPT_EEG_FEATURES\\SUBJECT_SPLIT_LOCKED.csv')))
D=D.merge(sp,on='subject',how='left',validate='many_to_one')
assert not D.partition.isna().any()
base=['FC5','FC3','FC1','C5','C3','C1','CP5','CP3']
repl=['FC6','FC4','FC2','C6','C4','C2','CP6','CP4']
assert len(set(base+repl))==16
BANDS=['theta','alpha','beta'];Y=D.state.map({'T0':0,'T1':1,'T2':2}).to_numpy()
S=D.subject.to_numpy();split=D.partition.to_numpy()
allcols=[f'{e}_{b}' for e in base+repl for b in BANDS]
assert not D[allcols].isna().any().any()
Xs=D[allcols].copy()
# Unsupervised per-session norm (requires the complete session signals, not state labels)
std=Xs.groupby(D.subject).transform('std').clip(lower=.05)
Xs=(Xs-Xs.groupby(D.subject).transform('mean'))/std
# Train-only global standardization, with no test learning
tr=split=='train'; va=split=='validation'; te=split=='test'
assert len(set(S[tr])&set(S[va]))==0 and len(set(S[tr])&set(S[te]))==0 and len(set(S[va])&set(S[te]))==0
Xfull=Xs.to_numpy();trainmu=Xfull[tr].mean(axis=0);trainsd=Xfull[tr].std(axis=0);trainsd[trainsd<.01]=1
Xfull=(Xfull-trainmu)/trainsd
colidx={c:i for i,c in enumerate(allcols)}
L={
 'balanced':np.array([[0,1,1],[1,0,1],[1,1,0]],float),
 'miss_T2_x3':np.array([[0,1,1],[1,0,1],[3,3,0]],float),
 'false_alarm_T2_x3':np.array([[0,1,3],[1,0,3],[1,1,0]],float),
 'reject035':np.array([[0,1,1,.35],[1,0,1,.35],[1,1,0,.35]],float)}
VALID_SUB=sorted(set(S[va]));TEST_SUB=sorted(set(S[te]));TRAIN_SUB=sorted(set(S[tr]))
KVAL={s:np.flatnonzero((S==s)) for s in VALID_SUB}
KTEST={s:np.flatnonzero((S==s)) for s in TEST_SUB}

def deficiency(P,Q):
 n,m=P.shape;assert n in (2,3) and m==3
 k=m*m;N=k+n*m+1;E=N-1
 Aeq=[];beq=[];Aub=[];b=[]
 for a in range(m):
  row=np.zeros(N);row[a*m:a*m+m]=1;Aeq.append(row);beq.append(1)
 for i in range(n):
  for j in range(m):
   row=np.zeros(N)
   for a in range(m):row[a*m+j]=P[i,a]
   row[k+i*m+j]=-1;Aub.append(row);b.append(Q[i,j])
   row=np.zeros(N)
   for a in range(m):row[a*m+j]=-P[i,a]
   row[k+i*m+j]=-1;Aub.append(row);b.append(-Q[i,j])
  row=np.zeros(N);row[k+i*m:k+i*m+m]=1;row[E]=-2;Aub.append(row);b.append(0)
 c=np.zeros(N);c[E]=1
 res=linprog(c,A_ub=np.array(Aub),b_ub=np.array(b),A_eq=np.array(Aeq),b_eq=np.array(beq),bounds=(0,None),method='highs')
 if not res.success:raise ValueError(res.message)
 return res.fun

def channels(band):
 out={}
 for e in base+repl:
  col=Xs[f'{e}_{band}'].to_numpy()
  cuts=np.quantile(col[tr],[1/3,2/3]);assert cuts[0]<cuts[1]
  yy=np.searchsorted(cuts,col,side='right')
  P=[]
  for state in range(3):
   freq=[np.bincount(yy[(S==s)&(Y==state)],minlength=3)/((S==s)&(Y==state)).sum() for s in TRAIN_SUB]
   P.append(np.array(freq).mean(axis=0))
  p=np.array(P);assert np.allclose(p.sum(axis=1),1)
  out[e]=p
 return out

audit=[]
for band in ['beta','alpha']:
 chan=channels(band)
 for j,(a,b) in enumerate(zip(base,repl)):
  P,Q=chan[a],chan[b]
  de=max(deficiency(P,Q),deficiency(Q,P))
  rr=max(max(deficiency(P[list(xx)],Q[list(xx)]),deficiency(Q[list(xx)],P[list(xx)])) for xx in ((0,1),(0,2),(1,2)))
  assert rr<=de+1e-8
  audit.append({'band':band,'index':j,'base':a,'replacement':b,'full_delta':de,'local_r2':rr,'full_minus_local':de-rr})
aud=pd.DataFrame(audit);aud.to_csv(OUT/'R540_EEG_16_electrode_information_metrics.csv',index=False)
print('DEFICIENCY',aud.to_string(index=False), 'time',time.time()-T,flush=True)

# Unchanged prior, fixed LDA fit to train subjects. Evaluate one complete eight-electrode portfolio.
def risks_by_subject(model,idx,labels,group):
 prob=model.predict_proba(idx)
 risk={}
 for lk,loss in L.items():
  action=np.argmin(prob@loss,axis=1)
  realized=loss[labels,action]
  vals=[]
  for subj in group:
   q=S[(split=='validation') if group==VALID_SUB else (split=='test')]
  # parallel across each class by subject
  for subj in group:
   local=S[(split=='validation') if group==VALID_SUB else (split=='test')]==subj
   sel=labels[local]; rr=realized[local]
   vals.append(np.mean([rr[sel==i].mean() for i in range(3)]))
  risk[lk]=np.array(vals)
 return risk

maskresults=[]; testsubject=[]; riskV={};riskT={};predcache={}
for mask in range(256):
 el=[repl[j] if (mask>>j)&1 else base[j] for j in range(8)]
 cols=[colidx[e+'_'+band] for e in el for band in BANDS]
 X=Xfull[:,cols]
 model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[1/3]*3).fit(X[tr],Y[tr])
 pV=model.predict_proba(X[va]);pT=model.predict_proba(X[te]);lv=Y[va];lt=Y[te]
 vparts=S[va];tparts=S[te];rV={};rT={}
 for key,loss in L.items():
  actionsV=np.argmin(pV@loss,axis=1);actionsT=np.argmin(pT@loss,axis=1)
  rV[key]=np.array([np.mean([loss[lv[(vparts==s)&(lv==c)],actionsV[(vparts==s)&(lv==c)]].mean() for c in range(3)]) for s in VALID_SUB])
  rT[key]=np.array([np.mean([loss[lt[(tparts==s)&(lt==c)],actionsT[(tparts==s)&(lt==c)]].mean() for c in range(3)]) for s in TEST_SUB])
 riskV[mask]=rV;riskT[mask]=rT
 maskresults.append({'mask':mask,'n_replacements':mask.bit_count(),'electrodes':';'.join(el),**{f'validation_{k}':a.mean() for k,a in rV.items()},**{f'test_{k}':a.mean() for k,a in rT.items()}})
 if mask%32==0:print('FIT',mask,'time',round(time.time()-T,1),flush=True)
M=pd.DataFrame(maskresults);M.to_csv(OUT/'R540_EEG_256_portfolio_validation_test.csv',index=False)

# Picks by information metric at fixed K, and target-specific val risk at fixed K or <=K.
select=[];per_subject=[]
for budget in [2,4,6]:
 for key in L:
  am=aud.query("band=='beta'").sort_values('index')
  chosen={}
  for pol,metric in [('local_r2','local_r2'),('full_delta','full_delta')]:
   chosen[pol]=sum(2**int(j) for j in am.sort_values([metric,'index']).index[:budget])
  # index sort stable, 0..7
  feasible=M[M.n_replacements==budget];less=M[M.n_replacements<=budget]
  fkey='validation_'+key
  chosen['target_match_exact']=int(feasible.sort_values([fkey,'mask']).iloc[0]['mask'])
  chosen['target_match_up_to']=int(less.sort_values([fkey,'n_replacements','mask']).iloc[0]['mask'])
  chosen['no_replacement']=0
  for pol,mask in chosen.items():
   sel=next(r for r in maskresults if r['mask']==mask)
   vals=riskT[mask][key]; baseline=riskT[0][key]; diff=vals-baseline
   row={'budget':budget,'contract':key,'policy':pol,'selected_mask':mask,'n_replacements':mask.bit_count(),
    'validation_risk':riskV[mask][key].mean(),'heldout_risk':vals.mean(),'heldout_minus_keep':diff.mean(),
    'n_subjects':len(TEST_SUB)}
   select.append(row)
   for s,v,d in zip(TEST_SUB,vals,diff):per_subject.append({'budget':budget,'contract':key,'policy':pol,'test_subject':s,'risk':v,'minus_keep':d})
sel=pd.DataFrame(select);sel.to_csv(OUT/'R540_EEG_selected_policies_3budgets_4contracts.csv',index=False)
per=pd.DataFrame(per_subject);per.to_csv(OUT/'R540_EEG_test_subject_policy_risks.csv',index=False)

# Bootstrap fixed policies with subject unit, replicable deterministic. Not post-selection corrected.
rng=np.random.default_rng(540031);B=10000
compare=[]
for (b,key),g in sel.groupby(['budget','contract']):
  chosen={r.policy:int(r.selected_mask) for r in g.itertuples()}
  for pol in ['target_match_exact','target_match_up_to','full_delta','local_r2']:
   for base_pol in ['full_delta','local_r2','no_replacement']:
    if pol==base_pol:continue
    dif=riskT[chosen[pol]][key]-riskT[chosen[base_pol]][key]
    sample=dif[rng.integers(len(dif),size=(B,len(dif)))].mean(axis=1)
    compare.append({'budget':b,'contract':key,'policy':pol,'reference':base_pol,'mean_diff':dif.mean(),
     'ci_2p5':np.quantile(sample,.025),'ci_97p5':np.quantile(sample,.975),
     'subjects_better':(dif< -1e-10).sum(),'subjects_worse':(dif>1e-10).sum(),
     'subjects_tie':(abs(dif)<=1e-10).sum(),'n_test_subjects':len(dif)})
cm=pd.DataFrame(compare);cm.to_csv(OUT/'R540_EEG_19_subject_pair_bootstrap.csv',index=False)
# per-state balanced accuracy for no-swap and target exact in balanced 4
main=sel[(sel.budget==4)&(sel.contract=='balanced')]
print('EEG_PRIMARY',main.to_string(index=False),flush=True)
print('MAIN COMPARISONS',cm[(cm.budget==4)&(cm.contract=='balanced')].to_string(index=False),flush=True)
# summary useful
summ={'uploaded_sha256':hashlib.sha256(UP.read_bytes()).hexdigest(), 'subject_counts':D.groupby('partition').subject.nunique().to_dict(),'rows':len(D),'state_counts':D.state.value_counts().to_dict(),
 'train':len(TRAIN_SUB),'validation':len(VALID_SUB),'test':len(TEST_SUB),'electrode_pairs':dict(zip(base,repl)),
 'primary':'subject-normalized train-scaled LDA, 3 bands, uniform state prior; eight physical electrodes kept fixed, 2/4/6 replace',
 'caveat':'Subject normalization uses unlabeled full session per subject, this is offline calibration not prospective online. Earlier session prechecks saw test accuracy for four masks; this post-hoc development is exploratory, not preregistered prospective confirmation.',
 'elapsed_sec':time.time()-T}
(OUT/'R540_EEG_method_qc.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding='utf-8')
print('DONE',time.time()-T)
