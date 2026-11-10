"""R531 optional drift-aware rule: train months up to batch t-2, validate on batch t-1,
freeze policy selection, and test on batch t. Prespecified 4 folds t=7..10.
Both sensor/bin cutoffs and per-bin actions are trained on strictly earlier batches.
Validation only chooses BETWEEN the two policies (full Delta and task matched),
never uses future batch labels. Gates are defined separately for each science contract.
"""
import sys,itertools,importlib,json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(sys.argv[2]);sys.path.insert(0,str(Path(__file__).parent))
import R531_gas_temporal_external as base
X,b,y,ppm=base.load();UNIQUE=base.UNIQUE
completed=pd.read_csv(OUT/'R531_gas_800_tasks.csv')
report=[];summary=[];alltask=[]
for t in (7,8,9,10):
 train=np.isin(b,list(range(1,t-1)));val=b==t-1;test=b==t
 trclasses=y[train];va_classes=y[val];te_classes=y[test]
 label=f'train1to{t-2}_val{t-1}_test{t}'
 preflabel=f'forward_1to{t-2}_vs{t-1}'
 history=completed[completed.split==preflabel].copy().reset_index(drop=True)
 assert len(history)==160
 tasks=[]
 for triple in base.TRIPLES:
  it=np.isin(trclasses,triple)
  for aa,bb in base.PAIRS:
   ai=(aa-1)*8;bi=(bb-1)*8
   vals=[X[train,ai],X[train,bi],X[val,ai],X[val,bi],X[test,ai],X[test,bi]]
   ca=np.quantile(vals[0][it],[1/3,2/3]);cb=np.quantile(vals[1][it],[1/3,2/3]);
   P=base.channel(vals[0],trclasses,triple,ca);Q=base.channel(vals[1],trclasses,triple,cb)
   PV=base.channel(vals[2],va_classes,triple,ca);QV=base.channel(vals[3],va_classes,triple,cb)
   PT=base.channel(vals[4],te_classes,triple,ca);QT=base.channel(vals[5],te_classes,triple,cb)
   row={'split':label,'gases':'-'.join(map(str,triple)),'sensor_A':aa,'sensor_B':bb}
   for c,dd in base.LOSS.items():
    pri,L=dd['prior'],dd['L']
    row['train_signed_'+c]=base.risk(Q,pri,L)-base.risk(P,pri,L)
    row['val_signed_'+c]=base.risk(QV,pri,L,trained=Q)-base.risk(PV,pri,L,trained=P)
    row['test_signed_'+c]=base.risk(QT,pri,L,trained=Q)-base.risk(PT,pri,L,trained=P)
   tasks.append(row)
 df=pd.DataFrame(tasks)
 assert np.allclose(df[['train_signed_'+x for x in UNIQUE]].values,history[['train_signed_'+x for x in UNIQUE]].values,atol=1e-12)
 df['train_Delta']=history['train_Delta'].values
 alltask.append(df)
 for level in ('task','physical'):
  g=df if level=='task' else df.groupby('sensor_A').mean(numeric_only=True).reset_index()
  n=len(g)
  for frac in (.25,.5,.75):
   K=round(n*frac)
   idxFull=np.lexsort((np.arange(n),np.round(g.train_Delta.values,12)))[:K]
   for c in UNIQUE:
    idxMatch=np.lexsort((np.arange(n),np.round(g['train_signed_'+c].values,12)))[:K]
    # per-task metric = mean across selected tasks; physical = (sum of switches)/8 physical slots
    d= n if level=='physical' else K
    vfull=float(g.iloc[idxFull]['val_signed_'+c].sum()/d)
    vmatch=float(g.iloc[idxMatch]['val_signed_'+c].sum()/d)
    testFull=float(g.iloc[idxFull]['test_signed_'+c].sum()/d)
    testMatch=float(g.iloc[idxMatch]['test_signed_'+c].sum()/d)
    gate=vmatch-vfull < -1e-12
    testGate=testMatch if gate else testFull
    report.append({'split':label,'level':level,'K':K,'budget':frac,'contract':c,
      'val_target_minus_full':vmatch-vfull,'test_target_minus_full':testMatch-testFull,
      'gate_using_validation':int(gate),'test_gate_minus_full':testGate-testFull,
      'test_gate_minus_always_target':testGate-testMatch})
 print('PREQUENTIAL FINISHED',label,flush=True)
R=pd.DataFrame(report);A=pd.concat(alltask,ignore_index=True)
assert len(R)==4*2*3*4 and len(A)==4*160
R.to_csv(OUT/'R531_prequential_96_policy_checks.csv',index=False)
A.to_csv(OUT/'R531_prequential_640_task_channels.csv',index=False)
for level,grp in R.groupby('level'):
 print('\nLEVEL',level)
 print(grp.groupby('split').agg(target_better=('test_target_minus_full',lambda x:sum(x< -1e-10)),target_worse=('test_target_minus_full',lambda x:sum(x>1e-10)),gate_used=('gate_using_validation','sum'),gate_better_than_full=('test_gate_minus_full',lambda x:sum(x< -1e-10)),gate_worse_than_full=('test_gate_minus_full',lambda x:sum(x>1e-10)),mean_target_diff=('test_target_minus_full','mean'),mean_gate_diff=('test_gate_minus_full','mean')).to_string())
