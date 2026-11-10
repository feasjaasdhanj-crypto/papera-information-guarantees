#!/usr/bin/env python3
"""Independent reproduction in a Python/SciPy environment for the *six earlier* sources.
Usage: python R546_REPLAY_SIX_SOURCE_ONE_COMMAND.py <PaperA_R544_COMPLETE.zip> <output_directory>
No network, no external downloads. Extracts original data and immutable code from nested parents.
Preserves subject-feature-level limitation for WISDM/DSADS/MEX/MHEALTH.
Separate EEG Windows R feature-level audit is in R545 and is NOT claimed to be run here.
"""
import io,zipfile,sys,subprocess,json,os,hashlib
from pathlib import Path
import pandas as pd
import numpy as np
if len(sys.argv)!=3:raise SystemExit(__doc__)
ARCH=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
SRC=OUT/'scripts';INP=OUT/'input';GEN=OUT/'generated';LOG=OUT/'logs'
for p in [SRC,INP,GEN,LOG]:p.mkdir(exist_ok=True)
root=zipfile.ZipFile(ARCH)
assert root.testzip() is None,'R544 parent ZIP corrupt'
Z={0:root}
for i in range(1,17):
 prev=Z[i-1]; choices=[n for n in prev.namelist() if n.lower().endswith('.zip') and ('parent' in n.lower() or 'original' in n.lower() or 'complete' in n.lower())]
 assert choices,('Missing historical parent',i)
 key=max(choices,key=lambda n:prev.getinfo(n).file_size)
 Z[i]=zipfile.ZipFile(io.BytesIO(prev.read(key)))

def extract(version, member, output):
 data=Z[version].read(member);path=Path(output);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);return path
# Original public datasets, present in the historical parent chain.
HAR=extract(12,'00_必须保留_完整科研交接包/USER_UPLOADED_UCI_HAR_ORIGINAL.zip',INP/'HAR_original.zip')
GAS=extract(11,'00_必须保留_原始完整包/USER_UPLOADED_GAS_SENSOR_ORIGINAL.zip',INP/'Gas_original.zip')
# Minimal, deterministic feature input archive from archived RDS (no raw untouched original windows implied).
b526=io.BytesIO()
with zipfile.ZipFile(b526,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for ds in ['WISDM','DSADS','MEX','MHEALTH']:
  a=f'03_直接保存的四来源轻量化数据/{ds}_ANALYSIS_READY.rds';z.writestr(a,Z[16].read(a))
 a='01_新的结果与研究结论/R526_task_level_natural_bayes_vs_delta.csv';z.writestr(a,Z[16].read(a))
b527=io.BytesIO()
with zipfile.ZipFile(b527,'w',compression=zipfile.ZIP_STORED) as z:z.writestr('00_必须保留_完整科研交接包/ORIGINAL_R526_COMPLETE.zip',b526.getvalue())
with zipfile.ZipFile(INP/'four_source_feature_archive.zip','w',compression=zipfile.ZIP_STORED) as z:z.writestr('00_必须保留_完整科研交接包/ORIGINAL_R527_COMPLETE.zip',b527.getvalue())
script={
 'HAR':extract(12,'02_可复算代码/R530_UCI_HAR_external.py',SRC/'R530_UCI_HAR_external.py'),
 'GAS':extract(11,'02_复算代码/R531_gas_temporal_external.py',SRC/'R531_gas_temporal_external.py'),
 'OLDER':extract(13,'02_可复算代码/R529_multi_decision.py',SRC/'R529_multi_decision.py'),
 'BUDGET':extract(13,'02_可复算代码/R529_budget_robustness.py',SRC/'R529_budget_robustness.py'),
 'MILP':extract(13,'02_可复算代码/R529_global_minimax.py',SRC/'R529_global_minimax.py'),
 'CERT':extract(13,'04_严格数学证书与核验/R528_VERIFY_EXACT_PURE_PYTHON.py',SRC/'R528_VERIFY_EXACT_PURE_PYTHON.py')}
cert=extract(14,'02_严格数学/R528_R524_16项精确对偶证书.json',INP/'certificates_16.json')
locked={
'HAR':(12,'03_新实证结果/',[
 'R530_UCI_HAR_780_真实通道任务.csv','R530_UCI_HAR_351_策略结果.csv','R530_UCI_HAR_39_整数优化.csv','R530_UCI_HAR_624_测试比较.csv']),
'GAS':(11,'03_实证结果/',[
 'R531_gas_800_tasks.csv','R531_120_policies.csv','R531_15_milp.csv','R531_240_comparisons.csv']),
'OLDER':(13,'03_数值结果/',[
 'R529_five_settings_1008_tasks.csv','R529_3budgets_8policies_5settings.csv','R529_budget_840_target_comparisons.csv','R529_global_minimax_168_portfolios.csv'])}
for group,(v,folder,names) in locked.items():
 for n in names:extract(v,folder+n,INP/'locked'/group/n)
# Archived R529 budget replay validates its own 50%-budget selection against the historical locked file.
extract(13,'03_数值结果/R529_56split_7policies_5decision_outcomes.csv',GEN/'OLDER'/'R529_56split_7policies_5decision_outcomes.csv')
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(tag,args):
 p=subprocess.run([sys.executable,str(script[tag]),*map(str,args)],cwd=str(OUT),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (LOG/f'{tag}.log').write_text(p.stdout,encoding='utf-8');assert p.returncode==0, f'{tag}: {p.stdout[-2000:]}'
run('HAR',[HAR,GEN/'HAR']);run('GAS',[GAS,GEN/'GAS']);run('OLDER',[INP/'four_source_feature_archive.zip',GEN/'OLDER'])
run('BUDGET',[GEN/'OLDER']);run('MILP',[GEN/'OLDER'])
run('CERT',[cert])
checks=[]
for grp,(version,folder,names) in locked.items():
 for name in names:
  a=(GEN/grp/name).read_bytes();b=(INP/'locked'/grp/name).read_bytes()
  d1=pd.read_csv(io.BytesIO(a));d2=pd.read_csv(io.BytesIO(b));assert d1.shape==d2.shape
  numeric=d1.select_dtypes(include='number').columns
  dif=(d1[numeric]-d2[numeric]).abs();maxerr=float(dif.max().max())
  assert maxerr<1e-10 and all(d1[k].equals(d2[k]) for k in d1 if k not in numeric),(grp,name,maxerr)
  checks.append({'collection':grp,'table':name,'rows':len(d1),'max_numeric_error':maxerr,'byte_identical':a==b})
assert len(checks)==12 and all(x['byte_identical'] for x in checks)
result={'status':'PASS','rebuild_scope':{'original_raw_ZIP':['UCI_HAR','GasDrift'],'archived_subject_level_features':['WISDM','DSADS','MEX','MHEALTH']},'all_12_tables_byte_identical':True,'n_tasks':2588,'n_policy_rows':1815,'n_portfolio_MILP':222,'all_16_rational_certificates_pass':True,'source_archive_sha256':hashlib.sha256(ARCH.read_bytes()).hexdigest(),'table_checks':checks,'not_executed':'Raw EEG EDF and all-seven-source R-only Windows single-entry workflow'}
(OUT/'R546_INDEPENDENT_RAW_REPLAY_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'status':'PASS','tables':len(checks),'tasks':2588,'policy_rows':1815,'portfolios':222,'certificates':16},indent=2))
