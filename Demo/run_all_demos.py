#!/usr/bin/env python3
"""Run all three compact demos; verify outputs; record test evidence and timings."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent

def run(label,command,output):
    t0=time.perf_counter()
    proc=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,check=False)
    elapsed=time.perf_counter()-t0
    if proc.returncode != 0:
        raise RuntimeError(f'{label}: exit {proc.returncode}\n{proc.stdout}\n{proc.stderr}')
    (output/(label+'.txt')).write_text(proc.stdout,encoding='utf-8')
    if proc.stderr.strip():
        (output/(label+'.stderr.txt')).write_text(proc.stderr,encoding='utf-8')
    return proc.stdout,round(elapsed,6)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='Demo/OUTPUT',help='Directory for output logs and test report')
    p.add_argument('--custom-input',default='Demo/INPUT/example_matrices.json',help='Custom 3x3 probability matrices for decision risk')
    a=p.parse_args()
    out=Path(a.output)
    if not out.is_absolute():out=ROOT/out
    out.mkdir(parents=True,exist_ok=True)
    tests=[]
    stdout,elapsed=run('exact_certificates',[sys.executable,'Scripts/35_R528_VERIFY_EXACT_PURE_PYTHON.py','Demo/INPUT/certificates_16.json'],out)
    value=json.loads(stdout)
    expected=json.loads((ROOT/'Demo/EXPECTED/exact_verification_output.json').read_text(encoding='utf-8'))
    if value != expected:raise AssertionError('Exact certificate results do not match expected JSON')
    tests.append({'name':'exact_certificates','result':'PASS','elapsed_seconds':elapsed})
    stdout,elapsed=run('finite_sample_bounds',[sys.executable,'Scripts/20_R535_FINITE_SAMPLE_CERTIFICATE_PYTHON_STDLIB.py'],out)
    expected=(ROOT/'Demo/EXPECTED/finite_sample_bounds.txt').read_text(encoding='utf-8').strip()
    if stdout.strip() != expected:raise AssertionError('Finite-sample bounds differ from expected output')
    tests.append({'name':'finite_sample_bounds','result':'PASS','elapsed_seconds':elapsed})
    inp=Path(a.custom_input)
    if not inp.is_absolute():inp=ROOT/inp
    stdout,elapsed=run('three_state_decision_risk',[sys.executable,'Demo/run_custom_matrices.py','--input',str(inp)],out)
    value=json.loads(stdout)
    is_example=inp.resolve()==(ROOT/'Demo/INPUT/example_matrices.json').resolve()
    if is_example:
        expected=json.loads((ROOT/'Demo/EXPECTED/expected_matrices_risks.json').read_text(encoding='utf-8'))
        if value != expected:raise AssertionError('Three-state decision risks differ from expected')
    tests.append({'name':'three_state_decision_risk','result':'PASS','elapsed_seconds':elapsed,'input':('Demo/INPUT/example_matrices.json' if is_example else str(inp))})
    stdout,elapsed=run('panel_alignment_self_test',[sys.executable,'Scripts/39_audit_panel_alignment.py','--self-test'],out)
    if stdout.strip()!='Panel alignment self-test: PASS':raise AssertionError('Panel alignment self-test did not pass')
    tests.append({'name':'panel_alignment_self_test','result':'PASS','elapsed_seconds':elapsed})
    result={
      'status':'PASS','scope':'compact exact-mathematics and 3x3-decision-risk demo only',
      'python_version':sys.version.split()[0], 'operating_system':platform.platform(),
      'architecture':platform.machine(),'tests':tests,
      'total_demo_subprocess_seconds':round(sum(item['elapsed_seconds'] for item in tests),6),
      'expected_demo_input_sha256':hashlib.sha256((ROOT/'Demo/INPUT/certificates_16.json').read_bytes()).hexdigest()
    }
    (out/'test_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
