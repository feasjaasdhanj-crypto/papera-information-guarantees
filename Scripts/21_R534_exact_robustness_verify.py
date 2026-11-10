#!/usr/bin/env python3
"""R534 independent exact rational verification of decision-reversal margins.
Uses only Python stdlib. The proof of robustness is symbolic (TV contraction),
not based on finite simulation. Exact LP certificates are reverified using
R528_VERIFY_EXACT_PURE_PYTHON.py from the parent.
"""
from fractions import Fraction as F
from itertools import product
import json,sys
U=[[F(1,3) for _ in range(3)] for _ in range(3)]
I=[[F(int(i==j)) for j in range(3)] for i in range(3)]
t,w=F(1,10),F(1,215)
PA=[[U[i][j]+t*(I[i][j]-U[i][j]) for j in range(3)] for i in range(3)]
QA=[[U[i][j]-(t-w)*(I[i][j]-U[i][j]) for j in range(3)] for i in range(3)]
PB=[r.copy() for r in U]
QB=[[F(103,300),F(97,300),F(1,3)],[F(97,300),F(103,300),F(1,3)],U[0].copy()]
def bayes_risk_zero_one(E):
    return 1-sum(max(E[i][z] for i in range(3)) for z in range(3))/3
risks={x:bayes_risk_zero_one(E) for x,E in [('PA',PA),('QA',QA),('PB',PB),('QB',QB)]}
rA=risks['QA']-risks['PA'];rB=risks['QB']-risks['PB']
localA,localB=F(1,430),F(1,100)
fullA,fullB=F(3,86),F(1,100)
assert (rA,rB)==(F(3,86),-F(1,150)),(rA,rB)
lg=localB-localA;fg=fullA-fullB;rg=rA-rB
eta=lg/4
assert eta==F(33,17200)
assert fg>lg and rg>lg and rA>2*eta and -rB>2*eta
assert all(sum(row)==1 and all(x>=0 for x in row) for E in [PA,QA,PB,QB] for row in E)
report={'state':'PASS','prior':'uniform_three_class','loss':'zero_one',
    'matrix_PA':[[str(x) for x in row] for row in PA],
    'matrix_QA':[[str(x) for x in row] for row in QA],
    'matrix_PB':[[str(x) for x in row] for row in PB],
    'matrix_QB':[[str(x) for x in row] for row in QB],
    'baseline_risks':{k:str(v) for k,v in risks.items()},
    'rA':str(rA),'rB':str(rB),
    'local_gap':str(lg),'global_gap':str(fg),'risk_gap':str(rg),
    'sufficient_open_TV_row_radius':str(eta),
    'strict_claim':'for max per-row TV perturbation eta<33/17200 on each of PA,QA,PB,QB, both score rankings and signs of risk changes persist',
    'scope':'sufficient, not sharp'}
if len(sys.argv)>1:open(sys.argv[1],'w',encoding='utf8').write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS', {k:report[k] for k in ['rA','rB','local_gap','global_gap','risk_gap','sufficient_open_TV_row_radius']})
