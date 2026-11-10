#!/usr/bin/env python3
"""R535 independent exact + conservative multinomial sampling audit, stdlib only."""
from fractions import Fraction as F
from math import ceil, log, exp
import json, itertools
M=12; alpha=0.05
for x in range(-20,21):
 for y in range(-20,21):
  z=-x-y
  assert sum(map(abs,(x,y,z)))/2==max(map(abs,(x,y,z)))
local=F(1,100)-F(1,430)
full=F(3,86)-F(1,100)
risk=F(3,86)+F(1,150)
assert (local,full,risk)==(F(33,4300),F(107,4300),F(134,3225))
limits={'relative_decision_risk':risk/4,'opposite_risk_signs':F(1,300),'all_ranking_and_signs':local/4}
for k,e in limits.items():
 n=int(log(6*M/alpha)/(2*float(e)**2))+1
 assert 6*M*exp(-2*n*float(e)**2)<alpha
 print(k,'epsilon <',str(e),'n per row >=',n,'total',12*n)
assert [int(log(6*M/alpha)/(2*float(e)**2))+1 for e in limits.values()]==[33700,327258,987818]
print('PASS: R535 exact margins, ternary TV identity, three sample-size thresholds')
