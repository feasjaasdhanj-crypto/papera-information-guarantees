"""Reconstruct exact rational primal-dual certificates for all 16 of R524's LPs.
Uses scipy linprog only to locate candidate basic solutions; cert validity is checked
with fractions.Fraction EXACTLY, independently of floating point solver tolerances.
"""
from fractions import Fraction as F
import itertools,json,sys
import numpy as np
from scipy.optimize import linprog
from pathlib import Path

def rational_lp(P,Q):
 n,m=len(P),len(P[0]);N=m*m+n*m+1;k=m*m
 c=[F(0)]*N;c[-1]=F(1)
 eq=[];beq=[];ub=[];bub=[]
 for a in range(m):
  row=[F(0)]*N
  for j in range(m):row[a*m+j]=F(1)
  eq.append(row);beq.append(F(1))
 for i in range(n):
  for j in range(m):
   row=[F(0)]*N
   for a in range(m):row[a*m+j]=P[i][a]
   row[k+i*m+j]=F(-1)
   ub.append(row);bub.append(Q[i][j])
   row=[F(0)]*N
   for a in range(m):row[a*m+j]=-P[i][a]
   row[k+i*m+j]=F(-1)
   ub.append(row);bub.append(-Q[i][j])
  row=[F(0)]*N
  for j in range(m):row[k+i*m+j]=F(1)
  row[-1]=F(-2)
  ub.append(row);bub.append(F(0))
 return c,eq,beq,ub,bub

def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))

def reconstruct(P,Q,label,limit):
 c,eq,beq,ub,bub=rational_lp(P,Q)
 rr=linprog(np.array(c,dtype=float), A_ub=np.array(ub,dtype=float),b_ub=np.array(bub,dtype=float),A_eq=np.array(eq,dtype=float),b_eq=np.array(beq,dtype=float),bounds=(0,None),method='highs')
 assert rr.success,(label,rr.message)
 approx=lambda arr:[F(float(v)).limit_denominator(limit) for v in arr]
 x=approx(rr.x); y=approx(rr.eqlin.marginals); z=approx(rr.ineqlin.marginals)
 primal_eq=all(dot(row,x)==v for row,v in zip(eq,beq))
 primal_ineq=all(dot(row,x)<=v for row,v in zip(ub,bub))
 primal_nonnegative=all(t>=0 for t in x)
 dual_nonpositive=all(t<=0 for t in z)
 # dual lower bound: c>=Aeq^T y+Aub^T z
 dual_feas=all(c[j]>=sum((row[j]*v for row,v in zip(eq,y)),F(0))+sum((row[j]*v for row,v in zip(ub,z)),F(0)) for j in range(len(c)))
 p=dot(c,x);d=dot(beq,y)+dot(bub,z)
 if not (primal_eq and primal_ineq and primal_nonnegative and dual_nonpositive and dual_feas and p==d):
  return dict(label=label,success=False,objective=p,dual=d,checks=(primal_eq,primal_ineq,primal_nonnegative,dual_nonpositive,dual_feas),LP=float(rr.fun))
 return dict(label=label,success=True,objective=str(p),primal=[str(a) for a in x],dual_eq=[str(a) for a in y],dual_ub=[str(a) for a in z],LP=float(rr.fun),n_vars=len(x),n_ineq=len(ub))

def main(out):
 u=F(1,3);t=F(1,10);w=F(1,215);c=F(1,100)
 P_A=[[u+t*((1 if i==j else 0)-u) for j in range(3)] for i in range(3)]
 Q_A=[[u-(t-w)*((1 if i==j else 0)-u) for j in range(3)] for i in range(3)]
 P_B=[[u]*3 for _ in range(3)]
 Q_B=[[u+c,u-c,u],[u-c,u+c,u],[u,u,u]]
 allrows=[]
 for name,p,q in [('A',P_A,Q_A),('B',P_B,Q_B)]:
  for pair in [(0,1,2),(0,1),(0,2),(1,2)]:
   for dire,source,target in [('P_to_Q',p,q),('Q_to_P',q,p)]:
    rec=reconstruct([source[i] for i in pair],[target[i] for i in pair],name+'_'+''.join(str(i) for i in pair)+'_'+dire,10000000)
    allrows.append(rec)
    print(rec['label'],rec['success'],rec['objective'],rec['LP'],rec.get('checks'))
 assert len(allrows)==16
 assert all(r['success'] for r in allrows)
 out.write_text(json.dumps({'type':'exact fraction primal dual feasibility plus matching objective certificates','count':16,'results':allrows},ensure_ascii=False,indent=2),encoding='utf8')
 return allrows

if __name__=='__main__':main(Path(sys.argv[1]))
