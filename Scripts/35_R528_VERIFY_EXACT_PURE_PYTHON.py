"""No external packages. Independently check all 16 rational primal-dual LP certificates,
and exact R524 one-slot ranking plus fixed three-class classification regret.
Usage: python R528_VERIFY_EXACT_PURE_PYTHON.py R528_R524_16项精确对偶证书.json
"""
from fractions import Fraction as F
from pathlib import Path
import json,sys

def dot(a,b): return sum((x*y for x,y in zip(a,b)),F(0))
def matrices():
 u=F(1,3);t=F(1,10);w=F(1,215);c=F(1,100)
 p=[[u+t*((1 if i==j else 0)-u) for j in range(3)] for i in range(3)]
 q=[[u-(t-w)*((1 if i==j else 0)-u) for j in range(3)] for i in range(3)]
 pb=[[u]*3 for _ in range(3)];qb=[[u+c,u-c,u],[u-c,u+c,u],[u,u,u]]
 return {'A':(p,q),'B':(pb,qb)}
def build(P,Q):
 n=len(P);m=3; N=m*m+n*m+1;k=m*m
 c=[F(0)]*N;c[-1]=F(1)
 eq=[];beq=[];ub=[];bub=[]
 for i in range(m):
  r=[F(0)]*N
  for j in range(m):r[i*m+j]=F(1)
  eq.append(r);beq.append(F(1))
 for i in range(n):
  for j in range(m):
   r=[F(0)]*N
   for a in range(m):r[a*m+j]=P[i][a]
   r[k+i*m+j]=-1
   ub.append(r);bub.append(Q[i][j])
   r=[F(0)]*N
   for a in range(m):r[a*m+j]=-P[i][a]
   r[k+i*m+j]=-1
   ub.append(r);bub.append(-Q[i][j])
  r=[F(0)]*N
  for j in range(m):r[k+i*m+j]=F(1)
  r[-1]=F(-2);ub.append(r);bub.append(F(0))
 return c,eq,beq,ub,bub

def validate(d):
 mats=matrices(); objs={}
 for item in d['results']:
  label=item['label']; name,states,direction=label.split('_',2)
  ids=[int(v) for v in states]
  p,q=mats[name]; source,target=(p,q) if direction=='P_to_Q' else (q,p)
  source=[source[i] for i in ids];target=[target[i] for i in ids]
  c,eq,beq,ub,bub=build(source,target)
  x=list(map(F,item['primal'])); y=list(map(F,item['dual_eq']));z=list(map(F,item['dual_ub']))
  assert len(x)==len(c) and len(y)==len(eq) and len(z)==len(ub),label
  assert all(v>=0 for v in x),label
  assert all(dot(a,x)==b for a,b in zip(eq,beq)),label
  assert all(dot(a,x)<=b for a,b in zip(ub,bub)),label
  assert all(v<=0 for v in z),label
  assert all(c[j]>=sum((a[j]*w for a,w in zip(eq,y)),F(0))+sum((a[j]*w for a,w in zip(ub,z)),F(0)) for j in range(len(c))),label
  lhs=dot(c,x);rhs=dot(beq,y)+dot(bub,z)
  assert lhs==rhs==F(item['objective']),label
  objs[label]=lhs
 assert len(objs)==16
 full={}
 pair={}
 for name in ['A','B']:
  full[name]=max(objs[f'{name}_012_P_to_Q'],objs[f'{name}_012_Q_to_P'])
  pair[name]=max(objs[f'{name}_{ij}_{dr}'] for ij in ['01','02','12'] for dr in ['P_to_Q','Q_to_P'])
 assert full['A']==F(3,86) and pair['A']==F(1,430)
 assert full['B']==pair['B']==F(1,100)
 assert pair['A']<pair['B']<full['A'] and full['B']<full['A']
 def risk01(M):return 1-sum(max(M[i][j] for i in range(3)) for j in range(3))/3
 losses={name:risk01(q)-risk01(p) for name,(p,q) in mats.items()}
 assert losses['A']==F(3,86) and losses['B']==-F(1,150)
 assert losses['A']-losses['B']==F(134,3225)
 out={'exact_primal_dual_certificates_passed':len(objs),'A_full':str(full['A']),'A_local':str(pair['A']),
      'B_full':str(full['B']),'B_local':str(pair['B']),
      'A_classification_loss_Q_minus_P':str(losses['A']),
      'B_classification_loss_Q_minus_P':str(losses['B']),
      'one_slot_regret_local_vs_full':str(losses['A']-losses['B']),
      'all_pass':True}
 print(json.dumps(out,ensure_ascii=False,indent=2));return out
if __name__=='__main__':validate(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')))
