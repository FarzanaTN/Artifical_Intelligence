from environment import *
gamma=0.9;theta=1e-3
S=states();V={s:0 for s in S}
while True:
 d=0
 for s in S:
  if s[0]==GOAL: continue
  vals=[]
  for a in valid_actions(s):
   ns,r,t=step(s,a)
   vals.append(r+gamma*V[ns]*(0 if t else 1))
  nv=max(vals)
  d=max(d,abs(nv-V[s]));V[s]=nv
 if d<theta: break
print("Value(A,5)=",round(V[("A",5)],2))
