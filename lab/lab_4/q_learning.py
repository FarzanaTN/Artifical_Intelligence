from environment import *
import random
alpha=.5;gamma=.9;eps=.2
Q={}
for s in states():
 for a in ACTIONS:Q[(s,a)]=0
for ep in range(2000):
 s=("A",5)
 for _ in range(20):
  acts=valid_actions(s)
  a=random.choice(acts) if random.random()<eps else max(acts,key=lambda x:Q[(s,x)])
  ns,r,t=step(s,a)
  mx=max(Q[(ns,x)] for x in valid_actions(ns))
  Q[(s,a)]+=alpha*(r+gamma*(0 if t else mx)-Q[(s,a)])
  s=ns
  if t: break
print("Best action at (A,5):",max(valid_actions(("A",5)),key=lambda x:Q[(("A",5),x)]))
