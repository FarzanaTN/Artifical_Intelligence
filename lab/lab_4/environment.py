import numpy as np

CITIES=["A","B","C","D","E"]
CAP=5
GOAL="E"
GRAPH={
"A":{"B":2,"C":3},
"B":{"A":2,"E":2},
"C":{"A":3,"D":2},
"D":{"C":2,"E":1},
"E":{}
}
ACTIONS=["REFUEL","STOP","GO_B","GO_C","GO_D","GO_E","GO_A"]
def states():
    return [(c,f) for c in CITIES for f in range(CAP+1)]
def valid_actions(s):
    c,f=s
    acts=["REFUEL"]
    if c==GOAL: acts.append("STOP")
    for n,d in GRAPH[c].items():
        acts.append("GO_"+n)
    return acts
def step(s,a):
    c,f=s
    if a=="REFUEL":
        return (c,CAP),-5,False
    if a=="STOP":
        return s,(0 if c==GOAL else -50), c==GOAL
    if a.startswith("GO_"):
        n=a[3:]
        if n not in GRAPH[c]: return s,-100,False
        d=GRAPH[c][n]
        if f<d: return s,-100,False
        ns=(n,f-d)
        r=100 if n==GOAL else -2
        return ns,r,n==GOAL
    return s,-100,False
