"""Find a minimum-cut witness on the complete permitted material graph, in physical kilograms."""
import json,subprocess
from pathlib import Path
import numpy as np
def check(binary,out,qid,quota,O,external,ch,qq,inside,P,E,op,dest,parent,floor,cap):
    Q=len(qid);K=Q+P;R=len(cap);T=K+R;S=T+1;SS=S+1;TT=SS+1;n=TT+1
    ids=np.where(inside,np.searchsorted(O,parent),P+np.searchsorted(external,parent))+Q
    # Link capacities at least the entire supply make this an independent feasibility certificate.
    total=quota.sum();positive=np.arange(len(qid))[quota>0]
    arcs=[(qq,ids,np.full(len(qq),total)),(Q+np.searchsorted(O,op),K+E+np.searchsorted(ch,dest),np.full(len(op),total)),
     (np.arange(R)+K,np.full(R,T),np.maximum(0,cap-floor)),(np.array([T]),np.array([S]),np.array([total*2]))]
    balance=np.zeros(n);balance[S]-=quota.sum();balance[:Q]+=quota;balance[K:T]-=floor;balance[T]+=floor.sum()
    pos=np.flatnonzero(balance>0);neg=np.flatnonzero(balance<0)
    arcs.extend([(np.full(len(pos),SS),pos,balance[pos]),(neg,np.full(len(neg),TT),-balance[neg])])
    size=sum(len(v) for a,b,v in arcs);dt=np.dtype([('a','<i4'),('b','<i4'),('cap','<f8')]);buf=np.empty(size,dt);offset=0
    for a,b,v in arcs:
        sl=buf[offset:offset+len(v)];sl['a']=a;sl['b']=b;sl['cap']=v;offset+=len(v)
    proc=subprocess.run([str(binary.resolve())],input=np.asarray([n,SS,TT,size],'<i4').tobytes()+buf.tobytes(),capture_output=True,check=True)
    missed=[];cut=[]
    def identify(i):
        if i<Q:return 'carrier',int(qid[i])
        if i<K:return 'parent',int(O[i-Q])
        if i<K+E:return 'external',int(external[i-K])
        if i<T:return 'child',int(ch[i-K-E])
        return 'terminal',int(i)
    for row in proc.stdout.decode().splitlines():
        mode,i,*amt=row.split();i=int(i)
        if mode=='c':cut.append(i)
        else:missed.append((mode,*identify(i),float(amt[0])))
    unfilled=sum(s[3] for s in missed if s[0]=='r');unspent=sum(s[3] for s in missed if s[0]=='s')
    np.savez_compressed(out.with_suffix('.npz'),reachable=np.array(cut),O=O,external=external,children=ch,carrier=qid)
    out.write_text(json.dumps(dict(unfilled_kg=unfilled,unspent_kg=unspent,reach=len(cut),largest_unmet=sorted(missed,key=lambda v:-v[3])[:50],nodes=n,arcs=size),indent=2)+'\n')
    return unfilled
