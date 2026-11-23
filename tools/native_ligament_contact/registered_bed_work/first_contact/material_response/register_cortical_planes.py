"""Recover each conditioned cortical triangle's actual parent bone plane and emitted material nodes."""
from pathlib import Path
import json,sys
import numpy as np
from audit_registered_skeleton import digest

def planes(plc,initial,current,bones,collapses):
    c=np.load(plc);z=np.load(initial);d=np.load(current);w=np.load(bones)
    local={p.tobytes():i for i,p in enumerate(z['x'])}
    nodes=np.array([local[row.tobytes()] for row in c['vertex_world_m']])
    changes=json.loads(collapses.read_bytes())
    for op in changes['operations']:nodes[nodes==op['drop']]=op['keep']
    parent={};bankplanes={}
    for k in sorted(set(c['cortex_bank'])-{0}):
        y,inv=np.unique(w['cortex'+str(k)],axis=0,return_inverse=True);y=y.astype(float);f=inv[w['faces'+str(k)]]
        if np.einsum('ij,ij->',y[f[:,0]]-y.mean(0),np.cross(y[f[:,1]]-y.mean(0),y[f[:,2]]-y.mean(0)))<0:f=f[:,[0,2,1]]
        pts=y[f];n=np.cross(pts[:,1]-pts[:,0],pts[:,2]-pts[:,0]);n/=np.linalg.norm(n,axis=1)[:,None];bankplanes[k]=(pts[:,0],n)
    for row,k,j in zip(c['faces'],c['cortex_bank'],c['parent_face']):
        if k:
            key=tuple(sorted(nodes[row]))
            if len(set(key))!=3:continue
            if key in parent and parent[key]!=(k,j):raise ValueError('Ambiguous cortical ancestry')
            parent[key]=k,j
    ids=[];normals=[];origins=[];seen=set()
    for face in d['facets'][np.isin(d['markers'],[115,116])]:
        k,j=parent[tuple(sorted(face))];p,n=bankplanes[k]
        for i in face:
            if (i,k,j) in seen:continue
            seen.add((i,k,j));ids.append(i);normals.append(n[j]);origins.append(p[j])
    return np.array(ids),np.array(normals),np.array(origins)
if __name__=='__main__':
    a=list(map(Path,sys.argv[1:]));ids,n,p=planes(*a[:-1]);x=np.load(a[2])['x'].astype(float);g=np.einsum('ij,ij->i',n,x[ids]-p)
    print(len(ids),np.percentile(g*1e6,[0,1,25,50,75,99,100]))
    np.savez_compressed(a[-1],node=ids,normal=n,on_plane=p,initial=digest(a[1]),source=digest(a[2]),cortex=digest(a[3]))
