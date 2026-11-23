"""Resolve a newly intersected cortical face by adding an actual source edge to the free span."""
import numpy as np
from scipy.optimize import minimize,nnls
from certify_cortical_paths import line_length,verify

def crossing_edges(surface,a,b):
    contacts=surface.solid.ray_cast(a,b)
    verts=surface.xyz;tri=surface.tris;ids=set()
    for h in contacts:
        p=np.asarray(h.position);candidate=np.flatnonzero(((surface.face_lo<=p+1e-5)&(surface.face_hi>=p-1e-5)).all(1))
        for j in candidate:
            face=tri[j];A=(face[1:]-face[0]).T;uv=np.linalg.lstsq(A,p-face[0],rcond=None)[0]
            if np.linalg.norm(A@uv+face[0]-p)>1e-5 or min(1-uv.sum(),*uv)<-1e-5:continue
            ids.update(surface.inverse.reshape(-1,3)[j].tolist())
    return sorted(ids)

def insert_cortical_contact(y,intervals,normals,features,weights,owners,witnesses):
    if not witnesses:return None
    result=[]
    for seg,sid,depth in witnesses:
        o=owners[sid]
        if o.solid is None:continue
        for edge in crossing_edges(o,y[seg],y[seg+1]):
            pair=o.edges[edge];raw=o.first[pair].tolist();identity=(sid,tuple(sorted(raw)))
            if any(identity==(q['stable_id'],tuple(sorted(q['source_edge']))) for q in features):continue
            incident=np.flatnonzero(o.inverse==edge)//3
            if len(incident)!=2:continue
            E=np.insert(intervals,seg,o.verts[edge],axis=0);delta=E[:,1]-E[:,0]
            N=normals[:seg]+[o.normal[incident]]+normals[seg:]
            item=dict(stable_id=sid,body=o.body,source_edge=raw,incident_faces=incident.tolist())
            chain=features[:seg]+[item]+features[seg:]
            t=np.insert(weights,seg,.5);path=np.insert(y,seg+1,E[seg].mean(0),axis=0)
            def objective(x):
                path[1:-1]=E[:,0]+x[:,None]*delta
                length,d=line_length(path);return length,np.einsum('ij,ij->i',d,delta)
            opt=minimize(objective,t,method='L-BFGS-B',jac=True,bounds=[(0,1)]*len(t),
                options={'ftol':2e-15,'gtol':2e-9,'maxiter':500,'maxls':64})
            L,_=objective(opt.x);_,grad=line_length(path);res=[]
            for j,(t0,n,info) in enumerate(zip(opt.x,N,chain)):
                if min(t0,1-t0)<1e-6:
                    surf=owners[info['stable_id']];v=info['source_edge'][int(t0>.5)];n=surf.normal[(surf.faces==np.flatnonzero(surf.first==v)[0]).any(1)]
                res.append(nnls(n.T,grad[j])[1])
            if max(res)>1e-4 or verify(path,owners.values()):continue
            result.append((L,max(res),path.copy(),chain,opt.x.copy(),item))
    if not result:return None
    return min(result,key=lambda v:v[0])
