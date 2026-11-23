"""Permit Steiner vertices inside the material PLC and verify the untouched exterior joining faces."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
from build_carpal_cavity import boundary,FACES
from verify_carpal_mesh import classify
from measure_forearm_enclosure import Enclosure
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from fit_carpal_clearance import geom
from audit_registered_skeleton import digest


def material_cells(x,t,f,marks,z,lib):
    # TetGen may mesh voids bounded by multiple disjoint source components.
    # Classify whole connected regions separated by actual recovered PLC facets.
    where={tuple(k):int(m) for k,m in zip(np.sort(f,axis=1),marks) if int(m)!=1000}
    facets=np.sort(t[:,FACES].reshape(-1,3),axis=1)
    keys,inv,cnt=np.unique(facets,axis=0,return_inverse=True,return_counts=True)
    if cnt.max()>2:raise ValueError('Nonmanifold tetrahedral adjacency')
    at=np.argsort(inv,kind='stable');begin=np.cumsum(cnt)-cnt
    shared=[i for i in np.flatnonzero(cnt==2) if tuple(keys[i]) not in where]
    pair=np.column_stack((at[begin[shared]],at[begin[shared]+1]))//4
    graph=coo_matrix((np.ones(len(pair)),pair.T),shape=(len(t),len(t)))
    count,labels=connected_components(graph,directed=False)
    shells={}
    for k in (15,16):
        bounds=z['faces'][np.isin(z['marker'],[k,100+k,1516])]
        shells[k]=Enclosure(lib,z['vertex_world_m'],bounds)
    try:
        volume,_=geom(x[t].astype(float))
        tag=np.zeros(count,'i4')
        for label in range(count):
            ids=np.flatnonzero(labels==label);pick=ids[np.argsort(-np.abs(volume[ids]))[:5]]
            center=x[t[pick]].astype(float).mean(1)
            inside=np.array([s.query(center)[0] for s in shells.values()])
            if (inside<0).any() or (inside.sum(0)>1).any() or not (inside==inside[:,0,None]).all():
                raise ValueError(f'Unresolved material component {label}')
            if inside[0,0]==1:tag[label]=15
            if inside[1,0]==1:tag[label]=16
    finally:
        for s in shells.values():s.close()
    return tag[labels]


def run(mesh,plc,source,tetgen,lib,out,quality=False):
    out.mkdir();z=np.load(plc/'cavity.npz')
    info=json.loads((plc/'cavity.json').read_bytes())
    if digest(plc/'cavity.npz')!=info['cavity']:raise ValueError('Wrong cortical volume')
    y=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4')
    t=np.loadtxt(mesh/'mesh.ele',skiprows=1,dtype='i4')[:,1:5]-1;old=z['source_tetrahedra']
    bound=boundary(t[old]);adj=np.flatnonzero(np.isin(t,np.unique(bound)).sum(1)>=3);adj=np.setdiff1d(adj,old)
    outer={tuple(k) for k in np.sort(t[adj][:,FACES].reshape(-1,3),axis=1)}
    fence=np.array([v for v in bound if tuple(sorted(v)) in outer])
    wanted={tuple(sorted(y[key].tobytes() for key in row)) for row in fence}
    cfg=dict(source=digest(source),patch=digest(plc/'cavity.npz'),tetgen=digest(tetgen),source_joining_faces=len(wanted),trials=[])
    opts=([('improve','-pYzQfO2/7'),('refine','-pYzQfq3S1800'),('refine-fine','-pYzQfq2S6000')] if quality
          else [('interior','-pYzQf'),('adaptive','-pzQf')])
    for label,flags in opts:
        dest=out/label;dest.mkdir();poly=dest/'shoulder.poly';poly.write_bytes(source.read_bytes())
        cmd=[str(tetgen),flags,str(poly)];proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=160)
        (dest/'tetgen.log').write_bytes(proc.stdout)
        trial=dict(command=cmd,exit=proc.returncode);cfg['trials'].append(trial)
        if proc.returncode:continue
        node=np.loadtxt(dest/'shoulder.1.node',skiprows=1);ele=np.loadtxt(dest/'shoulder.1.ele',skiprows=1,dtype='i8')
        faces=np.loadtxt(dest/'shoulder.1.face',skiprows=1,dtype='i8');markers=faces[:,4];keep=markers>0
        # TetGen may emit -1 on unmarked convex hull faces. All material
        # surfaces must retain their supplied markers and partition by adjacency.
        x=(node[:,1:4]*.001+np.asarray(info['center_m'])).astype('f4')
        tc=ele[:,1:5];f=faces[keep,1:4];mk=markers[keep]
        if not np.array_equal(node[:,0],np.arange(len(x))) or not np.array_equal(ele[:,0],np.arange(len(tc))):raise ValueError('TetGen renumbered without mapping')
        tags=material_cells(x,tc,f,mk,z,lib)
        tc=tc[tags>0];expected=tags[tags>0]
        role,external=classify(x,tc,f,mk)
        if not np.array_equal(role,expected):raise ValueError('Conflicting material ownership')
        trial['excluded_void_cells']=int((tags==0).sum())
        got={tuple(sorted(x[k].tobytes() for k in row)) for row in external}
        # classify returns triangle row IDs as canonical node triples.
        vv,shape=geom(x[tc].astype(float));bad=(vv<=1.e-18)|(shape<1.e-5)
        trial.update(nodes=len(x),cells=len(tc),added_nodes=len(x)-len(z['vertex_world_m']),missing_joining_faces=len(wanted-got),
                     inverted=int((vv<=0).sum()),bad_cells=int(bad.sum()),layer_counts={str(k):int((role==k).sum()) for k in (15,16)})
        locked_set={y[int(k)].tobytes() for k in np.unique(fence)}
        locked=np.array([pt.tobytes() in locked_set for pt in x])
        np.savez_compressed(dest/'domain.npz',x=x,t=tc,role=role,facets=f,markers=mk,outer=external,locked=locked,fence=fence)
        print(label,trial,flush=True)
        if not trial['missing_joining_faces'] and (not quality or not trial['bad_cells']):break
    (out/'result.json').write_text(json.dumps(cfg,indent=2)+'\n')


if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
