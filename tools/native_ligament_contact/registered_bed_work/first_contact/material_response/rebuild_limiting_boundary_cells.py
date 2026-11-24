"""Find conservative, boundary-preserving alternatives for the active preload-limiting cells."""
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from material_cavity_flips import replace_pass,boundary
from material_star_cavity import grow_pass
from supported_material_admission import geom
from compile_connected_joint import digest

def run(job, volume, fit, audit, out):
    proof=json.loads(audit.read_bytes())
    if digest(fit/'current.node')!=proof['current_node_sha256']:
        raise ValueError('Wrong registered configuration')
    read=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3)).astype('f4').astype('f8')
    x=read(volume/'mesh.node');y=read(fit/'current.node')
    rows=np.loadtxt(volume/'mesh.ele',usecols=(1,2,3,4,5),skiprows=1,dtype='i4')
    mesh,roles=rows[:,:4]-1,rows[:,4]
    boundary_nodes=np.load(job/'registered-boundary/registered-boundary.npz')
    layer=np.zeros(len(x),dtype='i4')
    for number,name in enumerate('OID',1):
        layer[boundary_nodes[name+'_node_joint']]=number
    ids=np.array([item['local_id'] for item in proof['cells']])
    if not np.array_equal(mesh[ids],np.array([item['nodes'] for item in proof['cells']])):
        raise ValueError('Witness topology changed')
    out.mkdir()
    history=[];patches={};results={}
    for role in (15,16):
        targets=ids[roles[ids]==role]
        inside=(roles==role)&np.isin(mesh,np.unique(mesh[targets])).any(1)
        domain=mesh[inside].copy()
        if not len(targets):continue
        original=domain.copy()
        corners=np.unique(domain)
        outside=mesh[(~inside)&(np.isin(mesh,corners).sum(1)>=2)]
        forbidden={v for cell in outside for v in combinations(sorted(map(int,cell)),2)}
        def face_only(t):
            return (layer[t[:,0]]!=0)&(layer[t]==layer[t[:,:1]]).all(1)
        def measure(t):
            v,q=geom(x,t);w,s=geom(y,t)
            ok=(v>1e-18)&(w>1e-18)&(q>=1e-5)&(s>=1e-5)&(w>.4*v)&(w<2.5*v)
            return ok,np.minimum(q,s)
        def score(t):
            ok,quality=measure(t);return np.where(ok,np.where(face_only(t),.15,1.)*(1+np.clip(quality,0,.1)),0.)
        def stretch(t):
            f=np.linalg.solve(x[t[:,1:]]-x[t[:,:1]],y[t[:,1:]]-y[t[:,:1]])
            sv=np.linalg.svd(f,compute_uv=False);return (sv[:,0]>2)|(sv[:,2]<.5)
        for method in [replace_pass,grow_pass]:
            for _ in range(4):
                _,ops=method(x,domain,forbidden,score)
                safe=[]
                for old,new in ops:
                    oa,oq=measure(old);na,nq=measure(new)
                    if not na.all() or nq.min()<oq.min() or stretch(new).sum()>stretch(old).sum():continue
                    if face_only(new).sum()>=face_only(old).sum():continue
                    if boundary(old)!=boundary(new) or set(old.ravel())!=set(new.ravel()):continue
                    if any(abs(geom(s,old)[0].sum()-geom(s,new)[0].sum())>abs(geom(s,old)[0].sum())*1e-9 for s in (x,y)):continue
                    safe.append((old,new))
                if not safe:break
                removed={tuple(v) for o,n in safe for v in o}
                domain=np.vstack([domain[[tuple(v) not in removed for v in domain]],*(v for o,v in safe)])
                for old,new in safe:history.append(dict(material=int(role),old=old.tolist(),new=new.tolist()))
        if boundary(original)!=boundary(domain) or set(original.ravel())!=set(domain.ravel()):
            raise ValueError('External boundary changed')
        final={tuple(sorted(c))for c in domain}
        results[str(role)]=dict(witnesses=targets.tolist(),remaining=[int(i) for i in targets if tuple(sorted(mesh[i])) in final],
                all_boundary_only_before=int(face_only(original).sum()),all_boundary_only_after=int(face_only(domain).sum()))
        for key,v in [('source',np.flatnonzero(inside)),('before',original),('after',domain)]:patches[key+str(role-15)]=v
    np.savez_compressed(out/'patches.npz',**patches)
    (out/'ancestry.json').write_text(json.dumps(history,separators=(',',':'))+'\n')
    report=dict(reference=digest(volume/'mesh.node'),loaded=digest(fit/'current.node'),elements=digest(volume/'mesh.ele'),
                roles=results,operations=len(history),audit=digest(audit))
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(results,len(history),flush=True)
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
