"""Choose representable interface coordinates near measured crossing witnesses and recertify the whole PLC."""
from pathlib import Path
import itertools,json,sys
import numpy as np
from scipy.spatial import cKDTree
from fast_skin_intersection_audit import install_for_float32,owner
from exact_plc_witnesses import run as audit
from audit_registered_skeleton import digest


def run(mesh,source,bad,out):
    z=np.load(source/'cavity.npz');doc=json.loads((source/'cavity.json').read_bytes())
    if digest(source/'cavity.npz')!=doc['cavity']:raise ValueError('Wrong patch')
    install_for_float32();f=z['faces'];x=z['vertex_world_m'].copy();start=x.copy()
    points=np.loadtxt(mesh/'mesh.node',skiprows=1,usecols=(1,2,3)).astype('f4')
    tree=cKDTree(points);movable=tree.query(x)[0]>0.
    # Only coordinates created at the intersection with a cortical source face
    # are eligible. No parent material vertex, exterior facet, or joining face moves.
    clip=np.unique(f[z['cortex_bank']!=0]);movable&=np.isin(np.arange(len(x)),clip)
    witness=json.loads(bad.read_bytes())
    origpairs={tuple(d['rows']) for d in witness['pairs']}
    def test(xx,rows):
        frows=np.array(sorted(rows),dtype=int)
        vtx,remap=np.unique(f[frows],return_inverse=True)
        rec,_=owner.float32_surface_records(xx[vtx],remap.reshape(-1,3))
        q=owner._audit_pair_prepared_first(owner._prepare_surface_aabb(rec),rec,same_surface=True)
        return {tuple(sorted((int(frows[a]),int(frows[b])))) for a,b in q['triangle_pairs']}
    tri=x[f].astype(float);lo=tri.min(1);hi=tri.max(1)
    pairs=origpairs.copy();history=[]
    for step in range(18):
        if not pairs:break
        active=sorted({k for pair in pairs for k in pair});nodes=np.unique(f[active]);nodes=nodes[movable[nodes]]
        # Test the current failures first. A proposed coordinate earns a full
        # neighboring-facet audit only if it removes one of those failures.
        best=None
        for j in nodes:
            affected=np.flatnonzero((f==j).any(1));seen=set(affected.tolist())
            for i in affected:
                seen.update(np.flatnonzero(np.all((hi+2e-6>=lo[i])&(lo-2e-6<=hi[i]),axis=1)).tolist())
            orig=x[j].copy()
            for size in (1,2,4,8,16,32):
                for axis,sign in itertools.product(range(3),(-1,1)):
                    v=orig.copy();v[axis]+=sign*size*abs(np.spacing(v[axis]))
                    if np.linalg.norm(v.astype(float)-start[j])>1e-6:continue
                    x[j]=v
                    try:
                        changed=test(x,active)
                        if len(changed)>=len(pairs):continue
                        newpairs=changed|test(x,seen)
                        if len(newpairs)>=len(pairs):continue
                    except (ValueError,RuntimeError):continue
                    norm=float(np.linalg.norm(v.astype(float)-start[j]))
                    score=(len(newpairs),norm)
                    if best is None or score<best[0]:best=score,int(j),v.copy(),newpairs
                x[j]=orig
                if best is not None and best[0][0]==0:break
        if best is None:break
        _,j,v,pairs=best;x[j]=v
        history.append(dict(vertex=j,remaining=len(pairs),displacement_m=float(np.linalg.norm(v.astype(float)-start[j]))))
        print(step,j,len(pairs),flush=True)
    out.mkdir();arr={k:z[k] for k in z.files};arr['vertex_world_m']=x;dst=out/'cavity.npz'
    np.savez_compressed(dst,**arr);doc.update(cavity=digest(dst),original_cavity=digest(source/'cavity.npz'),conditioning=history)
    (out/'cavity.json').write_text(json.dumps(doc,indent=2)+'\n')
    audit(dst,out/'exact-crossings.json')
    if json.loads((out/'exact-crossings.json').read_bytes())['count']:raise ValueError('Cortical PLC still has self-crossings')


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
