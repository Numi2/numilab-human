"""Measure the emitted material against every participating cortex, exterior skin, and fixed join."""
from pathlib import Path
import json,sys
import numpy as np
from build_carpal_cavity import boundary,shell
from audit_registered_skeleton import digest
from fit_carpal_clearance import geom
from rebuild_shoulder_cavities import canonical
from partition_carpal_material import solid


def run(domain,baseline,conflicts,out):
    z=np.load(domain);ref=np.load(baseline);c=np.load(conflicts/'witnesses.npz')
    x=z['x'];t=z['t'];roles=z['role'];old=ref['x']
    if len(x)!=len(old):raise ValueError('Quality verifier requires preserved node correspondence')
    faces=ref['facets'];fixed=ref['locked'];skin=faces[ref['markers']==15]
    if not np.array_equal(canonical(x,skin),canonical(old,skin)) or np.any(x[fixed]!=old[fixed]):raise ValueError('Protected surface/outer join changed')
    center=x.astype(float).mean(0);banks={}
    for sid in (8,9,10,11,12,13):
        y,inv=np.unique(c['cortex'+str(sid)],axis=0,return_inverse=True);fy=inv[c['faces'+str(sid)]]
        y=(y.astype(float)-center)*1000
        if np.einsum('ij,ij->',y[fy[:,0]],np.cross(y[fy[:,1]],y[fy[:,2]]))<0:fy=fy[:,[0,2,1]]
        banks[sid]=solid(y,fy)
    stats={}
    for mat in (15,16):
        choice=t[roles==mat];f=boundary(choice);ids,fi=np.unique(f,return_inverse=True)
        s=solid((x[ids].astype(float)-center)*1000,fi.reshape(-1,3))
        v,q=geom(x[choice].astype(float))
        stats[mat]=dict(cells=len(choice),inverted=int((v<=0).sum()),low=int(((v<=1e-18)|(q<1e-5)).sum()),
            overlap_mm3={sid:float((s^b).volume()) for sid,b in banks.items()},volume_mm3=float(s.volume()))
    report=dict(mesh=digest(domain),baseline=digest(baseline),registration=json.loads((conflicts/'result.json').read_bytes())['bones'],
                fixed_join_vertices=int(fixed.sum()),unchanged_outer_skin_faces=len(skin),layers=stats)
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(stats))


if __name__=='__main__':run(*map(Path,sys.argv[1:]))
