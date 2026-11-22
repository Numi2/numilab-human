"""Use the compiled Float32 lattice fit for initialization; verify its result on every source cell."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from screen_registered_dense_posture import load_geometry,measure
from construct_registered_donor_field import errors
from untangle_registered_donor_field import differentials
from recover_material_thickness import audit

def main(full,donor,seed,out,tool,warm=None):
    x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    if json.loads((seed/'result.json').read_text())['source']!=pins:raise ValueError('wrong material domain/pose')
    wanted=x+np.load(seed/'displacement.npy')
    ref=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    if audit(wanted,t,ref)[3].any():raise ValueError('uncertified starting displacement')
    out.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(dir=out) as td:
        temp=Path(td);a,b,c,d=[temp/n for n in ('xyz','tets','reference','result')]
        wanted.astype('f8').tofile(a);t.astype('i4').tofile(b);ref.tofile(c)
        args=[str(tool),str(a),str(b),str(c),str(d)]
        if warm is not None:
            if json.loads((warm/'result.json').read_text())['source']!=pins:raise ValueError('unmatched warm lattice pose')
            initial=temp/'warm';(x+np.load(warm/'displacement.npy')).astype('f4').tofile(initial);args.append(str(initial))
        boundary=seed/'construction-boundary.npz'
        if boundary.exists():
            data=np.load(boundary);assert data['axis'].all()
            fixed=np.zeros(len(x),np.uint8);fixed[data['node']]=1
            if not np.array_equal(wanted[data['node']],data['value']):raise ValueError('displaced attachment')
            path=temp/'fixed';fixed.tofile(path)
            if warm is None:args.append('-')
            args.append(str(path))
        subprocess.run(args,check=True)
        points=np.fromfile(d,'f4').reshape(x.shape).astype(float)
    if boundary.exists():
        import shutil
        if np.any(points[:,2]<0) or not np.array_equal(points[data['node']],data['value']):raise ValueError('unilateral bound or anchor lost in quantization')
        shutil.copyfile(boundary,out/boundary.name)
    if not np.isfinite(points).all():raise ValueError('nonfinite Float32 pose')
    j,h,det,bad=audit(points,t,ref)
    result=dict(source=pins,seed=str(seed),failed=int(bad.sum()),min_J=float(j.min()),max_J=float(j.max()),
                quality=measure(points,owner,t,mat),failure_by_role={str(k):int(np.count_nonzero(bad&(mat==k))) for k in np.unique(mat)},
                failed_rows=(np.flatnonzero(bad)+1).tolist(),correction_to_seed=errors(points,wanted),
                discrepancy_to_rigid_donors=errors(points,target),outer_error=errors(points[outer],target[outer]),
                scope='Exactly representable Float32 initial-pose proposal on unchanged skin/fat material reference and mass. Attachment, contact and motion are separate checks.')
    np.save(out/'displacement.npy',points-x)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main(*(Path(v).resolve() for v in sys.argv[1:]))
