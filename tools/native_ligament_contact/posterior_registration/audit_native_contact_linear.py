"""Recompute native contact subproblem residuals from captured GPU rows in Float64."""
from pathlib import Path
import json,sys,csv
import numpy as np
from material_context import sha
def audit(root):
    raw=root/'observed-contact-linear.bin'
    x=np.fromfile(raw,'f4').reshape(32,67600);events=[]
    for i,f in enumerate(x):
        ids=f[65536:65792].view('u4');keep=ids!=np.iinfo('u4').max
        W=f[:65536].astype(float).reshape(256,256)[np.ix_(keep,keep)]
        gap=f[65792:66048][keep].astype(float);p=f[66304:66560][keep].astype(float)
        if int(f[66816])!=1:continue
        if not np.all(np.isfinite(W)) or not np.all(np.isfinite(p)) or min(p)<0:raise ValueError('Invalid captured contact solve')
        h=gap+W@p;diag=W.diagonal()
        sym=.5*(W+W.T);use=diag>0;scale=1./np.sqrt(diag[use]);eig=np.linalg.eigvalsh(sym[np.ix_(use,use)]*scale[None,:]*scale[:,None])
        events.append(dict(iteration=i,admitted_slots=int(keep.sum()),nonzero_compliances=int(use.sum()),positive_solution_rows=int(sum(p>0)),
            material_direction=int(f[66817]),initial_minimum_gap_m=float(min(gap)),primal_penetration_m=float(max(0,-min(h))),
            max_complementarity_work_J=float(max(abs(p*h))),max_load_release_m=float(max(np.minimum(diag*p,abs(h)))),
            input_force_residual_N=float(f[66818]),input_force_scale_N=float(f[66819]),
            spectral_min_scaled=float(eig[0]),spectral_max_scaled=float(eig[-1]),
            relative_reciprocity_error=float(np.linalg.norm(W-W.T)/np.linalg.norm(W))))
    if not events:raise ValueError('No solves were observed')
    return {'source_sha256':sha(raw),'matrix':'actual GPU Float32 material compliance at current linearization, independently applied in Float64','side':'left','events':events,
        'max_linear_penetration_m':max(r['primal_penetration_m'] for r in events),'last':events[-1]}
if __name__=='__main__':
    p,o=map(Path,sys.argv[1:]);o.write_text(json.dumps(audit(p),indent=2)+'\n')
