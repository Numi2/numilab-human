"""Independently resolve accepted GPU contact slots back to registered bones and P1 parents."""
from pathlib import Path
import csv,json,sys
import numpy as np
from material_context import sha
def audit(assets,run):
    d=json.loads((assets/'manifest.json').read_text())
    meta=d['motion_contact_registry'];first=d['finite_facets']['start']+meta['original_pairs']*16
    sides=meta['slots_per_side'];keys={}
    cortex=np.fromfile(assets/'motion-cortical-owners.bin','u4').reshape(-1,4)
    faces=np.fromfile(assets/'closed-material-triangles.bin','u4').reshape(-1,3,4)
    shapes=np.fromfile(assets/'closed-material-shapes.bin','f4').reshape(-1,3,4).astype(float)
    fixed=set(map(tuple,np.fromfile(assets/'motion-registered-faces.bin','u4').reshape(-1,2)))
    with (run/'accepted-cortical-registry.csv').open() as f:
        for row in csv.DictReader(f):
            slot,b,m=map(int,(row['slot'],row['cortical_row'],row['material_face']))
            if slot in keys or (b,m) in fixed or not (slot//sides==int(cortex[b,2])==int(faces[m,0,3])):raise ValueError('Duplicated or anatomically mismatched pressure port')
            keys[slot]=(b,m)
    cap=run/'accepted-pelvic-state.csv';loaded={};worst=0.;total=0.;by={};pressure=[]
    with cap.open() as f:
        for row in csv.DictReader(f):
            idx=int(row['index'])
            if idx<first or row['field'] not in ('reverse_parent_nodes','reverse_shape','normal_and_force'):continue
            loaded.setdefault(idx,{})[row['field']]=np.array([float(row[k]) for k in 'xyzw'])
    for idx,rec in loaded.items():
        n=rec['normal_and_force'];force=n[3]
        if not force:continue
        if force<0 or not np.all(np.isfinite(n)) or abs(np.linalg.norm(n[:3])-1)>2e-5:raise ValueError('Non-compressive or unnormalized contact')
        b,m=keys[(idx-first)//16];nodes=rec['reverse_parent_nodes'];w=rec['reverse_shape']
        if not np.array_equal(nodes,faces[m,2]):raise ValueError('Force reached different tissue parents')
        mat=np.vstack((shapes[m].T,np.ones(3)));rhs=np.r_[w,1.]
        alpha=np.linalg.lstsq(mat,rhs,rcond=None)[0];err=max(abs(mat@alpha-rhs));worst=max(worst,float(err))
        if err>2e-6 or abs(w.sum()-1)>3e-6 or min(alpha)<-2e-3:raise ValueError('Feature outside the registered material face')
        owner=int(cortex[b,1]);by[owner]=by.get(owner,0.)+float(force);total+=force
        pressure.append({'row':idx,'cortical_row':b,'material_face':m,'bone':owner,'force_N':float(force)})
    return dict(accepted_capture_sha256=sha(cap),registry_sha256=sha(run/'accepted-cortical-registry.csv'),registered_dynamic_pairs=len(keys),
        loaded_dynamic_rows=len(pressure),sum_compressive_magnitudes_N=float(total),sum_magnitudes_by_bone_N=by,
        max_parent_interpolation_absolute_residual=worst,rows=pressure)
if __name__=='__main__':
    a,r,out=map(Path,sys.argv[1:]);out.write_text(json.dumps(audit(a,r),indent=2)+'\n')
