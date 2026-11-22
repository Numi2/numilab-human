"""Seat the admitted full reference without changing its joint coordinates or material reference."""
import json,sys
from pathlib import Path
import numpy as np
from evaluate_registered_bed_transfer import digest
from screen_registered_dense_posture import load_geometry
from recover_material_thickness import audit
from untangle_registered_donor_field import differentials

def main(full,out,distance,footprint=None):
    source=full/'launch.json';accepted=full/'native/accepted-geometry/step-16.receipt.json'
    cfg=json.loads(source.read_text());a=cfg['argv'];i=a.index('--root-pose')+3
    limit=.005
    if footprint:
        info=json.loads(footprint.read_text())
        assert info['source_receipt_sha256']==digest(accepted)
        limit=info['minimum_rigid_bone_height_m']
    if not 0<float(distance)<limit or json.loads(accepted.read_text())['accepted_step']!=16:raise ValueError('unbound reference')
    source_root=np.asarray(a[i-2:i+5],dtype='f4');target=source_root.copy();target[2]-=np.float32(distance)
    delta=(target[:3].astype(float)-source_root[:3].astype(float)).tolist()
    a[i]=format(float(target[2]),'.17g');cfg['environment']['NUMI_EXPERIMENT_INITIAL_ROOT_POSE']=','.join(a[i-2:i+5])
    cfg['placement_reference_receipt']=str(accepted);cfg['placement_reference_receipt_sha256']=digest(accepted)
    cfg['placement_reference_launch']=str(source);cfg['placement_reference_launch_sha256']=digest(source)
    cfg['construction_translation_m']=delta
    out.mkdir(exist_ok=False);path=out/'config.json';path.write_text(json.dumps(cfg,indent=2)+'\n')
    x,y,owner,t,mat,outer,R,pins=load_geometry(full,path);proposed=x+np.asarray(delta)
    if np.max(np.abs(proposed-y))>2e-6:raise ValueError('rigid body and material registration disagree')
    rr=np.concatenate([differentials(x,t[k:k+32768])[0] for k in range(0,len(t),32768)])
    assert not audit(proposed,t,rr)[3].any()
    np.save(out/'displacement.npy',proposed-x)
    (out/'result.json').write_text(json.dumps(dict(source=pins,original_pose=digest(accepted),translation_m=delta,
        native_match_required=True,initial_outer_penetrating=int((proposed[outer,2]<0).sum())),indent=2)+'\n')
if __name__=='__main__':main(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),float(sys.argv[3]),Path(sys.argv[4]) if len(sys.argv)>4 else None)
