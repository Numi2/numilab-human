"""Check the pelvic owner's pose stamp against independent FK from accepted q."""
import csv,json,sys
from pathlib import Path
import numpy as np
from material_context import sha
def run(state,receipt,out):
    rows=list(csv.DictReader(state.open()));doc=json.loads(receipt.read_text());n=int(rows[0]['accepted_step'])
    if doc['accepted_step']!=n:raise ValueError('different physical steps')
    F={r['body_index']:r for r in doc['accepted_body_poses']}
    fields={(r['field'],int(r['index'])):np.array([float(r[k]) for k in 'xyzw']) for r in rows if r['field'] in ('orientation','relative_position')}
    errors={}
    for i in (1,128,131,145):
        measured=fields['orientation',i];q=np.asarray(F[i]['quaternion_xyzw']);qerr=min(np.linalg.norm(q-measured),np.linalg.norm(q+measured))
        delta=np.array(F[i]['position_m'])-F[128]['position_m']
        errors[i]=dict(relative_origin_error_m=float(np.linalg.norm(delta-fields['relative_position',i][:3])),quaternion_error=float(qerr))
    # CSV publication is Float32 rounded to nine significant digits. Stand presents
    # unpaired world positions; their subtraction may lose two Float32 ULPs.
    tol=max(2*abs(float(np.spacing(np.float32(v)))) for i in errors for v in [*F[i]['position_m'],*F[128]['position_m']])*np.sqrt(3)
    qt=4*float(np.finfo(np.float32).eps)
    d=dict(step=n,receipt=sha(receipt),state=sha(state),errors=errors,relative_origin_limit_m=tol,quaternion_limit=qt,
           passed=all(v['quaternion_error']<=qt and v['relative_origin_error_m']<=tol for v in errors.values()))
    out.write_text(json.dumps(d,indent=2)+'\n');print(d['passed'],errors,flush=True)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
