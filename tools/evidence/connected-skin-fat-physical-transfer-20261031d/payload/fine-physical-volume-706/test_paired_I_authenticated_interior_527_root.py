from pathlib import Path
import ast,json,hashlib,numpy as np
H=Path(__file__).resolve().parent;p=H/'repair_paired_I_authenticated_interior_526_root.py'
nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='admitted_coordinate_prefix']
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns);f=ns['admitted_coordinate_prefix']
a=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.2,.2,.2]],np.float32);pending=np.asarray([0,1,2,3]);fixed=np.asarray([1,1,1,0],bool)
b=a.copy();b[3]=[.2,.2,.3]
assert f(a,a,pending)
assert not f(b,a,pending)
assert not f(b,a,pending,b)
assert f(b,a,pending,b,fixed)
c=b.copy();c[0,0]=.01
assert not f(c,a,pending,c,fixed)
assert not f(c,a,pending,b,fixed)
assert not f(b[:-1],a,pending,b,fixed)
assert not f(b,a,pending,b,fixed[:-1])
assert f(np.vstack([b,[[.1,.1,.1]]]).astype(np.float32),a,pending,b,fixed)
r={'status':'PASS_AUTHENTICATED_INTERIOR_PENDING_GUARD_CONTROLS','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'controls':['legacy_parity','unqualified_rejected','fixed_mask_required','certified_interior_pending_allowed','fixed_pending_change_rejected','reference_mismatch_rejected','short_prefix_rejected','malformed_fixed_mask_rejected','appended_inferred_points_allowed'],'physical_admission':False,'native_steps':0}
out=H/'paired-I-authenticated-interior-controls-527-root';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
