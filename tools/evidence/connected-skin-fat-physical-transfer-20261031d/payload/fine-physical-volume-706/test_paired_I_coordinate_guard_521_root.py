from pathlib import Path
import ast,json,hashlib,numpy as np
H=Path(__file__).resolve().parent
p=H/'repair_paired_I_current_coordinates_520_root.py'
tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admitted_coordinate_prefix'];assert len(nodes)==1
ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
f=ns['admitted_coordinate_prefix'];a=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);pending=np.array([0,1,2])
b=a.copy();b[3]=[.1,.1,.5]
assert f(a,a,pending)
assert not f(b,a,pending)
assert f(b,a,pending,b)
c=b.copy();c[0,0]=.1
assert not f(c,a,pending,c)
assert not f(c,a,pending,b)
assert not f(b[:-1],a,pending,b)
assert not f(b,a,pending,a)
c=np.vstack([b,np.asarray([[.2,.2,.2]],np.float32)])
assert f(c,a,pending,b)
r={'status':'PASS_CERTIFIED_COORDINATE_PREFIX_GUARD_CONTROLS','controls':['legacy_exact_prefix','unqualified_interior_change_rejected','certified_interior_only_change','pending_boundary_change_rejected','reference_mismatch_rejected','short_prefix_rejected','appended_inferred_nodes_allowed'],'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'physical_admission':False,'native_steps':0}
out=H/'paired-I-current-coordinate-521-controls-root';out.mkdir(exist_ok=False);(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
