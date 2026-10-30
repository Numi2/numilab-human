#!/usr/bin/env python3
import ast,copy,hashlib,json,pathlib,itertools
from fractions import Fraction
import numpy as np
H=pathlib.Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
T=H/"fat-construction-edge-controls-065-guarded";SRC=H/"collapse_exact_construction_edges_065.py";GUARDED=T/"collapse_exact_construction_edges_065_guarded.py"
def parse(path): return ast.parse(path.read_text())
def names(n): return {x.id for x in ast.walk(n) if isinstance(x,ast.Name)}
def extract(tree):
 defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="simplex_link"]
 assert len(defs)==1
 group_nodes=[]
 for i,n in enumerate(tree.body):
  if isinstance(n,ast.Assign) and {"inv","count"}<=names(n) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="np.unique":
   assert i+1<len(tree.body) and isinstance(tree.body[i+1],ast.Assign) and "groups" in names(tree.body[i+1])
   group_nodes=[copy.deepcopy(n),copy.deepcopy(tree.body[i+1])];break
 assert len(group_nodes)==2
 group_fn=ast.FunctionDef(name="duplicate_groups",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="p")],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[]),body=group_nodes+[ast.Return(value=ast.Name(id="groups",ctx=ast.Load()))],decorator_list=[])
 start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and "steps" in names(n))
 end=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and "tt" in names(n) and ast.unparse(n.value)=="compact[t]")
 body=[copy.deepcopy(n) for n in tree.body[start:end+1]]
 assert any(isinstance(n,ast.For) and ast.unparse(n.target)=="group" for n in body)
 keys=["steps","mapping","t","keep","removed_rows","initial","rows","compact","pp","tt","mat"]
 ret=ast.Return(value=ast.Dict(keys=[ast.Constant(k) for k in keys],values=[ast.Name(id=k,ctx=ast.Load()) for k in keys]))
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ["p","t","materials","original_count","groups"]],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
 core=ast.FunctionDef(name="contract_fixture",args=args,body=body+[ret],decorator_list=[])
 module=ast.Module(body=[copy.deepcopy(defs[0]),group_fn,core],type_ignores=[]);ast.fix_missing_locations(module)
 ns={"np":np,"itertools":itertools};exec(compile(module,"AST-extracted-frozen-contraction","exec"),ns)
 return ns["duplicate_groups"],ns["contract_fixture"]
def exact_det6(q):
 a=[[Fraction.from_float(float(v)) for v in row] for row in q]
 u=[a[1][k]-a[0][k] for k in range(3)];v=[a[2][k]-a[0][k] for k in range(3)];w=[a[3][k]-a[0][k] for k in range(3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
def result(name,ok,detail): return {"name":name,"pass":bool(ok),"detail":detail}
orig_groups,orig=extract(parse(SRC));groups,guarded=extract(parse(GUARDED));checks=[]
# A stellar 1-to-4 split whose construction point coincides exactly with source corner 0.
p=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0]],np.float32)
t=np.array([[4,1,2,3],[0,4,2,3],[0,1,4,3],[0,1,2,4]],np.int64);m=np.full(4,16,np.int32)
g=groups(p);out=guarded(p,t.copy(),m,4,g);tt=out["tt"];dets=[exact_det6(p[x]) for x in t];kept=[exact_det6(out["pp"][x]) for x in tt]
valid=(g==[[0,4]] and len(out["steps"])==1 and out["steps"][0]["simplicial_link_condition"] and len(out["removed_rows"])==3 and np.array_equal(out["mapping"][:4],np.arange(4)) and np.array_equal(out["pp"][:4].view(np.uint32),p[:4].view(np.uint32)) and len(tt)==1 and kept==[Fraction(1)] and sum(dets)==sum(kept) and np.array_equal(out["mat"],np.array([16],np.int32)) and all(exact_det6(p[out["initial"][r]])==0 for r in out["removed_rows"]) and np.all(np.isin(np.arange(4),np.unique(tt))))
checks.append(result("stellar_1_to_4_at_source_corner",valid,{"duplicate_group":g,"link_pass":out["steps"][0]["simplicial_link_condition"],"removed_zero_cells":out["removed_rows"],"retained_cells":tt.tolist(),"exact_parent_det6":str(sum(dets)),"exact_output_det6":str(sum(kept)),"original_prefix_identity":bool(np.array_equal(out["pp"][:4].view(np.uint32),p[:4].view(np.uint32)))}))
# A one-ULP/subnormal offset is not an exact construction duplicate.
near=np.nextafter(np.float32(0),np.float32(1));p2=p.copy();p2[4,0]=near
g2=groups(p2);o2=guarded(p2,t.copy(),m,4,g2);mapped=o2["compact"][o2["mapping"]]
false_ok=(g2==[] and len(o2["steps"])==0 and mapped[4]!=mapped[0] and np.array_equal(o2["tt"],t))
checks.append(result("reject_false_near_coincidence",false_ok,{"offset_f32":float(near),"groups":g2,"construction_row":int(mapped[4]),"source_corner_row":int(mapped[0]),"steps":len(o2["steps"])}))
# Two original source rows at the same position must never merge.
pm=np.array([[0,0,0],[0,0,0],[1,0,0],[0,1,0],[0,0,0]],np.float32);tm=np.array([[0,2,3,4]],np.int64);tm0=tm.copy()
try: guarded(pm,tm,np.array([16],np.int32),4,groups(pm));merge_rejected=False;message="no rejection"
except AssertionError as e:merge_rejected=True;message=str(e) or "original-row assertion"
checks.append(result("reject_merge_of_two_original_source_rows",merge_rejected and np.array_equal(tm,tm0),{"groups":groups(pm),"message":message,"input_cells_unchanged":bool(np.array_equal(tm,tm0))}))
# Two nondegenerate tetrahedra share the other three vertices but not the proposed edge.
pf=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,0]],np.float32);tf=np.array([[0,1,2,3],[4,1,2,3]],np.int64);mf=np.array([16,16],np.int32);tf0=tf.copy()
u=orig(pf,tf.copy(),mf,4,orig_groups(pf));duplicate_count=len(u["tt"])-len(np.unique(np.sort(u["tt"],axis=1),axis=0));duplicate_nonzero=all(exact_det6(pf[cell])!=0 for cell in u["tt"]);bad_link=not u["steps"][0]["simplicial_link_condition"] and duplicate_count>0 and duplicate_nonzero
try: guarded(pf,tf,mf,4,groups(pf));reject=False;msg="no rejection"
except RuntimeError as e:reject=True;msg=str(e)
checks.append(result("reject_nontrivial_failed_link",bad_link and reject and np.array_equal(tf,tf0),{"unguarded_link":u["steps"][0]["simplicial_link_condition"],"unguarded_duplicate_cells":duplicate_count,"unguarded_all_duplicate_cells_exact_nonzero":duplicate_nonzero,"guarded_rejected":reject,"guarded_error":msg,"guarded_input_cells_unchanged":bool(np.array_equal(tf,tf0))}))
report={"schema":"construction-edge-collapse-controls-065-guarded-v1","status":"PASS" if all(x["pass"] for x in checks) else "FAIL","frozen_065_source":str(SRC),"frozen_065_source_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),"guarded_derivative":str(GUARDED),"guarded_derivative_sha256":hashlib.sha256(GUARDED.read_bytes()).hexdigest(),"tests":checks,"physical_admission":False}
(T/"control-report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report,sort_keys=True))
if report["status"]!="PASS":raise SystemExit(1)
