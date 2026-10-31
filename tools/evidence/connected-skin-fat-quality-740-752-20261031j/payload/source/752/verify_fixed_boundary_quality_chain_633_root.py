"""Independent replay of whole-mesh and incremental fixed-boundary flip journals."""
from pathlib import Path
import collections,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
T=imp("verify633T",H/"transfer_surface_contractions_cross_tissue_470_root.py");M=T.m
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def read(p):return json.loads(Path(p).read_text())
def certify(p,old,new):
 assert M.certify(p,old,new)==(True,"PASS")
 od,oq=M.measurements(p,old);nd,nq=M.measurements(p,new);assert np.all(M.passes(nd,nq))
 return int((~M.passes(od,oq)).sum())
def verify(reportpath,kind):
 report=read(reportpath);rr=[]
 for r in report["roles"]:
  inp=r["input"];dest=r["candidate"];assert sha(inp["path"])==inp["sha256"] and sha(dest["path"])==dest["sha256"]
  a=load(inp["path"]);b=load(dest["path"]);p=a["points_world_m"];t=a["tetrahedra"].copy();mat=a["material_indices"].copy();assert np.array_equal(p.view("u4"),b["points_world_m"].view("u4"))
  original_n=len(t);gain=0;origin=np.arange(len(t));parents={}
  if kind=="sweep":
   groups=collections.defaultdict(list)
   for op in r["operations"]:groups[op["pass"]].append(op)
   for k,ops in sorted(groups.items()):
    keep=np.ones(len(t),bool);new=[];newmat=[];newanc=[];occupied=set()
    for op in ops:
     rows=np.asarray(op["current_old_cell_rows"]);nt=np.asarray(op["new_tetrahedra"]);old=t[rows];nodes=set(map(int,old.ravel()));assert not nodes&occupied;occupied.update(nodes)
     local=certify(p,old,nt);assert local==op["bad_gain"];gain+=local;assert len(np.unique(mat[rows]))==1
     union=set()
     for row in rows:union.update((int(origin[row]),) if origin[row]>=0 else parents[int(row)])
     assert sorted(union)==op["complete_initial_parent_union"]
     keep[rows]=False;new.extend(nt.tolist());newmat.extend([int(mat[rows[0]])]*len(nt));newanc.extend([tuple(sorted(union))]*len(nt))
    kept=np.flatnonzero(keep);nextparents={j:parents[int(old)] for j,old in enumerate(kept) if origin[old]<0}
    for j,anc in enumerate(newanc):nextparents[len(kept)+j]=anc
    parents=nextparents;origin=np.r_[origin[keep],np.full(len(new),-1,np.int64)];t=np.vstack((t[keep],np.asarray(new,np.int64)));mat=np.r_[mat[keep],np.asarray(newmat,mat.dtype)]
  else:
   ops=r["operations"];capacity=original_n+4*len(ops);cells=np.empty((capacity,4),np.int64);cells[:original_n]=t;mm=np.empty(capacity,mat.dtype);mm[:original_n]=mat
   active=np.zeros(capacity,bool);active[:original_n]=True;nextid=original_n
   for op in ops:
    rows=np.asarray(op["old_cell_ids"]);newrows=np.asarray(op["new_cell_ids"]);nt=np.asarray(op["new_tetrahedra"]);assert np.all(active[rows])
    assert np.array_equal(newrows,np.arange(nextid,nextid+len(nt)))
    local=certify(p,cells[rows],nt);assert local==op["native_bad_gain"];gain+=local;assert len(np.unique(mm[rows]))==1
    union=set()
    for row in rows:union.update((int(row),) if row<original_n else parents[int(row)])
    for row in rows:parents.pop(int(row),None)
    for row in newrows:parents[int(row)]=tuple(sorted(union))
    cells[newrows]=nt;mm[newrows]=mm[rows[0]];active[rows]=False;active[newrows]=True;nextid+=len(nt)
   ids=np.flatnonzero(active[:nextid]);t=cells[ids];mat=mm[ids];origin=np.where(ids<original_n,ids,-1)
   parents={i:parents[int(row)] for i,row in enumerate(ids) if row>=original_n}
   with np.load(r["active_cell_ids"]) as z:assert np.array_equal(ids,z["active_cell_ids"])
  assert np.array_equal(t,b["tetrahedra"]) and np.array_equal(mat,b["material_indices"])
  assert np.array_equal(origin,b["output_cell_to_input_cell"])
  changed=np.flatnonzero(origin<0);po=b["replacement_parent_offsets"];pr=b["replacement_parent_cell_rows"];assert len(po)==len(changed)+1 and po[-1]==len(pr)
  for j,row in enumerate(changed):assert np.array_equal(pr[po[j]:po[j+1]],parents[int(row)])
  bd=T.boundary(t);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(a["desired_boundary_faces"])) and np.array_equal(T.boundary_identity(bd),T.boundary_identity(b["boundary_faces"]))
  dd,qq=M.measurements(p,t);bad=int((~M.passes(dd,qq)).sum());assert np.all(dd>0) and bad==r["bad_after"]==r["bad_before"]-gain
  rr.append({"role":r["role"],"input":inp,"candidate":dest,"independent_exact_operations":len(r["operations"]),"bad_before":r["bad_before"],"bad_after":bad,"unchanged_points_materials_and_complete_boundary":True,"complete_transitive_parent_unions":True})
 return {"source_report":{"path":str(reportpath),"sha256":sha(reportpath)},"roles":rr}
def main():
 start=time.monotonic();out=H/"all-quality-633-root";out.mkdir(exist_ok=False)
 result={"status":"PASS_INDEPENDENT_FIXED_BOUNDARY_FLIP_CHAINS","checks":[verify(H/"all-quality-628-root/flips001/report.json","sweep"),verify(H/"all-quality-631-root/queue001/report.json","queue")],"physical_admission":False,"native_steps":0,"elapsed_s":time.monotonic()-start,"producer":{"path":str(Path(__file__)),"sha256":sha(Path(__file__))}}
 (out/"receipt.json").write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result),flush=True)
if __name__=="__main__":main()
