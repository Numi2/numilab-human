"""Incremental whole-volume fixed-boundary quality flips; only changed neighborhoods re-enter work queue."""
from pathlib import Path
from collections import deque,Counter
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
H=Path(__file__).resolve().parent
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
T=imp("queue631T",H/"transfer_surface_contractions_cross_tissue_470_root.py");M=T.m
L=imp("queue631L",H/"local_fixed_boundary_cavity_polygon_search_656_root.py")
E=np.asarray(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)))
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seconds",type=float,default=180);ap.add_argument("--maximum-operations",type=int,default=200000);args=ap.parse_args()
 state=json.loads(args.state.read_text());state=state.get("current_verified_state_20261031",state);out=args.output;out.mkdir(parents=True,exist_ok=False)
 result=dict(state);reports=[];started=time.monotonic()
 for role in ("Skin","Fat"):
  start=time.monotonic();e=state[role+"_current"];assert sha(e["path"])==e["sha256"];a=load(e["path"]);p=a["points_world_m"];t=a["tetrahedra"];mat=a["material_indices"]
  n=len(t);capacity=n+64*args.maximum_operations;cells=np.empty((capacity,4),np.int64);cells[:n]=t;materials=np.empty(capacity,mat.dtype);materials[:n]=mat
  active=np.zeros(capacity,bool);active[:n]=True;bad=np.zeros(capacity,bool);bad[:n]=~M.passes(*M.measurements(p,t));initial=int(bad.sum())
  flat=t.ravel();order=np.argsort(flat,kind="stable");off=np.r_[0,np.cumsum(np.bincount(flat,minlength=len(p)))]
  inc=[set(map(int,order[off[i]:off[i+1]]//4)) for i in range(len(p))];del flat,order,off
  queue=deque(map(int,np.flatnonzero(bad[:n])));queued=bad.copy();nextid=n;parents={};ops=[];rejections=Counter();tests=0;last_print=start
  def overlap(vertices):
   nodes=sorted(map(int,vertices),key=lambda v:len(inc[v]));return set.intersection(*(inc[v] for v in nodes))
  while queue and len(ops)<args.maximum_operations and time.monotonic()-start<args.seconds:
   row=queue.popleft();queued[row]=False
   if not active[row] or not bad[row]:continue
   candidates=[];seen=set();tet=cells[row]
   for ed in tet[E]:
    rows=tuple(sorted(overlap(ed)))
    if 3<=len(rows)<=32 and rows not in seen:seen.add(rows);candidates.append(("3-2" if len(rows)==3 else "4-4" if len(rows)==4 else "polygon",np.asarray(rows,np.int64)))
   for ff in tet[M.PAT]:
    rows=tuple(sorted(overlap(ff)))
    if len(rows)==2 and rows not in seen:seen.add(rows);candidates.append(("2-3",np.asarray(rows,np.int64)))
   for kind,rows in candidates:
    assert np.all(active[rows]);old=cells[rows];verts=np.unique(old)
    if len(np.unique(materials[rows]))!=1:rejections["mixed_material"]+=1;continue
    idx={int(v):i for i,v in enumerate(verts)};lt=np.asarray([[idx[int(v)] for v in tt] for tt in old],np.int64)
    try:
     rr=L.repair_local_cavity(M,p[verts],lt,materials[rows],np.arange(len(rows)),kind,parent_offsets=np.arange(len(rows)+1),parent_rows=rows,retained_cell_keys=set())
    except ValueError as ex:rejections[str(ex)]+=1;continue
    tests+=1
    if not rr["accepted"]:rejections["no_certified_quality_gain"]+=1;continue
    nt=verts[rr["new_tets"]];cavity=set(map(int,rows))
    oldedges={tuple(sorted(map(int,x))) for x in old[:,E].reshape(-1,2)};newedges={tuple(sorted(map(int,x))) for x in nt[:,E].reshape(-1,2)}
    if any(overlap(ed)-cavity for ed in newedges-oldedges):rejections["external_new_edge"]+=1;continue
    oldfaces={tuple(sorted(map(int,x))) for x in old[:,M.PAT].reshape(-1,3)};newfaces={tuple(sorted(map(int,x))) for x in nt[:,M.PAT].reshape(-1,3)}
    if any(overlap(ff)-cavity for ff in newfaces-oldfaces):rejections["external_new_face"]+=1;continue
    if any(overlap(tt)-cavity for tt in nt):rejections["external_duplicate"]+=1;continue
    union=set()
    for r in rows:union.update((int(r),) if r<n else parents[int(r)])
    ancestor=tuple(sorted(union));assert ancestor
    newrows=list(range(nextid,nextid+len(nt)));assert newrows[-1]<capacity
    affected=set(map(int,verts))
    for r in rows:
     active[r]=False
     for v in cells[r]:inc[int(v)].remove(int(r))
     parents.pop(int(r),None)
    material=materials[rows[0]]
    newbadmask=~M.passes(*M.measurements(p,nt))
    for localrow,(r,tt) in enumerate(zip(newrows,nt)):
     cells[r]=tt;materials[r]=material;active[r]=True;parents[r]=ancestor;bad[r]=newbadmask[localrow]
     for v in tt:inc[int(v)].add(r)
    nextid+=len(nt)
    ops.append({"kind":kind,"old_cell_ids":rows.tolist(),"new_cell_ids":newrows,"new_tetrahedra":nt.tolist(),"native_bad_gain":rr["native_bad_gain_local"],"native_log_deficit_before":rr["native_log_deficit_before"],"native_log_deficit_after":rr["native_log_deficit_after"],"exact_m_certify":"PASS"})
    for v in affected:
     for r in inc[v]:
      if bad[r] and not queued[r]:queue.append(r);queued[r]=True
    break
   if time.monotonic()-last_print>10:
    print(json.dumps({"role":role,"operations":len(ops),"removed_bad":sum(o["native_bad_gain"] for o in ops),"queue":len(queue),"elapsed_s":time.monotonic()-start}),flush=True);last_print=time.monotonic()
  ids=np.flatnonzero(active[:nextid]);tt=cells[ids];mm=materials[ids];d,q=M.measurements(p,tt);final=int((~M.passes(d,q)).sum());assert np.all(d>0) and final==initial-sum(o["native_bad_gain"] for o in ops)
  bd=T.boundary(tt);assert np.array_equal(T.boundary_identity(bd),T.boundary_identity(a["desired_boundary_faces"]));assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
  keep=ids[ids<n];change=ids[ids>=n];assert np.array_equal(ids,np.r_[keep,change]);po=[0];pr=[]
  for r in change:pr.extend(parents[int(r)]);po.append(len(pr))
  omap=np.r_[keep,np.full(len(change),-1,np.int64)];imap=np.full(n,-1,np.int64);imap[keep]=np.arange(len(keep))
  payload=dict(a);payload.update(tetrahedra=tt,material_indices=mm,boundary_faces=bd,current_geometry_parent_path=np.asarray([e["path"]]),current_geometry_parent_sha256=np.asarray([e["sha256"]]),output_cell_to_input_cell=omap,input_cell_to_output_cell=imap,retained_input_cell_rows=keep,replacement_parent_offsets=np.asarray(po,np.int64),replacement_parent_cell_rows=np.asarray(pr,np.int64),replacement_tet_offsets=np.asarray([0,len(change)],np.int64),replacement_tetrahedra=tt[len(keep):],canonical_parent_union_operation_order=np.asarray("incremental-polygon-quality657"))
  dest=out/(role.lower()+"-fixed-boundary-quality-queue.npz");np.savez_compressed(dest,**payload)
  result[role+"_current"]={**e,"path":str(dest),"sha256":sha(dest),"cells":len(tt),"native_quality_failures":final,"nonpositive":0}
  reports.append({"role":role,"input":e,"candidate":result[role+"_current"],"bad_before":initial,"bad_after":final,"operations":ops,"tests":tests,"rejections":dict(rejections),"queue_exhausted":not queue,"queue_remaining":len(queue),"elapsed_s":time.monotonic()-start,"initial_cells":n,"active_cell_ids":str(out/(role.lower()+"-active-cell-ids.npz"))})
  np.savez_compressed(out/(role.lower()+"-active-cell-ids.npz"),active_cell_ids=ids)
  print("ROLE_DONE",role,initial,final,len(ops),"queue",len(queue),flush=True)
 result["source_state"]={"path":str(args.state),"sha256":sha(args.state)};result["physical_admission"]=False;result["native_steps"]=0
 for k in ("operations","independent_replay","exact_context"):result.pop(k,None)
 (out/"state.json").write_text(json.dumps(result,indent=2)+"\n")
 (out/"report.json").write_text(json.dumps({"status":"PASS_INCREMENTAL_LEXICOGRAPHIC_QUALITY_SEARCH","roles":reports,"producer":{"path":str(Path(__file__)),"sha256":sha(Path(__file__))},"elapsed_s":time.monotonic()-started,"physical_admission":False,"native_steps":0},indent=2)+"\n")
 print("STATE",str(out/"state.json"),flush=True)
if __name__=="__main__":main()
