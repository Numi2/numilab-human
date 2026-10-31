"""Continue paired source-boundary diagonal repair with exact witness pruning."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
H=Path(__file__).resolve().parent;PY=sys.executable
def imp(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def pin(p):return {"path":str(p),"sha256":sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
def command(argv,log):
 with log.open("w") as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:print(log.read_text()[-5000:],flush=True);raise RuntimeError((argv,r.returncode))
def replay(base_path,new_path,out):
 C=imp("replay678C",H/"batch_paired_boundary_diagonal_quality_679_root.py");M=C.M;before=read(base_path);before=before.get("current_verified_state_20261031",before);now=read(new_path);items=[]
 for role in ("D","I","O"):
  a=load(before["source_surfaces"][role]["path"]);b=load(now["source_surfaces"][role]["path"]);f=a["faces"].copy();seen=set()
  assert np.array_equal(a["points_world_m_f32"].view("u4"),b["points_world_m_f32"].view("u4")) and np.array_equal(a["points_world_m"],b["points_world_m"])
  for oi,op in enumerate(now["operations"][role]):
   rows=np.asarray(op["source_face_rows"],np.int64);assert not seen.intersection(map(int,rows));seen.update(map(int,rows));assert np.array_equal(f[rows],op["old_source_faces"]);f[rows]=op["new_source_faces"]
   for row in rows:
    assert C.A.rows_union(b,int(row))==C.B.union_tuples(a,rows)
    assert list(map(tuple,C.A.rows_anc(b,int(row))))==list(map(tuple,C.B.ancestry_union(a,rows)))
  assert np.array_equal(f,b["faces"]);assert np.all(C.J.score(b["points_world_m_f32"],f)>=C.J.GATE)
 for vr in ("Skin","Fat"):
  a=load(before[vr+"_current"]["path"]);b=load(now[vr+"_current"]["path"]);p=a["points_world_m"];assert np.array_equal(p.view("u4"),b["points_world_m"].view("u4"));removed=[];news=[];newmat=[];po=[0];parents=[];target=Counter(C.L._face_chain(a["desired_boundary_faces"]));used=set()
  for role in ("D","I","O"):
   for op in now["operations"][role]:
    if vr not in op["volume_trials"]:continue
    tr=op["volume_trials"][vr];rows=np.asarray(tr["old_rows"],np.int64);old=a["tetrahedra"][rows];nt=np.asarray(tr["new_tetrahedra"],np.int64);verts=set(map(int,old.ravel()));assert not verts&used;used|=verts
    assert len(np.unique(a["material_indices"][rows]))==1
    expected=Counter(M.chain(old))
    for ff in tr["old_source_faces"]:C.addface(expected,ff,-1);C.addface(target,ff,-1)
    for ff in tr["new_source_faces"]:C.addface(expected,ff,1);C.addface(target,ff,1)
    assert dict(M.chain(nt))==dict(expected)
    oldbad,oldscore=C.score(p,old);newbad,newscore=C.score(p,nt);assert oldbad==tr["bad_before"] and newbad==tr["bad_after"] and newbad<=oldbad
    nodes=np.unique(np.r_[old.ravel(),nt.ravel()]);li={int(x):i for i,x in enumerate(nodes)};ip=M.integer_points(p[nodes]);ds0=sum(M.exact_det(ip,[li[int(x)] for x in tt]) for tt in old);ds1=[M.exact_det(ip,[li[int(x)] for x in tt]) for tt in nt];assert min(ds1)>0 and str(sum(ds1)-ds0)==tr["delta_exact"]
    removed.extend(rows.tolist());news.append(nt);newmat.extend([int(a["material_indices"][rows[0]])]*len(nt))
    for _ in nt:parents.extend(rows.tolist());po.append(len(parents))
  assert len(removed)==len(set(removed));keep=np.ones(len(a["tetrahedra"]),bool);keep[removed]=False;retained=np.flatnonzero(keep);new=np.vstack(news) if news else np.empty((0,4),np.int64);tt=np.vstack((a["tetrahedra"][keep],new));mat=np.r_[a["material_indices"][keep],np.asarray(newmat,a["material_indices"].dtype)]
  assert np.array_equal(tt,b["tetrahedra"]) and np.array_equal(mat,b["material_indices"])
  assert np.array_equal(retained,b["retained_input_cell_rows"]) and np.array_equal(np.asarray(po,np.int64),b["replacement_parent_offsets"]) and np.array_equal(np.asarray(parents,np.int64),b["replacement_parent_cell_rows"])
  bd=C.T.boundary(tt);assert dict(C.L._face_chain(bd))==dict(target);assert np.array_equal(C.T.boundary_identity(bd),C.T.boundary_identity(b["desired_boundary_faces"]))
  d,q=M.measurements(p,tt);assert np.all(d>0);bad=int((~M.passes(d,q)).sum());assert bad==now[vr+"_current"]["native_quality_failures"];assert len(np.unique(np.sort(tt,axis=1),axis=0))==len(tt)
  items.append({"role":vr,"quality_failures":bad,"exact_positive_replacements":len(new),"full_parent_union_rows":len(parents),"oriented_target_replayed":True})
 write(out,{"status":"PASS_INDEPENDENT_PAIRED_BOUNDARY_DIAGONAL_REPLAY","input_state":pin(base_path),"proposal_state":pin(new_path),"roles":items,"physical_admission":False,"native_steps":0})
def bad_operations(state,audit):
 st=read(state);r=read(audit);cursor=1901089;mapping={}
 for role in ("D","O","I"):
  a=load(st["source_surfaces"][role]["path"])
  for row in r["roles"][role]["changed_candidate_face_rows"]:
   oi=int(a["source_diagonal_flip_operation_rows"][row]);assert oi>=0;mapping[cursor]=(role,oi);cursor+=1
 assert cursor==1901089+r["changed_face_total"]
 bad=set()
 for group in ("changed_vs_complete_retained_OID_both_walls_all185","changed_OID_joint_self_pairs"):
  for pair in r[group]["triangle_pairs"]:
   touched=[mapping[int(x)] for x in pair if int(x) in mapping];assert touched;bad.update(touched)
 return bad
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--state",type=Path,required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--rounds",type=int,default=4);a=ap.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False);state=a.state;progress=[];start=time.monotonic()
 for nr in range(1,a.rounds+1):
  rd=root/("round%03d"%nr);rd.mkdir();base=read(state);base=base.get("current_verified_state_20261031",base);allowed=None;pruned=0
  for trial in range(1,10):
   out=rd/("trial%02d"%trial);argv=[PY,str(H/"batch_paired_boundary_diagonal_quality_679_root.py"),"--state",str(state),"--output",str(out),"--seconds","90"]
   if allowed is not None:
    af=rd/("allowed%02d.json"%trial);write(af,allowed);argv+=["--allowed",str(af)]
   command(argv,rd/("build%02d.log"%trial));candidate=out/"state.json";st=read(candidate);nops=sum(map(len,st["operations"].values()))
   if not nops:
    write(root/"stalled.json",{"status":"NO_VALID_PAIRED_BOUNDARY_DIAGONALS","state":pin(state),"round":nr,"progress":progress});print("DIAGONAL_STALLED",str(state),flush=True);return
   config=out/"context-config.json";write(config,{"schema":"h706.combined-current-exact-surface-diff-config.v1","baseline_candidates":base["source_surfaces"],"candidate_surfaces":st["source_surfaces"],"changed_roles":[role for role in ("D","O","I") if st["operations"][role]]})
   audit=out/"context001/report.json";command([PY,str(H/"combined-surface-diff-audit-270/audit_combined_declared_surface_diff_354.py"),"--config",str(config),"--out",str(audit.parent)],rd/("audit%02d.log"%trial))
   ar=read(audit)
   if ar["status"]=="PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185":
    independent=out/"independent-replay.json";replay(state,candidate,independent);st.update(surface_crossings=0,exact_context=pin(audit),independent_replay=pin(independent));dest=out/"verified-state.json";write(dest,st);state=dest
    item={"round":nr,"state":str(dest),"sha256":sha(dest),"Skin":st["Skin_current"]["native_quality_failures"],"Fat":st["Fat_current"]["native_quality_failures"],"operations":nops,"pruned_operations":pruned,"elapsed_s":time.monotonic()-start};progress.append(item);write(root/"latest-verified.json",{"current":item,"progress":progress});print("VERIFIED",json.dumps(item),flush=True);break
   bad=bad_operations(candidate,audit);assert bad;allowed=[[role,*sorted(op["source_edge"])] for role,ops in st["operations"].items() for oi,op in enumerate(ops) if (role,oi) not in bad];pruned+=len(bad);print("PRUNE",nr,trial,len(bad),"retaining",len(allowed),flush=True)
  else:raise RuntimeError("Exact diagonal witness pruning failed to converge")
  if st["Skin_current"]["native_quality_failures"]+st["Fat_current"]["native_quality_failures"]==0:print("ZERO_QUALITY_FAILURES",str(state),flush=True);return
 print("ROUND_BUDGET",str(state),flush=True)
if __name__=="__main__":main()
