from pathlib import Path
import sys,json,csv,importlib.util,numpy as np,mmap
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");C=A/"left-scm-masseters-outward-native-001";P=A/"left-scm-masseters-native-composition-003";N=C/"baseline/native-run";B=P/"baseline/native-run";O=C/"native-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=h.sha;T=C/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);proof=A/"left-scm-masseters-positive-winding-001/winding-proof.json"
pins={str(p):sha(p) for p in [Path(__file__),H,T,T.with_suffix(".manifest.json"),proof,h.INV,Path(ci.__file__),Path(ca.__file__),N/"run-metadata.json",N/"invocation.json",N/"resting-coupled.csv",B/"resting-coupled.csv",P/"native-verification-001/report.json",C/"baseline/execution.json"]}
ex=json.loads((C/"baseline/execution.json").read_text());assert ex["returncode"]==0 and not ex["changed_inputs"] and ex["native_argv_matches_prepared_cli_preview"] and ex["native_environment_matches_prepared_cli_preview"]
parent=json.loads((P/"native-verification-001/report.json").read_text());assert parent["all_native_checks_pass"] and parent["inputs_unchanged"]
trace=list(csv.DictReader((N/"resting-coupled.csv").open()));base=list(csv.DictReader((B/"resting-coupled.csv").open()));assert trace==base and len(trace)==1000
def volume(p,f):
 x=p.astype(float)-p.astype(float).mean(0);tri=x[f];return float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
def exact(p,f):
 used=np.unique(f);rec,dg=h.exact_rows(p[used],np.searchsorted(used,f),ci);assert not dg;return rec
def normals(pack,ids):
 with pack.open("rb") as f:
  m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);hdr=ca._PACK_HEADER.unpack_from(m)
  sec=[ca._PACK_DIRECTORY.unpack_from(m,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size) for i in range(hdr[2])];v=next(x for x in sec if x[0]==2);a=np.ndarray((v[4],20),dtype="<f4",buffer=m,offset=v[2]);out=a[ids,4:7].copy();del a;m.close()
 return out
sourcevol={str(sid):volume(h.row_data(t,sid)["positions"],h.row_data(t,sid)["faces"]) for sid in (151,152,159,160)};assert all(v>0 for v in sourcevol.values())
out={"scope":"Native outward-winding correction of mirrored left masseter160 in integrated bilateralSCM/masseter scene. All positions/bodyposes/physiology and all other863 surfaces exact parent. Existing SCM interfaces remain unresolved. No completebody/five-minutequalification.","pins":pins,"source_signed_volume_m3":sourcevol,"coupled_trace_exact_parent":True,"trace_rows":len(trace),"trace_columns":len(trace[0]),"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
keys,*_=ca._load_target_inventory(h.INV);keys=set(keys)|{(51005,sid) for sid in (151,152,159,160)}
for step in (0,4767,5023,5599,6207,6815,7423,8000):
 pack=N/"accepted-geometry"/f"step-{step}.mrvpack";pp=B/"accepted-geometry"/pack.name
 rc=json.loads(pack.with_suffix(".receipt.json").read_text());prc=json.loads(pp.with_suffix(".receipt.json").read_text());assert rc["accepted_body_poses"]==prc["accepted_body_poses"] and rc["physical_endpoint"]=="accepted"
 for pk,receipt in ((pack,rc),(pp,prc)):
  pins[str(pk)]=sha(pk);pins[str(pk.with_suffix(".receipt.json"))]=sha(pk.with_suffix(".receipt.json"));assert receipt["pack_file_sha256"]==pins[str(pk)]
 pos,surfs,_=ca._pack_surfaces(pack,keys);bp,bs,_=ca._pack_surfaces(pp,keys);assert set(surfs)==set(bs) and len(surfs)==864
 for key,target in surfs.items():
  f=np.asarray(target["faces"],int);bf=np.asarray(bs[key]["faces"],int)
  assert np.array_equal(pos[f],bp[bf[:,[0,2,1]]] if key==(51005,160) else bp[bf]),(step,key)
 sf=np.asarray(surfs[(51005,160)]["faces"],int);bf=np.asarray(bs[(51005,160)]["faces"],int);ids=np.unique(sf);bi=np.unique(bf);assert len(ids)==len(bi)
 normerr=float(np.linalg.norm(normals(pack,ids).astype(float)+normals(pp,bi).astype(float),axis=1).max());assert normerr<1e-6
 volumes={}
 for sid in (151,152,159,160):
  f=np.asarray(surfs[(51005,sid)]["faces"],int);used=np.unique(f);volumes[str(sid)]=volume(pos[used],np.searchsorted(used,f))
 assert all(v>0 for v in volumes.values())
 f=np.searchsorted(ids,sf);world=pos[ids];rec=exact(world,f);selfc=ci._audit_pair(rec,rec,same_surface=True);top=analyze_topology(world.astype(float).tolist(),f.tolist());lo=world.min(0);hi=world.max(0);pairs=[];count=0
 for key,target in surfs.items():
  if key==(51005,160):continue
  count+=1;tf=np.asarray(target["faces"],int);tri=pos[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
  if not len(sel):continue
  hit=ci._audit_pair(rec,exact(pos,tf[sel]),same_surface=False)["triangle_pairs"]
  if hit:pairs.append({"key":list(key),"pairs":[[int(a),int(sel[b])] for a,b in hit]})
 row={"step":step,"all_other863_surface_triangles_exact_parent":True,"corrected160_unoriented_triangles_exact_parent":True,"bodyposes_exact_parent":True,"signed_volume_m3":volumes,"max_normal_negation_error":normerr,"self_pairs":selfc["triangle_pairs"],"closed_oriented":top["closed_oriented_manifold_candidate"],"external_targets_scanned":count,"external_pairs":pairs};out["poses"].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ("external_pairs","self_pairs")}),flush=True)
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()),all_native_checks_pass=all(not p["self_pairs"] and p["closed_oriented"] and p["external_targets_scanned"]==863 and not p["external_pairs"] for p in out["poses"]));save();assert out["inputs_unchanged"] and out["all_native_checks_pass"]
