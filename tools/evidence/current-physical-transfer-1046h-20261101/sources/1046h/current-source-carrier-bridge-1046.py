#!/usr/bin/env python3
"""Current-433 rebinding for frozen 090 source-tet carrier quadratures.
Source-only packet preparation; never allocates mass or claims admission."""
import argparse,csv,hashlib,json,struct,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
H=Path(__file__).resolve().parent
P=H.parent
BASE=P/"native-physical-parent-mass-330/native-source-identity-control-003"
TAB=BASE/"compiled-parent-cell-mass.csv"
CAND=H/"current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz"
CREC=CAND.parent/"receipt.json"
STATE=H/"current-native-1024-root/volume-prefix-818c/verified-state.json"
P1BASE=H/"current-role-p1-support-938-1027-pinned-230-dependency/actual-001"
OLD={
 "Skin":P/"fractional-semantic-support-090-sparse-graph/skin-743497-run-004-source-batch/source-quadrature-007/source-quadrature.npz",
 "Fat":P/"fractional-semantic-support-090-sparse-graph/fat-853406-run-003-root/source-quadrature-162-root/source-quadrature.npz"}
MAT={"Skin":15,"Fat":16}; R=.019353
FIELDS=(("source_parent_tet_rows","<u4"),("global_source330_tet_rows","<u4"),("parent_offsets","<u4"),("carrier_weights","<f8"),("carrier_barycentric","<f8"),("carrier_target_child_rows","<u4"),("carrier_target_distance_m","<f8"))
def sha(p): return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def csvread(p):
 with open(p,newline="") as f:return list(csv.DictReader(f))
def packet(a):
 return b"FSSCARR01"+struct.pack("<QQ",len(a["source_parent_tet_rows"]),len(a["carrier_weights"]))+b"".join(np.ascontiguousarray(a[k],dtype=np.dtype(dt)).tobytes() for k,dt in FIELDS)
def verify_packet(binary,npz,global_rows):
 data=binary.read_bytes()
 if data[:9]!=b"FSSCARR01":raise ValueError("packet magic")
 np_,nc=struct.unpack_from("<QQ",data,9)
 if np_!=len(global_rows):raise ValueError("packet parent count")
 counts=[np_,np_,np_+1,nc,nc*4,nc,nc];pos=25;loaded={}
 for (name,dt),n in zip(FIELDS,counts):
  arr=np.frombuffer(data,dtype=np.dtype(dt),count=n,offset=pos).copy();pos+=n*np.dtype(dt).itemsize
  if name=="carrier_barycentric":arr=arr.reshape(nc,4)
  loaded[name]=arr
 if pos!=len(data):raise ValueError("packet trailing bytes")
 with np.load(npz,allow_pickle=False) as z:
  if set(z.files)!=set(loaded):raise ValueError("packet NPZ field set")
  for k,v in loaded.items():
   if z[k].dtype!=v.dtype or not np.array_equal(z[k],v):raise ValueError("packet binary/NPZ mismatch "+k)
 if not np.array_equal(loaded["source_parent_tet_rows"],np.arange(np_,dtype=np.uint32)):raise ValueError("local source order")
 if not np.array_equal(loaded["global_source330_tet_rows"],global_rows):raise ValueError("source330 row map")
 off=loaded["parent_offsets"].astype(np.int64);w=loaded["carrier_weights"];b=loaded["carrier_barycentric"]
 if off[0]!=0 or off[-1]!=nc or np.any(np.diff(off)<0):raise ValueError("packet CSR")
 if not np.isfinite(w).all() or np.any(w<=0) or not np.isfinite(b).all() or np.any(b<0) or np.any(b>1):raise ValueError("packet quadrature positivity")
 if np.any(np.abs(b.sum(axis=1)-1)>1e-12):raise ValueError("barycentric closure")
 for p in np.flatnonzero(np.diff(off)):
  a,e=off[p:p+2]
  if abs(float(w[a:e].sum())-1)>1e-12 or np.max(np.abs(w[a:e]@b[a:e]-.25))>2e-10:raise ValueError("mass/first moment")
 if np.any(loaded["carrier_target_distance_m"]>R+1e-12):raise ValueError("distance cap")
 return loaded,hashlib.sha256(data).hexdigest()
def build(role,outdir):
 mat=MAT[role];p1path=P1BASE/f"p1-{role.lower()}"/f"{role.lower()}-current-p1-support.npz"
 labs=csvread(TAB.with_name(TAB.name+".labels.csv"));tetr=csvread(TAB.with_name(TAB.name+".tets.csv"));nodes=csvread(TAB.with_name(TAB.name+".nodes.csv"))
 allmat=np.fromiter((int(x["material_index"]) for x in labs),dtype=np.int64)
 grows=np.flatnonzero(allmat==mat).astype(np.uint32)
 if len(grows)!=14316:raise ValueError(f"{role}: source role row count")
 node_rows=np.asarray([[int(x[f"parent_node{k}"]) for k in range(4)] for x in tetr],np.int64)
 rest=np.asarray([[float(x[f"rest_{a}_m"]) for a in "xyz"] for x in nodes],np.float64)
 verts=rest[node_rows[grows]];cent=verts.mean(axis=1)
 regs=np.fromiter((int(labs[int(g)]["anatomical_region_id"]) for g in grows),dtype=np.int64)
 with np.load(OLD[role],allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
 if set(old)!={x[0] for x in FIELDS}:raise ValueError("old packet keys")
 if not np.array_equal(old["global_source330_tet_rows"],grows) or not np.array_equal(old["source_parent_tet_rows"],np.arange(len(grows),dtype=np.uint32)):raise ValueError("old source map")
 with np.load(CAND,allow_pickle=False) as z:
  pts=z["points_reference_world_m_f64"].copy();tets=z["volume_tetrahedra"].astype(np.int64,copy=True);mats=z["material_index"].astype(np.int64,copy=True);primary=z["current_typed_p1_anchor_primary_group_tuple"].astype(np.int64,copy=True)
 if not np.array_equal(pts,pts.astype(np.float32).astype(np.float64)):raise ValueError("candidate positions are not widened F32")
 cglobal=np.flatnonzero(mats==mat);ct=tets[cglobal];cc=pts[ct].mean(axis=1)
 with np.load(p1path,allow_pickle=False) as z:
  p={k:z[k].copy() for k in ("cell_group_offsets","cell_group_ids","cell_admissible_region_offsets","cell_admissible_region_ids","group_tuples","admissibility_max_source_region_hops","current_state_path","current_state_sha256","source330_mass_sha256")}
 if len(cglobal)!=len(p["cell_group_offsets"])-1 or int(p["admissibility_max_source_region_hops"])!=1:raise ValueError("P1 child domain or hop")
 if str(np.asarray(p["current_state_path"]).reshape(-1)[0])!=str(STATE) or str(np.asarray(p["current_state_sha256"]).reshape(-1)[0])!=sha(STATE):raise ValueError("P1 state pin")
 if str(np.asarray(p["source330_mass_sha256"]).reshape(-1)[0])!=sha(TAB):raise ValueError("P1 parent mass pin")
 tuples=[tuple(map(int,x)) for x in p["group_tuples"]];t2g={x:i for i,x in enumerate(tuples)}
 if len(t2g)!=len(tuples):raise ValueError("duplicate P1 tuple")
 go=p["cell_group_offsets"];gids=p["cell_group_ids"];matched=0
 for i,key in enumerate(primary[cglobal]):
  g=t2g.get(tuple(map(int,key)))
  if g is None or not np.any(gids[go[i]:go[i+1]]==g):raise ValueError(f"candidate/P1 row-order mismatch {i}")
  matched+=1
 missing=set(tuple(map(int,x)) for x in np.asarray([[int(labs[int(g)][k]) for k in ("material_index","geometric_donor_body","anatomical_region_id")] for g in grows]))-set(t2g)
 if missing:raise ValueError(f"source tuples absent from P1 domain: {len(missing)}")
 ro=p["cell_admissible_region_offsets"];rid=p["cell_admissible_region_ids"].astype(np.int64)
 if np.any(np.diff(ro)<=0):raise ValueError("empty one-hop region row")
 child=np.repeat(np.arange(len(cc),dtype=np.int32),np.diff(ro));order=np.argsort(rid,kind="stable");sr=rid[order];sc=child[order]
 ur,st,n=np.unique(sr,return_index=True,return_counts=True);rr={};trees={}
 for reg,start,count in zip(ur,st,n):
  rows=np.unique(sc[start:start+count]);rr[int(reg)]=rows;trees[int(reg)]=cKDTree(cc[rows])
 off=old["parent_offsets"].astype(np.int64);w0=old["carrier_weights"].astype(np.float64);b0=old["carrier_barycentric"].astype(np.float64)
 if off.shape!=(len(grows)+1,) or off[0]!=0 or off[-1]!=len(w0) or b0.shape!=(len(w0),4):raise ValueError("old packet CSR")
 no=[0];nw=[];nb=[];nt=[];nd=[];fallback=[];maxd=0.;nonempty=0
 for i,reg0 in enumerate(regs):
  reg=int(reg0);rows=rr.get(reg);tree=trees.get(reg);a,e=map(int,off[i:i+2])
  if rows is None or tree is None or not len(rows):
   if a!=e:raise ValueError(f"no admissible region child for carrier parent {int(grows[i])}")
   no.append(len(nw));continue
  if a==e:
   d,j=tree.query(cent[i])
   if float(d)>R+1e-12:raise ValueError(f"uncarrierized parent centroid out of cap {int(grows[i])}:{float(d)}")
   no.append(len(nw));continue
  qpts=b0[a:e]@verts[i];ds,js=tree.query(qpts,k=1);ds=np.asarray(ds,float).reshape(-1);js=np.asarray(js,np.int64).reshape(-1)
  if np.any(~np.isfinite(ds)):raise ValueError("nonfinite nearest carrier distance")
  if float(ds.max(initial=0))>R+1e-12:
   d,j=tree.query(cent[i])
   if float(d)>R+1e-12:raise ValueError(f"carrier+centroid both exceed cap parent={int(grows[i])} carrier_max={float(ds.max())} centroid={float(d)}")
   fallback.append({"local_parent_row":i,"global_source330_tet_row":int(grows[i]),"region":reg,"discarded_carrier_rows":list(range(a,e)),"fallback_child_row":int(rows[int(j)]),"fallback_distance_m":float(d),"reason":"unchanged centroid route satisfies unchanged native transport cap"})
   no.append(len(nw));continue
  targets=rows[js];actual=np.linalg.norm(qpts-cc[targets],axis=1)
  if np.max(np.abs(actual-ds),initial=0)>1e-12 or np.any(actual>R+1e-12):raise ValueError("carrier nearest witness replay")
  nw.extend(map(float,w0[a:e]));nb.extend(b0[a:e].tolist());nt.extend(map(int,targets));nd.extend(map(float,actual));maxd=max(maxd,float(actual.max(initial=0)));no.append(len(nw));nonempty+=1
 arr={"source_parent_tet_rows":np.arange(len(grows),dtype="<u4"),"global_source330_tet_rows":grows.astype("<u4"),"parent_offsets":np.asarray(no,dtype="<u4"),"carrier_weights":np.asarray(nw,dtype="<f8"),"carrier_barycentric":np.asarray(nb,dtype="<f8").reshape(-1,4),"carrier_target_child_rows":np.asarray(nt,dtype="<u4"),"carrier_target_distance_m":np.asarray(nd,dtype="<f8")}
 outdir.mkdir(parents=True,exist_ok=True);stem=f"{role.lower()}-current-source-quadrature-1046";npz=outdir/(stem+".npz");binp=outdir/(stem+".bin")
 np.savez_compressed(npz,**arr);binp.write_bytes(packet(arr))
 loaded,bsha=verify_packet(binp,npz,grows)
 # Independently replay every stored current witness against candidate/P1.
 rp=loaded["parent_offsets"].astype(np.int64);points=[]
 for i in range(len(grows)):
  a,e=map(int,rp[i:i+2])
  if a==e:continue
  rpnts=loaded["carrier_barycentric"][a:e]@verts[i]
  tar=loaded["carrier_target_child_rows"][a:e].astype(np.int64)
  if np.any(mats[cglobal[tar]]!=mat):raise ValueError("witness material mismatch")
  if np.any([reg not in rid[ro[t]:ro[t+1]] for t,reg in zip(tar,np.repeat(regs[i],len(tar)))]):raise ValueError("witness region mismatch")
  dst=np.linalg.norm(rpnts-cc[tar],axis=1)
  if np.max(np.abs(dst-loaded["carrier_target_distance_m"][a:e]),initial=0)>1e-12 or np.any(dst>R+1e-12):raise ValueError("independent witness distance mismatch")
  points.extend(rpnts.tolist())
  if abs(float(loaded["carrier_weights"][a:e].sum())-1)>1e-12 or np.max(np.abs(loaded["carrier_weights"][a:e]@loaded["carrier_barycentric"][a:e]-.25))>2e-10:raise ValueError("independent quadrature moment")
 inp={"old_packet_sha256":sha(OLD[role]),"candidate_npz_sha256":sha(CAND),"candidate_receipt_sha256":sha(CREC),"current_state_sha256":sha(STATE),"p1_support_sha256":sha(p1path),"source330_mass_sha256":sha(TAB),"source330_labels_sha256":sha(TAB.with_name(TAB.name+".labels.csv")),"source330_nodes_sha256":sha(TAB.with_name(TAB.name+".nodes.csv")),"source330_tets_sha256":sha(TAB.with_name(TAB.name+".tets.csv")),"source330_source_contributions_sha256":sha(TAB.with_name(TAB.name+".source.csv"))}
 receipt={"schema":"numi.h706.current-source-carrier-rebind.1046.v1","status":"PASS_SOURCE_ONLY_CURRENT_433_CARRIER_REBIND","role":role,"material_index":mat,"native_steps":0,"physical_admission":False,"maximum_transport_distance_m":R,"source_parent_count":len(grows),"source_tuple_count":len(set(tuple(map(int,x)) for x in np.asarray([[int(labs[int(g)][k]) for k in ("material_index","geometric_donor_body","anatomical_region_id")] for g in grows]))),"carrierized_parent_count":nonempty,"carrier_count":len(nw),"centroid_fallback_parent_count":len(fallback),"centroid_fallbacks":fallback,"candidate_p1_row_alignment_count":matched,"witness_rule":"same role material and parent source region in authenticated one-hop P1 admissible-region CSR; nearest current F32-widened child centroid","quadrature_rule":"prior positive barycentric weights retained exactly, except fallback rows returned to unchanged centroid route","max_rebound_distance_m":maxd,"inputs_sha256":inp,"outputs":{"npz":{"path":str(npz),"sha256":sha(npz),"size_bytes":npz.stat().st_size},"binary":{"path":str(binp),"sha256":bsha,"size_bytes":binp.stat().st_size}},"limitations":["source-only packet preparation; no mass allocation or physical admission","exact tuple support gaps remain separately audited"]}
 recpath=outdir/(stem+"-receipt.json");recpath.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
 verification={"schema":"numi.h706.current-source-carrier-rebind.1046.readback.v1","status":"PASS_SOURCE_ONLY_PACKET_READBACK","role":role,"receipt_sha256":sha(recpath),"npz_sha256":sha(npz),"binary_sha256":bsha,"array_and_source_map_readback":"PASS","current_candidate_p1_row_alignment":"PASS","all_stored_current_witnesses_recomputed":"PASS","positive_mass_and_first_moment":"PASS","carrier_count":len(nw),"carrierized_parent_count":nonempty,"centroid_fallback_parent_count":len(fallback),"max_distance_m":maxd,"native_steps":0,"physical_admission":False}
 vpath=outdir/(stem+"-independent-verification.json");vpath.write_text(json.dumps(verification,sort_keys=True,indent=2)+"\n")
 return verification
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--self-test",action="store_true");ap.add_argument("--output",type=Path,default=H/"current-source-carrier-bridge-1046");a=ap.parse_args()
 if a.self_test:
  x={"source_parent_tet_rows":np.array([0,1],"<u4"),"global_source330_tet_rows":np.array([7,9],"<u4"),"parent_offsets":np.array([0,0,1],"<u4"),"carrier_weights":np.array([1.],"<f8"),"carrier_barycentric":np.array([[.25]*4],"<f8"),"carrier_target_child_rows":np.array([3],"<u4"),"carrier_target_distance_m":np.array([.01],"<f8")}
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);z=d/"f.npz";b=d/"f.bin";np.savez_compressed(z,**x);b.write_bytes(packet(x));y,_=verify_packet(b,z,np.array([7,9],"<u4"));assert all(np.array_equal(x[k],y[k]) for k in x)
   q=dict(x);q["carrier_weights"]=np.array([0.],"<f8");np.savez_compressed(z,**q);b.write_bytes(packet(q))
   try:verify_packet(b,z,np.array([7,9],"<u4"))
   except ValueError:pass
   else:raise AssertionError("nonpositive weight was accepted")
  print(json.dumps({"status":"PASS","packet_roundtrip":True,"nonpositive_weight_rejected":True}));return
 a.output.mkdir(parents=True,exist_ok=True);v={r:build(r,a.output) for r in ("Skin","Fat")}
 report={"schema":"numi.h706.current-source-carrier-rebind.1046.batch.v1","status":"PASS_SOURCE_ONLY_PACKET_READBACK","roles":v,"native_steps":0,"physical_admission":False}
 p=a.output/"batch-independent-verification.json";p.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"status":report["status"],"report":str(p),"roles":{r:{"carriers":x["carrier_count"],"fallbacks":x["centroid_fallback_parent_count"],"max_distance_m":x["max_distance_m"]} for r,x in v.items()}}))
if __name__=="__main__":main()

