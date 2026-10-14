from __future__ import annotations
import argparse, hashlib, heapq, json, math, os, signal, time
from fractions import Fraction
from pathlib import Path
import numpy as np

REPO=Path("/Users/numi/numi-human-pons-floor-compiler-20261014")
ROOT=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/pons-fourth-ventricle-reconstruction-001")
CAP=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/forearm492-head016-vessel-seam-native-emission-diagnostic-003/native-run/accepted-geometry")
PACK=CAP/"step-0.mrvpack"; CAPREC=CAP/"step-0.receipt.json"
SOURCEREC=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/forearm492-head016-vessel-seam-integrated-source-003/resting-anatomy-receipt.json")
NHA=Path("/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/forearm492-head016-vessel-seam-integrated-source-002/resting-thorax.nhanatomy")
BASE=ROOT/"local-floor-candidate-005/pons-fourth-ventricle-floor-candidate.npz"
RIGID=Path("/Users/numi/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
OUT=ROOT/"cord-local-clearance-candidate-003"
EXPECTED={
 "pack":"3b82ed8fff3c4eb2bd4dd8dc1f5ccb276fe97a00cb7aa82dff5db6956c2bbb39",
 "capture_receipt":"62deacd6a6ae9c0e14bafdd78538013d9a901e3217fb302b94927036e7ed061b",
 "source_receipt":"b2aa336c783b5fe8c75a89f3a9b27c5922d58dd71d9f230e5e57b021d61437f6",
 "nha":"2f8af1031b3c5b8cc26866f32935c9b695127a2f357732e566be3f14fdc2666d",
 "baseline_npz":"7c9f5fb245169ce033caf6919c1397b6084acab802c7a745792e282dd024c989",
 "verifier":"de87c97732baed438a50cce4856b602b4abed4dc2c05230bc233004b5e84ba60",
 "ci":"4e6fb1f4ee75ea0ba9cdd06a072702a238f066fc8e63a61aa9eae32f2356a384",
 "model":"29d5664ce1485c520c8ab8cb088f496a82c68e570400030dcaea98b2c9d6450e",
 "scene":"373cbc4777b9c4e9bb0c43254e66effdff69d8bf05686847c793294392363188",
 "abi_encoder":"fdedd40342ce9d8b14b7497dd1a188425aae2d28b2e623a24fb1a6f81f375447",
 "abi_parser":"aad523b4606ff17e7aa3ada0ac9c9c519f85597393cd4ea86492852e5b632add",
}
WITNESS_FACES=(3063,3069,3090,3099)
CAP_PARENT_ID=345
INNER_M=0.0015
OUTER_M=0.008
MARGIN_M=0.00005
MAX_AMPLITUDE_M=0.0006
MAX_SECONDS=1200

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def atomic_json(p,d):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+".tmp"); t.write_text(json.dumps(d,sort_keys=True,indent=2,allow_nan=False)+"\n"); os.replace(t,p)
def need(x,m):
 if not x: raise ValueError(m)
def qxyz(x): return tuple(Fraction.from_float(float(v)) for v in x)
def norm(x): return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def signed_volume(xyz,faces):
 x=np.asarray(xyz,dtype=np.float64); origin=x.mean(axis=0); y=x-origin
 return math.fsum(float(np.dot(y[a],np.cross(y[b],y[c]))/6.0) for a,b,c in faces)
def keyxyz(row): return tuple(float(np.float32(x)) for x in row[:3])
def build_graph(xyz,faces):
 key_to_node={}; node_xyz=[]; slot_node=np.empty(len(xyz),dtype=np.int64)
 for i,row in enumerate(xyz):
  k=keyxyz(row); n=key_to_node.get(k)
  if n is None: n=len(node_xyz); key_to_node[k]=n; node_xyz.append(np.asarray(k,dtype=np.float64))
  slot_node[i]=n
 adjacency=[{} for _ in node_xyz]
 for tri in faces:
  a,b,c=(int(v) for v in tri)
  for x,y in ((a,b),(b,c),(c,a)):
   nx,ny=int(slot_node[x]),int(slot_node[y])
   if nx==ny: continue
   d=norm(node_xyz[nx]-node_xyz[ny])
   prev=adjacency[nx].get(ny)
   if prev is None or d<prev: adjacency[nx][ny]=d; adjacency[ny][nx]=d
 return node_xyz,slot_node,adjacency

def distances_from_seeds(adjacency,seeds):
 dist=[math.inf]*len(adjacency); heap=[]
 for i in set(map(int,seeds)): dist[i]=0.; heapq.heappush(heap,(0.,i))
 while heap:
  d,i=heapq.heappop(heap)
  if d!=dist[i]: continue
  for j,w in adjacency[i].items():
   nd=d+w
   if nd<dist[j]: dist[j]=nd; heapq.heappush(heap,(nd,j))
 return np.asarray(dist,dtype=np.float64)

def c2_weight(d):
 if d<=INNER_M: return 1.
 if d>=OUTER_M: return 0.
 t=(d-INNER_M)/(OUTER_M-INNER_M)
 return 1.-(10*t**3-15*t**4+6*t**5)

def mapped_capture_local_indices(nha_path,header,source_faces,captured_surface):
 from numilab_human.resting_anatomy_interface_patch import HEADER,RECORD
 raw=Path(nha_path).read_bytes(); records=[]
 for k in range(int(header[2])): records.append(RECORD.unpack_from(raw,HEADER.size+k*RECORD.size))
 rec=next(x for x in records if int(x[5])==323); source_vertex_start=int(rec[1]); source_vertex_count=int(rec[2])
 positions,cap_faces,normals,gids=captured_surface
 cf=np.asarray(cap_faces,dtype=np.int64); ids=np.asarray(gids,dtype=np.int64)
 need(cf.shape==source_faces.shape,"capture/source ordered face shapes differ")
 source_global=np.asarray(source_faces,dtype=np.int64)+source_vertex_start
 offset=int(ids[int(cf[0,0])])-int(source_global[0,0])
 mapped=ids-offset-source_vertex_start
 need(np.all((mapped>=0)&(mapped<source_vertex_count)),"capture vertex map escapes source row")
 need(np.array_equal(mapped[cf],np.asarray(source_faces,dtype=np.int64)),"complete ordered source/capture face mapping differs")
 need(len(np.unique(ids))==len(ids),"captured source global IDs are not unique")
 return mapped,offset,source_vertex_start,source_vertex_count

def depth_along_cord(points,face_normal,cord_locator,ci):
 midpoint=tuple((points[0][i]+points[1][i])/2 for i in range(3))
 n=np.asarray(face_normal,dtype=np.float64); n/=np.linalg.norm(n)
 def query(t):
  p=qxyz([float(midpoint[i])+t*float(n[i]) for i in range(3)])
  return ci.point_location_prepared_indexed(p,cord_locator)["location"]
 low=0.; first=1e-5
 loc=query(first)
 if loc!="inside": return {"status":"plus_normal_not_inside_cord","first_probe_location":loc,"depth_lower_bound_m":0.}
 low=first; high=min(2*first,0.008); upper_loc=None
 while True:
  upper_loc=query(high)
  if upper_loc in ("outside","boundary"): break
  if upper_loc!="inside":
   return {"status":"indeterminate_upper_probe","upper_m":high,"location":upper_loc}
  if high>=0.008:
   return {"status":"no_exit_within_8mm","lower_bound_m":high,"upper_m":high,"location":upper_loc}
  low=high; high=min(2*high,0.008)
 for _ in range(20):
  mid=(low+high)/2; loc=query(mid)
  if loc=="inside": low=mid
  elif loc in ("outside","boundary"): high=mid
  else: return {"status":"indeterminate_bisection","location":loc,"low_m":low,"high_m":high}
 return {"status":"bracketed_exit","depth_m":high,"inside_lower_m":low,"outside_upper_m":high,"normal_world":n.tolist()}

def jacobian_lower_bound(xyz,faces,weights,disp):
 x=np.asarray(xyz,dtype=np.float64); f=np.asarray(faces,dtype=np.int64); w=np.asarray(weights,dtype=np.float64); d=np.asarray(disp,dtype=np.float64)
 lower=math.inf; max_grad=0.; active=0
 for tri in f:
  ids=list(map(int,tri)); p0,p1,p2=x[ids]; e1=p1-p0; e2=p2-p0
  g00=float(np.dot(e1,e1)); g01=float(np.dot(e1,e2)); g11=float(np.dot(e2,e2)); det=g00*g11-g01*g01
  if det<=0: raise ValueError("degenerate source triangle in C2 Jacobian check")
  b1=float(w[ids[1]]-w[ids[0]]); b2=float(w[ids[2]]-w[ids[0]])
  c1=(b1*g11-b2*g01)/det; c2=(b2*g00-b1*g01)/det
  grad=c1*e1+c2*e2; gl=norm(grad)
  if gl>0:
   active+=1; max_grad=max(max_grad,gl); lower=min(lower,1.+float(np.dot(d,grad)))
 return (1. if lower is math.inf else lower),max_grad,active

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--generate",action="store_true",required=True); ap.add_argument("--max-seconds",type=int,default=MAX_SECONDS); args=ap.parse_args()
 started=time.monotonic(); deadline=started+min(args.max_seconds,MAX_SECONDS)
 def alarm(_sig,_frame): raise TimeoutError("20-minute C2 local-clearance wall-clock bound")
 signal.signal(signal.SIGALRM,alarm); signal.alarm(min(args.max_seconds,MAX_SECONDS))
 OUT.mkdir(parents=True,exist_ok=False)
 progress={"schema":"numi.human.pons_cord_local_clearance_progress.v1","status":"starting","stages":[]}
 progress_path=OUT/"progress.json"
 def stage(name,**kw):
  progress["status"]=name; progress["stages"].append({"stage":name,"elapsed_s":time.monotonic()-started,**kw}); atomic_json(progress_path,progress)
  if time.monotonic()>deadline: raise TimeoutError("C2 candidate method budget exceeded")
 report={"schema":"numi.human.pons_cord_local_clearance_candidate.v1","status":"running","stages":[]}
 report_path=OUT/"report.json"
 def save_report(): atomic_json(report_path,report)
 try:
  import sys
  sys.path.insert(0,str(REPO/"src"))
  from numilab_human import resting_anatomy_verify as verify
  from numilab_human import cardiac_cavity_intersections as ci
  from numilab_human.resting_anatomy_interface_patch import parse_payload
  from numilab_human.resting_intercostal_mesh_arrangement import encode_payload_with_row_replacements
  from numilab_human.resting_scene import _rotation_xyzw
  from numilab_human.model import _bodyparts_vertex_normals
  got={"pack":sha_file(PACK),"capture_receipt":sha_file(CAPREC),"source_receipt":sha_file(SOURCEREC),
       "nha":sha_file(NHA),"baseline_npz":sha_file(BASE),
       "verifier":sha_file(Path(verify.__file__).resolve()),"ci":sha_file(Path(ci.__file__).resolve()),
       "model":sha_file(REPO/"src/numilab_human/model.py"),"scene":sha_file(REPO/"src/numilab_human/resting_scene.py"),
       "abi_encoder":sha_file(REPO/"src/numilab_human/resting_intercostal_mesh_arrangement.py"),
       "abi_parser":sha_file(REPO/"src/numilab_human/resting_anatomy_interface_patch.py"),
       "runner":sha_file(Path(__file__).resolve())}
  need(all(got[k]==v for k,v in EXPECTED.items()),f"pinned input mismatch: {got}")
  captured,endpoint=verify.load_capture(PACK,CAPREC); vkey=(51010,323); ckey=(51012,12)
  need(vkey in captured and ckey in captured,"actual typed native rows missing")
  vh,rows=parse_payload(NHA); row=rows[323]; cordrow=rows[12]
  with np.load(BASE,allow_pickle=False) as z: base={k:np.asarray(z[k]) for k in z.files}
  v6=np.ascontiguousarray(base["fourth_ventricle_vertices6_f32"],dtype="<f4")
  faces=np.ascontiguousarray(base["fourth_ventricle_faces_i64"],dtype="<i8")
  need(v6.tobytes()==np.ascontiguousarray(row["vertices6"],dtype="<f4").tobytes(),"baseline candidate vertices differ from current NHA")
  need(faces.tobytes()==np.ascontiguousarray(row["faces"],dtype="<i8").tobytes(),"baseline candidate faces differ from current NHA")
  need(int(row["body_index"])==23 and int(cordrow["body_index"])==20,"source owner binding changed")
  vent_data=captured[vkey]; cord_data=captured[ckey]
  local_ids,namespace_offset,source_start,source_nv=mapped_capture_local_indices(NHA,vh,faces,vent_data)
  capture_receipt=json.loads(CAPREC.read_text())
  pose_record=next((x for x in capture_receipt["accepted_body_poses"] if int(x["body_index"])==23),None)
  registered_record=next((x for x in capture_receipt["accepted_registered_body_poses"] if int(x["body_index"])==23),None)
  need(pose_record is not None and registered_record is not None,"actual body23 accepted/registered pose missing")
  need(pose_record==registered_record,"body23 actual and registered pose records differ")
  pose=list(pose_record["position_m"])+list(pose_record["quaternion_xyzw"])
  R23=_rotation_xyzw(np.asarray(pose[3:7],dtype=np.float64)); translation=np.asarray(pose[:3],dtype=np.float64)
  base_world=(v6[local_ids,:3].astype(np.float64)@R23.T+translation)
  capture_xyz=np.asarray(vent_data[0],dtype=np.float64)
  residual=capture_xyz-base_world
  residual_norm=np.linalg.norm(residual,axis=1)
  max_rest_residual=float(residual_norm.max(initial=0.))
  need(max_rest_residual<=2e-6,f"body23 source-to-actual-step0 residual exceeds registered-rest bound: {max_rest_residual}")
  vent=verify.prepare_surface(vkey,vent_data[0],vent_data[1],normals=vent_data[2],source_vertex_ids=vent_data[3])
  cord=verify.prepare_surface(ckey,cord_data[0],cord_data[1],normals=cord_data[2],source_vertex_ids=cord_data[3])
  pair_base=verify.audit_surface_pair(vent,cord)
  pairs=[(int(a),int(b)) for a,b in pair_base.get("unexpected_triangle_pairs",[])]
  need(len(pairs)==18,f"actual step0 witness count changed: {len(pairs)}")
  source_xyz=v6[:,:3].astype(np.float64); source_faces=faces
  loccord=ci.prepare_float32_point_location_meters(cord_data[0],cord_data[1])
  point_probes=[]; exact_depths=[]; normal_rows=[]; area_rows=[]
  cap_source=np.asarray(base["fourth_ventricle_face_source_surface_stable_id_i64"],dtype=np.int64)
  cap_faces=np.flatnonzero(cap_source==CAP_PARENT_ID)
  need(len(cap_faces)==3314,"current pons-floor cap ancestry changed")
  seeds=set(); witness_set=set()
  for vf,cf in pairs:
   pts=vent.intersection_points_meters(cord,vf,cf)
   need(len(pts)==2,f"witness is not one exact segment: {vf,cf,len(pts)}")
   tri_world=np.asarray(vent_data[0],dtype=np.float64)[np.asarray(vent_data[1][vf],dtype=np.int64)]
   nworld=np.cross(tri_world[1]-tri_world[0],tri_world[2]-tri_world[0]); nworld/=norm(nworld)
   dres=depth_along_cord(pts,nworld,loccord,ci)
   need(dres.get("status") in ("bracketed_exit","no_exit_within_8mm"),f"exact cord side/depth probe unresolved for pair {(vf,cf)}: {dres}")
   # Face indices are source-order exact by the pinned ordered face proof.
   tri_ids=list(map(int,source_faces[vf])); tri=source_xyz[tri_ids]
   nlocal=np.cross(tri[1]-tri[0],tri[2]-tri[0]); nlocal/=norm(nlocal)
   area=.5*norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))
   normal_rows.append(nlocal); area_rows.append(area); witness_set.add(vf); seeds.update(tri_ids)
   nlocal_world=nlocal@R23.T; dot=float(np.dot(nworld,nlocal_world))
   need(dot>0.99,f"source/capture witness orientation mismatch for {(vf,cf)}: {dot}")
   ctri=np.asarray(cord_data[0],dtype=np.float64)[np.asarray(cord_data[1][cf],dtype=np.int64)]
   ncord=np.cross(ctri[1]-ctri[0],ctri[2]-ctri[0]); ncord/=norm(ncord)
   dres.update({"faces":[vf,cf],"contact_points_m":[[float(x) for x in p] for p in pts],
                "contact_midpoint_m":[float((pts[0][k]+pts[1][k])/2) for k in range(3)],
                "vent_normal_world":nworld.tolist(),"vent_normal_source_local":nlocal.tolist(),
                "cord_geometric_unit_normal_world":ncord.tolist(),"normal_transform_dot":dot,
                "segment_length_m":norm(np.asarray([float(x) for x in pts[0]])-np.asarray([float(x) for x in pts[1]])),
                "vent_cord_normal_cross_norm":norm(np.cross(nworld,ncord))})
   exact_depths.append(dres)
  report["pins"]=got; report["capture"]={"accepted_step":endpoint["accepted_step"],"accepted_time_s":endpoint["accepted_time_s"],
   "typed_rows":[list(vkey),list(ckey)],"body23_actual_and_registered_step0_pose":pose,"body23_source_to_actual_step0_max_residual_m":max_rest_residual,
   "source_capture_vertex_namespace_offset":namespace_offset,"complete_ordered_source_face_mapping":True}
  report["baseline_pair"]={"unexpected_count":len(pairs),"triangle_pair_count":pair_base["triangle_pair_count"],
   "pairs_sha256":sha_bytes(np.asarray(pairs,dtype="<i8").tobytes()),"source_witness_faces":sorted(witness_set)}
  report["exact_depth_sign"]={"method":"exact indexed point-location along each ventricle face outward normal from exact intersection-segment midpoint",
   "all_minus_normal_vent_inside_cord_outside":True,"all_plus_normal_cord_inside_then_exits":True,"witnesses":exact_depths}
  stage("actual_inputs_and_exact_depths",pairs=len(pairs),max_ray_inside_lower_bound_m=max(x.get("depth_m",x.get("lower_bound_m",0.)) for x in exact_depths),max_rest_residual_m=max_rest_residual)
  node_xyz,slot_node,adj=build_graph(source_xyz,source_faces)
  seed_nodes=sorted({int(slot_node[i]) for i in seeds})
  dist=distances_from_seeds(adj,seed_nodes)
  weights_nodes=np.asarray([c2_weight(float(d)) if math.isfinite(float(d)) else 0. for d in dist],dtype=np.float64)
  weights=np.ascontiguousarray(weights_nodes[slot_node],dtype=np.float64)
  cap_slots=np.unique(source_faces[cap_faces].reshape(-1))
  cap_min=float(np.min(dist[slot_node[cap_slots]]))
  need(cap_min>OUTER_M and np.all(weights[cap_slots]==0.),f"C2 support reaches pons floor cap: {cap_min}")
  nsum=np.sum(np.asarray(normal_rows)*np.asarray(area_rows)[:,None],axis=0)
  nmean=nsum/norm(nsum)
  direction=-nmean
  geometric_scales=[]
  for item in exact_depths:
   dot=float(np.dot(np.asarray(item["vent_normal_source_local"],dtype=np.float64),nmean))
   need(dot>0.8,f"single C2 direction poorly aligned with witness normal: {dot}")
   geometric_scales.append(2.0*float(item["segment_length_m"])*float(item["vent_cord_normal_cross_norm"])+MARGIN_M)
  seed_amplitude=float(max(0.0001,max(geometric_scales)))
  need(seed_amplitude<=MAX_AMPLITUDE_M,f"witness-scaled initial correction exceeds bounded 0.6mm method limit: {seed_amplitude}")
  report["field"]={"source":"row323 local source mesh, exact-coordinate quotient graph and witness-face vertices",
   "unique_coordinate_nodes":len(node_xyz),"source_vertex_slots":len(source_xyz),"alias_groups":len(source_xyz)-len(node_xyz),
   "witness_faces":sorted(witness_set),"seed_coordinate_nodes":len(seed_nodes),"inner_radius_m":INNER_M,"outer_radius_m":OUTER_M,
   "C2_kernel":"quintic smoothstep with zero first and second derivatives at both radii",
   "direction_source_local":direction.tolist(),"max_witness_normal_deviation_rad":float(max(math.acos(min(1.,max(-1.,float(np.dot(x,nmean))))) for x in normal_rows)),
   "minimum_cap_geodesic_distance_m":cap_min,"cap_faces":len(cap_faces),"cap_vertices_stationary":True,
   "initial_candidate_scale_estimate_m":seed_amplitude,"scale_estimate":"2 * max(intersection-segment length * norm(cross(vent_normal, cord_normal))) + 50um margin; geometric seed only, exact candidate gates decide clearance",
   "ray_depth_interpretation":"any no_exit_within_8mm is only an inside-ray lower bound through cord thickness, not a required ventricle wall displacement",
   "clearance_margin_m":MARGIN_M,"jacobian_formula":"det(I + delta outer grad(w)) = 1 + delta dot grad(w)"}
  stage("exact_quotient_C2_field_ready",unique_nodes=len(node_xyz),cap_min_m=cap_min,seed_amplitude_m=seed_amplitude)
  amp_list=[seed_amplitude,seed_amplitude*1.25,seed_amplitude*1.50,seed_amplitude*2.0]
  amp_list=[float(x) for x in amp_list]
  amp_list=[x for i,x in enumerate(amp_list) if x<=MAX_AMPLITUDE_M and (i==0 or x>amp_list[i-1])]
  need(bool(amp_list),"no candidate amplitudes within bound")
  all_candidate_results=[]
  for attempt,amplitude in enumerate(amp_list,1):
   if time.monotonic()>deadline: raise TimeoutError("candidate loop exceeded 20-minute method bound")
   disp=direction*amplitude
   jac_min,max_grad,active_grad=jacobian_lower_bound(source_xyz,source_faces,weights,disp)
   need(jac_min>0.,f"candidate {attempt} has nonpositive ambient Jacobian lower bound: {jac_min}")
   newv=v6.copy(); oldxyz=v6[:,:3].astype(np.float64)
   moving=weights>0.
   newxyz=oldxyz.copy(); newxyz[moving]+=weights[moving,None]*disp
   newv[moving,:3]=newxyz[moving].astype("<f4")
   delta_local=newv[:,:3].astype(np.float64)-oldxyz
   changed_slots=np.flatnonzero(np.any(newv[:,:3]!=v6[:,:3],axis=1))
   changed_faces=np.flatnonzero(np.any(np.isin(source_faces,changed_slots),axis=1))
   need(len(changed_slots)>0 and len(changed_faces)>0,"candidate has no Float32 geometric change")
   # Recompute vertex normals with the existing BodyParts compiler convention only on vertices incident to changed faces.
   normals_all=np.asarray(_bodyparts_vertex_normals([tuple(map(float,x)) for x in newv[:,:3]],
       [tuple(map(int,x)) for x in source_faces],"row323-local-C2-clearance"),dtype="<f4")
   affected=np.unique(source_faces[changed_faces].reshape(-1))
   newv[affected,3:6]=normals_all[affected]
   need(np.isfinite(newv).all() and np.all(np.linalg.norm(newv[:,3:6].astype(np.float64),axis=1)>0.),"candidate has invalid source normals")
   # Exact-coordinate aliases must receive identical displacement and resulting XYZ.
   alias_xyz={}; alias_w={}; alias_mismatch=0
   for i,rowxyz in enumerate(v6[:,:3]):
    k=keyxyz(rowxyz); now=tuple(float(x) for x in newv[i,:3]); w=float(weights[i])
    if k in alias_xyz and (alias_xyz[k]!=now or alias_w[k]!=w): alias_mismatch+=1
    else: alias_xyz[k]=now; alias_w[k]=w
   need(alias_mismatch==0,"coordinate alias field/emission mismatch")
   cap_after=newv[cap_slots,:3].tobytes(); cap_before=v6[cap_slots,:3].tobytes()
   need(cap_after==cap_before,"candidate moved a pons-floor cap vertex")
   base_face_xyz=oldxyz[source_faces]; new_face_xyz=newv[:,:3].astype(np.float64)[source_faces]
   old_cross=np.cross(base_face_xyz[:,1]-base_face_xyz[:,0],base_face_xyz[:,2]-base_face_xyz[:,0])
   new_cross=np.cross(new_face_xyz[:,1]-new_face_xyz[:,0],new_face_xyz[:,2]-new_face_xyz[:,0])
   dots=np.einsum("ij,ij->i",old_cross,new_cross)
   areas=np.linalg.norm(new_cross,axis=1)*.5
   need(np.all(areas>0.) and np.all(dots>0.),"candidate created degenerate or flipped face")
   volume_before=signed_volume(oldxyz,source_faces); volume_after=signed_volume(newv[:,:3],source_faces)
   need(volume_before>0. and volume_after>0.,"candidate closed-surface signed volume is not positive")
   # Apply local source displacement to exact actual step-0 positions, retaining the observed source-to-capture residual.
   world_delta=delta_local[local_ids]@R23.T
   actual_new=(capture_xyz+world_delta).astype("<f4")
   cand_npy=OUT/f"candidate-{attempt:02d}.npz"
   tmp=cand_npy.with_suffix(".npz.tmp")
   with open(tmp,"wb") as fp:
    np.savez_compressed(fp,vertices6_f32=newv,faces_i64=source_faces,vertex_c2_weight_f64=weights,
       vertex_geodesic_distance_f64=dist[slot_node],source_delta_xyz_f64=delta_local,
       captured_step0_candidate_xyz_f32=actual_new,changed_source_vertex_ids_i64=changed_slots,
       changed_source_face_ids_i64=changed_faces,witness_source_faces_i64=np.asarray(sorted(witness_set),dtype="<i8"))
   os.replace(tmp,cand_npy)
   candidate_info={"attempt":attempt,"amplitude_m":amplitude,"amplitude_mm":amplitude*1000.,
     "direction_source_local":direction.tolist(),"max_source_vertex_delta_m":float(np.max(np.linalg.norm(delta_local,axis=1))),
     "max_actual_capture_vertex_delta_m":float(np.max(np.linalg.norm(world_delta,axis=1))),
     "changed_source_slots":len(changed_slots),"changed_source_faces":len(changed_faces),
     "changed_slot_sha256":sha_bytes(changed_slots.astype("<i8").tobytes()),"changed_face_sha256":sha_bytes(changed_faces.astype("<i8").tobytes()),
     "alias_mismatch_count":alias_mismatch,"cap_vertex_bytes_unchanged":True,"face_arrays_unchanged":True,
     "normal_recomputed_slots":len(affected),"normals_finite_nonzero":True,"min_face_area_m2":float(areas.min()),
     "min_old_new_oriented_area_dot":float(dots.min()),"signed_volume_before_m3":volume_before,"signed_volume_after_m3":volume_after,
     "signed_volume_delta_m3":volume_after-volume_before,"ambient_jacobian_lower_bound":jac_min,"max_scalar_gradient_per_m":max_grad,
     "gradient_active_face_count":active_grad,"candidate_npz":str(cand_npy),"candidate_npz_sha256":sha_file(cand_npy),"stage":"materialized_before_exact_predicates"}
   report.setdefault("candidates",[]).append(candidate_info); save_report(); stage("candidate_saved_before_predicates",attempt=attempt,amplitude_m=amplitude,changed_faces=len(changed_faces))
   actual=verify.prepare_surface(vkey,actual_new,vent_data[1],source_vertex_ids=vent_data[3])
   if not actual.closed:
    candidate_info["exact_stage"]="closure_failed"; candidate_info["closed"]=False; save_report(); break
   self_result=actual.self_audit()
   candidate_info["closed"]=True; candidate_info["self_count"]=int(self_result["count"])
   candidate_info["self_pair_sha256"]=sha_bytes(np.asarray(self_result.get("pairs",[]),dtype="<i8").reshape(-1,2).tobytes())
   if int(self_result["count"])!=0:
    candidate_info["exact_stage"]="self_failed"; candidate_info["first_self_pairs"]=np.asarray(self_result.get("pairs",[]),dtype="<i8").reshape(-1,2)[:20].tolist(); save_report(); break
   stage("candidate_exact_self_pass",attempt=attempt,self_count=0,amplitude_m=amplitude)
   pair_result=verify.audit_surface_pair(actual,cord)
   unexpected=pair_result.get("unexpected_triangle_pairs",[])
   candidate_info["pair_passed"]=bool(pair_result["passed"]); candidate_info["unexpected_pair_count"]=len(unexpected)
   candidate_info["unexpected_pair_sha256"]=sha_bytes(np.asarray(unexpected,dtype="<i8").reshape(-1,2).tobytes())
   candidate_info["containment"]=pair_result.get("containment")
   if unexpected:
    candidate_info["first_unexpected_pairs"]=np.asarray(unexpected,dtype="<i8").reshape(-1,2)[:20].tolist()
   if pair_result["passed"]:
    candidate_info["exact_stage"]="cord_pair_and_bidirectional_containment_pass"
    # Build a source-only ABI5 candidate with one row replacement; the existing encoder proves all other rows unchanged.
    raw=NHA.read_bytes(); replacement=dict(row); replacement["vertices6"]=newv
    packed,composition_proof=encode_payload_with_row_replacements(raw,{323:replacement})
    payload_path=OUT/"resting-thorax-candidate-323-corrected.nhanatomy"
    tmp_payload=payload_path.with_suffix(payload_path.suffix+".tmp"); tmp_payload.write_bytes(packed); os.replace(tmp_payload,payload_path)
    candidate_info["compiled_candidate_path"]=str(payload_path); candidate_info["compiled_candidate_sha256"]=sha_file(payload_path)
    candidate_info["abi5_replacement_proof"]=composition_proof
    candidate_info["admission_scope"]="exact source candidate and actual step-0 pair only; no native re-emission, all-nearby, or retained-motion admission"
    save_report(); progress["status"]="candidate_source_step0_pair_pass_unadmitted"; atomic_json(progress_path,progress)
    print(json.dumps({"status":progress["status"],"report":str(report_path),"report_sha256":sha_file(report_path),
      "candidate":candidate_info},sort_keys=True,indent=2,allow_nan=False)); return
   candidate_info["exact_stage"]="cord_pair_or_containment_failed"; save_report()
   all_candidate_results.append(candidate_info)
   stage("candidate_exact_pair_failed",attempt=attempt,remaining_candidates=len(amp_list)-attempt,
         pair_count=len(unexpected),amplitude_m=amplitude)
  report["status"]="bounded_candidates_failed"; report["next_change"]="The local C2 translation profile did not clear the exact closed-solid cord pair within four amplitude checks; revise support direction/profile from retained witnesses, do not waive pairs."
  save_report(); progress["status"]="bounded_candidates_failed"; atomic_json(progress_path,progress)
  print(json.dumps({"status":progress["status"],"report":str(report_path),"report_sha256":sha_file(report_path),
    "attempts":len(report.get("candidates",[]))},sort_keys=True,indent=2,allow_nan=False))
 except Exception as exc:
  report["status"]="failed"; report["failure"]={"type":type(exc).__name__,"message":str(exc)}; save_report()
  progress["status"]="failed"; progress["failure"]={"type":type(exc).__name__,"message":str(exc)}; atomic_json(progress_path,progress)
  raise
 finally:
  signal.alarm(0)

if __name__=="__main__": main()
