#!/usr/bin/env python3
"""Selected-pair high-precision replay for 25 row-310 terminal self hits."""
import hashlib, importlib.util, json, math, pathlib, struct, sys
from fractions import Fraction
import numpy as np
E=pathlib.Path("/Users/n/numi-human-resting-evidence-20261005")
O=E/"native-lung-pleura-self-seam-diagnosis-952"
N=E/"lung-choroid-composition-924/choroid-v2/resting-thorax.nhanatomy"
M=E/"lung-choroid-composition-924/choroid-v2/resting-anatomy-manifest.json"
P=E/"native-terminal-cycle-931/skin-source-anatomy-parameters.bin"
R=E/"native-terminal-cycle-931/accepted-geometry/step-10000.receipt.json"
K=E/"native-terminal-cycle-931/accepted-geometry/step-10000.mrvpack"
L=E/"lung-current-precision-candidate-922/row-310-face-lineage.npy"
C=E/"native-lung-seam-terminal-classification-939/report.json"
F=E/"native-lung-seam-terminal-source-coplanarity-940/shared-feature-recheck.json"
PAR=pathlib.Path("/Users/n/numi-human-lung-source-publication-902/src/numilab_human/resting_anatomy_interface_patch.py")
PRED=pathlib.Path("/Users/n/numi-human-resting-final-integration-001/src/numilab_human/cardiac_cavity_intersections.py")
VIS=pathlib.Path("/Users/n/numi-human-performance-source-014/apps/NumiHumanRestingVisual.hpp")
MET=pathlib.Path("/Users/n/numi-human-performance-source-014/matter/src/human_respiration.metal")
HDR=pathlib.Path("/Users/n/numi-human-performance-source-014/include/metalrobo/numi_human_resting_visual_gpu.h")
PACKREADER=E/"cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py"
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
sys.path.insert(0,"/Users/n/numi-human-lung-source-publication-902/src")
from numilab_human.resting_anatomy_interface_patch import parse_payload
from numilab_human.cardiac_cavity_intersections import triangle_intersection_points,float32_point_lattice_key
pr=load("reader952",PACKREADER)
def tri32(t):return tuple(float32_point_lattice_key(tuple(float(x) for x in p)) for p in t)
def tri64(t):return tuple(tuple(Fraction.from_float(float(x)) for x in p) for p in t)
def hit(a,b,lattice=False):
 pts=list(dict.fromkeys(triangle_intersection_points(a,b)));span=0.
 if len(pts)>1:
  q=np.asarray([[float(x) for x in p] for p in pts])
  if lattice:
   q=np.ldexp(q,-149)
  span=max(float(np.linalg.norm(q[i]-q[j])) for i in range(len(q)) for j in range(i+1,len(q)))
 return {"hit":bool(pts),"points":len(pts),"span_m":span}
def ss(t):
 t=min(max(float(t),0.),1.);return t*t*(3.-2.*t)
def smooth(p,c):
 rim,rt=c["footprint_rim_ellipse_m"],c["footprint_rim_transition"];cr,ct=c["footprint_crural_ellipses_m"],c["footprint_crural_transition"]
 def el(e,tr,inside):
  x=(p[0]-e[0])/e[2];z=(p[2]-e[1])/e[3];r=math.sqrt(x*x+z*z);t=min(max((r-tr[0])/tr[1],0.),1.);v=ss(t);d=6*t*(1-t)/tr[1]*(-1. if inside else 1.)
  return (d*x/(e[2]*r) if r else 0.,d*z/(e[3]*r) if r else 0.,1.-v if inside else v)
 r=el(rim,rt,True);a=el(cr[0],ct,False);b=el(cr[1],ct,False);w=r[2]*a[2]*b[2]
 wx=r[0]*a[2]*b[2]+r[2]*a[0]*b[2]+r[2]*a[2]*b[0];wz=r[1]*a[2]*b[2]+r[2]*a[1]*b[2]+r[2]*a[2]*b[1]
 t=min(max((p[1]-c["basal_blend_start_m"])/c["basal_blend_span_m"],0.),1.);g=1.-ss(t);dg=-6*t*(1-t)/c["basal_blend_span_m"]
 return (g*wx,g*w*dg,g*wz,g*w)
def basis(p,c):
 h=float(c["conforming_grid_spacing_m"]);u=[float(x)/h for x in p];li=[math.floor(x) for x in u];low=[x*h for x in li];fr=[u[k]-li[k] for k in range(3)];order=sorted(range(3),key=lambda k:(-fr[k],k));corner=list(low);prev=smooth(corner,c)[3];out=[0.,0.,0.,prev]
 for k in order:
  corner[k]+=h;n=smooth(corner,c)[3];out[k]=(n-prev)/h;out[3]+=out[k]*(float(p[k])-low[k]);prev=n
 return out
def rot64(q,v):
 qq=np.asarray(q,dtype=np.float64);v=np.asarray(v,dtype=np.float64);return v+2.*np.cross(qq[:3],np.cross(qq[:3],v)+float(qq[3])*v)
def map64(p,c,axis,anchor,volume,area,motion,body):
 p=np.asarray(p,dtype=np.float64);w=basis(p,c)[3];d=motion[0]/area;after=volume+motion[0];rad=math.sqrt((after+motion[1])/after);off=p-anchor;along=np.dot(off,axis)*axis;across=off-along;m=p-axis*(d*w)+(rad-1.)*across
 return np.asarray(body["position_m"],dtype=np.float64)+rot64(body["quaternion_xyzw"],m)
def f(x):return np.asarray(x,dtype=np.float32)
def cross32(a,b):return f([f(a[1]*b[2]-a[2]*b[1]),f(a[2]*b[0]-a[0]*b[2]),f(a[0]*b[1]-a[1]*b[0])])
def rot32(q,v):return f(v+f(f(2.)*cross32(q[:3],f(cross32(q[:3],v)+f(q[3]*v)))))
def map32(p,c,axis,anchor,volume,area,motion,body):
 p=f(p);w=f(basis([float(x) for x in p],c)[3]);d=f(f(motion[0])/f(area));after=f(f(volume)+f(motion[0]));rad=f(np.sqrt(f(f(after+f(motion[1]))/after)));off=f(p-anchor);along=f(f(off[1])*axis);across=f(off-along);m=f(p-f(axis*f(d*w))+f(f(rad-f(1.))*across))
 return f(f(body["position_m"])+rot32(f(body["quaternion_xyzw"]),m))
def ulps(a,b):
 aa=np.asarray(a,dtype=np.float32).view(np.uint32).astype(np.uint64);bb=np.asarray(b,dtype=np.float32).view(np.uint32).astype(np.uint64)
 def ordered(x):return np.where((x&0x80000000)!=0,(~x)&0xffffffff,x|0x80000000)
 return np.abs(ordered(aa).astype(np.int64)-ordered(bb).astype(np.int64))
assert sha(N)=="3c444be7736c066a992988cc32b687917e1c4c5c3968a16b4d5f0106d5b5024e"
assert sha(K)=="40edd587f96eb27b26d882a74d60243086905fa55a4e65633f1467500f13d764"
assert sha(R)=="fe2869660da122f0042b8ebd9334d01cadb28aae14e6d5d1f692293ce57aa668"
assert sha(L)=="f86fbb7dd14b8ed285784d8f3dba3fe39750526f5ebb10831e6a3fca435aec36"
_,rows=parse_payload(N);lin=np.load(L,allow_pickle=False);cl=json.loads(C.read_text());re=json.loads(F.read_text());cfg=json.loads(M.read_text())["functional_bindings"]["respiratory_geometry_binding"];receipt=json.loads(R.read_text())
params=P.read_bytes();assert len(params)==1008;words=struct.unpack_from("<4I",params,0);assert words[0]==20
anchor=np.asarray(struct.unpack_from("<3f",params,16),dtype=np.float64);volume,=struct.unpack_from("<f",params,28);axis=np.asarray(struct.unpack_from("<3f",params,32),dtype=np.float64);area,=struct.unpack_from("<f",params,104)
body=next(x for x in receipt["accepted_registered_body_poses"] if x["body_index"]==20)
motion=np.asarray([receipt["accepted_respiratory_motion"]["diaphragm_swept_volume_m3"],receipt["accepted_respiratory_motion"]["rib_swept_volume_m3"]],dtype=np.float32)
assert all(rows[i]["body_index"]==20 for i in range(305,312))
mp,stream,vo,surfs=pr.read_pack(K)
try:
 pr.validate_accepted_receipt(K,R,10000,mp,vo,surfs)
 faces={i:surfs[(51023,i)] for i in range(305,310)};faces[310]=surfs[(51024,310)]
 def cap(sid,fid):
  ids=faces[sid]["faces"][int(fid)];return np.asarray([struct.unpack_from("<3f",mp,vo+int(i)*80) for i in ids],dtype=np.float32)
 def keys(t):return sorted(tuple(int(x) for x in np.asarray(p,dtype=np.float32).view(np.uint32)) for p in t)
 recidx={(tuple(x["owners"]),tuple(x["face_ids"])):x for x in re["cross_events"]}
 events=cl["pleura_self"]["events"];assert len(events)==25
 out=[]; exact_axes=total_axes=maxulp=0;maxf32=max64=0.;rel={}
 for ev in events:
  pf=ev["pleura_face_ids"];parent=ev["row310_face_lineage"];assert all(tuple(map(int,lin[int(f)]))==tuple(map(int,p)) for f,p in zip(pf,parent))
  owners=tuple(map(int,ev["parent_owner_pair"]));fids=tuple(map(int,ev["parent_face_ids"]));srcs=[];hi=[];emu=[];actual=[];vrows=[]
  for owner,fid,pfid in zip(owners,fids,pf):
   row=rows[owner];assert row["body_index"]==20;vid=row["faces"][fid];src=np.asarray(row["vertices6"][vid,:3],dtype=np.float32);srcs.append(src)
   ideal=np.asarray([map64(p,cfg,axis,anchor,volume,area,motion,body) for p in src],dtype=np.float64);hi.append(ideal)
   replay=np.asarray([map32(p,cfg,f(axis),f(anchor),f(volume),f(area),f(motion),body) for p in src],dtype=np.float32);emu.append(replay)
   actual.append(cap(owner,fid));assert keys(cap(310,pfid))==keys(actual[-1])
   for vi,sp,ip,ep,ap in zip(vid,src,ideal,replay,actual[-1]):
    e=float(np.linalg.norm(ap.astype(np.float64)-ip));m=float(np.linalg.norm(ap.astype(np.float64)-ep.astype(np.float64)));u=ulps(ap,ep);max64=max(max64,e);maxf32=max(maxf32,m);maxulp=max(maxulp,int(u.max()));exact_axes+=int(np.count_nonzero(u==0));total_axes+=3
    vrows.append({"owner":owner,"source_vertex":int(vi),"source_f32":[float(x) for x in sp],"capture_f32":[float(x) for x in ap],"map64":[float(x) for x in ip],"map32_replay":[float(x) for x in ep],"capture_minus_map64_m":e,"capture_minus_map32_m":m,"ulp_difference":[int(x) for x in u]})
  sh=hit(tri32(srcs[0]),tri32(srcs[1]),lattice=True);ih=hit(tri64(hi[0]),tri64(hi[1]));fh=hit(tri32(emu[0]),tri32(emu[1]),lattice=True);nh=hit(tri32(actual[0]),tri32(actual[1]),lattice=True);srec=recidx[(owners,fids)]
  assert sh["hit"]==bool(srec["source_intersection"]) and nh["hit"]
  rel[str((ih["hit"],ih["points"]))]=rel.get(str((ih["hit"],ih["points"])),0)+1
  out.append({"pleura_face_ids":pf,"parent_owner_pair":list(owners),"parent_face_ids":list(fids),"source_relation":sh,
   "source_feature":{"shared_coordinate_count":srec["shared_source_coordinate_count"],"confined_to_shared_feature":srec["source_intersection_confined_to_shared_feature"],"beyond_shared_feature":srec["source_intersection_beyond_shared_feature"],
                     "source_intersection":srec["source_intersection"]},
   "source_pair_diagnostic_939":{"closest_distance_m":ev["source_closest_distance_m"],"signed_gap_m":ev["source_signed_gap_m"],"side_status":ev["source_side_status"]},
   "high_precision_map_relation":ih,"float32_map_replay_relation":fh,"captured_native_relation":nh,
   "native_report":{"min_signed_gap_m":ev["native_signed_gap_m"]["min_m"],"opposite_source_side_intrusion_m":ev["opposite_source_side_intrusion_m"],"pair_half_ulp_coordinate_diagnostic_m":ev["pair_half_ulp_normal_bound_m"],"intrusion_exceeds_diagnostic":ev["intrusion_exceeds_coordinate_bound"],"intersection_span_m":ev["native_intersection_span_m"],"rigid_fit_residual_diagnostic_m":ev["max_rigid_pose_vertex_residual_m"]},
   "pair_max_capture_minus_map64_m":max(float(np.linalg.norm(actual[j][i].astype(np.float64)-hi[j][i])) for j in range(2) for i in range(3)),
   "pair_max_capture_minus_map32_replay_m":max(float(np.linalg.norm(actual[j][i].astype(np.float64)-emu[j][i].astype(np.float64))) for j in range(2) for i in range(3)),
   "vertices":vrows})
 summary={"pairs":len(out),"source_hits":sum(x["source_relation"]["hit"] for x in out),"source_shared_point_only":sum(x["source_feature"]["confined_to_shared_feature"] for x in out),"source_no_intersection":sum(not x["source_relation"]["hit"] for x in out),
  "map64_pair_relations":rel,"float32_replay_exact_pair_hits":sum(x["float32_map_replay_relation"]["hit"] for x in out),"float32_replay_relations":{str((x["float32_map_replay_relation"]["hit"],x["float32_map_replay_relation"]["points"])):sum(1 for y in out if (y["float32_map_replay_relation"]["hit"],y["float32_map_replay_relation"]["points"])==(x["float32_map_replay_relation"]["hit"],x["float32_map_replay_relation"]["points"])) for x in out},"native_capture_exact_pair_hits":sum(x["captured_native_relation"]["hit"] for x in out),"f32_replay_exact_coordinates":exact_axes,"f32_replay_coordinate_count":total_axes,"f32_replay_exact_fraction":exact_axes/total_axes,"f32_replay_max_ulp_difference":maxulp,"f32_replay_max_vertex_error_m":maxf32,"map64_vs_capture_max_vertex_error_m":max64,"map64_vs_capture_max_component_error_m":max(abs(float(v["capture_f32"][k])-float(v["map64"][k])) for e in out for v in e["vertices"] for k in range(3)),"float32_replay_vs_capture_max_component_error_m":max(abs(float(v["capture_f32"][k])-float(v["map32_replay"][k])) for e in out for v in e["vertices"] for k in range(3)),"source_disjoint_pair_distance_nm_range":[min(e["source_pair_diagnostic_939"]["closest_distance_m"] for e in out if not e["source_relation"]["hit"])*1e9,max(e["source_pair_diagnostic_939"]["closest_distance_m"] for e in out if not e["source_relation"]["hit"])*1e9],
  "max_native_intrusion_m":max(x["native_report"]["opposite_source_side_intrusion_m"] for x in out),"max_native_span_m":max(x["native_report"]["intersection_span_m"] for x in out),"all_native_hits_remain_unallowed":True}
 report={"schema":"numi.human.row310-terminal-self-seam-map-replay.v1","status":"complete_scoped_exact_replay","scope":"25 selected exact unallowed row-310 self pairs at accepted step 10000; no whole-mesh scan.",
  "inputs":{str(p):{"sha256":sha(p),"bytes":p.stat().st_size} for p in [N,M,P,R,K,L,C,F,PAR,PRED,PACKREADER,VIS,MET,HDR]},
  "runtime":{"accepted_step":receipt["accepted_step"],"accepted_time_s":receipt["accepted_time_s"],"body_pose_selected_by_body_index":20,"body20_pose":body,"body_and_flags":list(words),"motion_f32":motion.tolist(),"anchor_f32_m":anchor.tolist(),"lung_volume_f32_m3":volume,"axis_f32":axis.tolist(),"swept_area_f32_m2":area,
   "deformation_path":"Visual.hpp assigns kind 1 to functional lung and pleura rows; NHA rows 305-311 all declare source body_index 20. Metal kind-1 uses the shared source-local Kuhn basis, basal/radial vertex map, then body-20 rigid transform."},
  "methods":{"map64":"Double evaluation of the frozen C++ RespiratoryBasis::evaluate Kuhn interpolation and Metal kind-1 vertex-map equations using captured Float32 source points, uploaded anatomy coefficients, accepted diaphragm/rib motion, and accepted body20 pose; exact binary64 triangle predicates follow. This is a selected-pair higher-precision replay, not a proof over every representable input or a global injectivity proof.","native":"Actual accepted MRVPACK Float32 geometry, exact Float32-lattice predicate; row310 face coordinates checked bitwise against mapped parent face.","map32":"Offline Float32 transcription of the same equations, compared to native capture; not operation-bitwise identical to the compiled Metal kernel and not treated as capture authority. It is used to show sensitivity to the final Float32 operation/materialization path.","limits":["Selected-pair replay is not a global continuous-map injectivity proof.","The 939 half-ULP and rigid-fit quantities are diagnostics, not a complete theoretical bound for shader arithmetic. The measured capture-minus-higher-precision envelope in this selected 150-vertex set is empirical only, not a formal shader-error bound.","All exact native events remain unallowed; no geometry or gate changed."]},
  "summary":summary,"events":out}
 (O/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
 print(json.dumps(summary,sort_keys=True,indent=2))
finally:
 stream.close();mp.close()
