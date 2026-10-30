import csv,json,hashlib,heapq,time
from pathlib import Path
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
OUT=P/"protected-149-current-surface-crosswalk-001/run-009-correct-witness-operand-packet"
HEAD=P/"protected-149-current-surface-crosswalk-001/head-ring-135-root-008/head-ring-135-targets.csv"
HEAD_RECEIPT=P/"protected-149-current-surface-crosswalk-001/head-ring-135-root-008/receipt.json"
SKIN_MAP=P/"paired-fine-headwall-rim-635/run-013/output/closed-skin-source-map.npz"
FAT_MAP=P/"closed-fat-source-map-649/closed-fat-source-map.npz"
ATT=P/"native-physical-parent-mass-330/native-source-identity-control-003/parent-deep-attachments.csv"
PARENT=P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv"
POSE=P/"native-physical-parent-mass-330/native-source-identity-control-003/registered-skin-poses-100.json"
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
C=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
SEARCH=P/"protected-149-current-surface-crosswalk-001/run-007-corrected-parent-face-targets/corrected-search.json"
RECHECK=P/"protected-149-current-surface-crosswalk-001/run-004/native-bank-point-recheck.json"
REPLAY=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
BANK={23:P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/batch-007/bank-operands/bank-23-33.npz",139:P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/batch-017/bank-operands/bank-139-132.npz"}
TARGETS={29464:83239,14506:26444}
MAX_TARGET_OFFSET_M=0.010
ENVELOPE_MAX_OFFSET_M=4e-6*(3.0**0.5/2.0)
DONOR_MIN=0.85
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),bytes()):h.update(b)
 return h.hexdigest()
def load_csv(path):
 with open(path,newline="") as f:return list(csv.DictReader(f))
def json_sha(x):return sha(x)
def closest_point(p,t):
 a,b,c=t
 ab=b-a;ac=c-a;ap=p-a;n=np.cross(ab,ac);nn=float(np.dot(n,n))
 if nn>0:
  pl=p-np.dot(ap,n)/nn*n
  g00=float(np.dot(ab,ab));g01=float(np.dot(ab,ac));g11=float(np.dot(ac,ac));v=pl-a;h0=float(np.dot(v,ab));h1=float(np.dot(v,ac));det=g00*g11-g01*g01
  if det>0:
   u=(g11*h0-g01*h1)/det;w=(g00*h1-g01*h0)/det
   if u>=0 and w>=0 and u+w<=1:return pl,np.array([1-u-w,u,w])
 best=None
 for i,j in ((0,1),(1,2),(2,0)):
  e=t[j]-t[i];den=float(np.dot(e,e));z=0. if den==0 else min(1.,max(0.,float(np.dot(p-t[i],e)/den)))
  q=t[i]+z*e;d=float(np.dot(q-p,q-p));bc=np.zeros(3);bc[i]=1-z;bc[j]=z
  if best is None or d<best[0]:best=(d,q,bc)
 return best[1],best[2]
def face_graph(f,p):
 edge_rows={}
 for i,tri in enumerate(f):
  for a,b in ((0,1),(1,2),(2,0)):edge_rows.setdefault(tuple(sorted((int(tri[a]),int(tri[b])))),[]).append(i)
 adj=[{} for _ in range(len(f))]
 cent=p[f].mean(axis=1)
 for rs in edge_rows.values():
  if len(rs)>1:
   for i in rs:
    for j in rs:
     if i!=j:adj[i][j]=float(np.linalg.norm(cent[i]-cent[j]))
 return adj,edge_rows
def face_path(adj,start,target):
 d=[float("inf")]*len(adj);prev=[-1]*len(adj);h=[10**9]*len(adj);d[start]=0.;h[start]=0;q=[(0.,0,start)]
 while q:
  cost,hop,u=heapq.heappop(q)
  if cost!=d[u] or hop!=h[u]:continue
  if u==target:break
  for v,w in adj[u].items():
   nd=cost+w;nh=hop+1
   if nd<d[v] or (nd==d[v] and nh<h[v]):d[v]=nd;h[v]=nh;prev[v]=u;heapq.heappush(q,(nd,nh,v))
 assert np.isfinite(d[target]),(start,target)
 path=[];u=target
 while u!=-1:path.append(u);u=prev[u]
 path.reverse();return path,d[target]
def local_quality(v,p,f,orig,body,bone,sourceid,bodyid,boneid):
 rows=np.flatnonzero(np.any(f==v,axis=1));tri=p[f[rows]].astype(float)
 e01=np.linalg.norm(tri[:,1]-tri[:,0],axis=1);e12=np.linalg.norm(tri[:,2]-tri[:,1],axis=1);e20=np.linalg.norm(tri[:,0]-tri[:,2],axis=1)
 area=.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
 maxedge=np.maximum(np.maximum(e01,e12),e20); altitude=np.divide(2*area,maxedge,out=np.zeros_like(area),where=maxedge>0)
 labels=(body[rows]==bodyid)&(bone[rows]==boneid)
 return rows,{"incident_faces":int(len(rows)),"typed_body_bone_face_fraction":float(np.mean(labels)) if len(rows) else 0.,"same_envelope_id_fraction":float(np.mean(orig[rows]==sourceid)) if len(rows) else 0.,"min_edge_m":float(np.min(np.minimum(np.minimum(e01,e12),e20))) if len(rows) else None,"min_area_m2":float(np.min(area)) if len(rows) else None,"min_altitude_m":float(np.min(altitude)) if len(rows) else None,"min_angle_deg":float(np.min(np.concatenate([np.degrees(np.arccos(np.clip(np.sum((tri[:,(i+1)%3]-tri[:,i])*(tri[:,(i+2)%3]-tri[:,i]),axis=1)/(np.linalg.norm(tri[:,(i+1)%3]-tri[:,i],axis=1)*np.linalg.norm(tri[:,(i+2)%3]-tri[:,i],axis=1)), -1,1))) for i in range(3)]))) if len(rows) else None}
t0=time.time()
head=load_csv(HEAD); assert len(head)==135
head_receipt=json.load(open(HEAD_RECEIPT)); assert head_receipt["csv_sha256"]==sha(HEAD)
deep=load_csv(ATT); deep=[x for x in deep if x["role"]=="D"]; assert len(deep)==14
with np.load(BASE,allow_pickle=False) as base:
 pins=np.asarray(base["pinned_volume_node_rows"],dtype=int);parent=np.asarray(base["parent_volume_node_row"],dtype=int);basepts=np.asarray(base["points_current_world_m_f32"],dtype=float)
with np.load(C,allow_pickle=False) as c:
 cpts=np.asarray(c["points_world_m"],dtype=float); cfaces=np.asarray(c["faces"],dtype=int)
 corig=np.asarray(c["face_source_original_id"],dtype=int); cbody=np.asarray(c["face_source_body"],dtype=int); cbone=np.asarray(c["face_source_bone"],dtype=int); ctag=np.asarray(c["face_source_face_id"],dtype=int)
search=json.load(open(SEARCH)); recheck=json.load(open(RECHECK))
deep_by_source={int(x["source_vertex_row"]):x for x in deep}
# Source witness face tags, native replay local operand face rows and immutable donor weights.
witness_by_id={int(x["attachment_id"]):x for x in recheck["attachments"]} if "attachments" in recheck else {}
# Current run004 JSON stores rows in a body/attachment list; normalize deterministically.
if not witness_by_id:
 for x in recheck.get("rows",recheck.get("witnesses",[])):
  if "attachment_id" in x:witness_by_id[int(x["attachment_id"])]=x
poses=json.load(open(POSE));lane_body=[int(x["body_index"]) for x in poses["bodies"]]
weights_rows=load_csv(Path(str(PARENT)+".weights.csv")); weights_by_node={int(x["vertex"]):x for x in weights_rows}
tet_rows=load_csv(Path(str(PARENT)+".tets.csv")); label_rows=load_csv(Path(str(PARENT)+".labels.csv"))
label_by_tet={int(x["parent_tet_row"]):x for x in label_rows}
tets_by_node={}
for t in tet_rows:
 ti=int(t["parent_tet_row"]); nodes=[int(t[f"parent_node{i}"]) for i in range(4)]
 for n in nodes:tets_by_node.setdefault(n,[]).append(ti)
# Deep identity rows and canonical row mapping by the exact source330 parent node column.
canonical_by_parent={int(parent[int(pins[i])]):i for i in range(len(pins))}
assert len(canonical_by_parent)==len(pins)
deep_rows={}
for a in deep:
 src=int(a["source_vertex_row"]);drow=int(a["reduced_column_row"]);pnode=4822+drow
 canon=canonical_by_parent.get(pnode)
 assert canon is not None,(src,pnode)
 assert int(pins[canon])==int(a["volume_node_row"]),(src,canon,pins[canon],a["volume_node_row"])
 deep_rows[canon]=a
assert len(deep_rows)==14
# Load source ancestry maps; establish one-to-one row associations for all mapped boundary points.
sm=np.load(SKIN_MAP,allow_pickle=False); fm=np.load(FAT_MAP,allow_pickle=False)
o_idx={int(v):i for i,v in enumerate(sm["boundary_O_fine_vertex_rows"])}
i_idx={int(v):i for i,v in enumerate(sm["boundary_I_fine_vertex_rows"])}
fat_layer=np.asarray(fm["output_vertex_source_layer"],dtype=int);fat_source=np.asarray(fm["output_vertex_source_row"],dtype=int)
d_source_rows=np.flatnonzero(fat_layer==2); d_source_lookup={}
for r in d_source_rows:d_source_lookup.setdefault(int(fat_source[r]),[]).append(int(r))
# Candidate source-face ancestry and full local quality for the two remapped sites.
result_by_attachment={}
for run in search["results"]:
 aid=int(run["attachment_id"]);candidate=int(TARGETS[aid]);x=next(z for z in run["candidates_ranked"] if int(z["vertex_row"])==candidate)
 bdy=int(run["body"]);sourceid=int(run["source_envelope_id"]);boneid=int(run["bone"])
 assert x["anchor_distance_m"]<=MAX_TARGET_OFFSET_M
 assert x["source_parent_residual_m"]<=ENVELOPE_MAX_OFFSET_M
 assert x["quality"]["same_body_bone_fraction"]>=DONOR_MIN
 # Authenticate candidate current D triangle labels.
 inc=np.asarray(x["incident_output_face_rows"],dtype=int)
 assert np.all(cbody[inc]==bdy) and np.all(cbone[inc]==boneid) and np.all(corig[inc]==sourceid)
 bank=np.load(BANK[bdy],allow_pickle=False);bank_faces=np.asarray(bank["faces_f64"],dtype=int);bank_pts=np.asarray(bank["points_world_m_f64"],dtype=float)
 targetface=int(x["source_operand_face_row"]); wtag=int(run["canonical_witness_face_id_tag"])
 replay=np.load(REPLAY,allow_pickle=False);bi=65 if aid==29464 else 146
 # Resolve the native witness through the pinned global replay row; stable row
 # and face ID tag are distinct namespaces and are never array subscripts.
 witness_global=int(run["native_replay_global_face_row"])
 witness_face=int(run["native_replay_operand_face_row"])
 witness_stable=int(run["native_replay_stable_face_row"])
 assert int(replay["bank_face_offsets"][bi]) <= witness_global < int(replay["bank_face_offsets"][bi+1])
 assert int(replay["face_source_operand_face_row"][witness_global])==witness_face
 assert int(replay["face_source_stable_row"][witness_global])==witness_stable
 assert int(replay["bank_body"][bi])==bdy and int(replay["bank_bone"][bi])==boneid
 assert int(bank["face_id"][witness_face])==wtag
 witness_tag=int(bank["face_id"][witness_face])
 assert witness_tag==wtag
 adj,edge_rows=face_graph(bank_faces,bank_pts);fpath,flen=face_path(adj,witness_face,targetface)
 for u,v in zip(fpath,fpath[1:]):
  common=set(bank_faces[u])&set(bank_faces[v]);assert len(common)>=2
 fp=[int(bank["face_id"][r]) for r in fpath]
 source_q=np.asarray(x["target_world_m"],dtype=float)
 tri=bank_pts[bank_faces[targetface]]
 projected,bary=closest_point(source_q,tri)
 residual=float(np.linalg.norm(projected-source_q))
 assert abs(residual-float(x["source_parent_residual_m"]))<1e-9
 quality_rows,q=local_quality(candidate,cpts,cfaces,corig,cbody,cbone,sourceid,bdy,boneid)
 assert q["typed_body_bone_face_fraction"]>=DONOR_MIN
 # Full86 source donor confidence is inherited from immutable canonical source node; map lane to body from registered pose order.
 attachment=deep_by_source[aid];volume_node=int(attachment["volume_node_row"]); donor=int(attachment["donor_body"]); wr=weights_by_node[volume_node]
 lane_weights=[float(wr["w"+str(i)]) for i in range(86)];lane=int(np.argmax(lane_weights)); lane_body_id=lane_body[lane]; confidence=lane_weights[lane]
 assert lane_body_id==donor and confidence>=DONOR_MIN,(aid,lane,lane_body_id,donor,confidence)
 old=np.asarray([float(attachment[k]) for k in ("reference_x_m","reference_y_m","reference_z_m")])
 displacement=source_q-old
 # Source330 parent star remains the canonical input star; record exact rows and labels, do not claim candidate point lies inside these tets.
 star=tets_by_node[volume_node]
 star_records=[]
 for ti in star:
  tr=tet_rows[ti];lab=label_by_tet[ti]
  star_records.append({"parent_tet_row":ti,"material_index":int(tr["material_index"]),"donor_body":int(tr["donor_body"]),"geometric_donor_body":int(lab["geometric_donor_body"]),"anatomical_region_id":int(lab["anatomical_region_id"]),"stable_source_face_id":int(lab["stable_source_face_id"])})
 result_by_attachment[aid]={"stable_attachment_id":aid,"canonical_protected_row":int(canonical_by_parent[4822+int(attachment["reduced_column_row"])]),"canonical_source_volume_node_row":volume_node,"canonical_reduced_D_row":int(attachment["reduced_column_row"]),"canonical_donor_body":donor,"canonical_frame_owner_body":int(attachment["frame_owner_body"]),"source_role":"BONE_ENVELOPE","source_bone_id":boneid,"source_envelope_id":sourceid,"source_bank_path":str(BANK[bdy]),"source_bank_sha256":sha(BANK[bdy]),"source_native_witness_face_id_tag":wtag,"source_native_witness_operand_face_row":witness_face,"source_native_witness_stable_face_row":witness_tag,"source_target_operand_face_row":targetface,"source_target_face_id_tag":int(bank["face_id"][targetface]),"source_face_path_operand_rows":fpath,"source_face_path_face_id_tags":fp,"source_face_path_hops":len(fpath)-1,"source_face_centroid_path_length_m":flen,"target_to_parent_source_face_residual_m":residual,"target_source_face_barycentric":bary.tolist(),"source330_original_current_point_m":[float(attachment[k]) for k in ("current_x_m","current_y_m","current_z_m")],"source330_original_reference_point_m":old.tolist(),"target_world_m":source_q.tolist(),"old_reference_to_target_displacement_m":float(np.linalg.norm(displacement)),"old_reference_to_target_vector_m":displacement.tolist(),"full86_confidence_lane":lane,"full86_confidence_body":lane_body_id,"full86_confidence_weight":confidence,"full86_weight_lane_to_body_map_sha256":sha(POSE),"source330_parent_star_tets":star_records,"candidate_surface_vertex_row":candidate,"candidate_incident_output_faces":quality_rows.tolist(),"candidate_incident_source_face_id_tags":sorted(set(int(x) for x in ctag[quality_rows])),"candidate_surface_quality":q,"candidate_surface_quality_status":"finite_positive_measured; future tetrahedral incident-altitude gate NOT_RUN","candidate_volume_node_row":-1,"child_volume_binding_status":"pending candidate-specific retet; this surface vertex is not a child volume node"}
# Assemble 135 head boundary map with source ancestry.
rows=[]
for h in head:
 cr=int(h["canonical_protected_row"]);role=h["role"];v=int(h["current_surface_vertex_row"])
 base={"canonical_protected_row":cr,"mapping_kind":"head_ring_boundary_target","role":role,"base_volume_node_row":int(h["base_volume_node_row"]),"source330_parent_node_row":int(h["source330_parent_node_row"]),"source330_parent_star_tets_json":"","stable_attachment_id":"","canonical_donor_body":"","full86_confidence_lane":"","full86_confidence_body":"","full86_confidence_weight":"","current_surface_vertex_row":v,"target_world_x_m":float(h["current_x_m"]),"target_world_y_m":float(h["current_y_m"]),"target_world_z_m":float(h["current_z_m"]),"source_to_target_residual_m":float(h["source_to_current_distance_m"]),"source_map_path":"","source_map_sha256":"","source_map_row":"","source_face_path_operand_rows_json":"","source_face_path_face_id_tags_json":"","source_face_path_hops":"","source_face_centroid_path_length_m":"","target_to_source_face_residual_m":"","source_face_barycentric_json":"","candidate_surface_quality_json":"","binding_status":"source-only boundary ancestry mapped; volume binding pending"}
 if role=="O":
  i=o_idx[v]; assert int(sm["boundary_O_fine_vertex_rows"][i])==v
  base.update({"source_map_path":str(SKIN_MAP),"source_map_sha256":sha(SKIN_MAP),"source_map_row":i,"source330_parent_star_tets_json":json.dumps({"parent_tet_row":int(sm["boundary_O_parent_tet_rows"][i]),"parent_tet_nodes":sm["boundary_O_parent_tet_node_rows"][i].astype(int).tolist(),"parent_tet_barycentric":sm["boundary_O_parent_tet_barycentric_weights"][i].tolist(),"dominant_representative_row":int(sm["boundary_O_parent330_dominant_representative_rows"][i]),"dominant_weight_margin":float(sm["boundary_O_parent_tet_dominant_weight_margins"][i]),"material":int(sm["boundary_O_parent_material_indices"][i]),"geometric_donor":int(sm["boundary_O_parent_geometric_donor_bodies"][i]),"region":int(sm["boundary_O_parent_anatomical_region_ids"][i]),"stable_source_face_id":int(sm["boundary_O_parent_stable_source_face_ids"][i]),"registration_residual_m":float(sm["boundary_O_parent_tet_registration_residual_m"][i])})})
 elif role=="I":
  i=i_idx[v]; assert int(sm["boundary_I_fine_vertex_rows"][i])==v
  base.update({"source_map_path":str(SKIN_MAP),"source_map_sha256":sha(SKIN_MAP),"source_map_row":i,"source330_parent_star_tets_json":json.dumps({"parent330_node_row":int(sm["boundary_I_parent330_node_rows"][i]),"anchor_row":int(sm["boundary_I_anchor_rows"][i])})})
 else:
  candidates=d_source_lookup.get(v,[])
  assert len(candidates)==1,(v,len(candidates))
  base.update({"source_map_path":str(FAT_MAP),"source_map_sha256":sha(FAT_MAP),"source_map_row":candidates[0],"source330_parent_star_tets_json":json.dumps({"output_vertex_source_layer":int(fat_layer[candidates[0]]),"fine_D_source_vertex_row":int(fat_source[candidates[0]])})})
 rows.append(base)
# Add all 14 canonical deep attachment targets; 12 retain their exact canonical source volume row, 2 have explicit D surface references.
for cr,a in sorted(deep_rows.items()):
 aid=int(a["source_vertex_row"]);volume_node=int(a["volume_node_row"]);body=int(a["donor_body"]);drow=int(a["reduced_column_row"])
 old=np.asarray([float(a[k]) for k in ("current_x_m","current_y_m","current_z_m")])
 is_remap=aid in result_by_attachment
 if is_remap:
  x=result_by_attachment[aid];target=np.asarray(x["target_world_m"],dtype=float);surface_row=int(x["candidate_surface_vertex_row"]);kind="deep_attachment_reference_rebind_to_D_surface_vertex"
  source_path_json=json.dumps(x["source_face_path_operand_rows"])
  face_tags_json=json.dumps(x["source_face_path_face_id_tags"])
  star_json=json.dumps(x["source330_parent_star_tets"])
  quality_json=json.dumps(x["candidate_surface_quality"],sort_keys=True)
  srcmap=str(C);srcmapsha=sha(C);srcmaprow=surface_row
 else:
  target=basepts[int(pins[cr])];surface_row=-1;kind="deep_attachment_original_volume_node_retained"
  source_path_json="";face_tags_json="";star=tets_by_node[volume_node]
  star_json=json.dumps([{"parent_tet_row":ti,"material_index":int(tet_rows[ti]["material_index"]),"donor_body":int(tet_rows[ti]["donor_body"]),"geometric_donor_body":int(label_by_tet[ti]["geometric_donor_body"]),"anatomical_region_id":int(label_by_tet[ti]["anatomical_region_id"]),"stable_source_face_id":int(label_by_tet[ti]["stable_source_face_id"])} for ti in star])
  quality_json=""
  srcmap=str(BASE);srcmapsha=sha(BASE);srcmaprow=int(pins[cr])
 wr=weights_by_node[volume_node]; lane_weights=[float(wr["w"+str(i)]) for i in range(86)];lane=int(np.argmax(lane_weights));conf=lane_weights[lane];lane_b=lane_body[lane]
 assert lane_b==body and conf>=DONOR_MIN
 base={"canonical_protected_row":cr,"mapping_kind":kind,"role":"D","base_volume_node_row":int(pins[cr]),"source330_parent_node_row":int(parent[int(pins[cr])]),"source330_parent_star_tets_json":star_json,"stable_attachment_id":aid,"canonical_donor_body":body,"full86_confidence_lane":lane,"full86_confidence_body":lane_b,"full86_confidence_weight":conf,"current_surface_vertex_row":surface_row,"target_world_x_m":float(target[0]),"target_world_y_m":float(target[1]),"target_world_z_m":float(target[2]),"source_to_target_residual_m":float(np.linalg.norm(target-old)),"source_map_path":srcmap,"source_map_sha256":srcmapsha,"source_map_row":srcmaprow,"source_face_path_operand_rows_json":source_path_json,"source_face_path_face_id_tags_json":face_tags_json,"source_face_path_hops":result_by_attachment[aid]["source_face_path_hops"] if is_remap else "","source_face_centroid_path_length_m":result_by_attachment[aid]["source_face_centroid_path_length_m"] if is_remap else "","target_to_source_face_residual_m":result_by_attachment[aid]["target_to_parent_source_face_residual_m"] if is_remap else "","source_face_barycentric_json":json.dumps(result_by_attachment[aid]["target_source_face_barycentric"]) if is_remap else "","candidate_surface_quality_json":quality_json,"binding_status":"candidate reference only; volume-node correspondence pending retet" if is_remap else "canonical original volume node retained; no physical admission"}
 rows.append(base)
rows.sort(key=lambda x:int(x["canonical_protected_row"]))
assert len(rows)==149 and len({int(x["canonical_protected_row"]) for x in rows})==149
assert {int(x["canonical_protected_row"]) for x in rows}==set(range(149))
assert {29464,14506} <= {int(x["stable_attachment_id"]) for x in rows if x["stable_attachment_id"]}
assert sum(bool(x["stable_attachment_id"]) for x in rows)==14
out_csv=OUT/"protected-149-targets.csv";fields=list(rows[0])
with out_csv.open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
target_json={"schema":"numi.protected-149-source-and-operational-target-map.v1","canonical_rows":149,"head_boundary_rows":135,"deep_attachment_rows":14,"remapped_deep_attachment_targets":list(result_by_attachment.values()),"physical_admission":False,"runtime":0,"accepted_steps":0,"limitation":"This packet authenticates surface reference targets and immutable source330 parent stars. The two remapped rows are surface vertices, not volume nodes; a candidate-specific retet must bind them and pass all child-cell, mass, thickness, contact, and local tetra-altitude gates. No physical admission or runtime is claimed."}
(OUT/"deep-target-details.json").write_text(json.dumps(target_json,indent=2)+"\n")
inputs=[Path(__file__),HEAD,HEAD_RECEIPT,SKIN_MAP,FAT_MAP,ATT,PARENT,Path(str(PARENT)+".weights.csv"),Path(str(PARENT)+".tets.csv"),Path(str(PARENT)+".labels.csv"),POSE,BASE,C,SEARCH,RECHECK,REPLAY,*BANK.values()]
receipt={"schema":"numi.protected-149-candidate-target-packet-receipt.v1","output_csv":str(out_csv),"output_csv_sha256":sha(out_csv),"deep_details_json":str(OUT/"deep-target-details.json"),"deep_details_sha256":sha(OUT/"deep-target-details.json"),"input_sha256":{str(x):sha(x) for x in inputs},"head_receipt_sha256":head_receipt["csv_sha256"],"canonical_protected_rows":149,"head_source_ancestry":{"O_I_map_path":str(SKIN_MAP),"D_map_path":str(FAT_MAP),"O_I_role_rows_exact":True,"D_fine_source_rows_exact":True},"deep_attachment_candidates":{"29464":result_by_attachment[29464],"14506":result_by_attachment[14506]},"construction_bounds":{"max_old_to_target_euclidean_m":MAX_TARGET_OFFSET_M,"cube_envelope_max_residual_m":ENVELOPE_MAX_OFFSET_M,"full86_donor_min_weight":DONOR_MIN},"status":"source-only candidate target packet; physical admission false","physical_admission":False,"runtime":0,"accepted_steps":0,"elapsed_s":time.time()-t0}
(OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"output_csv":str(out_csv),"output_csv_sha256":sha(out_csv),"receipt":str(OUT/"receipt.json"),"receipt_sha256":sha(OUT/"receipt.json"),"deep_details_json":str(OUT/"deep-target-details.json"),"deep_details_sha256":sha(OUT/"deep-target-details.json"),"targets":[{"id":k,"vertex":v,"source_residual_um":result_by_attachment[k]["target_to_parent_source_face_residual_m"]*1e6,"anchor_displacement_mm":result_by_attachment[k]["old_reference_to_target_displacement_m"]*1e3,"source_face_hops":result_by_attachment[k]["source_face_path_hops"],"dual_centroid_path_mm":result_by_attachment[k]["source_face_centroid_path_length_m"]*1e3,"full86":result_by_attachment[k]["full86_confidence_weight"]} for k,v in TARGETS.items()],"physical_admission":False,"runtime":0,"accepted_steps":0},indent=2))
