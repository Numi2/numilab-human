#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time
from collections import defaultdict, Counter
from fractions import Fraction
from pathlib import Path
import numpy as np

D=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
P=D/"connected-obstacle-207/physical-transfer-209"
OUT=P/"root-connected-layer-fit-219/canonical-registration-diagnostic-228/attempt-007-parent-tet-bone-current-pose"
VOL=P/"P/native-kind-aware-source-preflight-226/run-006-native-emission"
BANKHELPER=Path("/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py")
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
EXACT=E/"collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py"
BANKDIR=E/"bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BANKKEYS=[(59,95),(92,15)] # highest reported I penetration and next independent D penetration region
FIT_REPORT=P/"root-connected-layer-fit-219/canonical-registration-diagnostic-228/attempt-005-connected-field-fit/fit-run-001/report.json"
REGION_RADIUS_M=0.010


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f"cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def read_dict_csv(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))

def exact_records(exact, triangles, face_ids):
    pts=[tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(triangles).reshape(-1,3)]
    faces=np.arange(len(pts),dtype=np.int64).reshape(-1,3)
    return [(tri,lo,hi,int(face_ids[i]),ids0) for i,(tri,lo,hi,row,ids0) in enumerate(exact._records(pts,faces))]

def exact_pairs(exact, surface_local, surface_ids, bone_local, bone_ids):
    S=np.asarray(surface_local,dtype=np.float64); B=np.asarray(bone_local,dtype=np.float64)
    if not len(S) or not len(B): return {"skin_aabb_active":0,"bone_aabb_active":0,"exact_pairs":[]}
    blo,bhi=B.min((0,1)),B.max((0,1)); slo,shi=S.min(1),S.max(1)
    active=np.flatnonzero(np.all(shi>=blo,axis=1)&np.all(slo<=bhi,axis=1))
    if not len(active): return {"skin_aabb_active":0,"bone_aabb_active":0,"exact_pairs":[]}
    rlo,rhi=slo[active].min(0),shi[active].max(0)
    ba=np.flatnonzero(np.all(B.max(1)>=rlo,axis=1)&np.all(B.min(1)<=rhi,axis=1))
    sr=exact_records(exact,S[active],np.asarray(surface_ids)[active])
    br=exact_records(exact,B[ba],np.asarray(bone_ids)[ba])
    result=exact._audit_pair(sr,br,same_surface=False)
    pairs=[]
    for a,b in result.get("triangle_pairs",[]):
        pairs.append({"parent_tet_face_id":int(a),"bone_triangle_stable_id":int(b)})
    return {"skin_aabb_active":int(len(active)),"bone_aabb_active":int(len(ba)),"aabb_candidate_pairs":int(result.get("aabb_candidate_pairs",0)),"exact_pairs":pairs}

start=time.monotonic()
# Read exact native-emitted parent cell arrays. Current positions are emitted F32 decimal values.
node_rows=read_dict_csv(VOL/"compiled-parent-cell-mass.csv.nodes.csv")
tet_rows=read_dict_csv(VOL/"compiled-parent-cell-mass.csv.tets.csv")
label_rows=read_dict_csv(VOL/"compiled-parent-cell-mass.csv.labels.csv")
world=np.asarray([[float(r[k]) for k in ("current_x_m","current_y_m","current_z_m")] for r in node_rows],dtype=np.float32).astype(np.float64)
fit_report=json.loads(FIT_REPORT.read_text())
fit_iter=fit_report["iterations"][0]
fit_witnesses=fit_iter["unmovable_witnesses"]
fit_domains=fit_iter["constraint_domains"]
tets=np.asarray([[int(r[f"parent_node{i}"]) for i in range(4)] for r in tet_rows],dtype=np.int64)
materials=np.asarray([int(r["material_index"]) for r in tet_rows],dtype=np.int32)
donors=np.asarray([int(r["donor_body"]) for r in tet_rows],dtype=np.int32)
labels={int(r["parent_tet_row"]):{"anatomical_region_id":int(r["anatomical_region_id"]),"stable_source_face_id":int(r["stable_source_face_id"])} for r in label_rows}

# Enumerate each actual tetra face once for the narrowphase packet while retaining its tet row.
local_faces=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
face_tris=[]; face_ids=[]; face_meta=[]
for tr,tet in enumerate(tets):
    for lf,ix in enumerate(local_faces):
        face_tris.append(tet[list(ix)])
        face_ids.append(tr*4+lf)
        face_meta.append((tr,lf))
face_tris=np.asarray(face_tris,dtype=np.int64)
face_ids=np.asarray(face_ids,dtype=np.int64)

# Derive actual material boundary and interface counts for context.
incidence=defaultdict(list)
for tr,tet in enumerate(tets):
    for lf,ix in enumerate(local_faces):
        tri=tuple(int(x) for x in tet[list(ix)])
        incidence[tuple(sorted(tri))].append((tr,int(materials[tr]),lf))
face_class_counts=Counter()
class_face_ids=defaultdict(list)
for key,occ in incidence.items():
    mats={x[1] for x in occ}
    if len(occ)==2 and mats=={15,16}:
        kind="I_mat15_16"
    elif len(occ)==1 and occ[0][1]==15:
        kind="O_mat15_boundary"
    elif len(occ)==1 and occ[0][1]==16:
        kind="D_mat16_boundary"
    elif len(occ)==1:
        kind=f"boundary_mat{occ[0][1]}"
    elif len(occ)>2:
        kind="nonmanifold_face"
    else:
        kind="internal_same_material"
    face_class_counts[kind]+=1
    if kind!="internal_same_material": class_face_ids[kind].append((key,occ))

# Pinned bank parser and exact narrowphase owner, same helpers as prior approved local audits.
sys.path.insert(0,str(E/"incremental-exact-audit-owner-253/src"))
helper=module("connected_bone_bank_159",BANKHELPER)
exact=module("numilab_human.connected_skin_bone_exact_159",EXACT)
T,ids,gids,bodies,M,off,pos,bankkeys,pins,unused=helper.read_bank_inputs(BANKDIR,E)
if len(bankkeys)!=185: raise ValueError(f"expected 185 typed banks, got {len(bankkeys)}")
native_pose_path=VOL/"registered-skin-poses-100.json"
native_pose=json.loads(native_pose_path.read_text())
native_bodies={int(x["body_index"]):x for x in native_pose["all_bodies"]}
bank_body_ids=set(int(x) for x in np.unique(ids[:,0]))
missing_pose_ids=sorted(bank_body_ids-set(native_bodies))
if missing_pose_ids: raise ValueError(f"native all_bodies pose export missing bank body ids: {missing_pose_ids}")
def current_world_to_body(q_xyzw):
    q=np.asarray(q_xyzw,dtype=np.float32).astype(np.float64).copy()
    q[:3]*=-1.0
    def rotate(v):
        return v+2.0*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
    return np.column_stack([rotate(v) for v in np.eye(3)])
pose_comparisons=[]
for body in sorted(bank_body_ids):
    cur=native_bodies[body]["current"]
    qpos=np.asarray(cur["position_world_m"],dtype=np.float32)
    qquat=np.asarray(cur["orientation_world_xyzw"],dtype=np.float32)
    bpos=np.asarray([float(bodies[body][f"position_{c}_m"]) for c in "xyz"],dtype=np.float32)
    bquat=np.asarray([float(bodies[body][f"orientation_q{c}"]) for c in "xyzw"],dtype=np.float32)
    qM=current_world_to_body(qquat)
    pose_comparisons.append({"body":body,"position_bitwise_equal_to_pinned039":bool(np.array_equal(qpos.view(np.uint32),bpos.view(np.uint32))),"orientation_bitwise_equal_to_pinned039":bool(np.array_equal(qquat.view(np.uint32),bquat.view(np.uint32))),"position_delta_m":float(np.linalg.norm(qpos.astype(np.float64)-bpos.astype(np.float64))),"world_to_body_max_abs_delta":float(np.max(np.abs(qM-M[body])))})

# Load libigl only for the selected bank point-sign checks, preserving the approved pinned frames.
authoring_deps=D/"connected-obstacle-207/authoring-deps"
sys.path.insert(0,str(authoring_deps))
import igl

results=[]
for body0,bone0 in BANKKEYS:
    body,bone=int(body0),int(bone0)
    mask=(ids[:,0]==body)&(ids[:,1]==bone)
    B=np.asarray(T[mask],dtype=np.float64)
    bank_ids=np.asarray(ids[mask,3],dtype=np.int64)
    if not len(B): raise ValueError(f"missing bank {(body,bone)}")
    current=native_bodies[body]["current"]
    world_to_body=current_world_to_body(current["orientation_world_xyzw"])
    off_body=np.asarray(off[body],dtype=np.float64) # pinned COM offset; local triangles retained exactly
    position=np.asarray(current["position_world_m"],dtype=np.float32).astype(np.float64)
    local=(world-position)@world_to_body.T-off_body
    # Choose the actual worst stored I/D witness for this bank as the local diagnostic center.
    matched=[]
    for w in fit_witnesses:
        if int(w["body"])!=body or int(w["bone_id"])!=bone: continue
        for dom in fit_domains:
            if dom["row_begin"]<=w["constraint_row"]<dom["row_end"] and dom.get("layer") in ("I","D"):
                q=dict(w); q["layer"]=dom["layer"]; matched.append(q); break
    if not matched: raise ValueError(f"no stored I/D witness for bank {(body,bone)}")
    worst=max(matched,key=lambda x:x["target_m"])
    center_world=np.asarray(worst["sample_points_world_m"],dtype=np.float64)
    center_local=(center_world-position)@world_to_body.T-off_body
    r=REGION_RADIUS_M
    region_lo,region_hi=center_local-r,center_local+r
    # Select only parent tetrahedra whose body-local AABB overlaps the 20 mm witness region.
    tet_local=local[tets]
    tet_lo,tet_hi=tet_local.min(axis=1),tet_local.max(axis=1)
    candidate_tets=np.flatnonzero(np.all(tet_hi>=region_lo,axis=1)&np.all(tet_lo<=region_hi,axis=1))
    candidate_face_ids=np.concatenate([np.arange(int(tr)*4,int(tr)*4+4,dtype=np.int64) for tr in candidate_tets]) if len(candidate_tets) else np.empty(0,dtype=np.int64)
    local_faces=local[face_tris]
    # AABBs further narrow selected triangles to this local region, then exact rational triangle tests.
    face_lo,face_hi=local_faces[candidate_face_ids].min(axis=1),local_faces[candidate_face_ids].max(axis=1)
    face_keep=np.flatnonzero(np.all(face_hi>=region_lo,axis=1)&np.all(face_lo<=region_hi,axis=1))
    candidate_face_ids=candidate_face_ids[face_keep]
    bank_lo,bank_hi=B.min(1),B.max(1)
    bank_keep=np.flatnonzero(np.all(bank_hi>=region_lo,axis=1)&np.all(bank_lo<=region_hi,axis=1))
    B_region=B[bank_keep]
    bank_ids_region=bank_ids[bank_keep]
    narrow=exact_pairs(exact,local_faces[candidate_face_ids],face_ids[candidate_face_ids],B_region,bank_ids_region)
    pairs=[]
    for pair in narrow["exact_pairs"]:
        face_id=pair["parent_tet_face_id"]; tr,lf=divmod(face_id,4)
        pairs.append({**pair,"parent_tet_row":tr,"local_tet_face":lf,"material_index":int(materials[tr]),"donor_body":int(donors[tr]),**labels.get(tr,{})})
    # Exact crossings on actual O, material-15/16 interface (I), and D volume-boundary facets.
    boundary_hits=[]
    for kind,faces0 in class_face_ids.items():
        rows=[]
        for key,occ in faces0:
            nodes0=np.asarray(key,dtype=np.int64)
            tri_local=local[nodes0]
            if np.all(tri_local.max(0)>=region_lo)&np.all(tri_local.min(0)<=region_hi):
                rows.append((nodes0,occ))
        if not rows: continue
        S=np.asarray([local[x[0]] for x in rows],dtype=np.float64)
        face_ids_boundary=np.arange(len(rows),dtype=np.int64)
        q=exact_pairs(exact,S,face_ids_boundary,B_region,bank_ids_region)
        mapped=[]
        for pp in q["exact_pairs"]:
            nodes0,occ=rows[int(pp["parent_tet_face_id"])]
            mapped.append({"bone_triangle_stable_id":pp["bone_triangle_stable_id"],"parent_node_rows":nodes0.tolist(),"incident_tets":[{"parent_tet_row":int(tr),"material_index":int(mat),"local_tet_face":int(lf),"donor_body":int(donors[tr]),**labels.get(int(tr),{})} for tr,mat,lf in occ]})
        boundary_hits.append({"surface_class":kind,"candidate_surface_faces":len(rows),"exact_pairs":mapped,"aabb_candidate_pairs":q.get("aabb_candidate_pairs",0)})
    # Query actual parent tet vertices/centroids plus stored detailed I/D witnesses against the full bank.
    bank_vertices=np.ascontiguousarray(B.reshape(-1,3),dtype=np.float64)
    bank_faces=np.arange(len(bank_vertices),dtype=np.int64).reshape(-1,3)
    dist_bvh=igl.AABB(); dist_bvh.init(bank_vertices,bank_faces)
    winding_bvh=igl.FastWindingNumberBVH(); winding_bvh.init(bank_vertices,bank_faces,2)
    tet_points=[]; tet_point_meta=[]
    for tr in candidate_tets:
        for node in tets[int(tr)]:
            tet_points.append(local[int(node)])
            tet_point_meta.append((int(tr),"vertex",int(node)))
        tet_points.append(local[tets[int(tr)]].mean(axis=0))
        tet_point_meta.append((int(tr),"centroid",-1))
    tet_points=np.ascontiguousarray(tet_points,dtype=np.float64)
    d2,nearest_face,closest=dist_bvh.squared_distance(bank_vertices,bank_faces,tet_points)
    wind=np.asarray(winding_bvh.winding_number(tet_points,2.0),dtype=np.float64)
    signed=np.where(np.abs(wind)>=0.5,-np.sqrt(np.maximum(d2,0.0)),np.sqrt(np.maximum(d2,0.0)))
    sample_rows=[]
    for w in sorted(matched,key=lambda x:x["target_m"],reverse=True)[:20]:
        p_world=np.asarray(w["sample_points_world_m"],dtype=np.float64).reshape(1,3)
        p_local=np.ascontiguousarray((p_world-position)@world_to_body.T-off_body)
        q2,qfi,qcp=dist_bvh.squared_distance(bank_vertices,bank_faces,p_local)
        qw=float(np.asarray(winding_bvh.winding_number(p_local,2.0)).reshape(-1)[0])
        sample_rows.append({"layer":w["layer"],"constraint_row":int(w["constraint_row"]),"required_displacement_m":float(w["target_m"]),"world_point_m":p_world[0].tolist(),"signed_distance_m":float(-np.sqrt(max(float(q2[0]),0.0)) if abs(qw)>=0.5 else np.sqrt(max(float(q2[0]),0.0))),"unsigned_distance_m":float(np.sqrt(max(float(q2[0]),0.0))),"winding_number":qw,"node_rows":w["node_rows"],"barycentric":w["barycentric"]})
    inside_by_material=Counter()
    tet_point_witnesses=[]
    for qrow,(tr,kind,node) in enumerate(tet_point_meta):
        if abs(float(wind[qrow]))>=0.5:
            mat=int(materials[tr]); inside_by_material[str(mat)]+=1
            if len(tet_point_witnesses)<100:
                tet_point_witnesses.append({"parent_tet_row":tr,"point_kind":kind,"parent_node_row":node,"material_index":mat,"donor_body":int(donors[tr]),"signed_distance_m":float(signed[qrow]),"winding_number":float(wind[qrow]),"world_point_m":((tet_points[qrow]+off_body)@np.linalg.inv(world_to_body).T+position).tolist(),**labels.get(tr,{})})
    # Check whether this bone triangle soup actually closes; winding signs are conditional on closure.
    edge_count=Counter()
    for tri in B:
        vv=[tuple(float(x) for x in v) for v in tri]
        for a,b in ((vv[0],vv[1]),(vv[1],vv[2]),(vv[2],vv[0])): edge_count[tuple(sorted((a,b)))]+=1
    edge_degrees=Counter(edge_count.values())
    # Re-run query on exact canonical surface witnesses only as context, not as a broad audit.
    results.append({"body":body,"stable_bone_id":bone,"bank_triangle_count":len(B),"bank_topology":{"unique_triangle_soup_edges":len(edge_count),"edge_incidence_histogram":dict(edge_degrees),"boundary_edge_count":edge_degrees.get(1,0),"nonmanifold_edge_count":sum(v for k,v in edge_degrees.items() if k>2),"winding_sign_closedness_caveat":"signed inside/outside is treated as a solid classification only if boundary_edge_count and nonmanifold_edge_count are zero"},"selected_fit_witness":{"layer":worst["layer"],"constraint_row":int(worst["constraint_row"]),"required_displacement_m":float(worst["target_m"]),"world_point_m":center_world.tolist(),"local_region_halfwidth_m":r},"region_candidate_parent_tet_count":int(len(candidate_tets)),"region_candidate_tet_face_count":int(len(candidate_face_ids)),"region_candidate_bone_face_count":int(len(bank_keep)),"tet_face_exact_audit":{"skin_aabb_active":narrow["skin_aabb_active"],"bone_aabb_active":narrow["bone_aabb_active"],"aabb_candidate_pairs":narrow.get("aabb_candidate_pairs",0),"exact_pair_count":len(pairs),"witnesses":pairs[:100]},"material_boundary_exact_audits":boundary_hits,"region_tet_sample_summary":{"sample_count":len(tet_points),"inside_winding_count_by_material":dict(inside_by_material),"ambiguous_winding_count_0p45_to_0p55":int(np.count_nonzero((np.abs(wind)>=0.45)&(np.abs(wind)<0.55))),"inside_winding_sample_witnesses":tet_point_witnesses},"stored_I_D_fit_witness_sign_checks":sample_rows})

out={"schema":"parent-tet-two-typed-bank-exact-diagnostic.v1","scope":"two selected typed bone banks only; exact triangle narrowphase over parent tetra faces whose local AABBs overlap selected banks; no full-scene audit or admission","native_cell_files":{"nodes_sha256":sha(VOL/"compiled-parent-cell-mass.csv.nodes.csv"),"tets_sha256":sha(VOL/"compiled-parent-cell-mass.csv.tets.csv"),"labels_sha256":sha(VOL/"compiled-parent-cell-mass.csv.labels.csv")},"counts":{"nodes":len(world),"tets":len(tets),"materials":dict(Counter(map(int,materials))),"all_tet_faces":len(face_ids),"material_face_classes":dict(face_class_counts)},"bank_parser_sha256":sha(BANKHELPER),"exact_owner_sha256":sha(EXACT),"bank_pins":pins,"native_pose_path":str(native_pose_path),"native_pose_sha256":sha(native_pose_path),"native_pose_body_ids":len(native_bodies),"bank_body_ids":len(bank_body_ids),"missing_bank_body_ids":missing_pose_ids,"pose_rebind_policy":"retained pinned 039 local triangle and COM-offset arrays; re-bound position and world_to_body from current all_bodies entries in native registered-skin-poses-100.json using Float32 quaternion/position promoted to Float64","pinned039_pose_comparison":{"bitwise_position_matches":sum(x["position_bitwise_equal_to_pinned039"] for x in pose_comparisons),"bitwise_orientation_matches":sum(x["orientation_bitwise_equal_to_pinned039"] for x in pose_comparisons),"max_position_delta_m":max(x["position_delta_m"] for x in pose_comparisons),"max_world_to_body_delta":max(x["world_to_body_max_abs_delta"] for x in pose_comparisons),"per_body":pose_comparisons},"selected_results":results,"elapsed_s":time.monotonic()-start,"physical_admission":False}
(OUT/"report.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps({"report":str(OUT/"report.json"),"counts":out["counts"],"banks":[{"body":r["body"],"stable_bone_id":r["stable_bone_id"],"region_candidate_tets":r["region_candidate_parent_tet_count"],"region_candidate_tet_faces":r["region_candidate_tet_face_count"],"exact_tet_pairs":r["tet_face_exact_audit"]["exact_pair_count"],"boundary_exact_pairs":{x["surface_class"]:len(x["exact_pairs"]) for x in r["material_boundary_exact_audits"]}} for r in results],"elapsed_s":out["elapsed_s"]},indent=2))
