#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D")
W=ROOT/"connected-obstacle-207/physical-transfer-209"
OUT=W/"root-connected-layer-fit-219/canonical-registration-diagnostic-228/attempt-009-physicalO-sampled-clearance-185bank"
O_PATH=W/"physical-outer-direct-placement-235/O-direct-native-placement.npz"
O_RECEIPT=W/"physical-outer-direct-placement-235/report.json"
POSE_PATH=W/"P/native-kind-aware-source-preflight-226/run-006-native-emission/registered-skin-poses-100.json"
QMOD=W/"geometry-worker/D-connectivity-218/bone-distance-query-001/bone_distance_query.py"
MARGIN_M=5e-6

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def current_world_to_body(q_xyzw):
    q=np.asarray(q_xyzw,dtype=np.float32).astype(np.float64).copy()
    q[:3]*=-1.0
    def rotate(v):
        return v+2.0*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
    return np.column_stack([rotate(v) for v in np.eye(3)])

def witness_meta(sample_index,nv,nf,edges,edge_offsets,edge_faces,z):
    i=int(sample_index)
    if i<nv:
        src=int(z["candidate_vertex_to_O033_row"][i]) if "candidate_vertex_to_O033_row" in z else -1
        return {"sample_kind":"vertex","sample_index":i,"source_O033_vertex_row":src}
    if i<nv+nf:
        fi=i-nv
        a=int(z["face_ancestry_offsets"][fi]);b=int(z["face_ancestry_offsets"][fi+1])
        return {"sample_kind":"face_centroid","sample_index":fi,"source_original_id":int(z["envelope_source_original_id"][fi]),"source_face_row":int(z["envelope_source_face_row"][fi]),"source_ancestry_parent_face_rows":np.asarray(z["face_ancestry_parent_face_rows"][a:b],dtype=np.int64).tolist()}
    ei=i-nv-nf
    a=int(edge_offsets[ei]);b=int(edge_offsets[ei+1])
    adjacent=np.asarray(edge_faces[a:b],dtype=np.int64)
    return {"sample_kind":"unique_edge_midpoint","sample_index":ei,"edge_vertices":edges[ei].astype(int).tolist(),"adjacent_face_rows":adjacent.tolist(),"adjacent_source_face_rows":np.asarray(z["envelope_source_face_row"][adjacent],dtype=np.int64).tolist()}

def main():
    start=time.monotonic()
    direct=json.loads(O_RECEIPT.read_text())
    if sha(O_PATH)!=direct.get("output_sha256"): raise ValueError("physical O hash differs from direct-placement receipt")
    z=np.load(O_PATH,allow_pickle=False)
    P=np.asarray(z["points_world_m"],dtype=np.float32)
    F=np.asarray(z["source_O_faces"],dtype=np.int64)
    if not np.array_equal(F,np.asarray(z["faces"],dtype=np.int64)): raise ValueError("O faces/source_O_faces mismatch")
    if not np.isfinite(P).all() or F.min()<0 or F.max()>=len(P): raise ValueError("invalid direct O input")
    nv,nf=len(P),len(F)
    # Sample all vertices, all face centroids, and each unique mesh edge midpoint, in Float32.
    tri=P[F].astype(np.float64)
    face_centers=tri.mean(axis=1).astype(np.float32)
    raw_edges=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]],axis=0)
    sorted_edges=np.sort(raw_edges,axis=1)
    edges,edge_inverse=np.unique(sorted_edges,axis=0,return_inverse=True)
    face_edge_ids=edge_inverse.reshape(3,nf).T
    edge_face_raw=np.tile(np.arange(nf,dtype=np.int64),3)
    order=np.argsort(edge_inverse,kind="stable")
    edge_face_sorted=edge_face_raw[order]
    counts=np.bincount(edge_inverse,minlength=len(edges))
    edge_offsets=np.concatenate([[0],np.cumsum(counts)]).astype(np.int64)
    edge_midpoints=((P[edges[:,0]].astype(np.float64)+P[edges[:,1]].astype(np.float64))*0.5).astype(np.float32)
    sample_points=np.concatenate([P,face_centers,edge_midpoints],axis=0).astype(np.float32,copy=False)
    if not np.isfinite(sample_points).all(): raise ValueError("nonfinite F32 sample points")
    pose=json.loads(POSE_PATH.read_text())
    pose_map={int(x["body_index"]):x for x in pose["all_bodies"]}
    sys.path.insert(0,str(QMOD.parent))
    from bone_distance_query import BoneDistanceQuery
    query=BoneDistanceQuery()
    if query.bank_count!=185: raise ValueError(f"expected 185 bank BVHs, got {query.bank_count}")
    body_ids=set(int(b["body"]) for b in query._banks)
    if body_ids-set(pose_map): raise ValueError(f"native pose export missing bank bodies {sorted(body_ids-set(pose_map))}")
    pose_comp=[]
    for bank in query._banks:
        body=int(bank["body"])
        cur=pose_map[body]["current"]
        qpos=np.asarray(cur["position_world_m"],dtype=np.float32)
        qquat=np.asarray(cur["orientation_world_xyzw"],dtype=np.float32)
        M39=np.asarray(bank["world_to_body"],dtype=np.float64)
        Mcur=current_world_to_body(qquat)
        bank["world_to_body"]=Mcur
        bank["body_to_world_row"]=np.linalg.inv(Mcur).T
        bank["position_world_m"]=qpos.astype(np.float64)
        pose_comp.append({"body":body,"matrix_delta":float(np.max(np.abs(Mcur-M39)))})
    # Per-bank branch-and-bound: exact AABB lower bounds prune points that cannot beat
    # the nearest positive clearance; all points inside the bank AABB are queried for inside tests.
    rows=[];total_queries=0;all_near=[];aabb_active=0
    for bi,bank in enumerate(query._banks,1):
        body=int(bank["body"]);bone=int(bank["stable_bone_id"])
        Pworld=np.ascontiguousarray(sample_points.astype(np.float64))
        local=np.ascontiguousarray((Pworld-bank["position_world_m"])@bank["world_to_body"].T-bank["com_offset_m"],dtype=np.float64)
        V=bank["vertices_local_m"];TF=bank["faces"]
        B=V[TF]
        lo=B.min(axis=(0,1));hi=B.max(axis=(0,1))
        delta=np.maximum(np.maximum(lo-local,local-hi),0.0)
        lower=np.linalg.norm(delta,axis=1)
        inside_aabb=np.flatnonzero(lower==0.0)
        dist=np.full(len(sample_points),np.nan,dtype=np.float64)
        signed=np.full(len(sample_points),np.nan,dtype=np.float64)
        winding=np.full(len(sample_points),np.nan,dtype=np.float64)
        closest=np.full((len(sample_points),3),np.nan,dtype=np.float64)
        face_idx=np.full(len(sample_points),-1,dtype=np.int64)
        queried=np.zeros(len(sample_points),dtype=bool)
        def query_idx(idx):
            nonlocal total_queries
            idx=np.asarray(idx,dtype=np.int64)
            if not len(idx): return
            pts=np.ascontiguousarray(local[idx],dtype=np.float64)
            d2,fi,cp=bank["distance_bvh"].squared_distance(V,TF,pts)
            wn=np.asarray(bank["winding_bvh"].winding_number(pts,2.0),dtype=np.float64).reshape(-1)
            dd=np.sqrt(np.maximum(np.asarray(d2,dtype=np.float64),0.0))
            ab=np.abs(wn)
            neg=(ab>=0.45) # match conservative closed-bank/ambiguous policy used by the pinned owner
            dist[idx]=dd;signed[idx]=np.where(neg,-dd,dd);winding[idx]=wn;closest[idx]=np.asarray(cp,dtype=np.float64);face_idx[idx]=np.asarray(fi,dtype=np.int64);queried[idx]=True
            total_queries+=len(idx)
        query_idx(inside_aabb)
        aabb_active+=int(bool(len(inside_aabb)))
        # Positive nearest sample: query outside-AABB points in ascending AABB lower bound
        # until no unqueried lower bound can beat the current exact point-to-triangle distance.
        def best_positive():
            mask=queried & np.isfinite(signed) & (signed>0.0)
            if not mask.any(): return float("inf"),-1
            idx=np.flatnonzero(mask);j=int(idx[np.argmin(dist[idx])])
            return float(dist[j]),j
        best, best_idx=best_positive()
        outside=np.flatnonzero(lower>0.0)
        if len(outside):
            order_out=outside[np.argsort(lower[outside],kind="stable")]
            cursor=0
            while cursor<len(order_out):
                if best_idx>=0 and lower[order_out[cursor]]>=best: break
                batch=order_out[cursor:cursor+2048]
                query_idx(batch);cursor+=len(batch)
                best,best_idx=best_positive()
        # Query every point whose AABB lower bound could be within the requested 5 um threshold.
        threshold_idx=np.flatnonzero((lower<=MARGIN_M)&(~queried))
        query_idx(threshold_idx)
        best,best_idx=best_positive()
        signed_query=np.flatnonzero(queried)
        inside_count=int(np.count_nonzero(np.abs(winding[signed_query])>=0.55))
        ambiguous_count=int(np.count_nonzero((np.abs(winding[signed_query])>=0.45)&(np.abs(winding[signed_query])<0.55)))
        near_positive=signed_query[(signed[signed_query]>=0)&(signed[signed_query]<=MARGIN_M)]
        negative=signed_query[signed[signed_query]<0]
        min_signed_idx=int(negative[np.argmin(signed[negative])]) if len(negative) else -1
        min_positive_witness=None
        if best_idx>=0:
            fi=int(face_idx[best_idx])
            min_positive_witness={"sample":witness_meta(best_idx,nv,nf,edges,edge_offsets,edge_face_sorted,z),"unsigned_distance_m":float(dist[best_idx]),"signed_distance_m":float(signed[best_idx]),"winding_number":float(winding[best_idx]),"closest_bone_point_world_m":((closest[best_idx]+bank["com_offset_m"])@bank["body_to_world_row"]+bank["position_world_m"]).tolist(),"bone_triangle_stable_id":int(bank["triangle_stable_id"][fi]),"bone_source_record_index":int(bank["source_record_index"][fi])}
        min_signed_witness=None
        if min_signed_idx>=0:
            fi=int(face_idx[min_signed_idx])
            min_signed_witness={"sample":witness_meta(min_signed_idx,nv,nf,edges,edge_offsets,edge_face_sorted,z),"signed_distance_m":float(signed[min_signed_idx]),"unsigned_distance_m":float(dist[min_signed_idx]),"winding_number":float(winding[min_signed_idx]),"bone_triangle_stable_id":int(bank["triangle_stable_id"][fi])}
        near_rows=[]
        for idx in near_positive[:1000]:
            fi=int(face_idx[idx])
            near_rows.append({"sample":witness_meta(idx,nv,nf,edges,edge_offsets,edge_face_sorted,z),"signed_distance_m":float(signed[idx]),"winding_number":float(winding[idx]),"bone_triangle_stable_id":int(bank["triangle_stable_id"][fi])})
        row={"body":body,"stable_bone_id":bone,"total_sample_points":int(len(sample_points)),"queried_points_for_minimum":int(len(signed_query)),"aabb_inside_sample_count":int(len(inside_aabb)),"outside_aabb_points_queried_until_minimum_certified":int(np.count_nonzero(queried & (lower>0))),"positive_minimum_sampled_clearance_m":float(best) if best_idx>=0 else None,"positive_minimum_witness":min_positive_witness,"most_negative_sampled_signed_distance_m":float(signed[min_signed_idx]) if min_signed_idx>=0 else None,"most_negative_witness":min_signed_witness,"inside_or_ambiguous_point_count":inside_count+ambiguous_count,"inside_winding_sample_count":inside_count,"ambiguous_winding_sample_count":ambiguous_count,"positive_samples_within_5um_count":int(len(near_positive)),"positive_samples_within_5um":near_rows}
        rows.append(row)
        if row["positive_minimum_sampled_clearance_m"] is not None and row["positive_minimum_sampled_clearance_m"]<=MARGIN_M:all_near.append({"body":body,"stable_bone_id":bone,"clearance_m":row["positive_minimum_sampled_clearance_m"],"witness":min_positive_witness})
        if bi%20==0 or bi==len(query._banks):
            (OUT/"progress.json").write_text(json.dumps({"completed_banks":bi,"expected_banks":185,"query_points_so_far":total_queries,"near_5um_banks":len(all_near),"elapsed_s":time.monotonic()-start})+"\n")
    positives=[r for r in rows if r["positive_minimum_sampled_clearance_m"] is not None]
    nearest=min(positives,key=lambda r:r["positive_minimum_sampled_clearance_m"]) if positives else None
    report={"schema":"physical-O235-current-pose-sampled-185-bank-clearance.v1","status":"COMPLETE_SAMPLED_CLEARANCE_DIAGNOSTIC","scope":"All Float32 physical O vertices, all Float32 face centroids and all unique-edge midpoints queried against 185 typed banks; body transforms rebound to native current all_bodies; query margin 5um with AABB branch-and-bound; not an exact surface or solid admission gate.","physical_O_path":str(O_PATH),"physical_O_sha256":sha(O_PATH),"physical_O_receipt_sha256":sha(O_RECEIPT),"physical_O_point_count":int(nv),"physical_O_face_count":int(nf),"unique_edge_count":int(len(edges)),"sample_counts":{"vertices":int(nv),"face_centroids":int(nf),"unique_edge_midpoints":int(len(edges)),"total":int(len(sample_points))},"sample_cast_policy":"Use exact Float32 physical O positions; compute face centers/edge midpoints in Float64 from Float32 endpoints then cast every sample to Float32 before frame transforms and queries.","native_pose_path":str(POSE_PATH),"native_pose_sha256":sha(POSE_PATH),"native_all_bodies_count":len(pose_map),"bank_count":query.bank_count,"bank_hashes":query.bank_hashes,"bank_pose_rebind":"Each body transform/position uses native all_bodies Float32 current quaternion/position; pinned local triangles and COM offsets remain unchanged.","pose_matrix_delta_max":max(x["matrix_delta"] for x in pose_comp),"query_margin_m":MARGIN_M,"total_point_bank_queries":int(total_queries),"aabb_active_bank_count":aabb_active,"global_min_positive_sampled_clearance_m":None if nearest is None else nearest["positive_minimum_sampled_clearance_m"],"global_min_positive_witness":None if nearest is None else {"body":nearest["body"],"stable_bone_id":nearest["stable_bone_id"],**nearest["positive_minimum_witness"]},"near_5um_bank_count":len(all_near),"near_5um_banks":all_near,"banks":rows,"elapsed_s":time.monotonic()-start,"physical_admission":False,"limitations":["Minima are over vertices, face centroids, and unique-edge midpoints, not all points on triangles; this can miss between-sample closest approaches.","Signs use the pinned Fast Winding Number policy; ambiguous samples are reported separately. The exact all-surface crossing audit is separate.","This report does not establish a solid containment certificate, layer thickness, positive cell Jacobians, mass conservation, native dynamics, or admission."]}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"progress.json").unlink(missing_ok=True)
    print(json.dumps({"status":report["status"],"samples":report["sample_counts"],"banks":query.bank_count,"global_min_positive_m":report["global_min_positive_sampled_clearance_m"],"global_min_positive_witness":report["global_min_positive_witness"],"near_5um_bank_count":report["near_5um_bank_count"],"elapsed_s":report["elapsed_s"],"report":str(OUT/"report.json")},indent=2))

if __name__=="__main__":main()
