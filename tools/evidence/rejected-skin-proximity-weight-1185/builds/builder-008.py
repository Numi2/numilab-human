#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, math, struct, sys, time, runpy
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/Users/n/numi-human-common-skin-multipose-001")
sys.path.insert(0, str(ROOT / "src"))
from numilab_human.skin_source_payload_preflight import decode_payload
from numilab_human.skin_surface_binding import reweight_registered_bone_proximity

E = Path("/Users/n/numi-human-resting-evidence-20261005")
OUT = E / "native-common-skin-bone-proximity-weight-candidate-1185/candidate-build-008"
if OUT.exists():
    raise SystemExit(f"refuse existing output: {OUT}")
OUT.mkdir(parents=True)
SKIN = E / "native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_MAN = SKIN.with_name("common-atlas-skin-geometry-registration.manifest.json")
RIGID = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-core-reference.nhrigid")
BONE = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/current-bone-registration-b1b410ad/bodyparts3d-myosim-major-bones.nhbones")
BONE_MAN = BONE.with_name("bodyparts3d-myosim-major-bones.manifest.json")
SCENE = E / "common-atlas-skin-composition-907/resting-scene/resting-supine-scene.manifest.json"
BASE_WEIGHTS_SHA="bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1"
BODY_NAMES={128:"pelvis",131:"femur_r",145:"femur_l"}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qrot(q,v):
    x,y,z,w=q; vx,vy,vz=v
    tx=2*(y*vz-z*vy); ty=2*(z*vx-x*vz); tz=2*(x*vy-y*vx)
    return (vx+w*tx+(y*tz-z*ty), vy+w*ty+(z*tx-x*tz), vz+w*tz+(x*ty-y*tx))
def add(a,b): return tuple(a[i]+b[i] for i in range(3))
def mul(a,s): return tuple(x*s for x in a)
def bone_world_point(record, vertex, rest):
    body=int(record[0]); t=tuple(record[6:9]); q=tuple(record[9:13]); scale=float(record[13])
    local=add(t,mul(qrot(q,tuple(vertex[:3])),scale))
    bp,bq=rest[body]
    return add(bp,qrot(bq,local))

def parse_rigid(path):
    b=path.read_bytes()
    magic,abi,engineabi,src_count,engine_count,joint_count,nq,nv,root,virtual,reserved,srcsha=struct.unpack_from("<8s10I32s",b,0)
    if magic!=b"NHRIGID2" or abi!=1: raise ValueError("wrong NHRIGID2 ABI")
    off=80+96+48+engine_count*160+joint_count*144+nv*64+nq*4+nv*4
    mapping=struct.unpack_from(f"<{src_count}I",b,off); off+=src_count*4
    poses=[struct.unpack_from("<7f",b,off+28*i) for i in range(src_count)]
    off+=src_count*28
    if off!=len(b): raise ValueError("NHRIGID2 extent mismatch")
    return {int(mapping[i]):((float(p[0]),float(p[1]),float(p[2])),
                             (float(p[3]),float(p[4]),float(p[5]),float(p[6])))
            for i,p in enumerate(poses)}

def closest_batch(points, tri):
    # points (N,3), tri (N,K,3,3); return squared distance to each triangle.
    p=points[:,None,:]
    a,b,c=tri[:,:,0,:],tri[:,:,1,:],tri[:,:,2,:]
    ab=b-a; ac=c-a; ap=p-a
    d00=np.einsum("nki,nki->nk",ab,ab)
    d01=np.einsum("nki,nki->nk",ab,ac)
    d11=np.einsum("nki,nki->nk",ac,ac)
    d20=np.einsum("nki,nki->nk",ap,ab)
    d21=np.einsum("nki,nki->nk",ap,ac)
    den=d00*d11-d01*d01
    safe=np.where(den>1e-30,den,1.0)
    u=(d11*d20-d01*d21)/safe
    v=(d00*d21-d01*d20)/safe
    inside=(u>=0)&(v>=0)&((u+v)<=1)&(den>1e-30)
    proj=a+u[:,:,None]*ab+v[:,:,None]*ac
    pd=np.einsum("nki,nki->nk",p-proj,p-proj)
    def edge_d2(x,y):
        e=y-x; denom=np.einsum("nki,nki->nk",e,e)
        t=np.einsum("nki,nki->nk",p-x,e)/np.maximum(denom,1e-30)
        t=np.clip(t,0.0,1.0)
        delta=p-(x+t[:,:,None]*e)
        return np.einsum("nki,nki->nk",delta,delta)
    ed=np.minimum(edge_d2(a,b),np.minimum(edge_d2(b,c),edge_d2(c,a)))
    return np.where(inside,pd,ed)

def parse_bone_triangles():
    # Reuse the established invocation-pinned 1159 parser and source-owner
    # checks from erratum 1182/bone-distance-005.py. Helper definitions load
    # from reproduce.py without executing its main report writer.
    diag=runpy.run_path(str(E/"native-vastus-lateralis-weight-row-erratum-1182/reproduce.py"),run_name="registered_bone_reader")
    rigid_info=diag["parse_rigid"](RIGID)
    raw=BONE.read_bytes()
    magic,abi,nb,nv,ni,reserved,srcsha=struct.unpack_from("<8s5I32s",raw,0)
    if magic!=b"NHBONES1" or abi!=3 or srcsha.hex()!=rigid_info["source_sha256"]:
        raise ValueError("NHBONES1 ABI/source identity mismatch")
    rb=RIGID.read_bytes()
    rh=struct.unpack_from("<8s10I32s",rb,0)
    source_count,engine_count,joint_count,nq,nv_rigid=rh[3],rh[4],rh[5],rh[6],rh[7]
    rigid_map_off=80+96+48+engine_count*160+joint_count*144+nv_rigid*64+nq*4+nv_rigid*4
    source_to_core=struct.unpack_from(f"<{source_count}I",rb,rigid_map_off)
    recs=[struct.unpack_from("<6I3f4ffI",raw,60+60*i) for i in range(nb)]
    off=60+60*nb
    verts=np.frombuffer(raw,"<f4",6*nv,off).reshape(nv,6); off+=24*nv
    inds=np.frombuffer(raw,"<u4",ni,off).astype(np.int64)
    man=json.loads(BONE_MAN.read_text())
    anchor_map=man["source"]["anchors"]
    rest=rigid_info["rest_by_engine"]
    member_owners={}; surfaces={body:[] for body in BODY_NAMES}
    for record in recs:
        body,firstv,countv,firsti,counti,stable=map(int,record[:6]); source_index=int(record[14])
        if source_index>=len(source_to_core) or int(source_to_core[source_index])!=body:
            raise ValueError(f"NHBONES source-to-core ownership mismatch stable={stable}")
        matches=[a for a in anchor_map if int(a["core_body_index"])==body and int(a["source_record_index"])==source_index and int(a["vertex_count"])==countv and int(a["triangle_count"])==counti//3]
        if len(matches)!=1: raise ValueError(f"NHBONES record provenance mismatch body={body} stable={stable}")
        member=matches[0]["member_id"]; member_owners[stable]=member
        if body not in surfaces: continue
        bodyverts=np.asarray([bone_world_point(record,verts[firstv+i],rest) for i in range(countv)],dtype=np.float64)
        localinds=inds[firsti:firsti+counti].reshape(-1,3)-firstv
        if np.any(localinds<0) or np.any(localinds>=countv): raise ValueError("NHBONES face escapes record vertex range")
        surfaces[body].append((bodyverts[localinds],member,stable))
    merged={}
    for body,parts in surfaces.items():
        if not parts: raise ValueError(f"registered bone surface absent for {body}")
        merged[body]=np.concatenate([x[0] for x in parts],axis=0)
    return raw,abi,merged,member_owners

def exact_surface_distances(points, triangles, initial_k=32, batch=64):
    # Use a centroid KD tree only to establish a finite upper bound. Then test
    # every triangle whose centroid-radius lower bound can beat that upper
    # bound; all omitted triangles are thereby certified farther away.
    centroids=triangles.mean(axis=1)
    radii=np.linalg.norm(triangles-centroids[:,None,:],axis=2).max(axis=1)
    tree=cKDTree(centroids)
    k=min(initial_k,len(triangles))
    out=np.empty(len(points),dtype=np.float64)
    candidate_counts=[]
    certificate_margins=[]
    exhaustive_counts=[]
    for start in range(0,len(points),batch):
        stop=min(start+batch,len(points)); pts=points[start:stop]
        _,nearest=tree.query(pts,k=k,workers=1)
        nearest=np.asarray(nearest,dtype=np.int64).reshape(stop-start,k)
        initial=closest_batch(pts,triangles[nearest]).min(axis=1)
        upper=np.sqrt(initial)
        center_d=np.linalg.norm(pts[:,None,:]-centroids[None,:,:],axis=2)
        lower=np.maximum(0.0,center_d-radii[None,:])
        possible=lower <= (upper[:,None]+2.0e-12)
        counts=possible.sum(axis=1).astype(np.int64)
        if np.any(counts==0): raise ValueError("exact distance lower-bound set unexpectedly empty")
        max_count=int(counts.max())
        candidate_ids=np.zeros((len(pts),max_count),dtype=np.int64)
        for row in range(len(pts)):
            ids=np.flatnonzero(possible[row])
            candidate_ids[row,:len(ids)]=ids
            if len(ids)<max_count: candidate_ids[row,len(ids):]=ids[0]
        exact=closest_batch(pts,triangles[candidate_ids])
        for row,count in enumerate(counts):
            exact[row,int(count):]=np.inf
        best2=exact.min(axis=1)
        best=np.sqrt(best2)
        omitted=np.where(possible, np.inf, lower)
        min_omitted=np.min(omitted,axis=1)
        margin=min_omitted-best
        # Rows with no omitted faces are exhaustive; otherwise the positive
        # lower-bound margin certifies every omitted face is farther.
        margin[~np.isfinite(min_omitted)]=np.inf
        if np.any(margin < -2.0e-12): raise ValueError("triangle lower-bound exactness certificate failed")
        out[start:stop]=best
        candidate_counts.extend(int(x) for x in counts)
        certificate_margins.extend(float(x) for x in margin)
        exhaustive_counts.extend(int(x==len(triangles)) for x in counts)
    return out, {
        "initial_centroid_upper_bound_k":k,
        "candidate_triangle_count_min":int(min(candidate_counts)),
        "candidate_triangle_count_median":float(np.median(candidate_counts)),
        "candidate_triangle_count_max":int(max(candidate_counts)),
        "all_faces_exhaustive_vertex_count":int(sum(exhaustive_counts)),
        "minimum_omitted_lower_bound_margin_m":float(min(certificate_margins)),
        "exact_lower_bound_certificate":True,
        "triangle_count":int(len(triangles)),
    }

def write_candidate(raw, decoded, weights, dest):
    b=bytearray(raw)
    n=decoded["vertex_count"]; nb=decoded["binding_count"]; ni=decoded["index_count"]
    vertex_off=60+36*nb; index_off=vertex_off+56*n; weights_off=index_off+4*ni
    f32=np.asarray(weights,dtype="<f4")
    if float(np.max(np.abs(f32.sum(axis=1)-1)))>2e-6 or np.any(f32<0): raise ValueError("packed candidate full-weight field invalid")
    for v,row in enumerate(f32):
        top=np.argsort(-row,kind="stable")[:4]
        vals=row[top].astype(np.float64); vals/=vals.sum()
        struct.pack_into("<4I4f",b,vertex_off+56*v+24,*(int(i) for i in top),*(float(x) for x in vals.astype("<f4")))
    b[weights_off:weights_off+f32.nbytes]=f32.tobytes()
    Path(dest).write_bytes(b)
    return {"path":str(dest),"sha256":hashlib.sha256(b).hexdigest(),"bytes":len(b),
            "full_weight_sha256":hashlib.sha256(f32.tobytes()).hexdigest()}

start=time.time()
raw=SKIN.read_bytes()
if sha(SKIN)!=BASE_WEIGHTS_SHA: raise ValueError("base NHSKIN hash mismatch")
decoded=decode_payload(raw)
reader=runpy.run_path(str(E/"native-vastus-lateralis-weight-row-erratum-1182/reproduce.py"),run_name="registered_skin_frame_reader")
registered_skin=reader["parse_skin"](SKIN)
rigid_info=reader["parse_rigid"](RIGID)
points=np.asarray(reader["skin_rest_points"](registered_skin,rigid_info["rest_by_engine"]),dtype=np.float64)
body_ids=decoded["bindings_u"][:,0].astype(np.int64)
if points.shape != (decoded["vertex_count"],3):
    raise ValueError("registered NHSKIN rest-point mapping changed source vertex order")
weights=decoded["full_weights"].astype(np.float64)
if len(points)!=54949 or len(body_ids)!=86 or points.shape[0]!=weights.shape[0]: raise ValueError("NHSKIN dimensions differ")
if sha(RIGID)!=json.loads(SKIN_MAN.read_text())["inputs"]["runtime_reference"]["rigid"]["sha256"]: raise ValueError("NHRIGID identity differs")
bone_raw,bone_abi,surfaces,members=parse_bone_triangles()
distances={}; search_stats={}
for body in BODY_NAMES:
    t0=time.time()
    distances[body],search_stats[body]=exact_surface_distances(points,surfaces[body])
    print(json.dumps({"phase":"bone_distance","body_id":body,"elapsed_seconds":time.time()-t0,"distance_min_m":float(distances[body].min()),"distance_max_m":float(distances[body].max()),"exact_search":search_stats[body]}),flush=True)
# Verify the fast centroid-indexed exact triangle projection against the retained
# four-witness brute-force surface distances before using it for all source points.
old_diag=json.loads((E/"native-vastus-lateralis-weight-row-erratum-1182/bone-distance-005.json").read_text())
witness_ids=[int(r["vertex_id"]) for r in old_diag["offending_skin_source_vertices"]]
witness_checks=[]
for row in old_diag["offending_skin_source_vertices"]:
    vid=int(row["vertex_id"])
    mapped={BODY_NAMES[body]:float(distances[body][vid]) for body in BODY_NAMES}
    expected={name:float(value["distance_m"]) for name,value in row["nearest_bones"].items() if name in BODY_NAMES.values()}
    common_names=set(mapped).intersection(expected)
    if not common_names: raise ValueError(f"no retained distance witnesses for vertex {vid}")
    witness_checks.append({"vertex_id":vid,"computed":mapped,"reference":expected,
       "max_abs_delta_m":max(abs(mapped[name]-expected[name]) for name in common_names)})
if max(row["max_abs_delta_m"] for row in witness_checks)>2e-8: raise ValueError("centroid-tree exact distances disagree with retained brute-force witness")
# Randomly compare 20 vertices to exhaustive all-triangle closest points for each body.
rng=np.random.default_rng(1185); sample_ids=np.sort(rng.choice(len(points),size=20,replace=False))
sample_checks=[]
for body,tris in surfaces.items():
    # Exhaustive reference in bounded batches.
    exhaustive=[]
    for vid in sample_ids:
                # Chunk all faces while keeping bounded scratch memory.
        best=float("inf")
        for lo in range(0,len(tris),512):
            dd=closest_batch(points[vid:vid+1],np.broadcast_to(tris[lo:lo+512][None,:,:,:],(1,min(512,len(tris)-lo),3,3)))
            best=min(best,float(dd.min()))
        exhaustive.append(math.sqrt(best))
    delta=np.abs(np.asarray(exhaustive)-distances[body][sample_ids])
    sample_checks.append({"body_id":body,"sample_count":len(sample_ids),"maximum_abs_distance_error_m":float(delta.max())})
    if float(delta.max())>2e-8: raise ValueError(f"centroid-tree distance index failed exhaustive check for body {body}")
scene=json.loads(SCENE.read_text())
protected=np.asarray([int(x["vertex_index"]) for x in scene["bed"]["support_witnesses"]],dtype=np.int64)
variants=[
 {"name":"compact-2.5x-half","maximum_femur_distance_m":0.085,"pelvis_to_femur_ratio_full":2.5,"transfer_fraction":0.5},
 {"name":"compact-2.5x-full","maximum_femur_distance_m":0.085,"pelvis_to_femur_ratio_full":2.5,"transfer_fraction":1.0},
 {"name":"compact-3x-full","maximum_femur_distance_m":0.085,"pelvis_to_femur_ratio_full":3.0,"transfer_fraction":1.0},
]
outputs=[]
for variant in variants:
 candidate,report=reweight_registered_bone_proximity(
   weights,body_ids,distances,pelvis_body_id=128,femur_body_ids=(131,145),
   minimum_femur_distance_m=0.020,maximum_femur_distance_m=variant["maximum_femur_distance_m"],
   pelvis_to_femur_ratio_start=1.5,pelvis_to_femur_ratio_full=variant["pelvis_to_femur_ratio_full"],
   transfer_fraction=variant["transfer_fraction"],protected_vertex_ids=protected)
 dst=OUT/f"{variant['name']}.nhskin"; output=write_candidate(raw,decoded,candidate,dst)
 # Exact source bytes outside the weights/quartet fields must remain unchanged.
 packed=dst.read_bytes(); vertex_off=60+36*decoded["binding_count"]; index_off=vertex_off+56*decoded["vertex_count"]
 weight_off=index_off+4*decoded["index_count"]
 if packed[:vertex_off]!=raw[:vertex_off] or packed[index_off:weight_off]!=raw[index_off:weight_off]:
   raise ValueError("candidate modified bindings, geometry, normals, or topology")
 check=decode_payload(packed)
 if not np.array_equal(check["vertices_u"][:,:6],decoded["vertices_u"][:,:6]) or not np.array_equal(check["indices"],decoded["indices"]):
   raise ValueError("candidate changed source positions/normals or topology")
 if not np.array_equal(check["full_weights"][:,[i for i,x in enumerate(body_ids) if int(x) not in (128,131,145)]],
                       decoded["full_weights"][:,[i for i,x in enumerate(body_ids) if int(x) not in (128,131,145)]]):
   raise ValueError("candidate changed a non-pelvis/femur weight")
 if not np.array_equal(check["full_weights"][:,[i for i,x in enumerate(body_ids) if int(x)==20]],
                       decoded["full_weights"][:,[i for i,x in enumerate(body_ids) if int(x)==20]]):
   raise ValueError("candidate changed the torso weight field")
 report.update({"variant":variant,"candidate_payload":output,
    "source_skin_sha256":sha(SKIN),"protected_support_vertex_count":int(len(protected)),
   "distance_coordinate_frame":"NHSKIN source points transformed by the exact pinned reproduce.py skin_rest_points(owner bindings, NHRIGID rest poses); compared with NHBONES registered surfaces in the same rest-world frame",
   "skin_rest_world_point_count":int(len(points)),
   "rest_positions_topology_normals_and_all_bindings_byte_identical":True,
   "torso_and_all_unrelated_owner_weight_columns_byte_identical":True,
   "nonpelvis_nonfemur_weights_byte_identical":True,
   "candidate_fullweight_row_sum_max_abs_error":float(np.max(np.abs(check["full_weights"].sum(axis=1)-1.0))),
   "candidate_new_pelvis_weight_quantiles_on_changed_vertices":np.quantile(check["full_weights"][report["changed_vertex_ids"],body_ids.tolist().index(128)],[0,.25,.5,.75,1]).tolist() if report["changed_vertex_ids"] else [],
   "candidate_new_femur_weight_quantiles_on_changed_vertices":np.quantile(check["full_weights"][report["changed_vertex_ids"]][:,[body_ids.tolist().index(131),body_ids.tolist().index(145)]].sum(axis=1),[0,.25,.5,.75,1]).tolist() if report["changed_vertex_ids"] else []})
 outputs.append(report)
np.savez_compressed(OUT/"registered-bone-surface-distances.npz",**{f"body_{k}":v.astype("<f8") for k,v in distances.items()})
report={
 "schema":"numi.human.registered-bone-proximity-thigh-reweight-candidate.v1",
 "status":"inferred_weight_candidates_built_pending_saved_pose_geometry_and_native_support_requalification",
 "inputs":{str(p):sha(p) for p in [SKIN,SKIN_MAN,RIGID,BONE,BONE_MAN,SCENE]},
 "code":{str(Path(__file__).resolve()):sha(Path(__file__).resolve())},
 "body_surfaces":{"body_name_by_core_id":{str(k):v for k,v in BODY_NAMES.items()},
   "nhtiss_registration_frame":"NHBONES record transform composed with NHRIGID accepted rest pose, checked against the exact 1159 NHSKIN common-atlas rest frame",
   "triangle_counts":{str(k):int(len(v)) for k,v in surfaces.items()},
   "all_record_owner_and_manifest_anchors_verified":True},
 "distance_validation":{"reference":str(E/"native-vastus-lateralis-weight-row-erratum-1182/bone-distance-005.json"),
   "reference_sha256":sha(E/"native-vastus-lateralis-weight-row-erratum-1182/bone-distance-005.json"),
   "known_witnesses":witness_checks,"exhaustive_randomized_surface_checks":sample_checks,"all_vertex_exact_search_certificate":{str(k):v for k,v in search_stats.items()},
   "query":"exact closest distance to each registered triangle mesh: centroid KD tree supplies an initial upper bound; exact point-to-triangle distances are evaluated for every face whose centroid-distance minus maximum vertex radius can beat the bound; all omitted faces have a positive certified lower-bound margin"},
 "protected_support_vertices":protected.tolist(),
 "variants":outputs,
 "input_sha256_before_after":sha(SKIN),
 "distance_array_path":str(OUT/"registered-bone-surface-distances.npz"),
 "distance_array_sha256":sha(OUT/"registered-bone-surface-distances.npz"),
 "elapsed_seconds":time.time()-start,
 "limits":["Weight field is an inferred bilateral thigh binding trial, not a measured dataset.","Only pelvis influence transfers to registered femora; torso column remains exact to preserve the respiratory selector map. This trial does not claim to fix other muscle crossings.","Rest positions, normals, bind records, topology, bed support witness weights, all nonpelvis/nonfemur weights, and mechanical NHTISS4 are preserved. Runtime contact support/Jacobians may still change because NHSKIN weights alter posed geometry.","No candidate has passed saved-pose exact geometry checks or native support/contact requalification."]
}
(OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(OUT/"report.json"),"report_sha256":sha(OUT/"report.json"),"elapsed_seconds":report["elapsed_seconds"],"variants":[{"name":x["variant"]["name"],"changed":x["changed_vertex_count"],"total_transferred_weight":x["total_transferred_weight"],"support":x["per_femur_support"]} for x in outputs]},indent=2))

