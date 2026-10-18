"""Repair inferred NHSKIN birth weights by positive mesh-local harmonic extension.

Offline asset preparation only. Original source rows, atlas transforms, topology,
mass and runtime owners are unchanged. Pull back the authenticated initial surface
through the new blend, rather than keeping a pose-specific inverse registration.
"""
from pathlib import Path
import argparse, hashlib, json, sys, struct
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from numilab_human import skin_source_payload_preflight as sp, resting_anatomy_verify as audit, model

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def harmonic_birth_weights(faces, points, weights, unknown):
    unknown=np.asarray(unknown,dtype=np.int64)
    if len(unknown)==0 or len(np.unique(unknown))!=len(unknown):
        raise ValueError("nonempty distinct inferred source slots required")
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    length=np.linalg.norm(points[edges[:,0]].astype(float)-points[edges[:,1]],axis=1)
    if np.any(length<=0):raise ValueError("zero source edge")
    conductance=1/length
    graph=coo_matrix((np.r_[conductance,conductance],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(points),len(points))).tocsr()
    lap=diags(np.asarray(graph.sum(axis=1)).ravel())-graph
    fixed=np.setdiff1d(np.arange(len(points)),unknown)
    rhs=-(lap[unknown][:,fixed]@weights[fixed].astype(float))
    solved=splu(lap[unknown][:,unknown].tocsc()).solve(rhs)
    if not np.isfinite(solved).all() or solved.min() < -1e-10:
        raise ValueError("unanchored or nonpositive harmonic extension")
    residual=np.max(np.abs(lap[unknown][:,unknown]@solved-rhs))
    solved=np.maximum(solved,0)
    solved/=solved.sum(axis=1)[:,None]
    out=weights.copy();out[unknown]=solved.astype("<f4")
    if np.max(np.abs(out.sum(axis=1)-1))>5e-7:raise ValueError("partition of unity")
    return out,float(residual)

def pose_affines(decoded, receipt):
    poses={int(row["body_index"]):(np.asarray(row["position_m"],dtype="<f4"),np.asarray(model._myosim_matrix_from_quaternion_xyzw(row["quaternion_xyzw"]),dtype="<f4")) for row in receipt["accepted_body_poses"]}
    linear=[];shift=[]
    for j,k in enumerate(decoded["bindings_u"][:,0]):
        t,r=poses[int(k)];b=decoded["bindings_f"][j]
        rotation=np.asarray(model._myosim_matrix_from_quaternion_xyzw(b[4:8].tolist()))
        linear.append(r.astype(float)@rotation*float(b[8]))
        shift.append(t.astype(float)+r.astype(float)@b[1:4].astype(float))
    return np.array(linear),np.array(shift)

def world_points(positions,weights,linear,shift):
    # Offline reference reconstruction, never used by the native step path.
    return np.einsum("nk,kij,nj->ni",weights.astype(float),linear,positions.astype(float))+weights.astype(float)@shift

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["skin","capture","receipt","output"]:p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--original-source-vertex-count",type=int,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError("retain previous candidate; choose fresh output")
    raw=a.skin.read_bytes();src=sp.decode_payload(raw)
    capture_receipt=json.loads(a.receipt.read_text())
    audit.validate_capture_receipt(capture_receipt,sha(a.capture))
    sections=audit._pack_sections(a.capture)
    vertices=np.frombuffer(sections[2][0],"<f4").reshape(-1,20)
    indices=np.frombuffer(sections[3][0],"<u4")
    primitives=np.frombuffer(sections[4][0],"<u4").reshape(-1,16)
    rows=[r for r in primitives if (int(r[4]),int(r[7]))==(51007,1)]
    if len(rows)!=1:raise ValueError("unique typed source skin required")
    r=rows[0];faces=src["indices"].reshape(-1,3)
    captured=indices[int(r[0]):int(r[0])+int(r[1])].reshape(-1,3)
    offset=int(captured[0,0])-int(faces[0,0])
    if not np.array_equal(captured-offset,faces):raise ValueError("exact source face emission required")
    points=vertices[offset:offset+src["vertex_count"],:3].copy()
    slots=np.arange(src["vertex_count"])
    # The source count is authenticated separately against the BP3D outer sheet.
    # Geometric support is evaluated in the accepted reference frame, not a future pose.
    body={int(x["body_index"]):np.asarray(x["position_m"]) for x in capture_receipt["accepted_body_poses"]}
    distance=np.full(len(points),np.inf)
    for lo,hi in [(45,76),(95,126)]:
        centers=np.array([body[k] for k in range(lo,hi+1) if k in body])
        distance=np.minimum(distance,np.linalg.norm(points[:,None,:]-centers[None,:,:],axis=2).min(axis=1))
    active=np.zeros(len(points),dtype=bool);active[np.unique(faces)]=True
    unknown=np.flatnonzero((slots>=a.original_source_vertex_count)&(distance<.075)&active)
    weights,residual=harmonic_birth_weights(faces,points,src["full_weights"],unknown)
    linear,shift=pose_affines(src,capture_receipt)
    old_world=world_points(src["vertices_f"][:,:3],src["full_weights"],linear,shift)
    # Preserve the original blend's reference surface; only attachment ancestry changes.
    target=old_world[unknown]
    aa=np.einsum("nk,kij->nij",weights[unknown].astype(float),linear)
    bb=weights[unknown].astype(float)@shift
    condition=np.linalg.cond(aa)
    if condition.max()>10:raise ValueError("ill-conditioned source pullback")
    source=src["vertices_f"][:,:3].copy()
    source[unknown]=np.linalg.solve(aa,(target-bb)[...,None])[...,0].astype("<f4")
    new_world=world_points(source,weights,linear,shift)
    error=np.linalg.norm(new_world[unknown]-target,axis=1)
    if error.max()>2e-7:raise ValueError("reference surface moved")
    v=src["vertices_u"].copy()
    v[unknown,:3]=source[unknown].view("<u4")
    # Diagnostic quartets remain consistent with the full ABI5 weights.
    quartet=np.argsort(weights[unknown],axis=1)[:,-4:][:,::-1].astype("<u4")
    qweights=np.take_along_axis(weights[unknown],quartet,axis=1)
    qweights/=qweights.sum(axis=1)[:,None]
    v[unknown,6:10]=quartet
    v[unknown,10:14]=qweights.astype("<f4").view("<u4")
    be=60+36*src["binding_count"]
    result=raw[:be]+v.tobytes()+src["indices"].tobytes()+weights.tobytes()
    dst=sp.decode_payload(result)
    fixed=np.setdiff1d(slots,unknown)
    assert np.array_equal(dst["vertices_u"][fixed],src["vertices_u"][fixed])
    assert np.array_equal(dst["full_weights"][fixed],src["full_weights"][fixed])
    assert np.array_equal(dst["indices"],src["indices"]) and result[:be]==raw[:be]
    assert np.array_equal(dst["vertices_u"][:a.original_source_vertex_count],src["vertices_u"][:a.original_source_vertex_count])
    a.output.mkdir()
    output=a.output/a.skin.name;output.write_bytes(result)
    np.savez_compressed(a.output/"candidate.npz",source_positions_f32=source,full_weights=weights,changed_inferred_source_slots=unknown,reference_positions_m=points,faces=faces)
    report=dict(status="PASS_OFFLINE_INFERRED_SKIN_BINDING_REPAIR_PENDING_NATIVE",source_payload_path=str(a.skin.resolve()),source_payload_sha256=sha(a.skin),output_payload_path=str(output.resolve()),output_payload_sha256=sha(output),capture_path=str(a.capture.resolve()),capture_sha256=sha(a.capture),capture_receipt_path=str(a.receipt.resolve()),capture_receipt_sha256=sha(a.receipt),original_source_vertex_count=a.original_source_vertex_count,changed_inferred_source_slots=unknown.tolist(),changed_inferred_vertex_count=len(unknown),all_original_source_records_and_weights_byte_exact=True,all_other_source_records_and_weights_byte_exact=True,atlas_binding_records_and_source_ids_byte_exact=True,triangle_indices_and_order_byte_exact=True,positive_harmonic_extension_residual_max_abs=residual,maximum_reference_position_error_m=float(error.max()),maximum_pullback_condition=float(condition.max()),reference_shape_not_repaired=True,mass_force_physiology_owners_unchanged=True,scope="Fix incorrect inferred birth attachment ancestry using current mesh graph; no measured-person, full-anatomy or native admission.")
    (a.output/"source-preparation-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ["status","changed_inferred_vertex_count","maximum_reference_position_error_m","maximum_pullback_condition"]}))
if __name__=="__main__":main()
