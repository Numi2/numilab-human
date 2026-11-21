"""Check the executed route transforms before screening alternate interpolation."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation

trial = Path(__file__).resolve().parent
prior = trial.parent / "paired-crossing-ownership-035"
reference = runpy.run_path(str(prior / "load_complete_reference.py"))
run = reference["run"]
banks = reference["banks"]
skin, skin_faces = reference["skin"], reference["skin_faces"]
clearance, exact = reference["clearance"], reference["exact"]
receipt = json.loads(Path(os.environ["NUMI_HUMAN_CAPTURED_TIME_ZERO_RECEIPT"]).read_text())
poses = {p["body_index"]:p for p in receipt["accepted_body_poses"]}
registered = {p["body_index"]:p for p in receipt["accepted_registered_body_poses"]}
args = json.loads((run / "launch.json").read_text())["argv"]
payload = Path(args[args.index("--soft-tissue-payload")+1])
source = payload.read_bytes()
manifest = json.loads(payload.with_suffix(".manifest.json").read_text())
sha = hashlib.sha256(source).hexdigest()
assert sha == manifest["payload"]["sha256"]
labels = {row["stable_id"]:row for row in manifest["source"]["surfaces"]}
magic, abi, nr, nb, nv, ni, _ = struct.unpack_from("<8s6I",source)
assert magic == b"NHTISS4\0" and abi == 5
record_offset = 64
binding_offset = record_offset+32*nr
vertex_offset = binding_offset+36*nb
index_offset = vertex_offset+56*nv
assert len(source) == index_offset+4*ni
records = np.frombuffer(source,dtype="<u4",count=8*nr,offset=record_offset).reshape(nr,8)
bindings = np.frombuffer(source,dtype=np.dtype([
    ("body","<u4"),("translation","<f4",3),("rotation","<f4",4),("scale","<f4")]),
    count=nb,offset=binding_offset)
vertices = np.frombuffer(source,dtype=np.dtype([
    ("position","<f4",3),("normal","<f4",3),("binding","<u4",4),("weight","<f4",4)]),
    count=nv,offset=vertex_offset)
indices = np.frombuffer(source,dtype="<u4",count=ni,offset=index_offset)

def pair_index(point, face):
    rec = clearance._exact_surface_records(point, face)
    return exact._prepare_surface_aabb(rec), clearance._float32_skin_aabb_index(rec), rec

def hits(index, points, faces):
    tree, broad, _ = index
    chosen, count = clearance._candidate_target_rows_for_skin_aabb(faces,points,broad)
    if not len(chosen):
        return set()
    used = np.unique(faces[chosen])
    records = clearance._exact_surface_records(points[used], np.searchsorted(used,faces[chosen]))
    audited = exact._audit_pair_prepared_first(tree,[(a,b,c,int(chosen[d]),e) for a,b,c,d,e in records])
    assert audited["aabb_candidate_pairs"] == count
    return set(map(tuple,audited["triangle_pairs"]))

def rotation(row):
    return Rotation.from_quat(row["quaternion_xyzw"])

def quat_multiply(a,b):
    vector = a[...,3,None]*b[...,:3] + b[...,3,None]*a[...,:3] + np.cross(a[...,:3],b[...,:3])
    return np.concatenate((vector,(a[...,3]*b[...,3]-np.sum(a[...,:3]*b[...,:3],axis=-1))[...,None]),axis=-1)

def bind_transforms(bound, *, use_registered):
    selected = registered if use_registered else poses
    scales, rot, shift = [], [], []
    for b in bound:
        body = int(b["body"])
        pose = selected[body]
        world = rotation(pose)
        local = Rotation.from_quat(b["rotation"])
        scales.append(float(b["scale"]))
        rot.append((world*local).as_quat())
        shift.append(world.apply(b["translation"])+pose["position_m"])
    return np.array(scales), np.array(rot), np.array(shift)

def linear(x, weights, scales, quat, shift):
    parts = np.stack([Rotation.from_quat(q).apply(x)*s+t for q,s,t in zip(quat,scales,shift)])
    return np.einsum("bn,bnd->nd",weights,parts)

def dual(x, weights, scales, quat, shift):
    assert np.ptp(scales)<1e-9
    quat = quat.copy()
    quat[(quat @ quat[0])<0] *= -1
    # Skip ambiguous sign neighborhoods. All evaluated rotations must lie in a
    # common hemisphere, with each pair consistently signed.
    assert (quat @ quat.T > 0).all()
    real = weights.T @ quat
    d = .5*quat_multiply(np.c_[shift,np.zeros(len(quat))],quat)
    dual_part = weights.T @ d
    norm = np.linalg.norm(real,axis=1)[:,None]
    real /= norm
    dual_part /= norm
    translation = 2*quat_multiply(dual_part,real*[-1.,-1.,-1.,1.])[:,:3]
    # Preserve the same weighted scale, including tiny authored weight-sum error.
    return (Rotation.from_quat(real).apply(x*scales[0])+translation)*weights.sum(axis=0)[:,None]

def normal_areas(x,f):
    tri=x[f].astype(np.float64)
    return np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])

def shape(x,y,faces):
    a,b = normal_areas(x,faces),normal_areas(y,faces)
    a_norm=np.linalg.norm(a,axis=1)
    b_norm=np.linalg.norm(b,axis=1)
    assert (a_norm>0).all()
    cosine=np.einsum("ij,ij->i",a,b)/np.maximum(a_norm*b_norm,np.finfo(float).tiny)
    w=clearance._signed_surface_integral(x,faces)
    z=clearance._signed_surface_integral(y,faces)
    return {
        "min_area_ratio":float((b_norm/a_norm).min()),
        "max_area_ratio":float((b_norm/a_norm).max()),
        "minimum_face_normal_cosine":float(cosine.min()),
        "relative_signed_volume_change":abs(z/w-1) if w else None,
    }

def neighborhood(reference_points, trial_points, faces, stable):
    lower = np.minimum(reference_points.min(axis=0),trial_points.min(axis=0))
    upper = np.maximum(reference_points.max(axis=0),trial_points.max(axis=0))
    original=pair_index(reference_points,faces)
    proposed=pair_index(trial_points,faces)
    details=[]
    for key,(points,triangles) in banks.items():
        if key==(51005,stable) or (points.min(axis=0)>upper).any() or (points.max(axis=0)<lower).any():
            continue
        old=hits(original,points,triangles)
        new=hits(proposed,points,triangles)
        if old!=new:
            details.append({"bank":key,"before":len(old),"after":len(new),
                            "new":len(new-old),"cleared":len(old-new)})
    return details

skin_index=pair_index(skin,skin_faces)
if __name__ == '__main__':
    wanted = list(map(int,sys.argv[2:])) or [137,68,67,110,109,138,74,73]
    out=[]
    for stable in wanted:
        rows=records[records[:,6]==stable]
        assert len(rows)==1
        start_b,n_b,start_v,n_v,start_i,n_i,_,_ = map(int,rows[0])
        bs=bindings[start_b:start_b+n_b]
        vs=vertices[start_v:start_v+n_v]
        raw_faces=(indices[start_i:start_i+n_i]-start_v).reshape(-1,3)
        used=np.unique(raw_faces)
        faces=np.searchsorted(used,raw_faces)
        vs=vs[used]
        x=vs["position"].astype(float)
        native,native_faces=banks[51005,stable]
        assert np.array_equal(faces,native_faces), stable
        weights=np.array([np.sum(np.where((vs["binding"]==i)&(vs["weight"]>2e-6),vs["weight"],0),axis=1)
                          for i in range(n_b)])
        assert np.allclose(weights.sum(axis=0),1,rtol=0,atol=.002)
        methods=[]
        for use_registered in (True,False):
            if any(int(b["body"]) not in (registered if use_registered else poses) for b in bs):
                continue
            scales,quat,shift=bind_transforms(bs,use_registered=use_registered)
            position=linear(x,weights,scales,quat,shift)
            delta=np.linalg.norm(position-native,axis=1)
            methods.append((float(delta.max()),use_registered,scales,quat,shift,position,
                            float(np.sqrt(np.mean(delta**2)))))
        assert methods, stable
        methods.sort(key=lambda m:m[0])
        maximum,use_registered,scales,quat,shift,decoded,rms=methods[0]
        control=hits(skin_index,native,faces)
        result={
            "stable_id":stable,
            "label":labels[stable]["label"],
            "owner_bodies":bs["body"].tolist(),
            "binding_scales":scales.tolist(),
            "executed_source_crossings":len(control),
            "reconstructed_pose_methods":[{"registered":m[1],"max_error_mm":1000*m[0],"rms_error_mm":1000*m[6]}
                                          for m in methods],
            "maximum_relative_body_rotation_degrees":float(max((Rotation.from_quat(quat[0]).inv()*
                                                     Rotation.from_quat(q)).magnitude() for q in quat)*180/np.pi),
        }
        if maximum>2e-6:
            result["stop"]="source reconstruction differs from native capture"
        elif np.ptp(scales)>=1e-9 or not np.all((quat @ quat.T != 0)):
            result["stop"]="incompatible rigid transforms"
        else:
            alternative=dual(x,weights,scales,quat,shift)
            diff=alternative-decoded
            multi=(weights>0).sum(axis=0)>1
            assert not np.any(np.linalg.norm(diff[~multi],axis=1)>1e-10)
            # The same native Float32 artifact supplies the base for all comparisons.
            # This preserves native evaluation-rounding differences in both arms.
            proposed=(native.astype(float)+diff).astype("<f4")
            new=hits(skin_index,proposed,faces)
            result.update(shape(native,proposed,faces))
            result.update(proposed_crossings=len(new),added_crossings=len(new-control),
                          cleared_crossings=len(control-new),max_position_change_mm=1000*float(np.linalg.norm(diff,axis=1).max()),
                          changed_vertices=int(np.any(proposed!=native,axis=1).sum()),
                          baseline_pair_replay_match=(hits(skin_index,np.ascontiguousarray(decoded,dtype="<f4"),faces)==control))
            if len(new)<len(control):
                result["neighbor_changes"]=neighborhood(native,proposed,faces,stable)
        out.append(result)
        print(json.dumps(result),file=sys.stderr,flush=True)

    print(json.dumps({"payload_sha256":sha,"reference_pack_sha256":receipt["base_pack_file_sha256"],
                      "scope":"Same topology, weights and fixed single-owner vertices. Native baseline plus decoded interpolation delta. Isolated screen only.",
                      "surfaces":out},indent=2))
