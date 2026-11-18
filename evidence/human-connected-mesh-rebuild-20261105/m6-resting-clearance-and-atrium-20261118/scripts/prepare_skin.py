"""Derive a bounded source-skin proposal from matched native capture witnesses."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "coupled-crossing-repair-001"))
sys.path.insert(0, "/Users/numi/numi-human-aortic-contact-opt-20261019/tools")
import bindings
from rebind_human_skin_reference import decode, rotation
from resting_anatomy_interface_patch import normals

HERE = Path(__file__).resolve().parent
RUN = ROOT / "coupled-thumb-reference-001"
BASE = ROOT / "regional-bed-layers-001/capture-4s/native/accepted-geometry/step-0.mrvpack"
END = RUN / "thumb-4s-capture/native/accepted-geometry/step-2000.mrvpack"

def capture(pack):
    receipt = json.loads(pack.with_suffix(".receipt.json").read_text())
    with tempfile.TemporaryDirectory(dir=HERE) as temporary:
        unpacked = Path(temporary) / pack.name
        with unpacked.open("wb") as out:
            subprocess.run(["/opt/homebrew/bin/zstd", "-dc", str(pack)+".zst"], stdout=out, check=True)
        assert bindings.audit.sha(unpacked) == receipt["pack_file_sha256"]
        points, faces = bindings.read_skin(unpacked, receipt)
    return receipt, points, faces

def full_self(points, faces):
    audit = bindings.audit
    rows = audit.clearance._exact_surface_records(points, faces)
    return audit.intersections._audit_pair_prepared_first(
        audit.intersections._prepare_surface_aabb(rows), rows, same_surface=True)

def main():
    launch = json.loads((RUN / "combined-12s-no-export/launch.json").read_text())
    argv = launch["argv"]
    skin = Path(argv[argv.index("--skin-payload")+1])
    assert bindings.audit.sha(skin) == launch["pins"][str(skin)]
    model = decode(skin.read_bytes())
    rest, p0, f0 = capture(BASE)
    settled, p1, faces = capture(END)
    assert np.array_equal(f0, faces)
    source_faces = model["inds"].astype(np.int64)
    lookup = {tuple(f): k for k, f in enumerate(source_faces)}
    assert len(lookup) == len(source_faces)
    assert all(tuple(face) in lookup for face in faces)
    baseline = json.loads((ROOT/"coupled-crossing-repair-001/thumb-exact.json").read_text())
    assert baseline["accepted_pack_sha256"] == settled["pack_file_sha256"]
    pairs = np.asarray(baseline["self_intersections"]["triangle_pairs"], dtype=np.int64)
    source = np.ascontiguousarray(model["verts"][:, :3], dtype="<f4")
    w = model["weights"].astype(float)
    local = np.array([rotation(b[3:7])*b[7] for b in model["binds"]["v"]])

    def jacobians(receipt):
        bodies = {b["body_index"]: b for b in receipt["accepted_body_poses"]}
        body_rot = np.array([rotation(bodies[int(i)]["quaternion_xyzw"]) for i in model["binds"]["body"]])
        return np.einsum("vi,ijk->vjk", w, body_rot @ local)
    j0, j1 = jacobians(rest), jacobians(settled)

    def proposed_poses(x):
        delta = np.subtract(x, source, dtype=np.float64)
        return tuple(np.ascontiguousarray(p + np.einsum("vij,vj->vi", j, delta), dtype="<f4")
                     for p, j in ((p0, j0), (p1, j1)))

    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adj = coo_matrix((np.ones(2*len(edges)), (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),
                     shape=(len(source),len(source))).tocsr()
    union = faces[pairs].reshape(len(pairs), -1)
    witnesses = np.unique(union)
    protect = np.unique(np.concatenate((edges[counts != 2].reshape(-1),
        np.flatnonzero((p0[:,2] < 0.015) | (p1[:,2] < 0.015)),
        bindings.audit.clearance._exact_coordinate_tied_source_vertex_ids(source))))
    # Use spatially connected witness neighborhoods so unrelated joints cannot
    # exchange corrections through the source rest surface.
    centers = p1[union].mean(axis=1)
    d = np.linalg.norm(centers[:,None] - centers[None,:], axis=2)
    _, groups = connected_components(coo_matrix(d < 0.014))
    candidate = source.copy()
    active = []
    moved = set()
    for group in np.unique(groups):
        group_pairs = pairs[groups==group]
        shared = np.array([len(set(faces[a]) & set(faces[b])) == 1 for a, b in group_pairs])
        used = np.unique(faces[group_pairs])
        free = np.setdiff1d(used, protect)
        if not len(free):
            active.append({"pairs":len(group_pairs),"error":"all witness vertices protected"})
            print(group,active[-1],flush=True)
            continue
        ring = np.unique(np.concatenate([adj[i].indices for i in free]))
        patch = np.setdiff1d(np.union1d(free,ring),np.r_[protect,list(moved)].astype(np.int64))
        if not len(patch):
            active.append({"pairs": len(group_pairs), "error": "protected source geometry"})
            continue
        ids = np.union1d(used, patch)
        try:
            proposal = bindings.audit.clearance._propose_local_self_separation_increment(
                source_positions_m=source, faces=faces, movable_vertex_ids=patch,
                self_patch_seed_vertex_ids=np.intersect1d(free,patch),
                required_target_seed_vertex_ids=np.empty(0,dtype=np.int64),
                active_target_face_vertex_ids=np.empty(0,dtype=np.int64),
                fixed_support_vertex_ids=protect,
                preserved_anchor_vertex_ids=np.empty(0,dtype=np.int64),
                pose_samples=[{"vertex_ids":ids, "baseline_world_positions":p0[ids],
                               "rejected_world_positions":p1[ids],"jacobians":j1[ids],
                               "self_pair_rows":group_pairs[~shared],
                               "shared_vertex_contact_pair_rows":group_pairs[shared]}])
        except ValueError as error:
            active.append({"pairs":len(group_pairs),"error":str(error)})
            print(group, active[-1], flush=True)
            continue
        candidate[patch] = proposal["proposal_source_positions_m_f32"][patch]
        moved.update(patch.tolist())
        active.append({"pairs":len(group_pairs),"patch_vertices":len(patch)})
        print(group,active[-1],flush=True)
    pose0, pose1 = proposed_poses(candidate)
    hit0, hit1 = full_self(pose0, faces), full_self(pose1, faces)
    changed = np.flatnonzero(np.any(candidate != source, axis=1))
    np.savez_compressed(HERE/"skin-proposal.npz", xyz=candidate, changed=changed,
                        p0=pose0,p1=pose1,faces=faces)
    result = {"source":str(skin), "sha256":bindings.audit.sha(skin),
              "baseline_counts":[0, len(pairs)], "posed_counts":[hit0["count"],hit1["count"]],
              "pose_mode":"First-order source Jacobian on frozen accepted captures; native remap required",
              "regions":active,"changed":len(changed),"remaining_pairs":hit1["triangle_pairs"],
              "rest_sha256":rest["pack_file_sha256"],"settled_sha256":settled["pack_file_sha256"]}
    (HERE/"skin-proposal.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="remaining_pairs"}),flush=True)
    if len(changed) and hit0["count"] == 0 and hit1["count"] < len(pairs):
        # Preserve stored normals outside the changed patch and all original face
        # identities. Rebuild the normals for faces incident to changed vertices.
        patched = bytearray(model["raw"])
        neighbors = np.unique(source_faces[np.any(np.isin(source_faces, changed),axis=1)])
        referenced = np.unique(source_faces)
        revised_normals = np.zeros_like(candidate)
        revised_normals[referenced] = normals(candidate[referenced],np.searchsorted(referenced,source_faces))
        for idx in neighbors:
            struct.pack_into("<3f",patched,model["vertoff"] + 56*int(idx) + 12,*revised_normals[idx])
        for idx in changed:
            struct.pack_into("<3f",patched,model["vertoff"] + 56*int(idx),*candidate[idx])
        decode(bytes(patched))
        (HERE/"skin-proposal.nhskin").write_bytes(patched)

if __name__ == "__main__":
    main()
