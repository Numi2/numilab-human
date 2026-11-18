"""Compare deformation-only alternatives against the same accepted body poses."""
import json
from pathlib import Path

import numpy as np

import bindings


def quat_mul(a, b):
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return np.concatenate((aw*bv + bw*av + np.cross(av, bv),
                           aw*bw - np.sum(av*bv, axis=-1, keepdims=True)), axis=-1)


def rotate(q, p):
    axis = q[..., :3]
    cross = np.cross(axis, p)
    return p + 2*(q[..., 3:]*cross + np.cross(axis, cross))


def pose(step):
    file = bindings.caps / f"step-{step}.mrvpack"
    receipt = json.loads(file.with_suffix(".receipt.json").read_text())
    p, faces = bindings.read_skin(file, receipt)
    bodies = sorted(receipt["accepted_body_poses"], key=lambda body:body["body_index"])
    return receipt, p.astype(float), faces, np.array([r["position_m"] for r in bodies]), np.array([
        r["quaternion_xyzw"] for r in bodies
    ])


def main():
    _, p0, faces, t0, q0 = pose(0)
    receipt, p1, later, t1, q1 = pose(2000)
    assert np.array_equal(faces, later)
    vi = receipt["skin_source_mapping"]
    mapping = np.memmap(vi["vertex_map"]["path"], dtype="<u4", mode="r").reshape(-1, 12)
    sources = np.memmap(vi["influences"]["path"], dtype="<u4", mode="r").reshape(-1, 12)
    counts = mapping[:, 1]
    rows = np.repeat(np.arange(len(counts)), counts)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    records = mapping[:, 0].astype(np.int64)[rows] - vi["influence_first_index"] + np.arange(len(rows)) - starts[rows]
    body = sources[records, 8].astype(np.int64)
    assert body.max() < len(t0) or np.all(body[body >= len(t0)] == 0xffffffff)
    invalid = body == 0xffffffff
    body[invalid] = len(t0)
    inv = np.array([-1,-1,-1,1])
    rotation = quat_mul(q1, q0 * inv)
    rotation /= np.linalg.norm(rotation, axis=1)[:, None]
    translation = t1 - rotate(rotation, t0)
    rotation = np.vstack((rotation, [0,0,0,1]))
    translation = np.vstack((translation, [0,0,0]))
    dual = 0.5 * quat_mul(np.pad(translation, ((0,0),(0,1))), rotation)
    weights = sources[records, 3].view("<f4").astype(float)

    dominant = np.zeros(len(counts), dtype=np.int64)
    maximum = np.maximum.reduceat(weights, starts)
    strongest = np.where(weights==maximum[rows], np.arange(len(rows)), len(rows))
    dominant[:] = body[np.minimum.reduceat(strongest, starts)]
    linear = np.zeros_like(p0)
    real = np.zeros((len(p0), 4))
    dual_sum = np.zeros_like(real)
    for j in range(0,len(rows),100_000):
        take = slice(j,j+100_000)
        v, b = rows[take], body[take]
        q = rotation[b]
        w = weights[take, None]
        sign = np.where(np.sum(q * rotation[dominant[v]], axis=1, keepdims=True) < 0, -1., 1.)
        np.add.at(linear, v, w*(rotate(q,p0[v]) + translation[b]))
        np.add.at(real, v, w * sign * q)
        np.add.at(dual_sum, v, w * sign * dual[b])
    norm = np.linalg.norm(real, axis=1, keepdims=True)
    assert norm.min() > 0.25
    real /= norm
    dual_sum /= norm
    skinned = rotate(real, p0) + 2*quat_mul(dual_sum, real*inv)[:, :3]
    # The change isolates blend math. The captured base retains the existing
    # respiratory/bed displacement and source registration.
    delta = skinned-linear
    free = (p1[:,2]>0.001) & (p0[:,2]>0.001)
    delta[~free] = 0.0
    corrected = np.ascontiguousarray(p1 + delta, dtype='<f4')
    a = bindings.audit
    r = a.clearance._exact_surface_records(corrected, faces)
    exact = a.intersections._audit_pair_prepared_first(a.intersections._prepare_surface_aabb(r), r, same_surface=True)
    result = dict(mode="offline DQS differential, dry run only",
                  initial_step=0, inspected_step=2000, original_self_pairs=610,
                  candidate_self_pairs=exact["count"], delta_max_mm=float(1000*np.linalg.norm(delta,axis=1).max()),
                  source_step0=receipt["base_pack_file_sha256"],
                  delta_rms_mm=float(1000*np.sqrt(np.mean(np.sum(delta**2, axis=1)))))
    out=Path(__file__).parent / "pose-blend.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
