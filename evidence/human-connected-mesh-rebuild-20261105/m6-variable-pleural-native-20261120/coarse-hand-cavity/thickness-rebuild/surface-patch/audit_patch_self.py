"""Compare exact self crossings along all changed scalp surface faces."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from rebuild import HERE, digest, install_for_float32, owner


def measured_pairs(points, faces, touched):
    triangles, _ = owner.float32_surface_records(points, faces)
    before = owner._audit_self_pairs_touching_faces(triangles, touched)
    return set(map(tuple, before["triangle_pairs"]))


def collapse(pairs, parents):
    return {tuple(sorted((int(parents[a]), int(parents[b])))) for a, b in pairs}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    install_for_float32()
    source = HERE / "cavity-batch-037/local-deep-restoration-03/rebuild.npz"
    path = args.candidate / "rebuild.npz"
    run = json.loads((args.candidate / "receipt.json").read_text())
    assert digest(source) == run["source_sha256"]
    assert digest(path) == run["candidate_sha256"]
    with np.load(source, allow_pickle=False) as f:
        old = dict(f)
    with np.load(path, allow_pickle=False) as f:
        new = dict(f)

    old_count = len(old["world"])
    faces = new["surface"]
    ancestry = new["surface_parent"]
    if "split_seed_faces" in new:
        split = np.flatnonzero(np.any(faces >= old_count, axis=1))
        old_split = set(new["split_seed_faces"][split].tolist())
        weights = np.repeat(new["split_seed_barycentric"], 3, axis=0)
        inserted = (old["world"][new["split_seed_nodes"]] * weights[:, :, None]).sum(axis=1)
        seed = np.vstack((old["world"], inserted.astype(np.float32)))
    else:
        split, old_split, seed = np.empty(0, dtype=np.intp), set(), old["world"]
    moved = set(np.flatnonzero(np.any(seed[faces] != new["world"][faces], axis=(1, 2))).tolist())
    if len(split):
        old_self = measured_pairs(old["world"], old["surface"], old_split)
        seed_self = measured_pairs(seed, faces, set(split.tolist()))
        old_keys = collapse(old_self, old["surface_parent"])
        seed_keys = collapse(seed_self, ancestry)
        geometry_changes = sorted(seed_keys - old_keys)
        assert not geometry_changes, geometry_changes
    else:
        old_self, seed_self, geometry_changes = set(), set(), []
    before = measured_pairs(seed, faces, moved)
    after = measured_pairs(new["world"], faces, moved)
    assert not after - before, sorted(after - before)
    print(json.dumps({
        "mesh_sha256": run["candidate_sha256"], "split_parent_faces": len(old_split),
        "moved_faces": len(moved), "source_pairs_touching_split": len(old_self),
        "seed_pairs_touching_split": len(seed_self), "new_seed_pair_identities": geometry_changes,
        "seed_pairs_touching_moved": len(before), "final_pairs_touching_moved": len(after),
        "new_final_raw_pairs": [],
    }, indent=2))


if __name__ == "__main__":
    main()
