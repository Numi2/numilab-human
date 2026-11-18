"""Localize exact late skin fold witnesses to source banks and pose neighborhoods."""
from collections import Counter
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from prepare_skin import HERE, capture, decode

def main():
    pack = HERE / "rebound-original-16/native/accepted-geometry/step-8000.mrvpack"
    capture_receipt, points, faces = capture(pack)
    exact = json.loads((HERE / "rebound-original-16-exact.json").read_text())
    assert capture_receipt["pack_file_sha256"] == exact["accepted_pack_sha256"]
    model = decode((HERE / "skin-rebound.nhskin").read_bytes())
    owner = model["binds"]["body"][np.argmax(model["weights"], axis=1)]
    pose_rows = {int(row["body_index"]): row for row in capture_receipt["accepted_body_poses"]}
    pairs = exact["self_intersections"]["triangle_pairs"]
    ids = faces[pairs]
    centers = points[ids].mean(axis=(1, 2))
    close = np.linalg.norm(centers[:, None] - centers[None, :], axis=2) <= .012
    _, groups = connected_components(coo_matrix(close))
    regions = []
    for i in np.unique(groups):
        mask = i == groups
        indices, cluster = ids[mask], centers[mask]
        region_owners = Counter(map(int, owner[np.unique(indices)]))
        regions.append(dict(
            witness_pairs=int(mask.sum()),
            min_world_m=cluster.min(axis=0).tolist(),
            max_world_m=cluster.max(axis=0).tolist(),
            owners=[{"body_index": k, "vertices": n, "name": pose_rows.get(k, {}).get("name")}
                    for k,n in region_owners.most_common()],
            source_face_pairs=[row for row, include in zip(pairs, mask) if include]))
    regions.sort(key=lambda row: -row["witness_pairs"])
    result = {"accepted_pack_sha256": exact["accepted_pack_sha256"],
              "accepted_time_s": exact["accepted_time_s"],
              "method": "Exact intersections accumulated in 12 mm connected components by pair barycenter. Owner is the dominant NHSKIN binding at each unique incident vertex.",
              "regions": regions}
    (HERE / "skin-late-witness-regions.json").write_text(json.dumps(result, indent=2)+"\n")
    print([(r["witness_pairs"], [(o["body_index"], o["vertices"]) for o in r["owners"][:5]]) for r in regions])
if __name__ == "__main__":
    main()
