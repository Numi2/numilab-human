"""Try topology-only repair of late skin self-crossings without moving supports."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.with_name("resting-clearance-and-atrium-007")
sys.path[:0] = [str(HERE.with_name("resting-clearance-and-atrium-006")), str(BASE)]
import repair_skin_connectivity as owner
from fit_edge_dependency_repair import capture_in_memory, dependencies


def main():
    owner.install_for_float32()
    active_run = BASE / "recomputed-support-native"
    model = owner.decode((BASE / "recomputed-support-skin.nhskin").read_bytes())
    reference = owner.decode((BASE.with_name("resting-clearance-and-atrium-006") /
                              "source-clearance-patches.nhskin").read_bytes())
    _, faces_before, _, legacy, supports, _ = dependencies(reference)
    points, faces = model["verts"][:, :3], model["inds"].copy()
    assert np.array_equal(faces, faces_before)
    protected = set(np.flatnonzero(legacy | supports))
    source_ties = owner.clear._exact_coordinate_tied_source_vertex_ids(points)
    protected.update(source_ties)
    topo = owner.topology(faces)
    report = json.loads((active_run / "step-8000-skin-exact.json").read_text())
    seeds = {i for pair in report["self_intersections"]["triangle_pairs"] for i in pair}
    active = set(seeds)
    for row in seeds:
        for a, b, _ in owner.edges(faces[row]):
            active.update(topo[min(a,b), max(a,b)])
    selected = np.array(sorted(active))
    source = owner.Geometry(points, faces, (), selected)
    native_pack = active_run / "native/accepted-geometry/step-8000.mrvpack"
    owner.capture = capture_in_memory
    frame = owner.prepare_case(native_pack, report, faces, selected)
    controls = [source, frame]
    fixed = set()
    edits = []
    for i in sorted(seeds):
        if i in fixed or protected.intersection(faces[i]):
            continue
        best = None
        for a,b,c in owner.edges(faces[i]):
            halves = topo[min(a,b),max(a,b)]
            if len(halves) != 2:
                continue
            j = next(row for row in halves if row != i)
            if j in fixed or protected.intersection(faces[j]):
                continue
            u,v,d = halves[j]
            if (u,v) != (b,a) or (min(c,d),max(c,d)) in topo:
                continue
            ids = np.array([i,j])
            changed = np.array([[c,a,d],[d,b,c]], dtype=faces.dtype)
            if not all(owner.quality(g.points, faces[ids], changed) for g in controls):
                continue
            results = [g.inspect(ids, changed) for g in controls]
            if any(result is None for result in results):
                continue
            rank = (len(results[1][1]), results[1][2])
            if rank[0] > 0 and (best is None or rank > best[0]):
                best = (rank, ids, changed, results)
        if best is None:
            continue
        rank, ids, rows, results = best
        for row in ids:
            for a,b,_ in owner.edges(faces[row]):
                topo[min(a,b),max(a,b)].pop(row)
        for row,face in zip(ids,rows):
            for a,b,c in owner.edges(face):
                topo[min(a,b),max(a,b)][row] = (a,b,c)
        faces[ids] = rows
        for g,result in zip(controls,results):
            g.accept(result)
        fixed.update(ids)
        edits.append({"face_ids": ids.tolist(), "replacements": rows.tolist(),
                      "late_self_pairs_removed": rank[0], "late_tissue_pairs_removed": rank[1]})
    identity = hashlib.sha256(model["raw"]).hexdigest()
    out = dict(source_sha256=identity, native_sha256=report["accepted_pack_sha256"],
               self_pairs_before=len(frame.baseline), self_pairs_after=len(frame.self_hits),
               remaining_self_pairs=sorted(frame.self_hits), protected_vertices=len(protected),
               changed_faces=len(fixed), operation=edits, native_qualified=False)
    if edits:
        start = model["indexoff"]
        indices = faces.astype("<u4").tobytes()
        trial = model["raw"][:start] + indices + model["raw"][start+len(indices):]
        decoded = owner.decode(trial)
        assert np.array_equal(decoded["inds"], faces)
        assert np.array_equal(decoded["verts"], model["verts"])
        assert np.array_equal(decoded["weights"], model["weights"])
        out["candidate_sha256"] = hashlib.sha256(trial).hexdigest()
        Path(sys.argv[1]).write_bytes(trial)
    (HERE / "late-self-diagonals.json").write_text(json.dumps(out, indent=2) + "\n")
    print({k:v for k,v in out.items() if k not in ("operation","remaining_self_pairs")}, flush=True)


if __name__ == "__main__":
    main()
