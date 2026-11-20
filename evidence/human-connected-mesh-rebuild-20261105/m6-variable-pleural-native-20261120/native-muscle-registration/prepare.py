"""Re-emit the two source-bank fits that passed both retained geometry screens."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
FIT = WORK / "muscle-shell-registration-018"
REFERENCE = WORK / "resting-clearance-and-atrium-017/floor-native-live-parent-026"
KEEP = {67, 164}
SOURCE_SHA = "8874b3fede93825d2e4ddb248231dc9d2c598e0155f8473c624fcb935d6b1042"
sys.path[:0] = [str(WORK / "resting-clearance-and-atrium-006"),
                str(WORK / "resting-surface-clearance-001/src")]
from register_tissue_bellies import quality, source
from tissue_payload_owner import VERTEX, _decode_nhtiss4, verify_source_preservation


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, payload):
    with path.open("xb") as out:
        out.write(payload)


def source_paths(plan):
    argv = plan["argv"]
    return tuple(Path(argv[argv.index(flag) + 1]).resolve()
                 for flag in ("--soft-tissue-payload", "--resting-anatomy-receipt"))


def candidate_vertices(vertices, bank, delta):
    rows, faces = bank["row_ids"], bank["source_faces"]
    original = bank["points"]
    assert delta.shape == original.shape and np.isfinite(delta).all()
    assert np.array_equal(original.astype("<f4"), vertices[rows, :3])
    revised = (original + delta).astype("<f4")
    changed = np.any(revised != original, axis=1)
    assert changed.any()
    fixed = ~np.any(delta, axis=1)
    assert np.array_equal(revised[fixed], original[fixed])
    area, volume = quality(original, revised.astype(np.float64), faces)
    assert area > 0.65 and volume < 0.001

    # Recompute the local normal increment on the same indexed triangles.
    # Accumulate across exact coordinate seams and preserve the original
    # normal records at pinned source rows.
    original_groups, grouped = np.unique(original, axis=0, return_inverse=True)
    revised_groups = np.unique(revised, axis=0)
    assert len(original_groups) == len(revised_groups)
    for group in np.unique(grouped[changed]):
        assert np.all(revised[grouped == group] == revised[np.flatnonzero(grouped == group)[0]])
    def normal_vectors(points):
        triangles = points[faces].astype(np.float64)
        return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])

    old_area = normal_vectors(original)
    new_area = normal_vectors(revised)
    affected = np.any(changed[faces], axis=1)
    assert (np.linalg.norm(new_area[affected], axis=1) > 0).all()
    accum = np.zeros((grouped.max() + 1, 3))
    increment = np.zeros_like(accum)
    for corner in range(3):
        np.add.at(accum, grouped[faces[:, corner]], old_area)
        np.add.at(increment, grouped[faces[affected, corner]], new_area[affected] - old_area[affected])
    adjusted = changed & (np.linalg.norm(accum[grouped], axis=1) > 1.e-20)
    g = grouped[adjusted]
    old_normals = vertices[rows[adjusted], 3:6].astype(np.float64)
    new_normals = old_normals + increment[g] / np.linalg.norm(accum[g], axis=1)[:, None]
    norm = np.linalg.norm(new_normals, axis=1)
    assert np.all(np.isfinite(norm)) and np.all(norm > 1.e-12)
    vertices[rows, :3] = revised
    vertices[rows[adjusted], 3:6] = (new_normals / norm[:, None]).astype("<f4")
    return {"moved_vertices": int(changed.sum()), "affected_faces": int(affected.sum()),
            "maximum_move_mm": float(1000 * np.linalg.norm(revised-original, axis=1).max()),
            "minimum_projected_area_ratio": area, "relative_volume_change": volume}


def main():
    plan_path = REFERENCE / "launch.json"
    plan = json.loads(plan_path.read_text())
    asset, receipt_path = source_paths(plan)
    assert digest(asset) == SOURCE_SHA
    assert json.loads((FIT / "registration-observation.json").read_text())["source_sha256"] == SOURCE_SHA
    verified = FIT / "targeted-first-surrounding.json"
    checks = json.loads(verified.read_text())
    assert set(checks["passes_surrounding"]) == KEEP

    source_receipt_text = receipt_path.read_text()
    source_receipt = json.loads(source_receipt_text)
    bindings = source_receipt["passive_tissue_parent_bindings"]
    assert bindings["child_payload_sha256"] == SOURCE_SHA
    assert not (KEEP & {r["stable_id"] for r in bindings["surfaces"]})
    assert not (KEEP & {t[1] for r in bindings["anatomy_surfaces"] for row in r["vertices"] for t in row})
    assert all(Path(s["path"]).is_absolute() for s in bindings["source_payloads"])
    assert all(digest(Path(s["path"])) == s["sha256"] for s in bindings["source_payloads"])

    raw = asset.read_bytes()
    header, _, _, start, _ = _decode_nhtiss4(raw)
    count = header[4]
    end = start + count * VERTEX.size
    payload = bytearray(raw)
    vertices = np.frombuffer(payload, dtype="<f4", count=count * 14, offset=start).reshape(count, 14)
    reference_vertices = np.frombuffer(raw, dtype="<u4", count=count * 14, offset=start).reshape(count, 14)
    banks = source(asset)
    results = {}
    fit = FIT / "targeted-first.npz"
    with np.load(fit, allow_pickle=False) as z:
        for stable_id in sorted(KEEP):
            results[stable_id] = candidate_vertices(vertices, banks[stable_id], z[f"bank{stable_id}"])
    assert bytes(payload[:start]) == raw[:start]
    assert bytes(payload[end:]) == raw[end:]
    assert np.array_equal(vertices.view("<u4")[:, 6:], reference_vertices[:, 6:])
    identity = verify_source_preservation(raw, bytes(payload), KEEP)

    digest_candidate = hashlib.sha256(payload).hexdigest()
    field = re.compile(r'"child_payload_sha256"\s*:\s*"(?P<digest>[0-9a-f]{64})"')
    matching = [match for match in field.finditer(source_receipt_text)
                if match.group("digest") == SOURCE_SHA]
    assert len(matching) == 1
    begin, finish = matching[0].span("digest")
    derived_receipt = source_receipt_text[:begin] + digest_candidate + source_receipt_text[finish:]
    assert len(derived_receipt) == len(source_receipt_text)
    replay = json.loads(derived_receipt)
    assert replay["passive_tissue_parent_bindings"]["child_payload_sha256"] == digest_candidate
    replay["passive_tissue_parent_bindings"]["child_payload_sha256"] = SOURCE_SHA
    assert replay == source_receipt

    save(HERE / "registered-muscle.nhtissue", payload)
    save(HERE / "resting-anatomy-receipt.json", derived_receipt.encode())
    report = {"source_asset": str(asset), "source_sha256": SOURCE_SHA,
              "reference_launch_sha256": digest(plan_path), "source_receipt": str(receipt_path),
              "source_receipt_sha256": digest(receipt_path), "fit_sha256": digest(fit),
              "surrounding_audit_sha256": digest(verified), "candidate_sha256": digest_candidate,
              "candidate_receipt_sha256": digest(HERE / "resting-anatomy-receipt.json"),
              "untouched_parents": len(bindings["source_payloads"]), "tissue_owner_preservation": identity,
              "banks": results}
    save(HERE / "source-replay.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(results))


if __name__ == "__main__":
    main()
