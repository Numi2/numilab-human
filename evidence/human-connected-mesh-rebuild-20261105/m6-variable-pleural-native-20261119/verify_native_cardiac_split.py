"""Verify each captured split cardiac vertex, anatomy lineage and accepted state."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

from fit_cardiac_volume_grid import HERE, hash_file
from verify_cardiac_tube_membership import exact_inside, read_tubes

PRIOR = HERE.parent / "resting-clearance-and-atrium-003"
sys.path[:0] = [str(PRIOR), str(HERE.parent / "resting-surface-clearance-001")]
from audit import clearance
from cardiac_partition_motion import load

ZSTD = "/opt/homebrew/bin/zstd"


def retain_input(path, target, sha256):
    if path.is_file():
        assert hash_file(path) == sha256
        return path
    with target.open("xb") as handle:
        subprocess.run([ZSTD, "-dc", str(path) + ".zst"], stdout=handle, check=True)
    assert hash_file(target) == sha256
    return target


def decode_capture(mapped):
    header = clearance._PACK_HEADER
    directory = clearance._PACK_DIRECTORY
    magic, version, count, *_ = header.unpack_from(mapped)
    assert magic == b"MRVPACK2" and version == 2
    sections = {}
    for i in range(count):
        section = directory.unpack_from(mapped, header.size + i * directory.size)
        sections[section[0]] = section
    vertex, index, primitive, instance = (sections[x] for x in range(2, 6))
    assert tuple(row[5] for row in (vertex, index, primitive, instance)) == (80, 4, 64, 80)
    positions = np.ndarray((vertex[4], 20), dtype="<f4", buffer=mapped, offset=vertex[2])[:, :3]
    indices = np.ndarray(index[4], dtype="<u4", buffer=mapped, offset=index[2])
    for i in range(primitive[4]):
        first, count, _, instance_id, semantic, _, _, stable = struct.unpack_from(
            "<8I", mapped, primitive[2] + i * 64)
        ids = indices[first:first+count].reshape(-1, 3)
        transform = struct.unpack_from("<8f12I", mapped, instance[2] + instance_id * 80)
        yield stable, semantic, transform, ids, positions


def validate(geometry, low, source, coefficients, metadata, q):
    surfaces = {}
    spec_by_id = {r["stable_id"]: r for r in metadata["vertex_ranges"]}
    exact_rows = 0
    for stable, semantic, transform, faces, positions in decode_capture(geometry):
        if stable not in spec_by_id:
            continue
        ids = np.unique(faces)
        if not np.all(low[ids, 3] == 1):
            continue
        assert stable not in surfaces and np.isfinite(positions[ids]).all()
        tracked = source[stable]
        spec = spec_by_id[stable]
        row = spec["first_vertex"]
        count = spec["vertex_count"]
        assert len(ids) == count == len(tracked["vertices6"])
        assert np.array_equal(tracked["faces"], np.searchsorted(ids, faces)), stable
        assert transform[10] == 0 and transform[3] == 1.0
        expected = tracked["vertices6"][:, :3].astype(float)
        magnitude = np.abs(expected)
        for column in range(7):
            term = coefficients[row:row+count, column].astype(float) * float(q[column])
            expected += term
            magnitude += np.abs(term)
        xy = positions[ids].astype(float) + low[ids, :3].astype(float)
        upper = np.abs(expected - xy) + magnitude * 2.**-48
        reserve = np.maximum(magnitude * 2.**-40, 2.**-122)
        assert (upper <= reserve).all(), (stable, float((upper/reserve).max()))
        t = xy[np.searchsorted(ids, faces)]
        t -= xy[0]
        signed_volume = np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) / 6
        surfaces[stable] = dict(semantic_id=semantic, frozen_torso_pose=list(transform[:8]),
            vertices=len(ids), faces=len(faces), signed_volume_ml=float(signed_volume * 1e6),
            max_position_error_bound_m=float(upper.max()),
            max_fraction_of_position_arithmetic_reserve=float((upper/reserve).max()))
        exact_rows += len(ids)
    assert set(surfaces) == set(spec_by_id)
    assert len({tuple(r["frozen_torso_pose"]) for r in surfaces.values()}) == 1
    assert exact_rows == int((low[:, 3] == 1).sum())
    return surfaces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    receipt = json.loads(args.receipt.read_text())
    payload = receipt["accepted_cardiac_geometry_buffers"]
    source, coefficients, metadata, _ = load()
    assert receipt["physical_endpoint"] == "accepted" and receipt["surface_audit_endpoint"] == "passed"
    assert receipt["common_field_source_anatomy_payload_sha256"] == metadata["anatomy_payload_sha256"]
    assert receipt["common_field_map_sha256"] == metadata["map"]["sha256"]
    coordinate = Path(payload["common_coordinates"]["path"])
    assert hash_file(coordinate) == payload["common_coordinates"]["sha256"]
    raw = np.fromfile(coordinate, "<f4")
    q = raw[:7]
    status, iterations, domain, _ = map(int, raw.view("<u4")[8:12])
    box_file = coordinate.parent.parent / "common-field-domain-boxes-f32.bin"
    assert hash_file(box_file) == metadata["domain_boxes"]["sha256"]
    assert box_file.stat().st_size % 64 == 0
    boxes = box_file.stat().st_size // 64
    assert status == 0 and np.isfinite(raw[12])
    if domain >= boxes:
        tubes = read_tubes()
        assert exact_inside(q, tubes[domain-boxes])
    with tempfile.TemporaryDirectory(dir=HERE) as work:
        work = Path(work)
        pack = retain_input(Path(receipt["accepted_pack_path"]), work / "capture.mrvpack",
                            receipt["pack_file_sha256"])
        low_metadata = payload["common_position_low"]
        low_path = retain_input(Path(low_metadata["path"]), work / "positions.bin", low_metadata["sha256"])
        low = np.memmap(low_path, "<f4", mode="r").reshape(receipt["vertex_count"], 4)
        with pack.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            surfaces = validate(mapped, low, source, coefficients, metadata, q)
    result = dict(receipt_sha256=hash_file(args.receipt), capture_sha256=receipt["pack_file_sha256"],
        step=receipt["accepted_step"], accepted_simulated_seconds=receipt["accepted_time_s"],
        domain=domain, tube_id=domain-boxes if domain>=boxes else None,
        coordinate=q.tolist(), normalized_volume_residual=float(raw[12]), surfaces=surfaces)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
