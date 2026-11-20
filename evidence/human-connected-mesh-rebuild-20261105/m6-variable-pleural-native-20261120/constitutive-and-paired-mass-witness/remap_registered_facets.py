"""Rebind native material boundary face rows after a connectivity-only tetrahedron splice."""
from collections import defaultdict
import csv
import hashlib
import io
import json
import subprocess

import numpy as np

from export_repaired_tetgen import HERE, array_hash, file_hash
from candidate import COMPILER, load

LOCAL_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
ZSTD = "/opt/homebrew/bin/zstd"


def oriented_face(tet, nodes):
    matches = [tuple(int(tet[i]) for i in face) for face in LOCAL_FACES
               if set(int(tet[i]) for i in face) == nodes]
    assert len(matches) == 1
    a, b, c = matches[0]
    return -1 if ((a > b) + (a > c) + (b > c)) % 2 else 1


def main():
    receipt = json.loads((HERE / "repaired-tetgen/geometry-receipt.json").read_text())
    _, _, tets, materials, origins, cavities, _, old_tets, old_materials = load()
    assert array_hash(tets, "<i4") == receipt["tetrahedron_array_sha256"]
    old_to_new = np.full(len(old_tets), -1, dtype=np.int32)
    retained = np.flatnonzero(origins >= 0)
    old_to_new[origins[retained]] = retained
    # Sequential splices can share an original ancestor. Search all surviving
    # descendants, then resolve the unique oriented material facet.
    descendants = defaultdict(set)
    for cavity, parents in cavities.items():
        replacements = set(np.flatnonzero(origins == cavity).tolist())
        for previous in parents:
            descendants[previous].update(replacements)
    source = COMPILER / "volume-export/volume/facet-ancestry.csv.zst"
    recovery = json.loads(source.with_suffix(".zst.recovery.json").read_text())
    source_sha = file_hash(source)
    assert source.stat().st_size == recovery["archive_bytes"]

    temporary = HERE / "repaired-tetgen/facet-ancestry.csv.zst.incomplete"
    emitted = HERE / "repaired-tetgen/facet-ancestry.csv.zst"
    assert not temporary.exists() and not emitted.exists()
    source_bytes = hashlib.sha256()
    output_bytes = hashlib.sha256()
    checked = changed_parent = changed_number = 0
    source_read = subprocess.Popen([ZSTD, "-qdc", source], stdout=subprocess.PIPE)
    process = subprocess.Popen([ZSTD, "-q", "-3", "-o", temporary], stdin=subprocess.PIPE)
    try:
        def lines():
            for line in source_read.stdout:
                source_bytes.update(line)
                yield line.decode("utf-8")
        header = (
            "source_partition,shell_face_row,face_marker,face_source_role,face_source_row,"
            "face_exact_source_row,face_source_original_id,face_source_layer,face_source_body,"
            "face_source_bone,face_ancestry_kind,face_run_original_id,face_source_face_id,"
            "face_ancestry_source_rows,face_parent_tet_row,face_parent_material_index,"
            "parent_source_node_rows,joint_element_id,child_tet_row,joint_node0,joint_node1,"
            "joint_node2,face_ancestry_source_original_ids,face_ancestry_parent_face_rows"
        ).split(",")
        source_rows = csv.DictReader(lines())
        assert source_rows.fieldnames == header
        chunk = io.StringIO(newline="")
        writer = csv.DictWriter(chunk, header, lineterminator="\n")
        writer.writeheader()

        def flush():
            data = chunk.getvalue().encode("utf-8")
            output_bytes.update(data)
            process.stdin.write(data)
            chunk.seek(0)
            chunk.truncate(0)

        for row in source_rows:
            previous = int(row["child_tet_row"])
            assert 0 <= previous < len(old_tets)
            assert int(row["joint_element_id"]) == previous + 1
            nodes = {int(row[f"joint_node{corner}"]) - 1 for corner in range(3)}
            assert len(nodes) == 3
            next_row = int(old_to_new[previous])
            if next_row < 0:
                expected_sign = oriented_face(old_tets[previous], nodes)
                candidates = [
                    tet for tet in descendants[previous]
                    if materials[tet] == old_materials[previous] and nodes.issubset(tets[tet])
                    and oriented_face(tets[tet], nodes) == expected_sign
                ]
                assert len(candidates) == 1, (previous, nodes, candidates)
                next_row = int(candidates[0])
                changed_parent += 1
            else:
                assert nodes.issubset(tets[next_row]), (previous, next_row, nodes)
            row["child_tet_row"] = next_row
            row["joint_element_id"] = next_row + 1
            writer.writerow(row)
            changed_number += int(previous != next_row)
            checked += 1
            if checked % 4096 == 0:
                flush()
        flush()
        source_read.stdout.close()
        process.stdin.close()
        assert source_read.wait() == 0
        assert process.wait() == 0
        assert source_bytes.hexdigest() == recovery["sha256"], "original facet byte hash differs"
        verify = hashlib.sha256()
        with subprocess.Popen([ZSTD, "-qdc", temporary], stdout=subprocess.PIPE) as restored:
            for data in iter(lambda: restored.stdout.read(1 << 20), b""):
                verify.update(data)
            assert restored.wait() == 0
        assert verify.hexdigest() == output_bytes.hexdigest()
        temporary.rename(emitted)
    finally:
        for child in (source_read, process):
            if child.poll() is None:
                child.terminate()
                child.wait()
    result = {
        "boundary_rows": checked,
        "rows_renumbered": changed_number,
        "rows_rebound_to_replacement_cells": changed_parent,
        "input_compressed_sha256": source_sha,
        "input_csv_sha256": source_bytes.hexdigest(),
        "emitted_compressed_sha256": file_hash(emitted),
        "emitted_csv_sha256": output_bytes.hexdigest(),
        "child_mesh_ele_sha256": receipt["mesh_ele_sha256"],
        "scope": "exact oriented topology and existing source-face row identities; posed registration and mass-flow gates remain",
    }
    (emitted.parent / "facet-ancestry-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
