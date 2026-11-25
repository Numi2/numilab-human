"""Rebuild final face IDs from the pinned donor tetrahedra of each winning tissue group."""

import argparse
import json
from pathlib import Path

import numpy as np
from verify_final_tissue_labels import digest, verify


def reconcile(volume, registration, scene, output, lineage):
    if output.exists() or lineage.exists() or output.resolve() == lineage.resolve():
        raise ValueError("Choose new paths for the labels and their lineage")
    invalid = {}
    verify(volume, registration, invalid)
    env = json.loads((scene / "launch.json").read_bytes())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    donors_file = Path(env[prefix + "PARENT_LABELS_CSV"])
    if digest(donors_file) != env[prefix + "PARENT_LABELS_SHA256"]:
        raise ValueError("Unpinned source tetrahedron labels")
    donors = np.loadtxt(donors_file, delimiter=",", skiprows=1, dtype=np.int64)
    labels_file = volume / "current-child.labels.csv"
    labels = np.loadtxt(labels_file, delimiter=",", skiprows=1, dtype=np.int64)
    if not np.array_equal(donors[:, 0], np.arange(len(donors))):
        raise ValueError("Mismatched source tetrahedron order")
    reg = json.loads((registration / "result.json").read_bytes())["environment"]
    changes = []
    for role, material in (("Skin", 15), ("Fat", 16)):
        tag = "NUMI_H706_FRACTIONAL_P1_" + role.upper()
        if env[tag + "_SHA256"] != reg[tag + "_SHA256"]:
            raise ValueError("P1 scene and registration disagree")
        with np.load(reg[tag + "_NPZ"]) as s:
            offsets, ids, groups, numerator, denominator, parent_offsets, parents = (
                s[k]
                for k in (
                    "cell_group_offsets",
                    "cell_group_ids",
                    "group_tuples",
                    "cell_group_p1_score_numerators",
                    "cell_group_p1_score_denominators",
                    "cell_group_parent_tet_offsets",
                    "cell_group_parent_tet_rows",
                )
            )
        rows = np.flatnonzero(labels[:, 1] == material)
        for row in invalid[role]:
            i = np.searchsorted(rows, row)
            cols = np.arange(offsets[i], offsets[i + 1])
            scores = numerator[cols] / denominator[cols]
            winners = cols[scores == scores.max()]
            matches = winners[(groups[ids[winners]] == labels[row, 1:4]).all(axis=1)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous anatomical ownership at child {row}")
            slot = matches[0]
            src = donors[parents[parent_offsets[slot] : parent_offsets[slot + 1]]]
            if not len(src) or not (src[:, 1:4] == labels[row, 1:4]).all():
                raise ValueError(f"Unmatched contributing donor at child {row}")
            faces = np.unique(src[:, 4])
            face = int(faces[0]) if len(faces) == 1 else 0x80000000 + int(ids[slot])
            if face == labels[row, 4]:
                raise ValueError(
                    f"Stable face conflicts with the source P1 ledger at child {row}"
                )
            changes.append((int(row), int(labels[row, 4]), face))
            labels[row, 4] = face
    with labels_file.open() as stream:
        header = stream.readline().strip()
    np.savetxt(output, labels, fmt="%d", delimiter=",", header=header, comments="")
    packet = np.asarray(changes, dtype=np.int64).reshape(-1, 3)
    np.savez_compressed(
        lineage,
        cell_row=packet[:, 0],
        previous_face=packet[:, 1],
        certified_face=packet[:, 2],
    )
    return len(packet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("volume", "registration", "scene", "output", "lineage"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    print(
        reconcile(args.volume, args.registration, args.scene, args.output, args.lineage)
    )
