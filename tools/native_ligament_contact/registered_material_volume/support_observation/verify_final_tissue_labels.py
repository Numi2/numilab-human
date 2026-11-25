"""Verify the final native skin and fat child labels against their registered P1 donor support."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(volume, registration, witnesses=None):
    report = json.loads((registration / "result.json").read_bytes())
    labels_file = volume / "current-child.labels.csv"
    labels = np.loadtxt(labels_file, delimiter=",", skiprows=1, dtype=np.int64)
    native = json.loads((volume / "native-import.json").read_bytes())
    if (
        not native["native_imported"]
        or len(labels) != native["cells"]
        or native["label_sha256"] != digest(labels_file)
    ):
        raise ValueError("Native import and final labels differ")
    if (
        not np.array_equal(labels[:, 0], np.arange(len(labels)))
        or not np.isin(labels[:, 1], [15, 16]).all()
    ):
        raise ValueError("Invalid or unrecognized material rows")
    output = {}
    for role, material in (("Skin", 15), ("Fat", 16)):
        prefix = "NUMI_H706_FRACTIONAL_P1_" + role.upper()
        path = Path(report["environment"][prefix + "_NPZ"])
        if digest(path) != report["environment"][prefix + "_SHA256"]:
            raise ValueError("Unpinned P1 donor support")
        rows = np.flatnonzero(labels[:, 1] == material)
        geometry = registration / (role.lower() + "-role-geometry.npz")
        if (
            digest(geometry)
            != report["environment"][
                "NUMI_H706_FRACTIONAL_GEOMETRY_" + role.upper() + "_SHA256"
            ]
        ):
            raise ValueError("Unpinned tissue row ordering")
        with np.load(geometry) as g:
            if not np.array_equal(g["combined_child_rows"], rows):
                raise ValueError("Material source rows differ")
        with np.load(path) as s:
            groups, face_offsets, faces = (
                s[k]
                for k in (
                    "group_tuples",
                    "cell_group_stable_face_offsets",
                    "cell_group_stable_face_rows",
                )
            )
            offsets, gids, scores = (
                s[k]
                for k in (
                    "cell_group_offsets",
                    "cell_group_ids",
                    "cell_group_p1_score_numerators",
                )
            )
            score_denominators = s["cell_group_p1_score_denominators"]
        if np.any(score_denominators <= 0):
            raise ValueError("Invalid P1 score denominator")
        scores = scores / score_denominators
        count = np.diff(offsets)
        if (
            len(count) != len(rows)
            or np.any(count <= 0)
            or offsets[0] != 0
            or offsets[-1] != len(gids)
            or len(face_offsets) != len(gids) + 1
            or face_offsets[-1] != len(faces)
        ):
            raise ValueError("Incomplete P1 support")
        owner = np.repeat(np.arange(len(rows)), count)
        winner = scores == np.maximum.reduceat(scores, offsets[:-1])[owner]
        candidate = np.flatnonzero(
            winner & (groups[gids] == labels[rows[owner], 1:4]).all(axis=1)
        )
        valid = np.zeros(len(rows), dtype=bool)
        for start in range(0, len(candidate), 65536):
            cols = candidate[start : start + 65536]
            fbase = face_offsets[cols].astype(np.int64)
            size = face_offsets[cols + 1] - fbase
            keep = size > 0
            cols, fbase, size = cols[keep], fbase[keep], size[keep]
            if not len(cols):
                continue
            cell = owner[cols]
            desired = labels[rows[cell], 4]
            cumulative = np.r_[0, np.cumsum(size)]
            actual = faces[
                np.repeat(fbase - cumulative[:-1], size) + np.arange(cumulative[-1])
            ]
            present = np.logical_or.reduceat(
                actual == np.repeat(desired, size), cumulative[:-1]
            )
            multi = (size > 1) & (actual[cumulative[:-1]] != actual[cumulative[1:] - 1])
            valid[
                cell[
                    present
                    | (multi & (desired == 0x80000000 + gids[cols].astype(np.int64)))
                ]
            ] = True
        missing = rows[~valid]
        if witnesses is not None:
            witnesses[role] = missing
        covered = np.zeros(len(rows), dtype=bool)
        covered[owner[candidate]] = True
        output[role] = {
            "wrong_donor_group": int((~covered).sum()),
            "cells": len(rows),
            "invalid_final_labels": len(missing),
            "first_witnesses": missing[:12].tolist(),
            "before_carrier_replay": len(
                report["roles"][role]["inadmissible_label_rows"]
            ),
        }
    return {"native_label_sha256": native["label_sha256"], "roles": output}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("volume", type=Path)
    parser.add_argument("registration", type=Path)
    args = parser.parse_args()
    result = verify(args.volume, args.registration)
    print(json.dumps(result, indent=2))
    raise SystemExit(
        any(role["invalid_final_labels"] for role in result["roles"].values())
    )
