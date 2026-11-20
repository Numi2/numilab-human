"""Reconstruct the native attachment stiffness and its static vertical load bound."""
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import ROOT, digest, read_model, read_table

HEADER = (ROOT.parent / "resting-clearance-and-atrium-017/floor-scene-build"
          / "NumiHumanWholeBodySkinReduction.hpp")
REFERENCE = ROOT / "thickness-rebuild-034/compiled-reference"
REPLAY = ROOT / "source-physical-bed-seed-031/contact-cache-128/stdout.log"


def event(path, prefix):
    with path.open() as stream:
        for line in stream:
            if line.startswith(prefix):
                return dict(re.findall(r"(\w+)=([^\s]+)", line))
    raise ValueError(f"{prefix} missing from {path}")


def header_rows(text, symbol, fields):
    block = text.split(symbol + "=")[1].split(";")[0]
    return np.array([list(map(int, row)) for row in re.findall(fields, block)], dtype=np.int32)


def main():
    model, _, world, _, _, pins, _ = read_model()
    n = model.outer
    text = HEADER.read_text()
    triplet = r"\{\s*(\d+)u,\s*(\d+)u,\s*(\d+)u\s*\}"
    faces = header_rows(text, "kWholeBodySkinFaces", triplet)
    sites = header_rows(text, "kWholeBodySkinDeepAttachmentSites", triplet)
    field = text.split("kWholeBodyFatThicknessM=")[1].split(";")[0].strip().strip("{}")
    thickness = np.array([float(item) for item in field.split(",") if item.strip()])
    assert len(thickness) == n and np.all(thickness > 0.)
    assert len(np.unique(sites[:, 0])) == len(sites) == 14

    startup = REFERENCE / "stdout.log"
    with startup.open() as stream:
        native_sites = {tuple(map(int, re.search(
            r"local_vertex=(\d+)", line).groups() + re.search(
            r"source_vertex=(\d+) donor_body=(\d+)", line).groups()))
            for line in stream if line.startswith("whole_body_skin_attachment ")}
    assert native_sites == set(map(tuple, sites))
    binding = event(startup, "whole_body_skin_elastic_attachment_binding")
    assert int(binding["explicit_bound_rows"]) == len(sites) and binding["hard_ties"] == "0"
    mu = float(event(startup, "whole_body_skin_anchor_quality")["shear_prior_pa"])
    positions = world[2*n:]
    triangles = positions[faces]
    area = .5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
    lumps = np.bincount(faces.ravel(), weights=np.repeat(area / 3., 3), minlength=n)
    indices = sites[:, 0]
    # Match the compiled shader's two Float32 factors. Exported current
    # vertices differ from the pre-compiler tributary areas by rounding.
    stiffness = (lumps[indices].astype(np.float32) * (mu / thickness[indices]).astype(np.float32)).astype(float)
    mass = np.array([float(r["compiled_lumped_mass_kg"])
                     for r in read_table(REFERENCE / "shell-mass.csv.nodes.csv")])
    observed = float(binding["maximum_explicit_k_dt2_over_node_mass"])
    reconstructed = float(np.max(stiffness * model.dt**2 / mass[2*n + indices]))
    assert abs(reconstructed / observed - 1.) < 1.e-4
    payload = model.body_weight - 9.81 * mass.sum()
    stiffness_by_body = defaultdict(float)
    for (_, _, body), k in zip(sites, stiffness, strict=True):
        stiffness_by_body[int(body)] += float(k)
    contact = event(REPLAY, "resting_fascia_fem_terminal_audit=passed")
    pins.update({str(p): digest(p) for p in (HEADER, startup, REPLAY)})
    result = {"input_sha256": pins, "elastic_rows": len(sites), "stiffness_by_donor_n_m": stiffness_by_body,
              "observed_maximum_k_dt2_over_m": observed, "reconstructed_maximum_k_dt2_over_m": reconstructed,
              "attachment_vertices": sites.tolist(), "attachment_stiffness_n_m": stiffness.tolist(),
              "shell_mass_kg": float(mass.sum()), "weight_excluding_shell_n": float(payload),
              "combined_stiffness_n_m": float(stiffness.sum()),
              "minimum_largest_attachment_extension_if_springs_alone_m": float(payload / stiffness.sum()),
              "generous_upper_vertical_force_at_one_local_layer_thickness_n": float(stiffness @ thickness[indices]),
              "native_terminal_contact_force_sum_n": float(contact["body_contact_normal_force_N"]),
              "native_terminal_closed_contact_rows": int(contact["body_contact_closed_samples"]),
              "native_replay_root_count": int(contact["accepted_steps"]),
              "bounds": "Static conditional force bounds for the spring rows only. Contact sum is across all closed tissue-to-body rows in a separate 2 ms native matched replay. Neither quantity is a universal limit on future distributed contact load."}
    Path("attachment-load-path.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["combined_stiffness_n_m"], result["minimum_largest_attachment_extension_if_springs_alone_m"],
          result["native_terminal_contact_force_sum_n"])


if __name__ == "__main__":
    main()
