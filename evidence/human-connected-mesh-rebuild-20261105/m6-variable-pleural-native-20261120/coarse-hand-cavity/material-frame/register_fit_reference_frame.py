"""Place the repaired parent rows in the invariant material reference frame."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from build_hand_trial import SOURCE, digest
from fit_hand_cavity import PARENT, table
from fit_hand_interface import coordinates

ROWS = 2411
HEADER = SOURCE / "NumiHumanWholeBodySkinReduction.hpp"


def source_layer(text, symbol):
    start = text.index(symbol+"={{")+len(symbol)+3
    end = text.index("}};",start)
    points = np.array([list(map(float,r.split(","))) for r in
                       re.findall(r"\{([^{}]+)\}",text[start:end])])
    assert points.shape == (ROWS,3) and np.isfinite(points).all(), symbol
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    nodes = PARENT.with_name(PARENT.name+".nodes.csv")
    fit = json.loads(args.fit.read_text())
    assert fit["tangent_plane_fit_pass"]
    assert fit["registered_parent_nodes_sha256"] == digest(nodes)
    original = coordinates(table(nodes),"rest")
    text = HEADER.read_text()
    local = np.concatenate([source_layer(text,k) for k in
                            ("kWholeBodySkinInnerWorld","kWholeBodyFatDeepWorld")])
    world = original[ROWS:]
    local_center, world_center = local.mean(axis=0),world.mean(axis=0)
    u,_,vt = np.linalg.svd((local-local_center).T @ (world-world_center))
    rotation = u @ vt
    assert np.linalg.det(rotation) > 0., "improper rigid placement"
    translation = world_center-local_center @ rotation
    max_error = float(np.linalg.norm(local @ rotation + translation-world,axis=1).max())
    assert max_error < 1.e-10, max_error
    for row in fit["candidate_nodes"]:
        ref = np.array(row.pop("source_world_m"),float)
        delta = np.array(row.pop("world_delta_m"),float)
        assert np.array_equal(ref,original[row["local_node"]])
        row["material_reference_m"] = ((ref-translation) @ rotation.T).tolist()
        row["material_delta_m"] = (delta @ rotation.T).tolist()
    fit["coordinate_frame"] = "undeformed_registered_skin_fat_material_reference"
    fit["coordinate_frame_provenance"] = {
        "fit_sha256": digest(args.fit),
        "material_layer_header_sha256": digest(HEADER),
        "world_reference_row_count": len(world),
        "maximum_reference_frame_residual_m": max_error,
    }
    with args.output.open("x") as f:
        json.dump(fit,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(fit["coordinate_frame_provenance"]))


if __name__=="__main__":
    main()
