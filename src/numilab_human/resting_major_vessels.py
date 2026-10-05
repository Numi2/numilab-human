"""Append named source major vessels to the existing resting anatomy payload.

The explicit side, source concept and existing MyoSim body frame are retained.
These surfaces identify regional CVSim compartments; they are not additional
hydraulic compartments or independently simulated vessel walls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .resting_passive_viscera import append_viscera


# concept, exact source name, existing body owner, source member count, CVSim ID
VESSELS = (
    ("FMA3941", "right common carotid artery", "neck", 1, 3),
    ("FMA4058", "left common carotid artery", "neck", 1, 3),
    ("FMA3949", "right internal carotid artery", "head", 1, 3),
    ("FMA4062", "left internal carotid artery", "head", 1, 3),
    ("FMA4754", "right internal jugular vein", "neck", 1, 4),
    ("FMA4762", "left internal jugular vein", "neck", 1, 4),
    ("FMA3953", "right subclavian artery", "torso", 1, 2),
    ("FMA4694", "left subclavian artery", "torso", 1, 2),
    ("FMA4755", "right subclavian vein", "torso", 1, 4),
    ("FMA4763", "left subclavian vein", "torso", 1, 4),
    ("FMA22655", "right axillary artery", "humerus_r", 1, 3),
    ("FMA22656", "left axillary artery", "humerus_l", 1, 3),
    ("FMA13330", "right axillary vein", "humerus_r", 1, 4),
    ("FMA13331", "left axillary vein", "humerus_l", 1, 4),
    ("FMA22691", "right brachial artery", "humerus_r", 1, 3),
    ("FMA22692", "left brachial artery", "humerus_l", 1, 3),
    ("FMA22935", "right medial brachial vein", "humerus_r", 1, 4),
    ("FMA22936", "left medial brachial vein", "humerus_l", 1, 4),
    ("FMA22733", "right radial artery", "radius_r", 1, 3),
    ("FMA22734", "left radial artery", "radius_l", 1, 3),
    ("FMA22948", "right radial vein", "radius_r", 1, 4),
    ("FMA22949", "left radial vein", "radius_l", 1, 4),
    ("FMA22797", "right ulnar artery", "ulna_r", 1, 3),
    ("FMA22798", "left ulnar artery", "ulna_l", 1, 3),
    ("FMA22951", "right ulnar vein", "ulna_r", 1, 4),
    ("FMA22952", "left ulnar vein", "ulna_l", 1, 4),
    ("FMA14765", "right common iliac artery", "pelvis", 1, 12),
    ("FMA14766", "left common iliac artery", "pelvis", 1, 12),
    ("FMA21387", "right common iliac vein", "pelvis", 1, 13),
    ("FMA21388", "left common iliac vein", "pelvis", 1, 13),
    ("FMA18806", "right external iliac artery", "pelvis", 1, 12),
    ("FMA18807", "left external iliac artery", "pelvis", 1, 12),
    ("FMA18885", "right external iliac vein", "pelvis", 1, 13),
    ("FMA18886", "left external iliac vein", "pelvis", 4, 13),
    ("FMA70249", "right femoral artery", "femur_r", 1, 12),
    ("FMA70250", "left femoral artery", "femur_l", 1, 12),
    ("FMA21188", "right femoral vein", "femur_r", 1, 13),
    ("FMA21189", "left femoral vein", "femur_l", 1, 13),
    ("FMA77380", "right popliteal artery", "tibia_r", 1, 12),
    ("FMA77381", "left popliteal artery", "tibia_l", 1, 12),
    ("FMA44328", "right popliteal vein", "tibia_r", 1, 13),
    ("FMA44329", "left popliteal vein", "tibia_l", 1, 13),
    ("FMA43896", "right anterior tibial artery", "tibia_r", 1, 12),
    ("FMA43897", "left anterior tibial artery", "tibia_l", 1, 12),
    ("FMA43898", "right posterior tibial artery", "tibia_r", 1, 12),
    ("FMA43899", "left posterior tibial artery", "tibia_l", 1, 12),
    ("FMA44336", "right anterior tibial vein", "tibia_r", 2, 13),
    ("FMA44337", "left anterior tibial vein", "tibia_l", 2, 13),
    ("FMA44338", "right posterior tibial vein", "tibia_r", 1, 13),
    ("FMA44339", "left posterior tibial vein", "tibia_l", 1, 13),
    ("FMA14752", "right renal artery", "Abdomen", 1, 8),
    ("FMA14753", "left renal artery", "Abdomen", 1, 8),
    ("FMA14335", "right renal vein", "Abdomen", 2, 9),
    ("FMA14336", "left renal vein", "Abdomen", 2, 9),
)


def append_major_vessels(base_receipt: Path, output: Path) -> dict:
    return append_viscera(base_receipt, output,
        families=tuple(row[:4] for row in VESSELS), hierarchy="is_a", layer=2,
        extension_key="passive_major_vessel_extension",
        compartment_by_name={row[1]: row[4] for row in VESSELS})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(append_major_vessels(args.base_receipt, args.output)["payload"], sort_keys=True))
