"""Compare the current launch skin with the native compiler's frozen source."""
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.with_name("resting-clearance-and-atrium-006")))
from fit_clearance_patches import decode


def source(path):
    document = json.loads(path.read_text())
    arguments = document["argv"]
    skin = Path(arguments[arguments.index("--skin-payload") + 1])
    raw = skin.read_bytes()
    identity = {"path": str(skin), "sha256": hashlib.sha256(raw).hexdigest(),
                "reference_mass_kg": float(arguments[arguments.index("--resting-reference-mass-kg") + 1])}
    return identity, decode(raw), Path(document["cwd"])


def main():
    current, first, owner = source(HERE / "full-contact-launch.json")
    registered, second, _ = source(HERE / "prior-full-admission.json")
    declaration = owner / "apps/NumiHumanFinal128CandidateRegistration.hpp"
    expected = re.search(r'kWholeBodySkinCandidateSourcePayloadSHA256="([0-9a-f]{64})"',
                         declaration.read_text()).group(1)
    assert registered["sha256"] == expected, registered
    current.update(vertices=len(first["verts"]), triangles=first["inds"].size // 3)
    registered.update(vertices=len(second["verts"]), triangles=second["inds"].size // 3)
    same_faces = bool(np.array_equal(first["inds"], second["inds"]))
    result = {"current": current, "frozen_source": registered,
              "declaration_sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(),
              "same_index_stream": same_faces,
              "required_candidate_hash_matches_current": current["sha256"] == expected}
    if same_faces and len(first["verts"]) == len(second["verts"]):
        distance = np.linalg.norm(first["verts"][:, :3].astype(float) - second["verts"][:, :3], axis=1)
        result["same_index_distance_mm"] = {str(q): float(np.quantile(distance, q)) * 1000.
                                            for q in (0., .5, .95, 1.)}
        result["changed_positions"] = int(np.count_nonzero(distance))
        result["weight_rows_equal"] = bool(np.array_equal(first["weights"], second["weights"]))
    (HERE / "physical-skin-source-identities.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
