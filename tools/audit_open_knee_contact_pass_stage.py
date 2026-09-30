"""Bind the rejected Open Knee contact pass to the native build and source."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "Docs/media/cruciate-source-contact-20260930"
LOG = FOLDER / "contact-pass-stage.log"
OUTPUT = FOLDER / "contact-pass-stage-receipt.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise RuntimeError("Open Knee contact pass: " + reason)


def run() -> dict:
    original = json.loads((FOLDER / "ptl-trajectory-native-receipt.json").read_text())
    witness_path = FOLDER / "ptl-trajectory-contact-witness.json"
    witness = json.loads(witness_path.read_text())
    binary = Path(original["binary"]["path"])
    build = binary.parents[1]
    matter = binary.parents[3]
    shader = build / "matter/shaders/NumiMatter.metallib"
    library = build / "lib/libmetalrobo.dylib"
    source_files = {
        "contact_shader": matter / "matter/src/metal/contact.metalinc",
        "runtime": matter / "matter/src/runtime.mm",
        "human_probe": matter / "apps/numilab_human_myosim_visual_probe.mm",
    }
    require(all(path.is_file() for path in
                (binary, shader, library, LOG, *source_files.values())),
            "source, executable, shader, library, or log missing")
    for label, source in original["inputs"].items():
        require(digest(Path(source["path"])) == source["sha256"],
                f"{label} input no longer matches the pinned payload")
    text = LOG.read_text()
    status = re.search(
        r"human_joint_failure completed_steps=(\d+) matter_status=(\d+) "
        r"matter_failing_index=(\d+) matter_object=(\d+) "
        r"matter_completed_microsteps=(\d+) matter_diagnostics="
        r"([\d.eE+-]+),([\d.eE+-]+),([\d.eE+-]+),([\d.eE+-]+)", text)
    captured = re.search(
        r"deformable_contact_failure step=(\d+) microtick=(\d+) "
        r"slot=(\d+) primitive_a=(\d+) primitive_b=(\d+) "
        r"thickness=([^ ]+) dt_a=([^ ]+) dt_b=([^ ]+) "
        r"solver_iteration=([^\n]+)", text)
    require(status is not None and captured is not None,
            "native failure stage is incomplete")
    require(int(status[1]) == int(captured[1]) == 5 and
            int(status[2]) == 6 and int(status[5]) == 0 and
            int(float(status[8])) == -2 and int(float(status[9])) == 2 and
            int(float(captured[9])) == 1,
            "expected second-pass predicted-finish contact rejection drifted")
    require((int(captured[4]), int(captured[5])) ==
            (306892, 603698) and
            witness["source_contact_pair"] == "PCL_To_ACL" and
            witness["source_contact_faces"] == {"ACL": 6832, "PCL": 3525} and
            witness["native_step_finish_strict_crossing"],
            "captured contact no longer maps to pinned PCL/ACL faces")
    revision = subprocess.check_output(
        ["git", "-C", str(matter), "rev-parse", "HEAD"], text=True).strip()
    return {
        "schema": "numi.human.open-knee-contact-pass-stage.v1",
        "matter_commit": revision,
        "binary_sha256": digest(binary),
        "metallib_sha256": digest(shader),
        "library_sha256": digest(library),
        "source_sha256": {key: digest(path)
                          for key, path in source_files.items()},
        "inputs": original["inputs"],
        "command": [*original["runs"]["eight"]["argv"][:4],
                    str(ROOT / "Build/cruciate-contact-pass-final-20260930"),
                    *original["runs"]["eight"]["argv"][5:]],
        "log_sha256": digest(LOG),
        "prior_contact_witness_sha256": digest(witness_path),
        "accepted_steps_before_rejection": int(status[1]),
        "status": {
            "code": int(status[2]),
            "slot": int(status[3]),
            "object": int(status[4]),
            "completed_microsteps": int(status[5]),
            "primitive_pair": [int(float(status[6])), int(float(status[7]))],
            "stage_code": int(float(status[8])),
            "one_based_nonlinear_contact_pass": int(float(status[9])),
        },
        "captured_pair": {
            "slot": int(captured[3]),
            "primitive_pair": [int(captured[4]), int(captured[5])],
            "zero_based_nonlinear_iteration": int(float(captured[9])),
            "source_faces": witness["source_contact_faces"],
        },
        "qualified": False,
        "boundary": "Five steps are accepted; the sixth predicts a strict PCL/ACL "
                    "crossing on nonlinear pass two. The contact solver rejects "
                    "the candidate before publication. This does not validate "
                    "the local ACL initialization or sustained knee mechanics.",
    }


if __name__ == "__main__":
    result = run()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"accepted_steps": result["accepted_steps_before_rejection"],
                      "contact_pass": result["status"]["one_based_nonlinear_contact_pass"],
                      "qualified": result["qualified"]}, sort_keys=True))
