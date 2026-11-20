"""Run the same-source skin/fat parent through the existing native GPU owner."""
from argparse import ArgumentParser
import json
import os
import subprocess
import time

from run_recruitment import CONTROL, HERE, digest

SOURCE = CONTROL.parent / "floor-live-parent-capture-002"


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--steps", default=16, type=int, choices=(2, 16))
    parser.add_argument("--compensated-contact", action="store_true")
    args = parser.parse_args()
    authored_launch = SOURCE / "launch.json"
    declaration = json.loads(authored_launch.read_text())
    fine = json.loads((CONTROL / "launch.json").read_text())
    suffix = "-compensated-contact" if args.compensated_contact else ""
    output = HERE / f"same-source-parent-{args.steps}{suffix}"
    assert os.statvfs(HERE).f_bavail * os.statvfs(HERE).f_frsize > 2 * 1024**3
    pins = {path: sha for path, sha in declaration["pins"].items()
            if path != declaration["argv"][0]}
    pins[fine["argv"][0]] = fine["pins"][fine["argv"][0]]
    for path, expected in pins.items():
        assert digest(path) == expected, path
    parent_table = SOURCE / "compiled-parent-cell-mass.csv.nodes.csv"
    assert str(parent_table) == fine["environment"]["NUMI_H706_FRACTIONAL_SOURCE330_NODES"]
    expected_parent_hash = fine["environment"]["NUMI_H706_FRACTIONAL_SOURCE330_NODES_SHA256"]
    assert digest(parent_table) == expected_parent_hash
    argv = [word.replace(str(SOURCE), str(output)) for word in declaration["argv"]]
    assert argv[1:4] == fine["argv"][1:4]
    argv[0] = fine["argv"][0]
    for flag in ("--root-pose", "--resting-registered-ligaments",
                 "--resting-release-initialization", "--muscle-step-seconds"):
        assert flag in argv and flag in fine["argv"]
    index = argv.index("--root-pose")
    fine_index = fine["argv"].index("--root-pose")
    assert argv[index+1:index+8] == fine["argv"][fine_index+1:fine_index+8]
    step_count = argv.index("--muscle-step-count") + 1
    argv[step_count] = str(args.steps)
    environment = dict(declaration["environment"])
    if args.compensated_contact:
        # Reuse the authoritative mixed-precision plane/clearance policy.
        for key in ("NM_FEM_COMPENSATED_CONTACT_PLANE_CCD",
                    "NM_FEM_CLEARANCE_ALIGNED_BARRIER", "NM_FEM_REFERENCE_SELF_CONTACT_WIDTH"):
            environment[key] = fine["environment"][key]
    assert environment["NUMI_HUMAN_FEM_SKIN_SUPPORT"] == "1"
    del environment["NUMI_HUMAN_SKIN_FAT_PARENT_MASS_LEDGER_CSV"]
    environment.update(
        DYLD_LIBRARY_PATH=fine["environment"]["DYLD_LIBRARY_PATH"],
        NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY="0",
        NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE="1",
        NUMI_HUMAN_TRAINING_PROFILE="0",
    )
    for name in ("NM_VASCULAR_STEP_EXTERNAL_PRESSURE",
                 "NUMI_HUMAN_RESTING_VARIABLE_PLEURAL_PRESSURE"):
        if name in fine["environment"]:
            assert environment.get(name) == fine["environment"][name]
    output.mkdir()
    declaration.update(argv=argv, environment=environment, pins=pins,
                       parent_launch_sha256=digest(authored_launch),
                       parent_table_sha256=expected_parent_hash,
                       native_binary_sha256=digest(argv[0]))
    (output / "launch.json").write_text(json.dumps(declaration, indent=2) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    env.update(environment)
    started = time.monotonic()
    with (output / "stdout.log").open("xb") as out, (output / "stderr.log").open("xb") as err:
        try:
            child = subprocess.run(argv, cwd=declaration["cwd"], env=env, stdout=out, stderr=err,
                                   timeout=540, check=False)
            result = dict(exit=child.returncode, stop="exit")
        except subprocess.TimeoutExpired:
            result = dict(exit=124, stop="timeout")
    result["wall_seconds"] = time.monotonic() - started
    (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
