"""Replay the registered full-contact Human scene on its native Numi host."""

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same_input(selected, selected_path, observed, observed_path):
    """Match recorded input hashes across independent native runs."""
    expected = selected["pins"].get(selected_path)
    return expected is not None and observed["pins"].get(observed_path) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--check",
        action="store_true",
        help="verify the native owner and all pinned inputs",
    )
    options.add_argument(
        "--steps",
        type=int,
        help="number of physical steps at the selected checkpoint timestep",
    )
    options.add_argument(
        "--seconds",
        type=int,
        choices=(16, 60),
        help="requested integrated duration in seconds",
    )
    options.add_argument(
        "--material-report",
        type=Path,
        help="write a cortical occupancy report and return failure on an invalid registered reference",
    )
    parser.add_argument(
        "--material-proposal",
        type=Path,
        help="Evaluate a sparse registered material correction in the material report",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="use the full registered skin/fat volume and its native timestep",
    )
    parser.add_argument(
        "--accepted-run",
        type=Path,
        help="Audit the terminal dense material and rigid capture in this run",
    )
    parser.add_argument(
        "--posterior-support", choices=("pressure-fit", "expanded", "covered-shoulder", "registered-shoulders", "registered-lumbar", "registered-lumbar-faces", "refined-lumbar", "connected-release", "joint-stops", "forearm-material"),
        help="select a matched dense posterior tissue and bed registration",
    )
    parser.add_argument("--wall-pairs", type=Path, help="Also enumerate accepted tissue-wall and cortical triangle pairs")
    args = parser.parse_args()
    if args.wall_pairs and not (args.material_report and (args.accepted_run or args.posterior_support)):
        parser.error("--wall-pairs requires --accepted-run and --material-report")
    if args.posterior_support and not args.dense:
        parser.error("--posterior-support requires --dense")
    if args.accepted_run and (not args.material_report or args.material_proposal):
        parser.error(
            "--accepted-run requires --material-report and supplies its own material positions"
        )
    if args.dense and args.material_report and not args.accepted_run and (not args.posterior_support or args.material_proposal):
        parser.error(
            "A dense material report needs --accepted-run. Use --current for an unrun proposal."
        )
    if args.material_proposal is not None and args.material_report is None:
        parser.error("--material-proposal requires --material-report")

    repo = Path(__file__).resolve().parents[1]
    evidence = repo / "evidence/human-connected-mesh-rebuild-20261105"
    reference_checkpoint = "m6-contact-feature-ownership-20261125"
    reference_sha = "f2c198650665551959263aabb2cd967088cabf56292151f573e83eae90a3624b"
    if args.dense:
        checkpoint = "m6-loaded-material-motion-20261125"
        expected_launch = (
            "c4d6c3e265d87ed8e664c5583a70822ef08a7ed18c58f34862c4d6f47e6282bd"
        )
    else:
        checkpoint = reference_checkpoint
        expected_launch = reference_sha
    cfg_path = evidence / checkpoint / "launch.json.gz"
    if args.posterior_support:
        cfg_path = evidence / checkpoint / "posterior-support" / f"{args.posterior_support}-launch.json.gz"
        expected_launch = {
            "pressure-fit": "324dcf87639e8dfd01e77d504afca338aa16fd603d9854666f7519154a319302",
            "expanded": "6ba126054085b3fac6d8ecd6d002dbb838b2c30f4039a9f780c4917822822cbf",
            "forearm-material": "f2a52ffa037e7a3294360d649a9d642c496ea8b81542286f83dd85275019394d",
            "joint-stops": "a31cd17b225c23a9c3219cdb4b5111d39d89f8ac6f27210919324b2763531148",
            "connected-release": "577640741029e3f87df1556e6f42607d77c55744a80d183610d4729968e76da1",
            "refined-lumbar": "07b1cb379fe2b352f6f79f57d0ab6f527af74cbc3c50b1597935ffa262091406",
            "registered-lumbar-faces": "158a15e86b31e32f786120a5d5e49db2fe4f040c4e74b8a4d13db8006f79168c",
            "registered-lumbar": "32f804a4b5e8be8e320e6f72d04907e61b1cfdf9a4532466a5b1fe19733521c8",
            "registered-shoulders": "2056a5888386590054b9fc4cc75fc56e8d1561e55ad802399f5fa74a330ea605",
            "covered-shoulder": "0142f2c91c24ea5cbd7cfcfab13af5245043d77055102398bc706676692aca56",
        }[args.posterior_support]
    payload = gzip.decompress(cfg_path.read_bytes())
    if hashlib.sha256(payload).hexdigest() != expected_launch:
        raise SystemExit("The launch manifest differs from the verified checkpoint.")
    cfg = json.loads(payload)
    argv = cfg["argv"]
    settings = cfg["environment"]
    old_output = Path(argv[4]).parent
    steps = (
        args.steps
        if args.steps is not None
        else int(argv[argv.index("--muscle-step-count") + 1])
    )
    if args.seconds is not None:
        ticks = args.seconds / Fraction(argv[argv.index("--muscle-step-seconds") + 1])
        if ticks.denominator != 1:
            parser.error("duration does not end on a physical step")
        steps = ticks.numerator
    if not 0 < steps < 2**32:
        parser.error("steps must be a positive 32-bit count")
    for name, expected in cfg["pins"].items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise SystemExit(f"Missing or changed native input: {path}")
    helper = (
        Path.home()
        / "Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/pelvic-contact-response-048/repartition/digital-corridor-049/posterior-equilibrium-050/thoracic-bed-port-051/run_cavity_motion.py"
    )
    if (
        not helper.is_file()
        or digest(helper)
        != "624d95ee5f007fd8de4c2479db94ab968f335e6382c8c64587f03d42a63a6893"
    ):
        raise SystemExit(f"Missing or changed process monitor: {helper}")
    print(
        f"Verified {len(cfg['pins'])} native input identities. Physical timestep {argv[argv.index('--muscle-step-seconds') + 1]} s.",
        flush=True,
    )
    print(
        f"Physical scope from the launch manifest: {cfg['scope']} "
        f"Variable pleural pressure={settings['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']}.",
        flush=True,
    )
    if args.material_report is not None:
        if args.dense and args.accepted_run is None:
            # The selected profile owns the retained capture. Resolve its original
            # launch before passing a directory to the terminal-state verifier.
            retained = Path(cfg["argv"][4]).parent
            launch = retained / "launch.json"
            if not launch.is_file() or digest(launch) != cfg.get("source_launch_sha256"):
                parser.error("This dense profile requires an explicit --accepted-run")
            args.accepted_run = retained
        if args.dense:
            observed = json.loads((args.accepted_run / "launch.json").read_bytes())
            for index in (0, 1, 3):
                if not same_input(cfg, cfg["argv"][index], observed, observed["argv"][index]):
                    parser.error("The accepted run and selected profile have different native owners or anatomy")
            for key in (
                "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE",
                "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ",
            ):
                requested = settings.get(key)
                captured = observed["environment"].get(key)
                if (requested or captured) and not same_input(cfg, requested, observed, captured):
                    parser.error("The accepted run and selected profile have different loaded material")
            print(f"Auditing the selected physical capture at {args.accepted_run}", flush=True)
        reference_pins = cfg["pins"]
        if args.dense:
            # The source registration is retained by the anatomical checkpoint.
            # Observations and the scene remain those selected by the dense launch.
            source = evidence / reference_checkpoint / "launch.json.gz"
            reference_bytes = gzip.decompress(source.read_bytes())
            if hashlib.sha256(reference_bytes).hexdigest() != reference_sha:
                raise SystemExit("The anatomical registration checkpoint changed")
            reference_pins = json.loads(reference_bytes)["pins"]
        registrations = [
            Path(name)
            for name in reference_pins
            if Path(name).name == "material-boundary.json"
        ]
        if len(registrations) != 1:
            raise SystemExit("Ambiguous registered material in the launch manifest.")
        registration = registrations[0]
        if digest(registration) != reference_pins[str(registration)]:
            raise SystemExit("The anatomical registration changed")
        volume_launch = Path(
            json.loads(registration.read_bytes())["initial_bodies_path"]
        ).with_name("launch.json")
        verifier = (
            repo
            / "tools/native_ligament_contact/registered_material_volume/audit_registered_cortical_occupancy.py"
        )
        volume = json.loads(volume_launch.read_bytes())["environment"]
        material_node = Path(
            volume["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"]
        )
        geometry = material_node.parents[2] / "offline-geometry"
        manifold = (
            Path.home()
            / "Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/material-bed-equilibrium-046/pelvic-interface-tools/lib"
        )
        modules = {
            geometry
            / "igl/__init__.py": "a7b3c75c4c68adea69e637ac1649a80a17cf3012e87cc31a9ee0978a15065e90",
            geometry
            / "igl/pyigl_core.abi3.so": "1284a530488349061c127b9030010671775ef3e8df3d3e87afd45a68cc3811b2",
            manifold
            / "manifold3d.cpython-313-darwin.so": "7f3a490aa64951ec8fecd12c65bfe76753f7c476e12b347853f4b1c28a5c9916",
        }
        pair_args = []
        if args.wall_pairs:
            wall = cfg.get("registered_material_walls")
            if not wall or wall not in cfg["pins"]:
                parser.error("The selected profile lacks a registered tissue-wall identity")
            if args.wall_pairs.suffix != ".json" or args.wall_pairs.resolve() == args.material_report.resolve():
                parser.error("--wall-pairs needs a separate JSON output path")
            pair_args = ["--wall-pairs", str(args.wall_pairs), "--registered-walls-sha256", cfg["pins"][wall]]
            modules[geometry / "igl/copyleft/cgal/__init__.py"] = "677e9b1ec585722d78685ad5e00e091ceb5ecef2ba2eecb877c5c8eb153fa1b5"
            modules[geometry / "igl/copyleft/cgal/pyigl_copyleft_cgal.abi3.so"] = "eb5b958e7c7a1adcf604e91ef3c429a44336f33f1ca4ed41542cf1b457adab5e"
        for path, expected in modules.items():
            if not path.is_file() or digest(path) != expected:
                raise SystemExit(
                    f"Missing or changed native geometry dependency: {path}"
                )
        if "material_registration" in cfg:
            registration = Path(cfg["material_registration"])
            if cfg["pins"].get(str(registration)) != digest(registration):
                raise SystemExit("The selected material registration changed")
            initial = Path(json.loads(registration.read_bytes())["initial_bodies_path"])
            volume_launch = initial.with_name("launch.json")
            if cfg["pins"].get(str(volume_launch)) != digest(volume_launch):
                raise SystemExit("The selected material volume launch changed")
        audit_env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(map(str, (geometry, manifold))),
            PYTHONDONTWRITEBYTECODE="1",
        )
        return subprocess.run(
            [
                sys.executable,
                str(verifier),
                "--scene",
                str(cfg_path),
                "--volume-launch",
                str(volume_launch),
                "--material-registration",
                str(registration),
                "--require-clear",
                "--require-deformation",
                *(
                    ["--proposal", str(args.material_proposal)]
                    if args.material_proposal
                    else []
                ),
                *(
                    ["--accepted-run", str(args.accepted_run)]
                    if args.accepted_run
                    else []
                ),
                *pair_args,
                "--output",
                str(args.material_report),
            ],
            check=False,
            env=audit_env,
        ).returncode
    if args.check:
        return 0

    out = Path(tempfile.mkdtemp(prefix="current-contact-", dir=old_output.parent))
    cfg["parent_launch"] = hashlib.sha256(payload).hexdigest()
    argv[4] = str(out / "native-traces")
    for flag in ("--muscle-step-count", "--persistent-metal-stand-steps"):
        if flag in argv:
            argv[argv.index(flag) + 1] = str(steps)
    for name, value in list(settings.items()):
        path = Path(value)
        if (
            path.is_absolute()
            and path.is_relative_to(old_output)
            and value not in cfg["pins"]
        ):
            settings[name] = str(out / path.relative_to(old_output))
    settings["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = str(steps)
    (out / "launch.json").write_text(json.dumps(cfg, indent=2) + "\n")
    environ = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("NM_", "NUMI_", "DYLD_"))
    }
    environ.update(settings)
    spec = importlib.util.spec_from_file_location("resting_contact_monitor", helper)
    monitor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(monitor)
    print(
        f"Running {steps} physical steps. Native capture and traces: {out}", flush=True
    )
    result = monitor.execute_native(
        argv,
        cfg["cwd"],
        environ,
        out,
        minimum_free_bytes=4 * 2**30,
        timeout_s=float("inf"),
    )
    (out / "execution.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
