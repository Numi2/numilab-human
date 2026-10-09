#!/usr/bin/env python3
"""Fail-closed adapter from sealed science v2 trials to the retained 1161 geometry owner."""
from __future__ import annotations
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005")
P18 = EVIDENCE / "final-integrated-study-readiness-1170/package-v018"
ANALYZER_PATH = P18 / "analyze_final_pair.py"
ANALYZER_SHA256 = "defe06f8ded56d66fe5ba903445bb8dd81ef96ad57fa59e3be4c615c5892cb5c"
PREPARE_PATH = P18 / "prepare_final_plan.py"
PREPARE_SHA256 = "f5d57e46afce90ea756e8eb8e23bfd1f4a6a6e9fcb971659f946ce29768531be"
TERMINAL_STEP = 155000
DT = 0.002


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned registered-study helper: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_successful_exit(receipt, accepted_terminal, expected_terminal=TERMINAL_STEP):
    """Validate the registered process outcome and terminal capture, not a scene metadata file."""
    if not isinstance(receipt, dict):
        raise ValueError("registered exit receipt is missing")
    if receipt.get("returncode") != 0 or receipt.get("failure") is not None or not receipt.get("ended_at"):
        raise ValueError("registered exit receipt is failed, killed, or incomplete")
    if accepted_terminal is not None and int(accepted_terminal) != int(expected_terminal):
        raise ValueError("registered accepted geometry does not include the exact terminal step")
    return {"returncode": 0, "failure": None,
            "accepted_terminal_step": (int(accepted_terminal) if accepted_terminal is not None else None)}


def validate_receipt_chain_links(registration_sha, started_doc_sha, process_doc_sha,
                                 started, process, receipt):
    if started.get("registration_sha256") != registration_sha:
        raise ValueError("started receipt is not bound to this registration")
    if started_doc_sha != process.get("binding") or receipt.get("started_sha256") != started_doc_sha:
        raise ValueError("started/process/exit receipts do not bind the same trial")
    if process_doc_sha != receipt.get("process_sha256"):
        raise ValueError("exit receipt does not bind the recorded process")
    return True


def verify_receipt_files(trial_root, files):
    """Hash-check every registered output path; reject traversal, symlinks, and missing files."""
    trial_root = Path(trial_root).resolve()
    if not isinstance(files, dict) or not files:
        raise ValueError("registered exit receipt has no sealed output file map")
    verified = []
    for relative, expected in sorted(files.items()):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("registered output path escapes trial: " + str(relative))
        path = trial_root / "output" / rel
        if path.is_symlink() or not path.is_file():
            raise ValueError("registered output file missing or symlinked: " + str(relative))
        actual = sha(path)
        if actual != expected:
            raise ValueError("registered output hash mismatch: " + str(relative))
        verified.append({"path": str(path.resolve()), "sha256": actual})
    return verified


def load_registered_context(trial_path, arm):
    """Return validated current-scene inputs from a successful registered v2 trial."""
    trial_path = Path(trial_path).resolve()
    if arm not in ("control", "treatment"):
        raise ValueError("registered arm must be control or treatment")
    for path, expected in ((ANALYZER_PATH, ANALYZER_SHA256), (PREPARE_PATH, PREPARE_SHA256)):
        if not path.is_file() or sha(path) != expected:
            raise ValueError("pinned P18 registered-study helper changed: " + str(path))
    analyzer = _load(ANALYZER_PATH, "registered_final_pair_1171")
    prepare = _load(PREPARE_PATH, "registered_prepare_final_pair_1171")
    # Validate the sealed receipt chain and successful process outcome before the
    # P18 analyzer opens any scene output. This ensures a failed registered trial
    # is rejected for its recorded failure even when it has no complete trace.
    audit = analyzer.audit
    study_path = trial_path.parent.parent
    registration_doc, registration = audit.read_wrapped(study_path / "registration.json", "study registration")
    started_doc, started = audit.read_wrapped(trial_path / "started.json", "started receipt")
    process_doc, process = audit.read_wrapped(trial_path / "process.json", "process receipt")
    receipt_doc, receipt = audit.read_wrapped(trial_path / "receipt.json", "registered exit receipt")
    validate_receipt_chain_links(
        registration_doc.get("sha256"), started_doc.get("sha256"), process_doc.get("sha256"),
        started, process, receipt,
    )
    validate_successful_exit(receipt, None)
    schedule = prepare.accepted_geometry_capture_schedule()
    item = analyzer.derived_capture_trial(trial_path, schedule, arm)
    geometry = analyzer.accepted_geometry_summary(item, schedule, arm)
    terminal = geometry.get("terminal_step")
    exit_status = validate_successful_exit(item.get("receipt"), terminal)
    if item.get("steps") != TERMINAL_STEP or abs(float(item.get("dt")) - DT) > 1e-12:
        raise ValueError("registered scene is not the exact 155000-root, 2 ms run")
    if len(geometry.get("captures", [])) != 8:
        raise ValueError("registered scene must have exactly eight accepted capture receipts")
    steps = [int(x["accepted_step"]) for x in geometry["captures"]]
    if steps != list(schedule["arms"][arm]["step_ids"]) or steps[-1] != TERMINAL_STEP:
        raise ValueError("registered capture schedule is incomplete or does not end at N=155000")
    scene = trial_path / "output" / "scene"
    invocation = item["inv"]
    if not isinstance(invocation.get("asset_sha256"), dict):
        raise ValueError("registered scene invocation has no asset hash map")
    assets = []
    for raw, expected in sorted(invocation["asset_sha256"].items()):
        path = Path(raw)
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("registered scene asset missing or hash mismatch: " + raw)
        assets.append({"path": str(path.resolve()), "sha256": expected})
    receipt_files = verify_receipt_files(trial_path, item["receipt"].get("files"))
    scene_invocation = scene / "invocation.json"
    scene_log = scene / "native.log"
    if not scene_invocation.is_file() or not scene_log.is_file():
        raise ValueError("registered scene invocation/log is missing")
    # Deliberately no scene/run-metadata.json lookup: the sealed registered receipt chain
    # and P18 loaded-runtime check are the authority for a run-native science trial.
    runtime = item.get("scene_loaded_metal_runtime")
    if not isinstance(runtime, dict) or runtime.get("verified") is not True:
        raise ValueError("registered trial does not prove the loaded native Metal runtime")
    helper_paths = {ANALYZER_PATH.resolve(), PREPARE_PATH.resolve()}
    for module in (analyzer, prepare, getattr(analyzer, "audit", None), getattr(prepare, "audit", None)):
        path = getattr(module, "__file__", None) if module is not None else None
        if path:
            helper_paths.add(Path(path).resolve())
    runtime_source = inspect.getsourcefile(analyzer.human_loaded_metal_runtime)
    if runtime_source:
        helper_paths.add(Path(runtime_source).resolve())
    helper_identities = [{"path": str(path), "sha256": sha(path)}
                         for path in sorted(helper_paths)]
    tracked = set(helper_paths)
    tracked.update({
        trial_path.parent.parent / "registration.json",
        trial_path / "started.json", trial_path / "process.json", trial_path / "receipt.json",
        item["launch_template_path"], item["owner_preflight_path"],
        item["owner_preflight_path"].with_name("run-metadata.json"),
        item["owner_preflight_path"].with_name("native.log"),
        scene_invocation, scene_log,
    })
    for capture in geometry["captures"]:
        tracked.update((Path(capture["pack_path"]), Path(capture["receipt_path"])))
    tracked.update(Path(x["path"]) for x in assets)
    tracked.update(Path(x["path"]) for x in receipt_files)
    for path in tracked:
        if path.is_symlink() or not path.is_file():
            raise ValueError("registered evidence dependency missing or symlinked: " + str(path))
    hashes = {str(path.resolve()): sha(path) for path in sorted(tracked)}
    return {
        "trial_path": trial_path, "scene_path": scene, "item": item,
        "geometry": geometry, "schedule": schedule, "steps": steps,
        "requested_roots": TERMINAL_STEP, "dt_seconds": DT,
        "invocation": invocation, "runtime": runtime, "exit_status": exit_status,
        "tracked": tracked, "input_hashes": hashes,
        "receipt_chain": {
            "schema": "numi.science.registration.v2",
            "study_registration": str(trial_path.parent.parent / "registration.json"),
            "started_receipt": str(trial_path / "started.json"),
            "process_receipt": str(trial_path / "process.json"),
            "exit_receipt": str(trial_path / "receipt.json"),
            "arm": arm, "trial_id": item["started"]["payload"]["trial"]["id"],
            "terminal_step": TERMINAL_STEP, "capture_count": len(steps),
            "scene_run_metadata_required": False,
            "loaded_metal_runtime_verified": True,
            "p18_analyzer": {"path": str(ANALYZER_PATH), "sha256": ANALYZER_SHA256},
            "p18_prepare": {"path": str(PREPARE_PATH), "sha256": PREPARE_SHA256},
            "p18_helper_modules": helper_identities,
            "receipt_files_verified": len(receipt_files),
            "asset_files_verified": len(assets),
        },
    }
