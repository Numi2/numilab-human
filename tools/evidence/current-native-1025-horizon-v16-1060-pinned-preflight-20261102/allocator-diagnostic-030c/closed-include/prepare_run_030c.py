#!/usr/bin/env python3
import copy
import hashlib
import json
import shlex
from pathlib import Path

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
NATIVE = ROOT / "current-fractional-semantic-native-1047"
TRIAL = NATIVE / "optimization-trial-030c-gzip-fullcut-witness-closed-include"
BASE_RUN = NATIVE / "actual-029-adaptive-joint-031-baseline-route/run-001/run-manifest.json"
PACKET_RUN = NATIVE / "optimization-trial-032-adaptive-joint-packet/run-005"
LP_PROPOSAL = ROOT / "current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-032-all-witness-adaptive-joint-lp-001/run-008/proposal"
RUN = NATIVE / "actual-030c-032d-diagnostic/run-001"

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def pin(path):
    path = Path(path)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

def main():
    if RUN.exists() and any(RUN.iterdir()):
        raise SystemExit("run directory is not fresh")
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)

    cli = TRIAL / "NumiHumanCurrentFractionalMassMappingCLI_030c.cpp"
    binary = TRIAL / "bin/current-fractional-mass-mapping-cli-030c-o3"
    depfile = TRIAL / "controls/compile-cli-030c.d"
    cc_path = TRIAL / "controls/compile-control-030c.json"
    sc_path = TRIAL / "source-control-manifest-030c.json"
    launcher = TRIAL / "launch_run_030c.py"
    stdout = TRIAL / "controls/compile-cli.stdout.txt"
    stderr = TRIAL / "controls/compile-cli.stderr.txt"

    dep_text = depfile.read_text().replace("\\\n", " ")
    dep_body = dep_text.split(":", 1)[1]
    dependencies = [Path(token) for token in shlex.split(dep_body) if token]
    dependencies = [p for p in dependencies if p.resolve() != binary.resolve()]
    dependency_pins = [pin(p) for p in dependencies]
    by_name = {Path(p["path"]).name: p for p in dependency_pins}
    expected = {
        "ConservativeSpatialFractionalSemanticSupport.hpp": "fe75054637a78716ee015e74e82fb4e5c7d190690b6b04ffb2383e4426dfaa91",
        "ConservativeSpatialSourceCarrierQuadrature.hpp": "9d0afe443ec1d753eb5daaa82475b1dbd2afd12979812bc85af57ff3a22fa47a",
        "NumiHumanPhysicalSkinChildMassMapping.hpp": "1fce135c3fd0c5564f8358d464c54283f3164e70d66bbf34c86e8991774d304d",
        "NumiHumanCurrentFractionalSupportNPZ_029.hpp": "84f23778ccde7cfddd85564a62056906b66dba09f9e57924a9eeae3a7bf14ca0",
        "NumiHumanPhysicalSkinChildTetGen.hpp": "ada53384d415c85f3367938753c1bad4041adddb8da405a3536558eaebd73d2c",
        "FullCutWitnessGzip_030.hpp": "d1201a9c76234d4e5576239a37b17cbc80d9f88189978496351c488392d82430",
    }
    for name, expected_sha in expected.items():
        record = by_name.get(name)
        if not record or record["sha256"] != expected_sha or Path(record["path"]).parent != TRIAL:
            raise SystemExit(f"dependency closure mismatch for {name}")

    compile_argv = [
        "/usr/bin/clang++", "-std=c++20", "-O3", "-Wall", "-Wextra", "-Werror",
        "-DNUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1",
        "-I" + str(TRIAL), "-I" + str(NATIVE / "source/apps"),
        "-I/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/direct-source-parent-retet-control-654/source/matter/include",
        "-I/Users/numi/Work/numi-human-connected-surface-069/publication-140/dependencies/include",
        "-I/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/direct-source-parent-retet-control-654/source/include",
        "-MMD", "-MF", str(depfile), str(cli), "-lz", "-o", str(binary)
    ]
    cc = {
        "schema": "numi.h706.current-fractional-cli-compile-030c.v1",
        "status": "PASS_COMPILED_BACKEND_MATCHES_029",
        "exit_code": 0,
        "command": compile_argv,
        "source": pin(cli),
        "binary": pin(binary),
        "dependency_file": pin(depfile),
        "dependencies": dependency_pins,
        "stdout": pin(stdout),
        "stderr": pin(stderr),
        "backend_identity": {
            "actual_029_support_sha256": expected["ConservativeSpatialFractionalSemanticSupport.hpp"],
            "actual_029_carrier_quadrature_sha256": expected["ConservativeSpatialSourceCarrierQuadrature.hpp"],
            "actual_029_mass_mapping_sha256": expected["NumiHumanPhysicalSkinChildMassMapping.hpp"],
            "writer_change": "030b CLI adds durable first-line/header manifest before gzip full-cut publication; solver/support headers are byte-identical to actual029 and 030.",
            "physical_admission": False,
            "native_steps": 0
        }
    }
    save_json(cc_path, cc)

    base = json.loads(BASE_RUN.read_text())
    packet_batch_path = PACKET_RUN / "batch-packet-verification.json"
    packet_exec_path = PACKET_RUN / "execution.json"
    batch = json.loads(packet_batch_path.read_text())
    packet_exec = json.loads(packet_exec_path.read_text())
    if batch.get("status") != "PASS_SOURCE_ONLY_ADAPTIVE_PACKET_READBACK" or batch.get("physical_admission") is not False or batch.get("native_steps") != 0:
        raise SystemExit("032d packet verification is not source-only PASS")
    if packet_exec.get("source", {}).get("sha256") != batch.get("bridge_source", {}).get("sha256"):
        raise SystemExit("032d execution and packet bridge source pins differ")

    proposal = LP_PROPOSAL / "adaptive-joint-weight-proposal-032d.npz"
    proposal_report = LP_PROPOSAL / "joint-lp-report.json"
    proposal_source = ROOT / "current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-032-all-witness-adaptive-joint-lp-001/adaptive_anchor_joint_lp_032d.py"
    expected_proposal = {
        proposal: "acde169351151266192500724ebf332fb9b6c1beba6c1c0245263603ed85fffa",
        proposal_report: "f26da4a56d4fc4e7414b1963e4daee7e45674fcb2c064ae8f4643586c7da1a7f",
        proposal_source: "1ca324fe7ff049ab4e5f5f4e01bff27dedc2ad3570bd36e3bcaf800520870974",
    }
    for path, expected_sha in expected_proposal.items():
        if sha(path) != expected_sha:
            raise SystemExit(f"032d artifact pin mismatch: {path}")

    source_control = {
        "schema": "numi.h706.current-fractional-cli-source-control-030c.v1",
        "status": "PASS_SOURCE_CONTROL_COMPILE_ONLY_FULL_ALLOCATOR_NOT_RUN",
        "variant": "030c-gzip-fullcut-witness-closed-include",
        "source": pin(cli),
        "binary": pin(binary),
        "compile_control": pin(cc_path),
        "dependency_file": pin(depfile),
        "dependency_count": len(dependency_pins),
        "dependencies": dependency_pins,
        "actual_029_backend_headers": {name: by_name[name] for name in expected},
        "prior_actual_029_run_manifest": pin(BASE_RUN),
        "prior_actual_029_source_control": base.get("source_control_manifest"),
        "physical_admission": False,
        "native_steps": 0,
        "full_allocator_run": False,
        "scope": "compile and include-closure evidence only; full allocator result not yet run"
    }
    save_json(sc_path, source_control)

    source_control_pin = pin(sc_path)
    batch_pin = pin(packet_batch_path)
    packet_exec_pin = pin(packet_exec_path)
    proposal_pin = pin(proposal)
    proposal_report_pin = pin(proposal_report)
    proposal_source_pin = pin(proposal_source)
    matrix_dir = ROOT / "current-source-carrier-joint-initialization-1054/cut-closure-1054/actual-028-complete-predicate-joint-lp-001/run-001"
    matrix_path = matrix_dir / "anchor-cut-matrix.csv"
    matrix_manifest = matrix_dir / "run-manifest.json"
    matrix_execution = matrix_dir / "execution.json"

    config = {
        "schema": "numi.h706.diagnostic-run-config-030c.v1",
        "base_actual_029_run_manifest": pin(BASE_RUN),
        "source": pin(cli),
        "binary": pin(binary),
        "source_control_manifest": source_control_pin,
        "packet_batch": batch_pin,
        "packet_execution": packet_exec_pin,
        "proposal": proposal_pin,
        "proposal_report": proposal_report_pin,
        "proposal_source": proposal_source_pin,
        "output_dir": str(RUN / "verified-flow"),
        "maximum_wall_seconds": 600,
        "scope": "diagnostic only; physical_admission=false; native_steps=0"
    }
    config_path = RUN / "run-config.json"
    save_json(config_path, config)
    config_pin = pin(config_path)

    manifest = copy.deepcopy(base)
    manifest["base_manifest"] = pin(BASE_RUN)
    manifest["prior_actual_029_run_manifest"] = pin(BASE_RUN)
    manifest["prior_actual_029_execution"] = pin(BASE_RUN.parent / "execution.json")
    manifest["prior_actual_029_stdout_capture"] = pin(BASE_RUN.parent / "stdout-capture-manifest.json")
    manifest["source"] = pin(cli)
    manifest["binary"] = pin(binary)
    manifest["variant"] = "030c-gzip-fullcut-witness-closed-include"
    manifest["source_control_manifest"] = source_control_pin
    manifest["compile_control"] = pin(cc_path)
    manifest["adaptive_joint_proposal"] = {
        "proposal_report": proposal_report_pin,
        "proposal_npz": proposal_pin,
        "proposal_source": proposal_source_pin,
        "matrix": pin(matrix_path),
        "matrix_run_manifest": pin(matrix_manifest),
        "matrix_execution": pin(matrix_execution),
        "packet_batch": batch_pin,
        "packet_execution": packet_exec_pin,
        "adapter_source": batch["bridge_source"],
        "scope": "032d adaptive quadrature diagnostic; source-only packet readback; full allocator diagnostic required; no physical admission"
    }
    manifest["baseline_route_blend"] = batch.get("baseline_route_blend", {})
    manifest["named_input_pins"] = copy.deepcopy(base["named_input_pins"])
    for role in ("Skin", "Fat"):
        outputs = batch["roles"][role]["outputs"]
        key = role.lower()
        manifest["named_input_pins"]["carrier_" + key + "_packet"] = outputs["npz"]
        manifest["named_input_pins"]["carrier_witness_" + key] = {
            "path": outputs["witness_csv"]["path"],
            "sha256": outputs["witness_csv"]["sha256"],
            "bytes": outputs["witness_csv"]["bytes"]
        }
    manifest["named_input_pins"]["adaptive_packet_batch"] = batch_pin
    manifest["named_input_pins"]["adaptive_packet_execution"] = packet_exec_pin
    stale_packet_paths = {
        base["named_input_pins"][k]["path"]
        for k in ("carrier_skin_packet", "carrier_fat_packet", "carrier_witness_skin", "carrier_witness_fat")
    }
    manifest["file_pins"] = [x for x in base.get("file_pins", []) if x["path"] not in stale_packet_paths]
    for role in ("Skin", "Fat"):
        for item in batch["roles"][role]["outputs"].values():
            manifest["file_pins"].append(item)
    extra_pins = [
        batch_pin, packet_exec_pin, proposal_pin, proposal_report_pin, proposal_source_pin,
        pin(matrix_path), pin(matrix_manifest), pin(matrix_execution),
        source_control_pin, pin(cc_path), config_pin
    ]
    manifest["file_pins"].extend(extra_pins)

    argv = list(base["argv"])
    old_binary = base["binary"]["path"]
    binary_index = argv.index(old_binary)
    env = {}
    for token in argv[1:binary_index]:
        key, value = token.split("=", 1)
        env[key] = value
    env.update({
        "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_NPZ": manifest["named_input_pins"]["carrier_skin_packet"]["path"],
        "NUMI_H706_FRACTIONAL_CARRIERS_SKIN_SHA256": manifest["named_input_pins"]["carrier_skin_packet"]["sha256"],
        "NUMI_H706_FRACTIONAL_CARRIERS_FAT_NPZ": manifest["named_input_pins"]["carrier_fat_packet"]["path"],
        "NUMI_H706_FRACTIONAL_CARRIERS_FAT_SHA256": manifest["named_input_pins"]["carrier_fat_packet"]["sha256"],
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_CSV": manifest["named_input_pins"]["carrier_witness_skin"]["path"],
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_SKIN_SHA256": manifest["named_input_pins"]["carrier_witness_skin"]["sha256"],
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_CSV": manifest["named_input_pins"]["carrier_witness_fat"]["path"],
        "NUMI_H706_FRACTIONAL_CARRIER_WITNESS_FAT_SHA256": manifest["named_input_pins"]["carrier_witness_fat"]["sha256"],
        "NUMI_H706_FRACTIONAL_OUTPUT_DIR": str(RUN / "verified-flow"),
        "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256": source_control_pin["sha256"],
        "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256": config_pin["sha256"]
    })
    manifest["argv"] = ["/usr/bin/env"] + [f"{k}={v}" for k, v in sorted(env.items())] + [str(binary)] + argv[binary_index + 1:]
    manifest["cwd"] = str(NATIVE)
    manifest["maximum_wall_seconds"] = 600
    manifest["physical_admission"] = False
    manifest["native_steps"] = 0
    manifest["experimental_status"] = "READY_DIAGNOSTIC_FULL_ALLOCATOR_RUN_032D_NO_ADMISSION_CLAIM"
    manifest["stdout_log"] = str(RUN / "logs/cli.stdout.log")
    manifest["stderr_log"] = str(RUN / "logs/cli.stderr.log")
    manifest["execution_json"] = str(RUN / "execution.json")
    manifest["launcher_source"] = str(launcher)
    manifest["wrapper"] = "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/run_geometry_serial.py"
    manifest["diagnostic_scope"] = {
        "status": "DIAGNOSTIC_ONLY",
        "physical_admission": False,
        "native_steps": 0,
        "reason": "The source-only 032d proposal remains short on some source-side cuts. This run measures the actual allocator residual graph and full cut for this packet."
    }
    manifest_path = RUN / "run-manifest.json"
    save_json(manifest_path, manifest)
    print(json.dumps({
        "run_manifest": pin(manifest_path),
        "source_control_manifest": source_control_pin,
        "run_config": config_pin,
        "batch_packet_verification": batch_pin,
        "binary": pin(binary),
        "output_dir": str(RUN / "verified-flow"),
        "dependency_count": len(dependency_pins),
        "status": manifest["experimental_status"]
    }, sort_keys=True))

if __name__ == "__main__":
    main()

