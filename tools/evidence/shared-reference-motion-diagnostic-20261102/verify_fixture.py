#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    pins = load_json(ROOT / "package-pins.json")
    closure = load_json(ROOT / "header-closure.json")
    report_path = ROOT / "evidence" / "diagnostic-report.json"
    report = load_json(report_path)
    if report["status"] != pins["status"]:
        raise SystemExit("report status differs from package pin")
    if report["physical_admission"] or report["native_steps"] != 0 or report["gpu_executed"]:
        raise SystemExit("diagnostic unexpectedly claims native or physical execution")

    owner = closure["owner"]
    repo_root = Path(owner["root_path"])
    env = dict(os.environ)
    env["GIT_NO_LAZY_FETCH"] = "1"
    revision = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
        env=env,
    ).strip()
    if revision != owner["revision"]:
        raise SystemExit(f"owner revision mismatch: {revision}")

    for header in closure["headers"]:
        path = repo_root / header["path"]
        if path.stat().st_size != header["bytes"] or sha256(path) != header["sha256"]:
            raise SystemExit(f"pinned dependency mismatch: {path}")
    notice = repo_root / closure["repository_notice"]["path"]
    if notice.stat().st_size != closure["repository_notice"]["bytes"]:
        raise SystemExit("owner notice size mismatch")
    if sha256(notice) != closure["repository_notice"]["sha256"]:
        raise SystemExit("owner notice hash mismatch")

    source = ROOT / "source" / "shared_reference_motion.cpp"
    fixture = ROOT / "fixture" / "selected-rest-f32.bin"
    row_file = ROOT / "fixture" / "selected-tet-rows.csv"
    for key, path in {"cpp": source, "fixture": fixture, "rows": row_file}.items():
        if sha256(path) != report["inputs"][key]["sha256"]:
            raise SystemExit(f"packaged {key} does not match diagnostic report")

    with row_file.open(newline="", encoding="utf-8") as stream:
        rows = [line for line in stream if line.strip()]
    selected = pins["expected_arithmetic"]["selected_tets"]
    if len(rows) != selected + 1 or rows[0].strip() != "tet_row":
        raise SystemExit("selected-row table domain mismatch")
    fixture_bytes = fixture.read_bytes()
    if len(fixture_bytes) < 8:
        raise SystemExit("truncated fixture header")
    count = struct.unpack("<Q", fixture_bytes[:8])[0]
    if count != selected or len(fixture_bytes) != 8 + count * 12 * 4:
        raise SystemExit("fixture point payload size mismatch")

    compile_info = closure["compile"]
    with tempfile.TemporaryDirectory(prefix="shared-reference-motion-") as temp:
        binary = Path(temp) / "shared-reference-motion"
        argv = [
            compile_info["compiler"],
            *compile_info["arguments"],
            "-I" + str(repo_root / compile_info["include_roots"][0]),
            "-I" + str(repo_root / compile_info["include_roots"][1]),
            str(source),
            "-o",
            str(binary),
        ]
        subprocess.run(argv, check=True, capture_output=True, text=True)
        replay = subprocess.run([str(binary), str(fixture)], check=True, capture_output=True, text=True)
    actual = json.loads(replay.stdout)
    if actual != report["arithmetic"]:
        raise SystemExit("compiled fixture replay differs from pinned report")

    print(json.dumps({
        "status": "PASS_CPU_FIXTURE_REPLAY_NO_NATIVE",
        "report_sha256": sha256(report_path),
        "selected_tets": selected,
        "gpu_executed": False,
        "native_steps": 0,
        "physical_admission": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
