from __future__ import annotations
import hashlib
import subprocess
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_current_native_1025_direct_o_first_run_010c_v16.sh"

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def invoke_with_invalid_pin(tmp_path: Path, which: str):
    files = {}
    for name in ("run.json", "closure.json", "build.json", "checkpoint.json"):
        p = tmp_path / name
        p.write_text("{}\n")
        files[name] = p
    audit = tmp_path / "orientation.json"
    audit.write_text("not parsed because the pin must fail first\n")
    owner = tmp_path / "orientation-owner.py"
    owner.write_text("# nonlaunch test owner\n")
    audit_sha = digest(audit) if which == "owner" else "0" * 64
    owner_sha = "0" * 64 if which == "owner" else digest(owner)
    job = tmp_path / f"job-{which}"
    cmd = [str(RUNNER), str(files["run.json"]), str(files["closure.json"]),
           str(files["build.json"]), str(files["checkpoint.json"]), str(audit),
           audit_sha, str(owner), owner_sha, str(job)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result, job

def test_runner_rejects_wrong_audit_sha_before_creating_job_or_launching(tmp_path):
    result, job = invoke_with_invalid_pin(tmp_path, "audit")
    assert result.returncode == 2
    assert "fresh orientation audit SHA mismatch" in result.stderr
    assert not job.exists()

def test_runner_rejects_wrong_owner_sha_before_creating_job_or_launching(tmp_path):
    result, job = invoke_with_invalid_pin(tmp_path, "owner")
    assert result.returncode == 2
    assert "fresh orientation owner SHA mismatch" in result.stderr
    assert not job.exists()

def test_runner_has_correct_nine_argument_and_v16_expansion_contract():
    text = RUNNER.read_text()
    assert "[[ $# -ne 9 ]]" in text
    for token in ('RUN=$(resolve_file "$1")', 'CLOSURE=$(resolve_file "$2")',
                  'BUILD=$(resolve_file "$3")', 'CHECKPOINT=$(resolve_file "$4")',
                  'AUDIT=$(resolve_file "$5")', 'AUDIT_SHA=$6',
                  'OWNER=$(resolve_file "$7")', 'OWNER_SHA=$8', 'JOB_INPUT=$9'):
        assert token in text
    assert "\\${" not in text
    assert "--orientation-audit \"$AUDIT\" --orientation-audit-sha256 \"$AUDIT_SHA\"" in text
    assert "--orientation-owner \"$OWNER\" --orientation-owner-sha256 \"$OWNER_SHA\"" in text
    assert "first-horizon-gate.json" in text
