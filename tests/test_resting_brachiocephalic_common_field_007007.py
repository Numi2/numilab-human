from pathlib import Path

import pytest

from numilab_human import resting_brachiocephalic_common_field_007007 as driver


def test_header_preflight_reads_only_abi5_header_and_checks_declared_length(tmp_path):
    header = driver.NHANAT_HEADER.pack(
        b"NHANAT1\0", 5, 2, 4, 6, 123, bytes(range(32)))
    raw = header + bytes(2 * driver.NHANAT_RECORD.size + 4 * 24 + 6 * 4)
    path = tmp_path / "fixture.nhanatomy"
    path.write_bytes(raw)

    report = driver._read_nhanat_header(path)

    assert report["magic"] == "NHANAT1"
    assert report["abi"] == 5
    assert report["surface_count"] == 2
    assert report["vertex_count"] == 4
    assert report["index_count"] == 6
    assert report["geometry_arrays_decoded"] is False
    assert report["header_bytes_sha256"]


def test_header_preflight_rejects_wrong_abi_and_truncated_payload(tmp_path):
    path = tmp_path / "fixture.nhanatomy"
    path.write_bytes(driver.NHANAT_HEADER.pack(
        b"NHANAT1\0", 4, 1, 3, 3, 123, bytes(32)))
    with pytest.raises(ValueError, match="not NHANAT ABI5"):
        driver._read_nhanat_header(path)

    path.write_bytes(driver.NHANAT_HEADER.pack(
        b"NHANAT1\0", 5, 1, 3, 3, 123, bytes(32)))
    with pytest.raises(ValueError, match="byte length"):
        driver._read_nhanat_header(path)


def test_historical_owner_pins_match_the_published_revision():
    import hashlib
    import subprocess
    from pathlib import Path

    repo = Path(driver.__file__).resolve().parents[2]
    for name, expected in driver.OWNER_SOURCE_SHA256.items():
        result = subprocess.run(
            ["git", "-C", str(repo), "show",
             f"{driver.PUBLISHED_OWNER_REVISION}:src/numilab_human/{name}"],
            check=True, capture_output=True)
        assert hashlib.sha256(result.stdout).hexdigest() == expected


def test_historical_preflight_fails_closed_on_changed_current_owner():
    with pytest.raises(ValueError, match="published source owner hash changed"):
        driver._owner_import_pins()


def test_python_preflight_binds_qualified_venv_not_framework_symlink():
    identity = driver._python_executable_identity()

    assert identity["virtual_environment"].endswith(
        "/numi-human-prep-venv-20261005")
    assert identity["base_prefix"] != identity["virtual_environment"]
    assert Path(identity["executable"]).parent.name == "bin"
    assert Path(identity["executable"]).name in {"python", "python3"}
    assert identity["resolved_executable"] != identity["executable"]


def test_python_preflight_rejects_another_venv(monkeypatch):
    monkeypatch.setattr(driver.sys, "prefix", "/tmp/other-venv")

    with pytest.raises(ValueError, match="qualified Numi interpreter"):
        driver._python_executable_identity()


@pytest.mark.parametrize("present", [False, True])
def test_unconsumed_historical_p6_pin_does_not_block_common_preflight(
    tmp_path, monkeypatch, present):
    p6_path = tmp_path / "resting_anatomy_verify.py"
    if present:
        p6_path.write_text("current verifier differs from historical source pin")
    monkeypatch.setattr(driver, "UNCONSUMED_P6_VERIFIER", p6_path)
    monkeypatch.setattr(driver.source007, "P6_ROOT", tmp_path)
    monkeypatch.setattr(driver.source007, "INPUT_SHA", {str(p6_path): "old-pin"})

    files = driver._input_files()
    context = driver._current_unconsumed_p6_context()

    assert p6_path.resolve() not in {path.resolve() for path in files}
    assert driver._verify_file_pins({}) == {}
    assert context["status"] == "HISTORICAL_UNCONSUMED_CONTEXT_ONLY"
    assert context["historical_source007_verifier_sha256"] == (
        driver.source007.P6_VERIFIER_SHA)
    assert context["current_sha256"] == (
        driver._sha_file(p6_path) if present else None)
    assert context["consumed_by_common_field_preparation"] is False
    assert context["p6_predicates_invoked"] is False


def test_common_preparation_rejects_drift_in_a_consumed_input(tmp_path):
    consumed = tmp_path / "consumed-owner.py"
    consumed.write_text("pinned source owner")
    expected = driver._sha_file(consumed)

    assert driver._verify_file_pins({consumed: expected}) == {
        str(consumed.resolve()): expected}
    consumed.write_text("changed source owner")
    with pytest.raises(ValueError, match="input hash changed"):
        driver._verify_file_pins({consumed: expected})
