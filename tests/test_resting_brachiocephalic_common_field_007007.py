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


def test_preflight_imports_the_pinned_published_owner_sources():
    pins = driver._owner_import_pins()

    assert pins["common_field_compiler_code_sha256"] == {
        name: digest for name, digest in driver.OWNER_SOURCE_SHA256.items()
        if name != "myosim_bone_proximity.py"
    }
    assert pins["modules"][
        "numilab_human.myosim_bone_proximity"]["sha256"] == (
            driver.OWNER_SOURCE_SHA256["myosim_bone_proximity.py"])
    assert pins["modules"][
        "numilab_human.resting_brachiocephalic_common_field"]["origin"].endswith(
            "/resting_brachiocephalic_common_field.py")


def test_python_preflight_binds_qualified_venv_not_framework_symlink():
    identity = driver._python_executable_identity()

    assert identity["virtual_environment"].endswith(
        "/numi-human-prep-venv-20261005")
    assert identity["base_prefix"] != identity["virtual_environment"]
    assert identity["executable"].endswith("/bin/python3")
    assert identity["resolved_executable"] != identity["executable"]


def test_python_preflight_rejects_another_venv(monkeypatch):
    monkeypatch.setattr(driver.sys, "prefix", "/tmp/other-venv")

    with pytest.raises(ValueError, match="qualified Numi interpreter"):
        driver._python_executable_identity()
