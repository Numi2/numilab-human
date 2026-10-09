import errno
import hashlib
import os
import stat
import sys

import pytest

from numilab_human import resting_anatomy


@pytest.mark.parametrize("link_error", [errno.EPERM, errno.EXDEV])
def test_checked_sidecar_copies_and_checks_both_hashes_after_link_fallback(
    tmp_path, monkeypatch, link_error
):
    source = tmp_path / "owner-sidecar.bin"
    destination = tmp_path / "composed-sidecar.bin"
    payload = b"immutable but readable owner sidecar"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def fail_link(_source, _destination):
        raise OSError(link_error, os.strerror(link_error))

    monkeypatch.setattr(resting_anatomy.os, "link", fail_link)
    resting_anatomy._copy_checked_sidecar(source, destination, expected)

    assert destination.read_bytes() == payload
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
    assert not os.path.samefile(source, destination)


def test_fallback_copy_uses_exclusive_destination_creation(tmp_path, monkeypatch):
    source = tmp_path / "owner-sidecar.bin"
    destination = tmp_path / "composed-sidecar.bin"
    payload = b"source"
    competitor = b"created after the destination preflight"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def create_destination_then_fail(_source, target):
        target.write_bytes(competitor)
        raise OSError(errno.EPERM, os.strerror(errno.EPERM))

    monkeypatch.setattr(resting_anatomy.os, "link", create_destination_then_fail)
    with pytest.raises(FileExistsError):
        resting_anatomy._copy_checked_sidecar(source, destination, expected)

    assert destination.read_bytes() == competitor


def test_unrelated_hardlink_errors_propagate_without_copy(tmp_path, monkeypatch):
    source = tmp_path / "owner-sidecar.bin"
    destination = tmp_path / "composed-sidecar.bin"
    source.write_bytes(b"source")

    def fail_link(_source, _destination):
        raise OSError(errno.EACCES, os.strerror(errno.EACCES))

    monkeypatch.setattr(resting_anatomy.os, "link", fail_link)
    with pytest.raises(OSError) as raised:
        resting_anatomy._copy_checked_sidecar(
            source, destination, hashlib.sha256(source.read_bytes()).hexdigest()
        )

    assert raised.value.errno == errno.EACCES
    assert not destination.exists()


def test_destination_and_source_identity_checks_remain_fail_closed(tmp_path):
    source = tmp_path / "owner-sidecar.bin"
    destination = tmp_path / "composed-sidecar.bin"
    payload = b"sidecar"
    source.write_bytes(payload)
    destination.write_bytes(b"existing")

    with pytest.raises(ValueError, match="destination already exists"):
        resting_anatomy._copy_checked_sidecar(
            source, destination, hashlib.sha256(payload).hexdigest()
        )
    assert destination.read_bytes() == b"existing"

    destination.unlink()
    with pytest.raises(ValueError, match="identity differs"):
        resting_anatomy._copy_checked_sidecar(source, destination, "0" * 64)
    assert not destination.exists()


@pytest.mark.skipif(
    sys.platform != "darwin"
    or not hasattr(os, "chflags")
    or not getattr(stat, "UF_IMMUTABLE", 0),
    reason="Darwin immutable-file flags are required for this regression",
)
def test_checked_sidecar_copies_readable_uchg_source(tmp_path):
    source = tmp_path / "owner-sidecar.bin"
    destination = tmp_path / "composed-sidecar.bin"
    probe = tmp_path / "hardlink-probe.bin"
    payload = b"readable source with Darwin uchg flag"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    original_flags = source.stat().st_flags

    try:
        os.chflags(source, original_flags | stat.UF_IMMUTABLE)
        with pytest.raises(OSError) as raised:
            os.link(source, probe)
        assert raised.value.errno == errno.EPERM

        resting_anatomy._copy_checked_sidecar(source, destination, expected)

        assert source.read_bytes() == payload
        assert destination.read_bytes() == payload
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == expected
        assert not os.path.samefile(source, destination)
    finally:
        # Clear uchg even if an assertion or the copy path fails so pytest can
        # remove its temporary directory.
        os.chflags(source, original_flags)
