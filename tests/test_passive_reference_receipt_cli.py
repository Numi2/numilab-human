"""The source CLI must reject mismatched evidence before creating output."""
import hashlib
import pytest
from numilab_human import passive_reference_receipt_composition as owner


def inputs(tmp_path):
    receipt, patch, report = [tmp_path/name for name in ("receipt.json", "patch.npz", "report.json")]
    for path in (receipt, patch, report):
        path.write_bytes(path.name.encode())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    argv = ["--receipt", str(receipt), "--receipt-sha256", sha(receipt),
            "--output", str(tmp_path/"out"), "--replacement", "103", str(patch),
            sha(patch), str(report), sha(report), "--extend", "103", "107"]
    return argv, receipt, patch, report


def test_pinned_cli_delegates_to_existing_owner(tmp_path, monkeypatch):
    argv, receipt, patch, report = inputs(tmp_path)
    calls = []
    def compose(*args, **kwargs):
        calls.append((args, kwargs))
        return {"native_admission": False}
    monkeypatch.setattr(owner, "compose_reference_receipt", compose)
    assert owner.main(argv) == 0
    assert calls == [((receipt, tmp_path/"out", [(103, patch, report)]),
                      {"extensions": {103: 107}})]


@pytest.mark.parametrize("fault", ["hash", "duplicate", "extension", "conflicting_pin"])
def test_invalid_input_rejected_before_owner_or_output(tmp_path, monkeypatch, fault):
    argv, receipt, patch, report = inputs(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("composition must not start")
    monkeypatch.setattr(owner, "compose_reference_receipt", forbidden)
    if fault == "hash":
        argv[argv.index("--replacement")+3] = "0"*64
    elif fault == "duplicate":
        at = argv.index("--replacement")
        argv += argv[at:at+6]
    elif fault == "extension":
        argv += ["--extend", "103", "109"]
    else:
        argv += ["--source-pin", str(receipt), "0"*64]
    with pytest.raises(ValueError):
        owner.main(argv)
    assert not (tmp_path/"out").exists()


def test_preflight_never_composes(tmp_path, monkeypatch, capsys):
    argv, *_ = inputs(tmp_path)
    monkeypatch.setattr(owner, "compose_reference_receipt",
                        lambda *a, **k: pytest.fail("preflight composed sources"))
    assert owner.main(argv+["--preflight-only"]) == 0
    assert '"geometry_predicates_executed": false' in capsys.readouterr().out
    assert not (tmp_path/"out").exists()
