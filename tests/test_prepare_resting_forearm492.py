import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "forearm492", Path(__file__).parents[1] / "tools/prepare_resting_forearm492.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def owner():
    return {
        "source_payloads": [{"path": f"source{i}", "sha256": str(i)*64} for i in range(8)],
        "child_payload_sha256": "a"*64,
        "surfaces": [
            {"stable_id": 132, "vertices": [[[0, 132, 1, 1.0]]]},
            {"stable_id": 134, "vertices": [[[5, 134, 1, 1.0]]]},
            {"stable_id": 157, "vertices": [[[2, 157, 1, 1.0]]]},
        ],
        "anatomy_surfaces": [{"stable_id": 477, "vertices": [[[7, 477, 1, 1.0]]]}],
    }


def test_replaces_only_existing_fds_source_and_preserves_fdp_back_vessels():
    original = owner()
    frozen = copy.deepcopy(original)
    replacement = {"stable_id": 132, "vertices": [[[0, 132, 4, .5], [0, 132, 5, .5]]]}
    result = compiler.replace_owner(original, replacement, "/new/carrier", "b"*64, "c"*64)
    assert original == frozen
    assert len(result["source_payloads"]) == 8
    assert result["source_payloads"][1:] == frozen["source_payloads"][1:]
    assert result["surfaces"][1:] == frozen["surfaces"][1:]
    assert result["anatomy_surfaces"] == frozen["anatomy_surfaces"]
    assert result["surfaces"][0] == replacement
    assert result["child_payload_sha256"] == "c"*64


@pytest.mark.parametrize("group", ["surfaces", "anatomy_surfaces"])
def test_rejects_replacement_if_unrelated_row_uses_source_zero(group):
    original = owner()
    original[group][-1]["vertices"] = [[[0, 99, 1, 1.0]]]
    with pytest.raises(ValueError, match="unrelated surface"):
        compiler.replace_owner(original, {"stable_id": 132}, "/new/carrier", "b"*64, "c"*64)


def test_cannot_append_duplicate_fdp_or_exceed_native_source_capacity():
    original = owner()
    with pytest.raises(ValueError, match="FDS132"):
        compiler.replace_owner(original, {"stable_id": 134}, "/new/carrier", "b"*64, "c"*64)
    original["source_payloads"].append({"path": "ninth", "sha256": "9"*64})
    with pytest.raises(ValueError, match="capacity"):
        compiler.replace_owner(original, {"stable_id": 132}, "/new/carrier", "b"*64, "c"*64)
