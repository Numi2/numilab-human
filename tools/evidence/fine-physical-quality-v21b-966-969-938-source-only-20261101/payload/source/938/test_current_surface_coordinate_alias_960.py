from __future__ import annotations
import hashlib
import importlib.util
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
OWNER = HERE / "current_fine_217_owner_938_v2_coordinate_alias_guard.py"
BRIDGE = HERE / "current_fine_volume_to_217_938_v2_coordinate_alias_guard.py"
EXPECTED_OWNER_SHA = "1fc5d6278aa0aa9bd76105d4bf7caa2c1ada6431428e20fe16c28e4afad3f6e4"
EXPECTED_BRIDGE_SHA = "1756c7126f708141b06b57e9b653a121d038d215d8801bb7e28f5220edcc8c66"
STATE = HERE.parent / "all-quality-959-remesh-883h-fixed-775-qualified-root" / "verified-state.json"
EXPECTED_STATE_SHA = "c3e51d1d4b1c9c15f639b564c677c57c13d3361803f6289b2c4ac34d55ee487e"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_bridge_and_owner():
    if sha(OWNER) != EXPECTED_OWNER_SHA or sha(BRIDGE) != EXPECTED_BRIDGE_SHA:
        raise AssertionError("versioned 938 owner/caller source pin mismatch")
    spec = importlib.util.spec_from_file_location("current_fine_volume_to_217_938_v2", BRIDGE)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    owner = bridge.import_owner()
    if owner.CAPACITY_PROFILE != "full-reference-500k-nodes-2m-cells":
        raise AssertionError("versioned caller did not configure the established capacity profile")
    return bridge, owner

def write_surface(path: Path, points64: np.ndarray, points32: np.ndarray | None) -> None:
    payload = {"points_world_m": points64, "faces": np.asarray([[0, 1, 2]], np.int64),
               "face_cache_row": np.asarray([0], np.int64), "face_exact_source_row": np.asarray([0], np.int64)}
    if points32 is not None:
        payload["points_world_m_f32"] = points32
    np.savez_compressed(path, **payload)

def test_alias_guard() -> dict:
    bridge, owner = load_bridge_and_owner()
    p32 = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    p64 = p32.astype(np.float64)
    checks = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        valid = root / "valid.npz"
        write_surface(valid, p64, p32.copy())
        got = owner.load_surface(valid, "I")["points"]
        assert got.dtype == np.float32 and got.view(np.uint32).tobytes() == p32.view(np.uint32).tobytes()
        checks["matching_full_and_f32_views_pass_and_f32_lane_is_consumed"] = True

        missing = root / "missing-f32.npz"
        write_surface(missing, p64, None)
        fallback = owner.load_surface(missing, "I")["points"]
        assert fallback.view(np.uint32).tobytes() == p32.view(np.uint32).tobytes()
        checks["missing_f32_lane_uses_full_precision_cast"] = True

        stale = root / "stale-f32.npz"
        bad = p32.copy()
        bad[1, 0] = np.nextafter(bad[1, 0], np.float32(np.inf))
        write_surface(stale, p64, bad)
        try:
            owner.load_surface(stale, "I")
        except ValueError as e:
            assert "coordinate views disagree" in str(e)
            checks["one_ulp_stale_f32_alias_is_rejected"] = True
        else:
            raise AssertionError("stale F32 alias unexpectedly passed surface admission")

        wrong_dtype = root / "wrong-dtype.npz"
        write_surface(wrong_dtype, p64, p32.astype(np.float64))
        try:
            owner.load_surface(wrong_dtype, "I")
        except ValueError as e:
            assert "invalid dtype" in str(e)
            checks["non_f32_alias_dtype_is_rejected"] = True
        else:
            raise AssertionError("non-F32 alias unexpectedly passed surface admission")
    return checks

def test_selected_959_assets() -> dict:
    import json
    if sha(STATE) != EXPECTED_STATE_SHA:
        raise AssertionError("selected 959 state pin mismatch")
    state = json.loads(STATE.read_text())
    pins = state["source_surfaces"] | {r: state[r + "_current"] for r in ("Skin", "Fat")}
    checks = {}
    for role, pin in pins.items():
        if sha(Path(pin["path"])) != pin["sha256"]:
            raise AssertionError(f"{role} source/volume asset pin mismatch")
        with np.load(pin["path"], allow_pickle=False) as z:
            if "points_world_m" not in z.files or "points_world_m_f32" not in z.files:
                raise AssertionError(f"{role} lacks both coordinate views")
            full = np.asarray(z["points_world_m"])
            stored = np.asarray(z["points_world_m_f32"])
            cast = np.ascontiguousarray(full, dtype=np.float32)
            if stored.dtype != np.float32 or stored.shape != full.shape or cast.view(np.uint32).tobytes() != np.ascontiguousarray(stored).view(np.uint32).tobytes():
                raise AssertionError(f"{role} coordinate views differ")
            checks[role] = {"rows": int(len(full)), "full_dtype": full.dtype.str, "f32_dtype": stored.dtype.str, "bitwise_f32_alias_match": True}
    return checks

if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    result = {
        "schema": "numi.h706.current-surface-coordinate-alias-guard-960.v1",
        "status": "PASS_SOURCE_ONLY_COORDINATE_ALIAS_ADMISSION_CONTROLS",
        "owner": {"path": str(OWNER), "sha256": sha(OWNER)},
        "bridge": {"path": str(BRIDGE), "sha256": sha(BRIDGE)},
        "test": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        "synthetic_controls": test_alias_guard(),
        "selected_state": {"path": str(STATE), "sha256": sha(STATE)},
        "selected_959_assets": test_selected_959_assets(),
        "physical_admission": False,
        "native_steps": 0,
        "mass_reconciliation": "NOT_RUN",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")
