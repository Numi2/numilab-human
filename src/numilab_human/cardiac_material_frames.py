"""Prepare explicitly derived material frames; never step or repair anatomy.

The pinned Rodero source arrays remain immutable. This opt-in policy normalizes
the fibre and uses Gram–Schmidt on the sheet. The derived quaternion field is an
offline input to native authoring, not regional material assembly, a Matter
package, measured microstructure, or calibrated anatomical mechanics.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys
import tempfile

from . import cardiac_wall_source as wall
from .model import ImportError as HumanImportError

SCHEMA = "HumanPack.cardiac-material-frames.v1"
POLICY_ID = "normalize-fibre-gram-schmidt-sheet-v1"
QUATERNION_BUFFER = "material-frame-rotations.f64le"
POLICY = {
    "id": POLICY_ID,
    "arithmetic": "IEEE754 binary64; source field components are unchanged",
    "basis": "Q columns are fibre, sheet, normal; material to reference-world",
    "algorithm": [
        "f = raw_fibre / hypot(raw_fibre)",
        "s0 = raw_sheet / hypot(raw_sheet)",
        "r = s0 - dot(f,s0)*f; require hypot(r) > minimum_sheet_sine",
        "s = r / hypot(r)",
        "n = cross(f,s) / hypot(cross(f,s))",
        "Q = columns(f,s,n)",
        "Hamilton xyzw quaternion via positive trace, otherwise largest diagonal (x,y,z tie order)",
        "normalize quaternion; sign w>=0, at w=0 first nonzero x,y,z positive; canonical positive zero",
        "verify quaternion norm and reconstructed Q before writing little-endian binary64 xyzw",
    ],
    "minimum_sheet_sine": 1.0e-8,
    "orthonormal_absolute_tolerance": 1.0e-12,
    "quaternion_norm_squared_absolute_tolerance": 1.0e-12,
    "reconstruction_absolute_tolerance": 1.0e-12,
    "source_geometry_or_cell_order_changes": False,
    "source_frame_fields_modified": False,
    "source_axis_direction_is_measured": False,
}

# Exact import buffers for the versioned case18 source. Pinning these prevents
# a coherently rehashed fake manifest from claiming the original archive ID.
# The preliminary and final retained importer manifests have identical buffers.
SOURCE_BUFFERS = {
    "boundary.u32le": ("821ad6dedb5f32cce78314d704e35f03726188d9d56978ebb23646e4690b3c60", [230364, 3], 2764368),
    "boundary_components.u32le": ("8e07999b113bd619fe8f0cf0fc5435730fb962ed13bead55e18c9f759352ba0a", [230364], 921456),
    "boundary_owners.u32le": ("7c3f0bf847d4d5c1aa0badc5ff83345c0165cd3aab516d8576b55474d5495392", [230364], 921456),
    "fibres.f64le": ("5ec6aade89b8a86bd8d3529282565edc77386b9dbe7902b13cf698fb327605e8", [1470083, 3], 35281992),
    "labels.u32le": ("304d0d9d9a99d51be4367529e9598815361018a446ed047923c85b96786b569a", [1470083], 5880332),
    "nodes.f64le": ("4c8e6e7d9adfa5180ba0730c99222044c6a1aa51901823cf0b4dd861ae732716", [300965, 3], 7223160),
    "sheets.f64le": ("8171a504b443a3f8267de1bea7ad02e61862c530c8b3855570df968cd416c523", [1470083, 3], 35281992),
    "source_reversed_cells.u32le": ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", [0], 0),
    "tetrahedra.u32le": ("3f571c7815f0fd2e1f9d2735046d3bb7677a1a011215fca6737c6310b5b29066", [1470083, 4], 23521328),
    "uvc_phi.f64le": ("b214751e9ad04c5ca0b11d3271d6077a798f943ef1788665014cefe20ee66682", [300965], 2407720),
    "uvc_rho.f64le": ("7b5615e0d7492505bb65a4a78af9fc5b1a41e7965d3eb7fd5711883b3455374b", [300965], 2407720),
    "uvc_v.f64le": ("5d8d8a19e39717cd1369eefa924db62623bdc600479169b80be95c667773baa7", [300965], 2407720),
    "uvc_z.f64le": ("080789f06648ff82eb323b28d786cf5383c37ceb3cc4fa06c7ec1d5f6086f0e3", [300965], 2407720),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac material frames: " + message)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field {key}")
        result[key] = value
    return result


def _read_json(data: bytes) -> dict:
    def nonfinite(value: str) -> None:
        raise HumanImportError("cardiac material frames: nonfinite JSON " + value)
    result = json.loads(data, object_pairs_hook=_object, parse_constant=nonfinite)
    require(isinstance(result, dict), "manifest must be an object")
    # JSON exponent overflow (e.g. 1e999) does not invoke parse_constant.
    canonical(result)
    return result


def _open_source(path: Path):
    require(not path.is_symlink(), f"symlink input {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"nonregular input {path.name}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _digest_source(path: Path, expected: tuple) -> None:
    h = hashlib.sha256(); size = 0
    with _open_source(path) as stream:
        while data := stream.read(1 << 20):
            size += len(data); h.update(data)
    require(size == expected[2] and h.hexdigest() == expected[0], f"source buffer changed: {path.name}")


def _validate_asset(asset: Path, manifest_bytes: bytes, expected_manifest_sha256: str,
                    configuration: dict, configuration_sha256: str, pins: dict) -> dict:
    require(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is not None,
            "expected manifest SHA256 must be 64 lowercase hexadecimal characters")
    require(sha256(manifest_bytes) == expected_manifest_sha256, "asset manifest hash mismatch")
    manifest = _read_json(manifest_bytes)
    require(manifest.get("schema") == wall.SCHEMA, "unsupported asset manifest schema")
    require(manifest.get("source_config_sha256") == configuration_sha256 and
            manifest.get("source_config") == configuration, "source configuration mismatch")
    buffers = manifest.get("buffers")
    require(isinstance(buffers, dict) and set(buffers) == set(pins), "source buffer inventory mismatch")
    for name, expected in pins.items():
        record = buffers[name]
        require(isinstance(record, dict) and record.get("sha256") == expected[0] and
                record.get("shape") == expected[1] and type(record.get("bytes")) is int and
                record["bytes"] == expected[2], f"source buffer identity/shape mismatch: {name}")
        require(all(type(v) is int for v in record["shape"]), f"noninteger shape: {name}")
        _digest_source(asset / name, expected)
    n = configuration["mesh"]["cells"]
    require(type(n) is int and n > 0 and pins["fibres.f64le"][1] == [n, 3] and
            pins["sheets.f64le"][1] == [n, 3] and pins["labels.u32le"][1] == [n] and
            pins["tetrahedra.u32le"][1] == [n, 4] and
            manifest.get("topology", {}).get("positive_oriented_tetrahedra") == n,
            "source cell counts mismatch")
    return manifest


def quaternion_matrix(q: tuple) -> tuple:
    """Row-major material-to-reference-world matrix of a unit Hamilton xyzw q."""
    x, y, z, w = q
    return (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
            2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
            2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))


def _quaternion(m: tuple) -> tuple:
    trace = m[0] + m[4] + m[8]
    if trace > 0:
        scale = 2 * math.sqrt(1 + trace)
        q = ((m[7]-m[5])/scale, (m[2]-m[6])/scale, (m[3]-m[1])/scale, scale/4)
    elif m[0] >= m[4] and m[0] >= m[8]:
        scale = 2 * math.sqrt(1 + m[0] - m[4] - m[8])
        q = (scale/4, (m[1]+m[3])/scale, (m[2]+m[6])/scale, (m[7]-m[5])/scale)
    elif m[4] >= m[8]:
        scale = 2 * math.sqrt(1 + m[4] - m[0] - m[8])
        q = ((m[1]+m[3])/scale, scale/4, (m[5]+m[7])/scale, (m[2]-m[6])/scale)
    else:
        scale = 2 * math.sqrt(1 + m[8] - m[0] - m[4])
        q = ((m[2]+m[6])/scale, (m[5]+m[7])/scale, scale/4, (m[3]-m[1])/scale)
    norm = math.hypot(*q)
    q = tuple(v/norm for v in q)
    first = next((v for v in (q[3], q[0], q[1], q[2]) if v != 0), 1)
    if first < 0:
        q = tuple(-v for v in q)
    return tuple(0.0 if v == 0 else v for v in q)


def convert_axes(fibre: tuple, sheet: tuple) -> tuple[tuple, dict]:
    """Apply the declared policy to one row; return xyzw plus numeric diagnostics."""
    require(len(fibre) == len(sheet) == 3 and
            all(math.isfinite(v) for v in (*fibre, *sheet)), "nonfinite or malformed source axes")
    fn = math.hypot(*fibre); sn = math.hypot(*sheet)
    require(math.isfinite(fn) and math.isfinite(sn) and fn > 0 and sn > 0, "zero or overflowing source axis")
    f = tuple(v/fn for v in fibre); s0 = tuple(v/sn for v in sheet)
    dot = sum(x*y for x, y in zip(f, s0))
    residual = tuple(s0[i]-dot*f[i] for i in range(3))
    sine = math.hypot(*residual)
    require(sine > POLICY["minimum_sheet_sine"], "parallel or numerically degenerate source axes")
    s = tuple(v/sine for v in residual)
    cross = (f[1]*s[2]-f[2]*s[1], f[2]*s[0]-f[0]*s[2], f[0]*s[1]-f[1]*s[0])
    nn = math.hypot(*cross)
    require(math.isfinite(nn) and nn > 0, "degenerate right-handed normal")
    n = tuple(v/nn for v in cross)
    axes = (f, s, n)
    orthogonal_error = max(abs(sum(x*y for x, y in zip(a, b)) - (i == j))
                           for i, a in enumerate(axes) for j, b in enumerate(axes))
    require(orthogonal_error <= POLICY["orthonormal_absolute_tolerance"], "unstable orthonormal construction")
    matrix = tuple(axes[j][i] for i in range(3) for j in range(3))
    q = _quaternion(matrix)
    norm_error = abs(sum(v*v for v in q)-1)
    reconstruction_error = max(abs(a-b) for a, b in zip(matrix, quaternion_matrix(q)))
    require(norm_error <= POLICY["quaternion_norm_squared_absolute_tolerance"] and
            reconstruction_error <= POLICY["reconstruction_absolute_tolerance"], "quaternion reconstruction failed")
    sheet_cross = (s0[1]*s[2]-s0[2]*s[1], s0[2]*s[0]-s0[0]*s[2], s0[0]*s[1]-s0[1]*s[0])
    angle = math.atan2(math.hypot(*sheet_cross), sum(x*y for x, y in zip(s0, s)))
    return q, {"raw_fibre_norm_error": abs(fn-1), "raw_sheet_norm_error": abs(sn-1),
        "raw_normalized_abs_dot": abs(dot), "sheet_correction_radians": angle,
        "fibre_direction_correction_radians": 0.0, "orthonormal_error": orthogonal_error,
        "quaternion_norm_squared_error": norm_error, "quaternion_reconstruction_error": reconstruction_error}


def native_identity_words(identity: str) -> tuple[int, int, int, int]:
    require(re.fullmatch(r"[0-9a-f]{64}", identity) is not None, "invalid source identity SHA256")
    return struct.unpack("<4Q", bytes.fromhex(identity))


def _convert_streams(asset: Path, staging: Path, pins: dict, cell_count: int) -> tuple[dict, dict]:
    names = ("fibres.f64le", "sheets.f64le", "labels.u32le")
    consumed = {name: hashlib.sha256() for name in names}
    output_hash = hashlib.sha256(); regions = {}; maximum = {}
    pack = struct.Struct("<4d")
    with ExitStack() as stack:
        streams = {name: stack.enter_context(_open_source(asset/name)) for name in names}
        output = stack.enter_context((staging/QUATERNION_BUFFER).open("xb"))
        for offset in range(0, cell_count, 4096):
            count = min(4096, cell_count-offset)
            raw = {name: streams[name].read(count*(4 if name == "labels.u32le" else 24)) for name in names}
            for name, data in raw.items():
                require(len(data) == count*(4 if name == "labels.u32le" else 24), f"truncated consumed source: {name}")
                consumed[name].update(data)
            encoded = bytearray(count*32)
            rows = zip(struct.iter_unpack("<3d", raw[names[0]]), struct.iter_unpack("<3d", raw[names[1]]),
                       struct.iter_unpack("<I", raw[names[2]]))
            for local, (fibre, sheet, (label,)) in enumerate(rows):
                require(1 <= label <= 24, f"unknown source label at cell {offset+local}")
                try:
                    quaternion, values = convert_axes(fibre, sheet)
                except HumanImportError as error:
                    raise HumanImportError(f"{error}; source cell {offset+local}") from error
                pack.pack_into(encoded, local*32, *quaternion)
                region = regions.setdefault(str(label), {"count": 0, "maximum": {}})
                region["count"] += 1
                for key, value in values.items():
                    maximum[key] = max(maximum.get(key, 0.0), value)
                    region["maximum"][key] = max(region["maximum"].get(key, 0.0), value)
            output.write(encoded); output_hash.update(encoded)
        for name in names:
            require(not streams[name].read(1) and consumed[name].hexdigest() == pins[name][0],
                    f"consumed source bytes changed: {name}")
        output.flush(); os.fsync(output.fileno())
    return ({"bytes": cell_count*32, "sha256": output_hash.hexdigest(), "shape": [cell_count, 4],
             "encoding": "little-endian IEEE754 binary64 Hamilton quaternion x,y,z,w; source cell order"},
            {"count": cell_count, "maximum": maximum, "regions": regions,
             "consumed_source_sha256": {n: h.hexdigest() for n, h in consumed.items()}})


def _prepare(asset: Path, output: Path, expected_manifest_sha256: str, conversion_policy: str,
             configuration: dict, configuration_sha256: str, pins: dict, source_kind: str) -> dict:
    require(conversion_policy == POLICY_ID, "explicit supported conversion policy is required")
    policy_bytes = canonical(POLICY)
    implementation_sha256 = sha256(Path(__file__).read_bytes())
    require(asset.is_dir() and not asset.is_symlink(), "asset must be a real directory")
    require(not output.is_symlink() and not output.resolve().is_relative_to(asset.resolve()),
            "output must be separate from immutable source asset")
    if output.exists():
        require(output.is_dir() and {p.name for p in output.iterdir()} == {"manifest.json", QUATERNION_BUFFER},
                "refusing changed or unrelated output directory")
    with _open_source(asset/"manifest.json") as stream:
        manifest_bytes = stream.read()
    manifest = _validate_asset(asset, manifest_bytes, expected_manifest_sha256,
                               configuration, configuration_sha256, pins)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cardiac-material-frames-", dir=output.parent))
    try:
        converted, diagnostics = _convert_streams(asset, staging, pins, configuration["mesh"]["cells"])
        # Validate the named input again before publication, including buffers
        # not consumed by conversion. The actual consumed fields are separately
        # hashed above, so replacing a path cannot rebind opened-descriptor data.
        for name, pin in pins.items():
            _digest_source(asset/name, pin)
        with _open_source(asset/"manifest.json") as stream:
            require(stream.read() == manifest_bytes, "source manifest changed during conversion")
        require(canonical(POLICY) == policy_bytes and sha256(Path(__file__).read_bytes()) == implementation_sha256,
                "conversion policy or implementation changed during conversion")
        identity_record = {"schema": "HumanPack.cardiac-material-frame-identity.v1",
            "source_kind": source_kind, "asset_manifest_sha256": expected_manifest_sha256,
            "source_config_sha256": configuration_sha256,
            "source_buffers": {n: {"sha256": p[0], "shape": p[1], "bytes": p[2]} for n, p in pins.items()},
            "conversion_policy_sha256": sha256(policy_bytes), "converted_buffer": converted}
        identity = sha256(canonical(identity_record))
        result = {"schema": SCHEMA, "source_identity_sha256": identity,
            "source_identity_record": identity_record, "source_config": configuration,
            "conversion_policy": json.loads(policy_bytes), "buffer": {"path": QUATERNION_BUFFER, **converted},
            "native_source_identity": {"encoding": "raw SHA256 digest split into four little-endian uint64 words",
                "words_hex": [f"0x{word:016x}" for word in native_identity_words(identity)]},
            "diagnostics": diagnostics, "source_asset_qualification": manifest.get("qualification", {}),
            "implementation_sha256": implementation_sha256,
            "qualification": {"status": "derived_material_frames_only", "physical_steps": 0,
                "raw_source_fields_modified": False, "source_geometry_modified": False,
                "native_matter_package": False, "regional_material_assembly": False,
                "density_supplied": False, "measured_microstructure": False,
                "anatomical_wall_simulation": False, "subject_calibrated": False}}
        (staging/"manifest.json").write_bytes(canonical(result))
        if output.exists():
            require(output.is_dir() and not output.is_symlink() and
                    {p.name for p in output.iterdir()} == {"manifest.json", QUATERNION_BUFFER},
                    "output changed during conversion")
            for name in ("manifest.json", QUATERNION_BUFFER):
                require(not (output/name).is_symlink(), "symlink output member")
                _digest_source(output/name, (wall.digest(staging/name), [], (staging/name).stat().st_size))
        else:
            os.rename(staging, output)
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare_material_frames(asset: Path, output: Path, *, expected_manifest_sha256: str,
                            conversion_policy: str) -> dict:
    """Convert only the exact pinned Rodero18 asset with explicit policy consent.

    The output contains derived frames and provenance only. The input manifest
    SHA must be provided by the caller, and all thirteen source buffers must also
    match the independent case18 pins. Existing output is accepted only if exact.
    """
    configuration, configuration_sha256 = wall.load_config(wall.CONFIG)
    return _prepare(Path(asset), Path(output), expected_manifest_sha256, conversion_policy,
                    configuration, configuration_sha256, SOURCE_BUFFERS, "pinned_rodero_2021_ct_case18")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path, help="Existing immutable Rodero18 source asset")
    parser.add_argument("--output", required=True, type=Path, help="New immutable derived-frame directory")
    parser.add_argument("--asset-manifest-sha256", required=True, help="Exact SHA256 of input manifest bytes")
    parser.add_argument("--conversion-policy", required=True, choices=[POLICY_ID])
    args = parser.parse_args(argv)
    try:
        result = prepare_material_frames(args.asset, args.output,
            expected_manifest_sha256=args.asset_manifest_sha256, conversion_policy=args.conversion_policy)
        print(json.dumps({"status": result["qualification"]["status"], "output": str(args.output),
            "frames": result["diagnostics"]["count"], "source_identity_sha256": result["source_identity_sha256"],
            "quaternion_sha256": result["buffer"]["sha256"], "physical_steps": 0}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, ValueError, TypeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
