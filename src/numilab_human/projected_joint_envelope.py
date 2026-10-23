"""Prepare compatible auxiliary bounds for the native exact-projection policy.

This is asset preparation, never a physics step. It does not reproduce MuJoCo
soft constraints. Primary ranges, equality bytes, poses, masses and other DOFs
remain unchanged. Only explicitly selected, bounded independent master joints
may authorize expansion of their terminal equality-dependent hard ranges.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import struct

POSITION_LIMIT = 1 << 2
U = F(1, 1 << 24)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _outward32(value: F, upper: bool) -> float:
    """Round an exact rational outward, including negative and zero bounds."""
    try:
        rounded = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("envelope exceeds finite Float32") from exc
    if not math.isfinite(rounded):
        raise ValueError("envelope exceeds finite Float32")
    if (upper and F(rounded) < value) or (not upper and F(rounded) > value):
        bits = struct.unpack("<I", struct.pack("<f", rounded))[0]
        if rounded == 0:
            bits = 1 if upper else 0x80000001
        else:
            bits += 1 if (rounded > 0) == upper else -1
        rounded = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(rounded):
        raise ValueError("envelope exceeds finite Float32")
    return rounded


def polynomial_envelope(coefficients, domain, dependent_reference=0.0,
                        master_reference=0.0, *, tiles=256):
    """Bound exact polynomial using rational Bernstein hulls on subintervals.

    Input Float32 values are exact dyadic rationals here. Bernstein convex hulls
    bound the whole domain, not samples or numerically recovered stationary roots.
    The domain includes rounded master-reference subtraction. A ten-operation
    Horner roundoff bound (also covering contracted FMA) and gradual/flush
    underflow allowance are added before outward Float32 emission.
    """
    values = [*coefficients, *domain, dependent_reference, master_reference]
    if len(coefficients) != 5 or len(domain) != 2 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("five finite polynomial coefficients and two bounds required")
    if domain[0] > domain[1] or type(tiles) is not int or not 1 <= tiles <= 4096:
        raise ValueError("invalid polynomial domain or subdivision count")
    # Require exact runtime values; silently rounding double inputs would hide
    # a mismatch between the measured source and the emitted NHEQ program.
    if any(struct.unpack("<f", struct.pack("<f", v))[0] != v for v in values):
        raise ValueError("polynomial inputs must be emitted Float32 values")
    a = [F(v) for v in coefficients]
    ref, origin = F(dependent_reference), F(master_reference)
    lower, upper = F(domain[0]) - origin, F(domain[1]) - origin
    magnitude = max(abs(lower), abs(upper))
    if magnitude > 1_000_000:
        raise ValueError("unsupported polynomial input magnitude")
    underflow = F(1, 1 << 126)
    subtraction_error = U * magnitude + underflow
    lower, upper = lower - subtraction_error, upper + subtraction_error
    minimum = maximum = None
    for tile in range(tiles):
        lo = lower + (upper - lower) * F(tile, tiles)
        hi = lower + (upper - lower) * F(tile + 1, tiles)
        span = hi - lo
        power = [sum((a[j] * math.comb(j, k) * lo ** (j-k) * span ** k
                      for j in range(k, 5)), F(0)) for k in range(5)]
        bernstein = [ref + sum((power[k] * F(math.comb(i, k), math.comb(4, k))
                               for k in range(i+1)), F(0)) for i in range(5)]
        low, high = min(bernstein), max(bernstein)
        minimum = low if minimum is None else min(minimum, low)
        maximum = high if maximum is None else max(maximum, high)
    magnitude = max(abs(lower), abs(upper))
    absolute_sum = abs(ref) + sum((abs(v) * magnitude ** i for i, v in enumerate(a)), F(0))
    gamma = 10 * U / (1 - 10 * U)
    roundoff = gamma * absolute_sum + 16 * underflow * (1 + magnitude) ** 4
    return {
        "range": [_outward32(minimum - roundoff, False),
                  _outward32(maximum + roundoff, True)],
        "exact_polynomial_hull": [float(minimum), float(maximum)],
        "float32_roundoff_allowance": float(roundoff),
        "master_subtraction_allowance": float(subtraction_error),
        "subintervals": tiles,
        "method": "exact_rational_piecewise_Bernstein_hull_plus_Float32_Horner_error_bound",
    }


def prepare_projected_joint_envelope(rigid: bytes, equalities: bytes, master_q_indices):
    """Return existing NHRIGID bytes plus an explicit changed-DOF receipt."""
    if len(rigid) < 224 or len(equalities) < 80:
        raise ValueError("truncated native input")
    header = struct.unpack_from("<8s10I32s", rigid)
    magic, abi, engine, sources, bodies, joints, nq, nv, r0, virtual, r1, digest = header
    if magic != b"NHRIGID2" or abi != 1 or engine != 5 or r0 or r1 or nq != nv+1:
        raise ValueError("unsupported rigid ABI")
    dofs_offset = 224 + bodies * 160 + joints * 144
    expected = dofs_offset + nv * 64 + (nq+nv) * 4 + sources * 32
    if len(rigid) != expected or struct.unpack_from("<8I", rigid, 80) != (
            5, bodies, 1, joints, 0, 0, nq, nv):
        raise ValueError("rigid dimensions or world header disagree")
    eh = struct.unpack_from("<8s10I32s", equalities)
    if eh[:4] != (b"NHEQ1\0\0\0", 1, nq, nv) or eh[5] != 96 or \
            eh[6] != eh[4] or any(eh[7:11]) or eh[-1] != digest or \
            len(equalities) != 80 + 96 * eh[4]:
        raise ValueError("requires matching legacy exact-projection NHEQ1 bytes")
    dofs = {}
    for v in range(nv):
        fields = struct.unpack_from("<8I8f", rigid, dofs_offset + 64*v)
        q, vi = fields[2:4]
        if vi != v or (v >= 6 and (q != v+1 or q in dofs)):
            raise ValueError("invalid native scalar DOF ownership")
        if v >= 6:
            if not all(math.isfinite(x) for x in fields[8:]):
                raise ValueError("nonfinite native DOF")
            dofs[q] = (v, fields)
    rows = [struct.unpack_from("<4I20f", equalities, 80+i*96) for i in range(eh[4])]
    dependents = {r[0] for r in rows}
    if len(dependents) != len(rows):
        raise ValueError("duplicate equality-dependent owner")
    selected = list(master_q_indices)
    if not selected or any(type(q) is not int for q in selected) or len(set(selected)) != len(selected):
        raise ValueError("unique explicit master q indices required")
    for q in selected:
        if q not in dofs or q in dependents:
            raise ValueError("selected master must be an independent scalar DOF")
        _, d = dofs[q]
        if not d[5] & POSITION_LIMIT or d[8] >= d[9]:
            raise ValueError("selected master needs a nonempty hard range")
    output = bytearray(rigid)
    receipts, touched, represented = [], set(), set()
    defaults_offset = dofs_offset + 64*nv
    defaults = struct.unpack_from(f"<{nq}f", rigid, defaults_offset)
    for row_index, row in enumerate(rows):
        dep_q, dep_v, master_q, master_v = row[:4]
        if master_q not in selected:
            continue
        represented.add(master_q)
        if dep_q not in dofs or dofs[dep_q][0] != dep_v or dofs[master_q][0] != master_v:
            raise ValueError("equality and rigid q/v ownership disagree")
        if dep_q in {r[2] for r in rows}:
            raise ValueError("chained equality requires composed-domain analysis")
        v, dependent = dofs[dep_q]
        if not dependent[5] & POSITION_LIMIT:
            # An already unbounded auxiliary coordinate needs no widening.
            continue
        if dependent[8] > dependent[9] or not dependent[8] <= defaults[dep_q] <= dependent[9]:
            raise ValueError("existing dependent range or default is invalid")
        master = dofs[master_q][1]
        dep_ref, master_ref, a0, a1, a2, a3, a4 = row[4:11]
        if any(row[i] != 0 for i in (11, 14, 15, 21, 22, 23)):
            raise ValueError("noncanonical equality padding")
        bound = polynomial_envelope([a0, a1, a2, a3, a4], master[8:10], dep_ref, master_ref)
        before = list(dependent[8:10])
        after = [min(before[0], bound["range"][0]), max(before[1], bound["range"][1])]
        if after == before:
            continue
        offset = dofs_offset + 64*v + 32
        struct.pack_into("<2f", output, offset, *after)
        touched.update(range(offset, offset+8))
        receipts.append({
            "equality_row": row_index, "dependent_q_index": dep_q,
            "dependent_v_index": dep_v, "master_q_index": master_q,
            "master_range_unchanged": list(master[8:10]),
            "original_hard_range": before, "derived_hard_range": after,
            "dependent_reference": dep_ref, "master_reference": master_ref,
            "polynomial_unchanged": [a0, a1, a2, a3, a4], **bound,
        })
    if represented != set(selected):
        raise ValueError("selected master has no equality-dependent rows")
    if any(a != b and i not in touched for i, (a,b) in enumerate(zip(rigid, output))):
        raise AssertionError("non-limit bytes changed")
    result = bytes(output)
    return result, {
        "status": "prepared_not_dynamic_qualification",
        "policy": "explicit_legacy_exact_projection_auxiliary_range_envelope",
        "source_archive_sha256": digest.hex(),
        "input_rigid_sha256": _sha(rigid), "output_rigid_sha256": _sha(result),
        "unchanged_equality_sha256": _sha(equalities),
        "selected_master_q_indices": selected, "changed_dof_count": len(receipts),
        "changed_dofs": receipts,
        "preserved": ["primary_joint_ranges", "source_polynomial_program", "default_q_v",
                      "body_mass_inertia", "attachments", "all_other_payload_bytes"],
        "limitation": "Compatibility of native hard ranges with its exact projection policy; "
                      "not source soft-limit dynamics, joint ROM, ligament or contact qualification.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--equalities", type=Path, required=True)
    parser.add_argument("--master-q", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse existing output directory")
    result, receipt = prepare_projected_joint_envelope(
        args.rigid.read_bytes(), args.equalities.read_bytes(), args.master_q)
    args.output.mkdir(parents=True)
    target = args.output / args.rigid.name
    target.write_bytes(result)
    receipt["input_paths"] = {"rigid": str(args.rigid.resolve()),
                            "equalities": str(args.equalities.resolve())}
    receipt["output_path"] = str(target.resolve())
    (args.output / "projected-joint-envelope.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"rigid": str(target), "sha256": _sha(result),
                      "changed_dof_count": receipt["changed_dof_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
