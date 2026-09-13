"""Compile a source-vessel mass-moment candidate without inventing lumen data.

The six BodyParts3D vessel surfaces already have source/world volume moments and
named MyoSim body links.  This module applies an explicitly supplied density to
those *surface-integral volumes* so the zeroth, first, and second mass moments
can be checked deterministically.  It is a source-surface mass proxy: it does
not reinterpret a surface as a lumen, assign a wall material, or couple blood
to tissue.  Those claims remain separate gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.vessel-mass-moment-owner-candidate.v1"
REGISTRATION = ROOT / "Docs/media/organ-vessel-registration-corrected-20260913/registration.json"
BODY_LINKS = ROOT / "Docs/media/organ-vessel-body-link-20260913/body-links.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("vessel mass moments: " + message)


def _finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} is not finite")
    return float(value)


def _vector(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 3, f"{label} must have three components")
    return [_finite(component, f"{label}[{index}]") for index, component in enumerate(value)]


def _matrix(value: Any, label: str) -> list[list[float]]:
    require(isinstance(value, list) and len(value) == 3, f"{label} must be 3x3")
    result = [_vector(row, f"{label}[{index}]") for index, row in enumerate(value)]
    return result


def _outer(vector: list[float]) -> list[list[float]]:
    return [[vector[i] * vector[j] for j in range(3)] for i in range(3)]


def _add_matrix(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def _scale_matrix(a: list[list[float]], scale: float) -> list[list[float]]:
    return [[scale * a[i][j] for j in range(3)] for i in range(3)]


def _matrix_sum(rows: list[list[list[float]]]) -> list[list[float]]:
    return [[math.fsum(row[i][j] for row in rows) for j in range(3)] for i in range(3)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    value = read_json(path)
    require(isinstance(value, dict), f"{label} is not an object")
    require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _check_identity(value: dict[str, Any], label: str, schema: str) -> None:
    require(value.get("schema") == schema, f"unsupported {label} schema")


def compile_candidate(
    *,
    registration: Path = REGISTRATION,
    body_links: Path = BODY_LINKS,
    density_kg_m3: float,
    density_provenance: str = "explicit_engineering_candidate_not_subject_calibrated",
    initial_velocity_mps: list[float] | None = None,
) -> dict[str, Any]:
    registration = Path(registration)
    body_links = Path(body_links)
    density = _finite(density_kg_m3, "density")
    require(density > 0.0, "density must be positive")
    require(isinstance(density_provenance, str) and density_provenance.strip(),
            "density provenance is required")
    velocity = [0.0, 0.0, 0.0] if initial_velocity_mps is None else _vector(initial_velocity_mps, "initial velocity")

    registration_doc, registration_sha = _read_canonical(registration, "vessel registration")
    links_doc, links_sha = _read_canonical(body_links, "vessel body links")
    _check_identity(registration_doc, "vessel registration", "HumanPack.organ-vessel-registration.v1")
    _check_identity(links_doc, "vessel body links", "HumanPack.organ-vessel-body-link-registration.v1")
    registration_rows = registration_doc.get("bindings")
    link_rows = links_doc.get("bindings")
    require(isinstance(registration_rows, list) and len(registration_rows) == 6,
            "registration must contain six vessels")
    require(isinstance(link_rows, list) and len(link_rows) == 6,
            "body links must contain six vessels")

    links_by_member: dict[str, dict[str, Any]] = {}
    for row in link_rows:
        require(isinstance(row, dict), "body link row is malformed")
        member_id = row.get("member_id")
        require(isinstance(member_id, str) and member_id not in links_by_member,
                "body links repeat or omit a member identity")
        require(row.get("body_link_registration") is True,
                f"body link is not registered: {member_id}")
        require(row.get("mechanical_mass_owner") is None,
                f"body link already owns mass: {member_id}")
        links_by_member[member_id] = row

    rows: list[dict[str, Any]] = []
    seen_members: set[str] = set()
    for source in sorted(registration_rows, key=lambda item: item.get("member_id", "")):
        require(isinstance(source, dict), "vessel registration row is malformed")
        member_id = source.get("member_id")
        require(isinstance(member_id, str) and member_id not in seen_members,
                "registration repeats or omits a member identity")
        seen_members.add(member_id)
        link = links_by_member.get(member_id)
        require(link is not None, f"missing body link for {member_id}")
        for key in ("source_name", "region_id", "myosim_body", "semantic_id"):
            require(source.get(key) == link.get(key), f"{key} disagrees for {member_id}")
        require(source.get("world_frame_registration") is True and
                source.get("tubular_field_registered") is False,
                f"unexpected vessel registration promotion: {member_id}")
        require(source.get("mechanical_mass_owner") is None,
                f"registration already owns mass: {member_id}")

        source_volume = _finite(source.get("source_surface_integral_volume_m3"),
                                f"source surface volume {member_id}")
        volume = _finite(source.get("registered_world_surface_integral_volume_m3"),
                         f"registered world surface volume {member_id}")
        centroid = _vector(source.get("registered_world_centroid_m"),
                           f"registered centroid {member_id}")
        central_volume = _matrix(source.get("registered_world_central_second_volume_moment_m5"),
                                 f"registered central moment {member_id}")
        require(source_volume > 0.0, f"source surface volume is not positive: {member_id}")
        require(volume > 0.0, f"registered world surface volume is not positive: {member_id}")
        mass = density * volume
        first = [mass * value for value in centroid]
        central_mass = _scale_matrix(central_volume, density)
        raw_mass = _add_matrix(central_mass, _scale_matrix(_outer(centroid), mass))
        momentum = [mass * value for value in velocity]
        rows.append({
            "member_id": member_id,
            "source_name": source["source_name"],
            "region_id": source["region_id"],
            "myosim_body": source["myosim_body"],
            "source_member_sha256": source["source_member_sha256"],
            "hydraulic_volume_owner_id": source["hydraulic_volume_owner_id"],
            "source_surface_integral_volume_m3": source_volume,
            "registered_world_surface_integral_volume_m3": volume,
            "density_kg_per_m3": density,
            "mass_kg": mass,
            "world_centroid_m": centroid,
            "first_mass_moment_kg_m": first,
            "central_second_mass_moment_kg_m2": central_mass,
            "raw_second_mass_moment_kg_m2": raw_mass,
            "initial_velocity_mps": velocity,
            "linear_momentum_kg_m_per_s": momentum,
            "mass_owner_count": 1,
            "lumen_or_tube_admitted": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_tissue_exchange": False,
            "subject_calibration": False,
        })

    require(len(rows) == len(links_by_member) == 6, "registration and body-link counts disagree")
    total_mass = math.fsum(row["mass_kg"] for row in rows)
    total_source_volume = math.fsum(row["source_surface_integral_volume_m3"] for row in rows)
    total_registered_world_volume = math.fsum(
        row["registered_world_surface_integral_volume_m3"] for row in rows
    )
    total_first = [math.fsum(row["first_mass_moment_kg_m"][i] for row in rows) for i in range(3)]
    total_momentum = [math.fsum(row["linear_momentum_kg_m_per_s"][i] for row in rows) for i in range(3)]
    total_raw_second = _matrix_sum([row["raw_second_mass_moment_kg_m2"] for row in rows])
    checkpoint = {
        "rows": rows,
        "total_mass_kg": total_mass,
        "total_first_mass_moment_kg_m": total_first,
        "total_raw_second_mass_moment_kg_m2": total_raw_second,
        "total_linear_momentum_kg_m_per_s": total_momentum,
    }
    checkpoint_bytes = canonical(checkpoint)
    restored = json.loads(checkpoint_bytes.decode("utf-8"))
    require(canonical(restored) == checkpoint_bytes, "checkpoint restore changed mass state")

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.vessel-mass-moments.1",
        "status": "partial",
        "source": {
            "registration": str(registration.relative_to(ROOT)) if registration.is_relative_to(ROOT) else str(registration),
            "registration_sha256": registration_sha,
            "body_links": str(body_links.relative_to(ROOT)) if body_links.is_relative_to(ROOT) else str(body_links),
            "body_links_sha256": links_sha,
            "vessel_count": len(rows),
            "subject": "one adult male source package",
        },
        "density": {
            "value_kg_per_m3": density,
            "provenance": density_provenance,
            "status": "candidate_not_subject_calibrated",
        },
        "owners": rows,
        "totals": {
            "source_surface_integral_volume_m3": total_source_volume,
            "registered_world_surface_integral_volume_m3": total_registered_world_volume,
            "mass_kg": total_mass,
            "first_mass_moment_kg_m": total_first,
            "raw_second_mass_moment_kg_m2": total_raw_second,
            "linear_momentum_kg_m_per_s": total_momentum,
            "owner_count": sum(row["mass_owner_count"] for row in rows),
            "unique_member_count": len(seen_members),
        },
        "qualification": {
            "zeroth_first_second_mass_moments": True,
            "momentum_at_authored_initial_velocity": True,
            "single_owner_per_registered_surface": True,
            "atomic_checkpoint_restore": True,
            "hydraulic_volume_storage_unchanged": True,
            "source_surface_is_lumen": False,
            "vessel_tube_or_lumen_mechanics": False,
            "pressure_gradient_momentum_transfer": False,
            "two_way_blood_tissue_transfer": False,
            "anatomical_blood_mass_owner": False,
            "material_density_calibrated": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "Six hash-bound source vessel surfaces receive an explicit density "
            "candidate and deterministic zeroth/first/second mass moments. "
            "The values are a surface-integral mass proxy, not a lumen or wall "
            "volume; no pressure reaction, tissue exchange, or subject claim is "
            "promoted."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    try:
        result = compile_candidate(
            registration=arguments.registration or REGISTRATION,
            body_links=arguments.body_links or BODY_LINKS,
            density_kg_m3=arguments.density_kg_m3,
            density_provenance=arguments.density_provenance,
            initial_velocity_mps=arguments.initial_velocity_mps,
        )
        output = arguments.output.resolve()
        digest = _immutable_write(output, result)
        print(json.dumps({
            "schema": SCHEMA,
            "output": str(output),
            "sha256": digest,
            "status": result["status"],
            "vessels": result["totals"]["unique_member_count"],
            "mass_kg": result["totals"]["mass_kg"],
            "owner_count": result["totals"]["owner_count"],
            "anatomical_blood_mass_owner": False,
        }, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"vessel mass moments: {error}")
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=REGISTRATION)
    parser.add_argument("--body-links", type=Path, default=BODY_LINKS)
    parser.add_argument("--density-kg-m3", type=float, required=True)
    parser.add_argument("--density-provenance", default="explicit_engineering_candidate_not_subject_calibrated")
    parser.add_argument("--initial-velocity-mps", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--output", type=Path, required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
