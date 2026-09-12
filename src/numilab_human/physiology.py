"""Source-bound physiology authoring and deterministic native index compilation.

This module does not evolve physical state. Anatomy membership is provenance,
not a volume measurement; executable parameters have independent provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError

SCHEMA = "HumanPack.physiology.v1"
NATIVE_SCHEMA = "HumanPack.physiology-native.v1"
ROOT = Path(__file__).resolve().parents[2]
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError) as error:
        raise HumanImportError("physiology requires finite JSON values") from error


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: Path) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise HumanImportError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        result = json.loads(path.read_text(), object_pairs_hook=pairs)
    except HumanImportError:
        raise
    except (ValueError, UnicodeError) as error:
        raise HumanImportError(f"invalid physiology JSON: {path.name}") from error
    canonical(result)
    if not isinstance(result, dict):
        raise HumanImportError("physiology document must be an object")
    return result


def _keys(value: Any, required: set[str], context: str, optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - required - (optional or set()):
        raise HumanImportError(f"invalid {context} fields")
    return value


def _id(value: Any, context: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise HumanImportError(f"invalid {context} identifier")
    return value


def _rows(value: Any, context: str, *, empty: bool = False) -> dict[str, dict]:
    if not isinstance(value, list) or (not value and not empty) or len(value) > 4096:
        raise HumanImportError(f"invalid {context} table")
    result = {}
    for row in value:
        if not isinstance(row, dict):
            raise HumanImportError(f"invalid {context} record")
        key = _id(row.get("id"), context)
        if key in result:
            raise HumanImportError(f"duplicate {context} ID: {key}")
        result[key] = row
    return result


def load_anatomy(sources: Path, source_lock: Path = ROOT / "sources.lock.json") -> dict:
    """Verify both source relationship tables against the pinned acquisition lock."""
    lock = read_json(source_lock)
    metadata = lock.get("sources", {}).get("bodyparts3d_4")
    if not isinstance(metadata, dict) or metadata.get("version") != "4.0":
        raise HumanImportError("missing pinned BodyParts3D 4.0 source")
    tables, records = {}, []
    for hierarchy, filename in (("part_of", "partof_element_parts.txt"), ("is_a", "isa_element_parts.txt")):
        expected = metadata.get("files", {}).get(filename, {}).get("sha256")
        path = sources / filename
        if not path.is_file():
            raise HumanImportError(f"missing anatomy source: {filename}")
        data = path.read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if not isinstance(expected, str) or actual != expected:
            raise HumanImportError(f"anatomy source hash mismatch: {filename}")
        relationships: dict[tuple[str, str], set[str]] = {}
        for line in data.decode("utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 3:
                raise HumanImportError(f"malformed anatomy source: {filename}")
            if fields[0] == "concept id":
                continue
            relationships.setdefault((fields[0], fields[1]), set()).add(fields[2])
        tables[hierarchy] = relationships
        records.append({"hierarchy": hierarchy, "file": filename, "sha256": actual})
    source = {"id": "bodyparts3d_4", "version": metadata["version"],
              "license": metadata["license"], "url": metadata["base_url"],
              "attribution": metadata["attribution"], "tables": records}
    return {"source": source, "tables": tables}


def _parameter(value: Any, unit: str, context: str, graph: dict, source_records: dict,
               *, positive: bool = False, nonnegative: bool = False) -> float | None:
    _keys(value, {"value", "unit", "provenance", "uncertainty"}, context)
    if value["unit"] != unit:
        raise HumanImportError(f"{context} must use SI unit {unit}")
    provenance = value["provenance"]
    if not isinstance(provenance, dict):
        raise HumanImportError(f"missing {context} provenance")
    kind = provenance.get("kind")
    if kind == "synthetic_fixture":
        _keys(provenance, {"kind", "description"}, context + " provenance")
        if graph["qualification"] != "fixture_only" or not isinstance(provenance["description"], str) or not provenance["description"].strip():
            raise HumanImportError("synthetic parameters require fixture_only qualification")
    elif kind == "source":
        _keys(provenance, {"kind", "source_id", "record_id"}, context + " provenance")
        if provenance["source_id"] not in source_records or not isinstance(provenance["record_id"], str) or not provenance["record_id"].strip():
            raise HumanImportError(f"unresolved {context} parameter source")
    elif kind == "unresolved":
        _keys(provenance, {"kind", "description"}, context + " provenance")
        if value["value"] is not None or graph["qualification"] != "uncalibrated" or not isinstance(provenance["description"], str) or not provenance["description"].strip():
            raise HumanImportError("unresolved parameter requires null value and uncalibrated qualification")
    else:
        raise HumanImportError(f"unsupported {context} provenance")
    number = value["value"]
    if number is not None and (type(number) not in (int, float) or not math.isfinite(number)):
        raise HumanImportError(f"nonfinite or nonnumeric {context}")
    uncertainty = value["uncertainty"]
    if not isinstance(uncertainty, dict) or uncertainty.get("status") not in {"unknown", "bounded", "exact_fixture"}:
        raise HumanImportError(f"missing {context} uncertainty")
    if uncertainty["status"] == "bounded":
        _keys(uncertainty, {"status", "lower", "upper", "unit"}, context + " uncertainty")
        lower, upper = uncertainty["lower"], uncertainty["upper"]
        if (uncertainty["unit"] != unit or type(lower) not in (int, float) or type(upper) not in (int, float)
                or not math.isfinite(lower) or not math.isfinite(upper) or lower > upper
                or value["value"] is None or not lower <= value["value"] <= upper):
            raise HumanImportError(f"invalid {context} uncertainty interval")
    else:
        _keys(uncertainty, {"status"}, context + " uncertainty")
        if uncertainty["status"] == "exact_fixture" and kind != "synthetic_fixture":
            raise HumanImportError("exact_fixture uncertainty requires synthetic provenance")
    number = value["value"]
    if number is None:
        if kind != "unresolved" or uncertainty["status"] != "unknown":
            raise HumanImportError(f"missing {context} value")
        return None
    if type(number) not in (int, float) or not math.isfinite(number):
        raise HumanImportError(f"nonfinite or nonnumeric {context}")
    if (positive and number <= 0) or (nonnegative and number < 0):
        raise HumanImportError(f"invalid {context} sign")
    return float(number)


def _validate_graph(graph: dict, *, sources: Path, source_lock: Path) -> dict:
    """Resolve exact source anatomy and validate a closed passive network, without stepping."""
    canonical(graph)
    _keys(graph, {"schema", "id", "qualification", "law", "anatomy_source", "regions", "parameter_sources",
                  "species", "compartments", "connections", "tissue_reservoirs", "exchanges", "residual_tolerance"}, "physiology graph")
    if graph["schema"] != SCHEMA or graph["law"] != "closed_linear_compliance_transport_v1":
        raise HumanImportError("unsupported physiology schema or law")
    _id(graph["id"], "graph")
    if graph["qualification"] not in {"fixture_only", "uncalibrated"}:
        raise HumanImportError("physiology authoring cannot promote biological qualification")
    anatomy = load_anatomy(sources, source_lock)
    if graph["anatomy_source"] != anatomy["source"]:
        raise HumanImportError("forged or stale anatomy source reference")
    regions = _rows(graph["regions"], "region")
    all_members: dict[str, list[str]] = {}
    for key, region in regions.items():
        _keys(region, {"id", "semantic_id", "concept_id", "source_name", "hierarchy", "member_ids", "selection"}, "region")
        if not isinstance(region["concept_id"], str) or not re.fullmatch(r"FMA[0-9]+", region["concept_id"]) or region["semantic_id"] != "FMA:" + region["concept_id"][3:]:
            raise HumanImportError("region semantic ID disagrees with source concept")
        members = region["member_ids"]
        if not isinstance(members, list) or not members or any(not isinstance(m, str) for m in members) or len(set(members)) != len(members):
            raise HumanImportError("invalid or duplicate anatomy member IDs")
        source_members = anatomy["tables"].get(region["hierarchy"], {}).get((region["concept_id"], region["source_name"]))
        if source_members is None or not set(members) <= source_members:
            raise HumanImportError("missing or forged source membership")
        if region["selection"] not in {"complete_membership", "explicit_subset"} or (region["selection"] == "complete_membership" and set(members) != source_members):
            raise HumanImportError("incomplete declared anatomy membership")
        for member in members:
            all_members.setdefault(member, []).append(key)
    parameter_sources = _rows(graph["parameter_sources"], "parameter source", empty=True)
    for source in parameter_sources.values():
        _keys(source, {"id", "url", "revision", "sha256", "license", "allowed_use"}, "parameter source")
        if (not all(isinstance(source[k], str) and source[k].strip() for k in source)
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) or not source["url"].startswith("https://")):
            raise HumanImportError("invalid parameter source provenance")
    residual_tolerance = _parameter(graph["residual_tolerance"], "1", "residual_tolerance", graph, parameter_sources, positive=True)
    species = _rows(graph["species"], "species")
    for row in species.values():
        _keys(row, {"id", "description", "amount_scale_mol"}, "species")
        if not isinstance(row["description"], str) or not row["description"].strip():
            raise HumanImportError("missing species description")
    compartments = _rows(graph["compartments"], "compartment")
    connections = _rows(graph["connections"], "connection")
    reservoirs = _rows(graph["tissue_reservoirs"], "tissue reservoir", empty=True)
    exchanges = _rows(graph["exchanges"], "exchange", empty=True)
    if len(compartments) < 2:
        raise HumanImportError("closed hydraulic network requires at least two compartments")
    owners, used_regions, numeric, missing = set(), set(), {}, []

    def number(value: Any, unit: str, context: str, **kwargs: Any) -> float | None:
        result = _parameter(value, unit, context, graph, parameter_sources, **kwargs)
        if result is None:
            missing.append(context)
        return result

    for table_name, table in (("compartments", compartments), ("tissue_reservoirs", reservoirs)):
        numeric[table_name] = []
        for key, row in sorted(table.items()):
            parameters = {"reference_volume_m3", "reference_pressure_pa", "external_pressure_pa", "compliance_m3_per_pa", "initial_volume_m3", "volume_scale_m3"} if table_name == "compartments" else {"volume_m3"}
            _keys(row, {"id", "anatomical_region_id", "physical_volume_owner_id", "initial_species_mol"} | parameters, table_name)
            if not isinstance(row["anatomical_region_id"], str) or row["anatomical_region_id"] not in regions:
                raise HumanImportError("unresolved anatomical region")
            used_regions.add(row["anatomical_region_id"])
            owner = _id(row["physical_volume_owner_id"], "physical volume owner")
            if owner in owners:
                raise HumanImportError("duplicate physical volume owner")
            owners.add(owner)
            initial = row["initial_species_mol"]
            if not isinstance(initial, dict) or initial.keys() != species.keys():
                raise HumanImportError("initial amount must cover every species exactly")
            out = {k: row[k] for k in ("id", "anatomical_region_id", "physical_volume_owner_id")}
            out["anatomical_region_id"] = regions[row["anatomical_region_id"]]["semantic_id"]
            out["initial_species_mol"] = [number(initial[s], "mol", key + "." + s, nonnegative=True) for s in sorted(species)]
            for name in sorted(parameters):
                unit = "Pa" if name in {"reference_pressure_pa", "external_pressure_pa"} else "m3/Pa" if name == "compliance_m3_per_pa" else "m3"
                out[name] = number(row[name], unit, key + "." + name, positive=name not in {"reference_pressure_pa", "external_pressure_pa"})
            if table_name == "compartments":
                out["volume_residual_tolerance"] = residual_tolerance
            numeric[table_name].append(out)
    indices = {key: i for i, key in enumerate(sorted(compartments))}
    reservoir_indices = {key: i for i, key in enumerate(sorted(reservoirs))}
    species_indices = {key: i for i, key in enumerate(sorted(species))}
    adjacency = {key: set() for key in compartments}
    pairs = set()
    numeric["connections"] = []
    for key, row in sorted(connections.items()):
        _keys(row, {"id", "from", "to", "resistance_pa_s_per_m3", "initial_flow_m3_per_s", "flow_scale_m3_per_s", "pressure_scale_pa"}, "connection", {"inertance_pa_s2_per_m3"})
        a, b = row["from"], row["to"]
        if not isinstance(a, str) or not isinstance(b, str) or a not in compartments or b not in compartments or a == b:
            raise HumanImportError("connection requires two distinct conservation endpoints")
        pair = tuple(sorted((a, b)))
        if pair in pairs:
            raise HumanImportError("duplicate hydraulic connection")
        pairs.add(pair)
        adjacency[a].add(b)
        adjacency[b].add(a)
        numeric["connections"].append({"id": key, "from": indices[a], "to": indices[b],
            "resistance_pa_s_per_m3": number(row["resistance_pa_s_per_m3"], "Pa*s/m3", key + ".resistance", positive=True),
            "inertance_pa_s2_per_m3": number(row["inertance_pa_s2_per_m3"], "Pa*s2/m3", key + ".inertance", nonnegative=True) if "inertance_pa_s2_per_m3" in row else 0.0,
            "initial_flow_m3_per_s": number(row["initial_flow_m3_per_s"], "m3/s", key + ".initial_flow"),
            "flow_scale_m3_per_s": number(row["flow_scale_m3_per_s"], "m3/s", key + ".flow_scale", positive=True),
            "pressure_scale_pa": number(row["pressure_scale_pa"], "Pa", key + ".pressure_scale", positive=True),
            "flow_residual_tolerance": residual_tolerance})
    seen, frontier = set(), [next(iter(compartments))]
    while frontier:
        node = frontier.pop()
        if node not in seen:
            seen.add(node)
            frontier.extend(adjacency[node] - seen)
    if seen != compartments.keys():
        raise HumanImportError("disconnected hydraulic topology")
    numeric["exchanges"], exchange_keys, used_reservoirs = [], set(), set()
    for key, row in sorted(exchanges.items()):
        _keys(row, {"id", "compartment", "tissue_reservoir", "species", "clearance_m3_per_s"}, "exchange", {"partition_coefficient"})
        a, b, s = row["compartment"], row["tissue_reservoir"], row["species"]
        if not all(isinstance(v, str) for v in (a, b, s)) or a not in compartments or b not in reservoirs or s not in species:
            raise HumanImportError("exchange requires blood, tissue and species conservation endpoints")
        if (a, b, s) in exchange_keys:
            raise HumanImportError("duplicate exchange endpoints")
        exchange_keys.add((a, b, s))
        used_reservoirs.add(b)
        numeric["exchanges"].append({"id": key, "compartment": indices[a], "tissue_reservoir": reservoir_indices[b], "species": species_indices[s],
            "clearance_m3_per_s": number(row["clearance_m3_per_s"], "m3/s", key, positive=True),
            "partition_coefficient": number(row["partition_coefficient"], "1", key + ".partition", positive=True) if "partition_coefficient" in row else 1.0})
    if used_reservoirs != reservoirs.keys():
        raise HumanImportError("tissue reservoir has no conservative exchange")
    if used_regions != regions.keys():
        raise HumanImportError("anatomical region has no physiological owner")
    native_species = [{"id": key, "description": species[key]["description"],
                       "amount_scale_mol": number(species[key]["amount_scale_mol"], "mol", key + ".amount_scale", positive=True),
                       "amount_residual_tolerance": residual_tolerance} for key in sorted(species)]
    next_identifier = 1
    for rows in (native_species, numeric["compartments"], numeric["connections"], numeric["tissue_reservoirs"], numeric["exchanges"]):
        for row in rows:
            row["stable_identifier"] = next_identifier
            next_identifier += 1
    for row in numeric["connections"]:
        row["from"] = numeric["compartments"][row["from"]]["stable_identifier"]
        row["to"] = numeric["compartments"][row["to"]]["stable_identifier"]
    for row in numeric["exchanges"]:
        row["compartment"] = numeric["compartments"][row["compartment"]]["stable_identifier"]
        row["tissue_reservoir"] = numeric["tissue_reservoirs"][row["tissue_reservoir"]]["stable_identifier"]
        row["species"] = native_species[row["species"]]["stable_identifier"]
    if residual_tolerance is None:
        missing.append("residual_tolerance")
    normalized = dict(graph)
    for name in ("regions", "parameter_sources", "species", "compartments", "connections", "tissue_reservoirs", "exchanges"):
        normalized[name] = sorted(graph[name], key=lambda row: row["id"])
    normalized["regions"] = [dict(row, member_ids=sorted(row["member_ids"])) for row in normalized["regions"]]
    return {"authored_graph_sha256": digest(normalized), "source_graph_sha256": digest({"source": anatomy["source"], "regions": normalized["regions"]}),
            "missing_parameters": sorted(missing), "calibration_required": graph["qualification"] == "uncalibrated",
            "overlapping_source_memberships": [{"member_id": member, "region_ids": sorted(ids)} for member, ids in sorted(all_members.items()) if len(ids) > 1],
            "native": {"schema": NATIVE_SCHEMA, "model_id": graph["id"], "qualification": graph["qualification"],
                       "law": graph["law"], "residual_tolerance": residual_tolerance, "species": native_species, **numeric}}


def validate_graph(graph: dict, *, sources: Path, source_lock: Path = ROOT / "sources.lock.json") -> dict:
    try:
        return _validate_graph(graph, sources=sources, source_lock=source_lock)
    except HumanImportError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise HumanImportError(f"malformed physiology authoring: {error}") from error


def compile_graph(graph: dict, *, sources: Path, source_lock: Path = ROOT / "sources.lock.json") -> dict:
    result = validate_graph(graph, sources=sources, source_lock=source_lock)
    if result["missing_parameters"]:
        raise HumanImportError("native compilation requires resolved parameters: " + ", ".join(result["missing_parameters"]))
    return {**result["native"], "authored_graph_sha256": result["authored_graph_sha256"],
            "source_graph_sha256": result["source_graph_sha256"]}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.set_defaults(handler=_run)


def _run(args: argparse.Namespace) -> int:
    graph = read_json(args.graph)
    result = validate_graph(graph, sources=args.sources, source_lock=args.source_lock)
    if not args.validate_only:
        result = compile_graph(graph, sources=args.sources, source_lock=args.source_lock)
    encoded = canonical(result) + b"\n"
    if args.output.exists() and args.output.read_bytes() != encoded:
        raise HumanImportError("physiology output is immutable; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists():
        with args.output.open("xb") as stream:
            stream.write(encoded)
    print(json.dumps({"schema": result.get("schema", "HumanPack.physiology-validation.v1"),
                      "output": str(args.output), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "qualification": graph["qualification"]}, indent=2))
    return 0
