"""Compile the pinned Shi/Hose CellML hydraulic model; never step physics.

This source-specific adapter parses the curated files directly. It shares no
implementation with the independent CellML trajectory oracle or native solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

from .model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/physiome/shi_hose_2009"
CONFIG = ROOT / "config/shi-hose-cardiac-source.v1.json"
SOURCE_LOCK_SHA256 = "3c1439ede07f5736520856f77a8cef11103e63471fa4ee264b24c8a04a90a39e"
REVISION = "a679cdc2e97429fb5280af8132c119758626c1f2"
CELL = "{http://www.cellml.org/cellml/1.1#}"
MATH = "{http://www.w3.org/1998/Math/MathML}"
HREF = "{http://www.w3.org/1999/xlink}href"
LAW = "closed_periodic_elastance_orifice_v2"
PHYSICAL = {"TempCDa.cellml", "TempCDv.cellml", "TempR.cellml", "TempRC.cellml", "TempRLC.cellml"}
FILES = {"Units.cellml", "ModelMain.cellml", "ModelHeart.cellml", "ModelSys.cellml", "ModelPul.cellml",
         "ParaHeart.cellml", "ParaSys.cellml", "ParaPul.cellml", "EAtrium.cellml", "EVentricle.cellml", *PHYSICAL}


def require(condition: Any, message: str) -> None:
    if not condition:
        raise HumanImportError(message)


def canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise HumanImportError("source authoring requires finite JSON") from error


def read_json(path: Path) -> dict:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, "duplicate source/configuration JSON field")
            result[key] = value
        return result
    try:
        result = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    except HumanImportError:
        raise
    except (ValueError, OSError) as error:
        raise HumanImportError(f"invalid source/configuration file: {path.name}") from error
    require(isinstance(result, dict), "source/configuration must be an object")
    canonical(result)
    return result


def _finite(text: str, context: str) -> float:
    try:
        number = float(text)
    except (ValueError, TypeError) as error:
        raise HumanImportError(f"invalid source number: {context}") from error
    require(math.isfinite(number), f"nonfinite source number: {context}")
    return number


def _load_source(directory: Path) -> tuple[dict, dict[str, ET.Element]]:
    lock_path = directory / "source-lock.json"
    require(lock_path.is_file() and hashlib.sha256(lock_path.read_bytes()).hexdigest() == SOURCE_LOCK_SHA256,
            "source manifest differs from the curated revision")
    lock = read_json(lock_path)
    require(lock.get("revision") == REVISION and lock.get("license") == "CC-BY-3.0", "source revision or license mismatch")
    records = lock.get("files")
    require(isinstance(records, list) and len(records) == 15 and {row["path"] for row in records} == FILES,
            "source must retain all fifteen exact CellML files")
    models = {}
    for record in records:
        name = record["path"]
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"missing/redirected curated source: {name}")
        data = path.read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"],
                f"curated source hash mismatch: {name}")
        require(b"<!DOCTYPE" not in data and b"<!ENTITY" not in data, "external XML declarations are unsupported")
        root = ET.fromstring(data)
        require(root.tag == CELL + "model", "source must retain CellML 1.1 namespace")
        for imported in root.findall(CELL + "import"):
            require(imported.get(HREF) in FILES, "unresolved or external CellML import")
        models[name] = root
    return lock, models


def _units(root: ET.Element) -> dict[str, tuple[float, tuple[float, float, float]]]:
    """Resolve source unit products into SI multiplier and mass/length/time powers."""
    definitions = {row.attrib["name"]: row for row in root.findall(CELL + "units")}
    require(len(definitions) == len(root.findall(CELL + "units")), "duplicate source unit definition")
    resolved = {"pascal": (1.0, (1.0, -1.0, -2.0)), "litre": (0.001, (0.0, 3.0, 0.0)),
                "second": (1.0, (0.0, 0.0, 1.0)), "dimensionless": (1.0, (0.0, 0.0, 0.0))}

    def resolve(name: str, active: set[str]) -> tuple[float, tuple[float, float, float]]:
        if name in resolved:
            return resolved[name]
        require(name in definitions and name not in active, "unresolved or cyclic source units")
        scale, dimensions = 1.0, [0.0, 0.0, 0.0]
        for unit in definitions[name]:
            require(unit.tag == CELL + "unit" and set(unit.attrib) <= {"units", "exponent", "multiplier"},
                    "unsupported source unit conversion")
            factor, powers = resolve(unit.attrib["units"], active | {name})
            exponent = _finite(unit.get("exponent", "1"), name)
            multiplier = _finite(unit.get("multiplier", "1"), name)
            require(multiplier > 0, "source unit multiplier must be positive")
            scale *= (factor * multiplier) ** exponent
            dimensions = [before + exponent * power for before, power in zip(dimensions, powers, strict=True)]
        require(math.isfinite(scale) and scale > 0, "invalid SI source multiplier")
        resolved[name] = (scale, tuple(dimensions))
        return resolved[name]

    for name in definitions:
        resolve(name, set())
    return resolved


def _connections(model: ET.Element) -> list[dict]:
    result = []
    for row in model.findall(CELL + "connection"):
        components = row.find(CELL + "map_components")
        require(components is not None, "CellML connection has no components")
        result.append({"a": components.attrib["component_1"], "b": components.attrib["component_2"],
                       "variables": [(item.attrib["variable_1"], item.attrib["variable_2"])
                                     for item in row.findall(CELL + "map_variables")]})
    return result


def _numerical(config: dict) -> dict[str, float]:
    require(set(config) == {"schema", "id", "source_manifest_sha256", "numerical_settings", "species", "tissue_reservoirs", "exchanges"},
            "source configuration contains unsupported fields or physiological overrides")
    require(config["schema"] == "HumanPack.shi-hose-source-config.v1" and config["id"] == "shi_hose_2009_curated_closed_loop",
            "unsupported source-model configuration")
    require(config["source_manifest_sha256"] == SOURCE_LOCK_SHA256, "configuration has a stale source manifest")
    require(all(config[key] == [] for key in ("species", "tissue_reservoirs", "exchanges")),
            "transport requires sourced absolute vascular volumes; hydraulic storage is not dilution volume")
    units = {"volume_scale_m3": "m3", "flow_scale_m3_per_s": "m3/s", "pressure_scale_pa": "Pa", "residual_tolerance": "1"}
    settings = config["numerical_settings"]
    require(isinstance(settings, dict) and settings.keys() == units.keys(), "missing numerical residual settings")
    result = {}
    for name, unit in units.items():
        record = settings[name]
        require(isinstance(record, dict) and set(record) == {"value", "unit", "role"} and record["unit"] == unit and
                record["role"] == "numerical_residual_control_not_source_physiological_parameter", "numerical setting provenance/units mismatch")
        require(type(record["value"]) in (int, float) and math.isfinite(record["value"]) and record["value"] > 0,
                "numerical settings must be finite positive scalars")
        result[name] = float(record["value"])
    require(result["residual_tolerance"] < 1, "dimensionless residual tolerance must be below one")
    return result


def compile_source(*, directory: Path = SOURCE, config: dict | None = None) -> tuple[dict, dict]:
    """Lower exact source identities and parameters; evaluate only algebraic t=0 initialization."""
    try:
        return _compile_source(directory=directory, config=read_json(CONFIG) if config is None else config)
    except HumanImportError:
        raise
    except (KeyError, TypeError, ValueError, OSError, ET.ParseError) as error:
        raise HumanImportError(f"invalid source-model compilation: {error}") from error


def _compile_source(*, directory: Path, config: dict) -> tuple[dict, dict]:
    canonical(config)
    numerical = _numerical(config)
    source, models = _load_source(directory)
    units = _units(models["Units.cellml"])
    imports, wiring = {}, {}
    for file, model in models.items():
        imports[file] = {}
        for imported in model.findall(CELL + "import"):
            for component in imported.findall(CELL + "component"):
                alias = component.attrib["name"]
                require(alias not in imports[file], "duplicate imported component")
                imports[file][alias] = (imported.attrib[HREF], component.attrib["component_ref"])
        wiring[file] = _connections(model)
    parameters = {}
    for file in ("ParaHeart.cellml", "ParaSys.cellml", "ParaPul.cellml"):
        for component in models[file].findall(CELL + "component"):
            for variable in component.findall(CELL + "variable"):
                name, unit = variable.attrib["name"], variable.attrib["units"]
                require("initial_value" in variable.attrib and unit in units, "source parameter is unresolved")
                key = (file, component.attrib["name"], name)
                require(key not in parameters, "duplicate source parameter")
                lexical = variable.attrib["initial_value"]
                parameters[key] = {"file": file, "component": component.attrib["name"], "symbol": name,
                                   "source_value": lexical, "source_unit": unit,
                                   "si_value": _finite(lexical, name) * units[unit][0],
                                   "si_dimensions_mass_length_time": list(units[unit][1])}
    require(len(parameters) == 59, "source parameter coverage must remain all 59 records")
    physical = {}
    for file in ("ModelHeart.cellml", "ModelSys.cellml", "ModelPul.cellml"):
        for alias, (imported, component) in imports[file].items():
            if imported in PHYSICAL:
                require(alias not in physical, "duplicate global source physical component")
                physical[alias] = {"model": file, "template": imported, "component": component}
    require(len(physical) == 14, "source physical component coverage must remain fourteen")
    bindings, used = {}, set()

    def parameter(file: str, alias: str, variable: str) -> float:
        matches = []
        for link in wiring[file]:
            for a, b in link["variables"]:
                if link["b"] == alias and b == variable:
                    matches.append((link["a"], a))
                elif link["a"] == alias and a == variable:
                    matches.append((link["b"], b))
        keys = []
        for peer, symbol in matches:
            if peer in imports[file]:
                owner_file, owner_component = imports[file][peer]
                key = (owner_file, owner_component, symbol)
                if key in parameters:
                    keys.append(key)
        require(len(keys) == 1, f"ambiguous or missing source parameter binding: {file}/{alias}/{variable}")
        key = keys[0]
        used.add(key)
        bindings[f"{file}/{alias}/{variable}"] = "/".join(key)
        return parameters[key]["si_value"]

    def endpoint(file: str, alias: str, variable: str, active: set[tuple[str, str, str]]) -> tuple[str, str]:
        state = (file, alias, variable)
        require(state not in active, "cyclic source wrapper connection")
        if alias in imports[file]:
            owner_file, owner_component = imports[file][alias]
            if owner_file in PHYSICAL:
                return alias, variable
            return endpoint(owner_file, owner_component, variable, active | {state})
        peers = []
        for link in wiring[file]:
            for a, b in link["variables"]:
                if link["a"] == alias and a == variable and link["b"] in imports[file]:
                    peers.append((link["b"], b))
                elif link["b"] == alias and b == variable and link["a"] in imports[file]:
                    peers.append((link["a"], a))
        require(len(peers) == 1, f"ambiguous wrapper port: {file}/{alias}/{variable}")
        return endpoint(file, *peers[0], active | {state})

    adjacency, source_edges = {}, []
    for file in ("ModelMain.cellml", "ModelHeart.cellml", "ModelSys.cellml", "ModelPul.cellml"):
        for link in wiring[file]:
            if ("Po", "Pi") not in link["variables"]:
                continue
            require(("Qo", "Qi") in link["variables"], "source pressure connection lacks conservative flow connection")
            a, pa = endpoint(file, link["a"], "Po", set())
            b, pb = endpoint(file, link["b"], "Pi", set())
            aq, qa = endpoint(file, link["a"], "Qo", set())
            bq, qb = endpoint(file, link["b"], "Qi", set())
            require((pa, pb, qa, qb) == ("Po", "Pi", "Qo", "Qi") and a == aq and b == bq and a != b,
                    "source pressure and flow endpoints disagree")
            require(a not in adjacency, "source hydraulic outlet has multiple owners")
            adjacency[a] = b
            source_edges.append({"file": file, "source_component": link["a"], "target_component": link["b"], "from": a, "to": b})
    require(adjacency.keys() == physical.keys() and set(adjacency.values()) == physical.keys(), "source closed-loop connectivity is incomplete")
    seen, cursor = set(), sorted(physical)[0]
    while cursor not in seen:
        seen.add(cursor)
        cursor = adjacency[cursor]
    require(seen == physical.keys(), "source topology has disconnected circulation loops")
    source_pi = {}
    for file in ("EAtrium.cellml", "EVentricle.cellml"):
        pi_nodes = [node for node in models[file].iter(MATH + "cn") if (node.text or "").strip() == "3.14159"]
        require(pi_nodes, "source periodic law literal pi was lost")
        source_pi[file] = _finite(pi_nodes[0].text.strip(), "source_pi")
    require(len(set(source_pi.values())) == 1, "source periodic laws disagree on literal pi")
    source_pi_value = next(iter(source_pi.values()))
    compartments, initial_pressure, connections, derivations = [], {}, [], []
    for alias, owner in sorted(physical.items()):
        file, template = owner["model"], owner["template"]
        if template == "TempR.cellml":
            continue
        row = {"id": alias, "anatomical_region_id": f"CellML:shi_hose_2009:{file[:-7]}:{alias}",
               "physical_volume_owner_id": f"source_storage:shi_hose_2009:{alias}", "initial_species_mol": [],
               "volume_scale_m3": numerical["volume_scale_m3"], "volume_residual_tolerance": numerical["residual_tolerance"],
               "external_pressure_pa": 0.0}
        if template in {"TempCDa.cellml", "TempCDv.cellml"}:
            matches = []
            for link in wiring[file]:
                if ("E", "E") in link["variables"] and alias in (link["a"], link["b"]):
                    matches.append(link["b"] if alias == link["a"] else link["a"])
            require(len(matches) == 1 and matches[0] in imports[file], "source chamber has no unique elastance owner")
            elastance = matches[0]
            law_file = imports[file][elastance][0]
            atrial = template == "TempCDa.cellml"
            require(law_file == ("EAtrium.cellml" if atrial else "EVentricle.cellml"), "source chamber elastance law mismatch")
            emin, emax = parameter(file, elastance, "Emin"), parameter(file, elastance, "Emax")
            period = parameter(file, elastance, "T")
            start = parameter(file, elastance, "Tpwb" if atrial else "Ts1")
            end = parameter(file, elastance, "Tpww" if atrial else "Ts2")
            reference_volume, reference_pressure = parameter(file, alias, "Vini"), parameter(file, alias, "Pini")
            initial_volume = parameter(file, alias, "V0")
            row.update(storage_kind="absolute_volume", pressure_law="atrial_elastance" if atrial else "ventricular_elastance",
                       reference_volume_m3=reference_volume, reference_pressure_pa=reference_pressure, compliance_m3_per_pa=0.0,
                       initial_volume_m3=initial_volume, elastance_min_pa_per_m3=emin, elastance_max_pa_per_m3=emax,
                       period_seconds=period, activation_start=start, activation_end=end, source_pi=source_pi_value)
            # Algebraic source initial-condition evaluation only. There is no
            # timestep, integration loop, candidate state or runtime authority.
            activation_at_zero = (1 - math.cos(2 * source_pi_value * (1 - start) / end)) if atrial and start + end >= 1 else 0.0
            initial_elastance = emin + activation_at_zero * (emax - emin) / 2
            initial_pressure[alias] = reference_pressure + initial_elastance * (initial_volume - reference_volume)
            derivations.append({"id": alias, "initial_pressure": "Pini + E(source_t=0) * (V0 - Vini)",
                                "source_template": template, "source_activation": law_file})
        else:
            compliance, pressure = parameter(file, alias, "C"), parameter(file, alias, "P0")
            row.update(storage_kind="storage_displacement", pressure_law="linear_compliance", reference_volume_m3=0.0,
                       reference_pressure_pa=0.0, compliance_m3_per_pa=compliance, initial_volume_m3=compliance * pressure,
                       elastance_min_pa_per_m3=0.0, elastance_max_pa_per_m3=0.0, period_seconds=0.0,
                       activation_start=0.0, activation_end=0.0, source_pi=0.0)
            initial_pressure[alias] = pressure
            derivations.append({"id": alias, "initial_storage": "C * P0; displacement only, absolute volume unspecified",
                                "source_template": template})
        compartments.append(row)
    indices = {row["id"]: index + 1 for index, row in enumerate(compartments)}
    for row in compartments:
        row["stable_identifier"] = indices[row["id"]]
    for alias in sorted(indices):
        owner, chain = physical[alias], [alias]
        target = adjacency[alias]
        while target not in indices:
            require(physical[target]["template"] == "TempR.cellml" and target not in chain, "unsupported zero-storage elimination")
            chain.append(target)
            target = adjacency[target]
        orifice = owner["template"] in {"TempCDa.cellml", "TempCDv.cellml"}
        row = {"id": alias + "_outflow", "stable_identifier": len(compartments) + len(connections) + 1,
               "from": indices[alias], "to": indices[target], "flow_scale_m3_per_s": numerical["flow_scale_m3_per_s"],
               "pressure_scale_pa": numerical["pressure_scale_pa"], "flow_residual_tolerance": numerical["residual_tolerance"]}
        if orifice:
            require(len(chain) == 1, "source valve cannot absorb a resistive segment")
            coefficient = parameter(owner["model"], alias, "CV")
            difference = initial_pressure[alias] - initial_pressure[target]
            flow = coefficient * math.sqrt(difference) if difference >= 0 else 0.0
            row.update(flow_law="one_way_orifice", resistance_pa_s_per_m3=0.0, inertance_pa_s2_per_m3=0.0,
                       orifice_coefficient_m3_per_s_sqrt_pa=coefficient, initial_flow_m3_per_s=flow)
            initial_method = "CV * sqrt(Pup-Pdown) for Pup>=Pdown, otherwise zero; source algebraic initialization"
        else:
            resistance = sum(parameter(physical[name]["model"], name, "R") for name in chain)
            inertive = owner["template"] == "TempRLC.cellml"
            inertance = parameter(owner["model"], alias, "L") if inertive else 0.0
            flow = parameter(owner["model"], alias, "Q0") if inertive else (initial_pressure[alias] - initial_pressure[target]) / resistance
            row.update(flow_law="resistance_inertance", resistance_pa_s_per_m3=resistance,
                       inertance_pa_s2_per_m3=inertance, orifice_coefficient_m3_per_s_sqrt_pa=0.0, initial_flow_m3_per_s=flow)
            initial_method = "source Q0 differential state" if inertive else "(Pi0-Po0)/R; source algebraic initialization"
        connections.append(row)
        derivations.append({"id": row["id"], "source_components": chain, "target_component": target,
                            "reduction": "exact series resistance sum; no added storage" if len(chain) > 1 else "source outlet law",
                            "initial_flow": initial_method})
    require(used == parameters.keys(), "source parameter coverage was truncated during lowering")
    native = {"schema": "HumanPack.physiology-native.v2", "model_id": config["id"], "qualification": "source_model_reproduction",
              "law": LAW, "authored_graph_sha256": hashlib.sha256(canonical(config)).hexdigest(),
              "source_graph_sha256": SOURCE_LOCK_SHA256, "residual_tolerance": numerical["residual_tolerance"],
              "species": [], "compartments": compartments, "connections": connections, "tissue_reservoirs": [], "exchanges": []}
    manifest = {"schema": "HumanPack.shi-hose-source-lowering.v1", "source_manifest_sha256": SOURCE_LOCK_SHA256,
                "revision": REVISION, "license": source["license"], "authored_config_sha256": native["authored_graph_sha256"],
                "native_content_sha256": hashlib.sha256(canonical(native) + b"\n").hexdigest(),
                "source_parameter_count": len(parameters), "parameters": [parameters[key] for key in sorted(parameters)],
                "parameter_bindings": dict(sorted(bindings.items())), "source_connections": source_edges,
                "derivations": derivations, "source_unit_si_multipliers": {key: value[0] for key, value in sorted(units.items())},
                "numerical_settings": config["numerical_settings"],
                "scientific_status": "source_model_reproduction_target_not_qualified_by_compilation",
                "boundary": "hydraulics_only; no absolute vascular baseline, dilution volume, species transport, organ mechanics or subject calibration"}
    return native, manifest


def _write_immutable(path: Path, value: dict) -> None:
    encoded = canonical(value) + b"\n"
    require(not path.exists() or path.read_bytes() == encoded, "source compiler output is immutable; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, default=SOURCE)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        native, manifest = compile_source(directory=args.source_directory, config=read_json(args.config))
        manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
        require(args.output.resolve() != manifest_path.resolve(), "native payload and source manifest require distinct output paths")
        for path, value in ((args.output, native), (manifest_path, manifest)):
            require(not path.exists() or path.read_bytes() == canonical(value) + b"\n", "source compiler output is immutable; choose a new path")
        _write_immutable(args.output, native)
        _write_immutable(manifest_path, manifest)
        print(json.dumps({"status": "compiled_source_model", "payload": str(args.output), "manifest": str(manifest_path),
                          "sha256": manifest["native_content_sha256"], "source_parameter_count": 59,
                          "compartments": 10, "connections": 10, "scientific_status": manifest["scientific_status"]}, indent=2))
    except (HumanImportError, OSError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
