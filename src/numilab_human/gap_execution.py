"""Join the Human ledger, execution dependencies and source target inventory.

This offline authoring view is not an evidence validator or qualification
authority. It never computes readiness or promotes a ledger status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .model import ImportError as HumanImportError
from .target_coverage import MANDATORY, canonical_bytes, digest, file_digest, validate_manifest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.gap-execution.v1"
REPORT_SCHEMA = "HumanPack.gap-execution-report.v1"
COMPILER = "numilab-human.gap-execution.1"
CATEGORIES = {"engineering", "source_data", "calibration", "validation", "performance"}
BOUNDARY = (
    "Execution metadata only. Ledger statuses are quoted, evidence files are referenced "
    "but not validated, and dependency order is not readiness or qualification. "
    "Source inventory and an implemented importer do not close scientific gates."
)


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise HumanImportError(f"{label} fields differ from the gap execution schema")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanImportError(f"{label} must be nonempty text")
    return value


def _strings(value: Any, label: str, *, empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not empty):
        raise HumanImportError(f"{label} must be an array")
    for item in value:
        _text(item, label)
    if len(set(value)) != len(value):
        raise HumanImportError(f"{label} contains duplicates")
    return value


def _path(root: Path, value: Any) -> Path:
    name = _text(value, "repository reference")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise HumanImportError(f"unsafe repository reference: {name}")
    resolved = (root / name).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise HumanImportError(f"missing or external repository reference: {name}")
    return resolved


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise HumanImportError(f"invalid execution JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise HumanImportError("execution JSON must contain an object")
    return value


def ledger_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read the authoritative top-level table, not dated narrative updates."""
    header = "| Workstream | Current evidence | Status | Gap that still matters | Completion gate |"
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines.count(header) != 1:
        raise HumanImportError("completion ledger must contain exactly one workstream table")
    start = lines.index(header)
    if start + 1 >= len(lines) or lines[start + 1].replace(" ", "") != "|---|---|---|---|---|":
        raise HumanImportError("completion ledger workstream table separator changed")
    result = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not all(cells):
            raise HumanImportError("malformed completion ledger workstream row")
        name, evidence, status, gap, closure = cells
        if name in result or status not in {"open", "partial", "proved"}:
            raise HumanImportError("duplicate ledger workstream or unsupported status")
        result[name] = {"current_evidence": evidence, "status": status,
                        "remaining_gap": gap, "completion_gate": closure}
    if not result:
        raise HumanImportError("completion ledger has no workstreams")
    return result


def dependency_layers(tasks: dict[str, dict[str, Any]]) -> list[list[str]]:
    remaining = {key: set(value["depends_on"]) for key, value in tasks.items()}
    if any(not parents <= tasks.keys() for parents in remaining.values()):
        raise HumanImportError("execution task references an unknown prerequisite")
    layers = []
    while remaining:
        frontier = sorted(key for key, parents in remaining.items() if not parents)
        if not frontier:
            raise HumanImportError("execution task dependency graph contains a cycle")
        layers.append(frontier)
        remaining = {key: parents - set(frontier) for key, parents in remaining.items()
                     if key not in frontier}
    return layers


def validate_registry(value: dict[str, Any], *, root: Path = ROOT) -> tuple[dict, dict, list]:
    _object(value, {"schema", "ledger", "workstreams"}, "execution registry")
    if value["schema"] != SCHEMA:
        raise HumanImportError("unsupported gap execution schema")
    rows = ledger_rows(_path(root, value["ledger"]))
    if not isinstance(value["workstreams"], list) or not value["workstreams"]:
        raise HumanImportError("execution workstreams must be a nonempty array")
    mandatory = {f"mandatory:{domain}/{name}" for domain, names in MANDATORY.items()
                 for name in names}
    identifiers, names, mapped, tasks = set(), set(), set(), {}
    for workstream in value["workstreams"]:
        _object(workstream, {"id", "ledger_workstream", "owner", "target_ids", "tasks", "references"},
                "execution workstream")
        identifier = _text(workstream["id"], "workstream ID")
        name = _text(workstream["ledger_workstream"], "ledger workstream")
        if identifier in identifiers or name in names:
            raise HumanImportError("duplicate execution workstream")
        identifiers.add(identifier)
        names.add(name)
        _text(workstream["owner"], "workstream owner")
        targets = set(_strings(workstream["target_ids"], "target IDs"))
        if not targets <= mandatory:
            raise HumanImportError("execution workstream references an unknown mandatory target")
        mapped |= targets
        for reference in _strings(workstream["references"], "evidence references"):
            _path(root, reference)
        if not isinstance(workstream["tasks"], list) or not workstream["tasks"]:
            raise HumanImportError("execution tasks must be a nonempty array")
        for task in workstream["tasks"]:
            _object(task, {"id", "category", "depends_on", "action", "acceptance"}, "execution task")
            task_id = _text(task["id"], "task ID")
            if task_id in tasks:
                raise HumanImportError("duplicate execution task ID")
            _text(task["action"], "task action")
            _strings(task["acceptance"], "task acceptance criteria")
            _strings(task["depends_on"], "task prerequisites", empty=True)
            if not isinstance(task["category"], str) or task["category"] not in CATEGORIES:
                raise HumanImportError("unsupported execution blocker category")
            tasks[task_id] = task
    if names != rows.keys():
        raise HumanImportError("execution registry does not account for every ledger workstream")
    if mapped != mandatory:
        raise HumanImportError("execution registry omits mandatory Human targets")
    return rows, tasks, dependency_layers(tasks)


def materialize(value: dict[str, Any], *, root: Path = ROOT,
                coverage: dict[str, Any] | None = None) -> dict[str, Any]:
    _object(value, {"schema", "ledger", "workstreams"}, "execution registry")
    ledger_sha256 = file_digest(_path(root, value["ledger"]))
    rows, tasks, layers = validate_registry(value, root=root)
    # The existing coverage validator owns source identity and claim admission.
    if coverage is not None:
        validate_manifest(coverage)
    references = sorted({value["ledger"], *(reference for workstream in value["workstreams"]
                                           for reference in workstream["references"])})
    snapshots = {reference: file_digest(_path(root, reference)) for reference in references}
    if snapshots[value["ledger"]] != ledger_sha256:
        raise HumanImportError("completion ledger changed while reading its workstreams")
    links: dict[str, list[str]] = {}
    for workstream in value["workstreams"]:
        for target in workstream["target_ids"]:
            links.setdefault(target, []).append(workstream["id"])
    leaf_by_id = {} if coverage is None else {
        leaf["semantic_id"]: leaf for leaf in coverage["leaves"] if leaf["kind"] == "mandatory_target"}
    unmapped = [] if coverage is None else [
        {key: leaf[key] for key in ("leaf_sha256", "semantic_id", "kind", "source_record_sha256")}
        for leaf in coverage["leaves"] if leaf["kind"] != "mandatory_target"]
    result = {
        "schema": REPORT_SCHEMA, "compiler": COMPILER, "registry_sha256": digest(value),
        "reference_sha256": snapshots, "evidence_boundary": BOUNDARY,
        "integrated_qualification": "not_assessed",
        "evidence_reference_validation": "not_performed",
        "workstreams": [{**workstream, "ledger": rows[workstream["ledger_workstream"]],
                         "task_assessment": "not_assessed"} for workstream in value["workstreams"]],
        "dependency_layers": layers,
        "target_links": [{"semantic_id": target, "workstream_ids": sorted(owners),
                          "leaf_sha256": leaf_by_id.get(target, {}).get("leaf_sha256")}
                         for target, owners in sorted(links.items())],
        "coverage": {
            "manifest_sha256": None if coverage is None else coverage["manifest_sha256"],
            "binding_status": "not_supplied" if coverage is None else "mandatory_targets_bound",
            "source_assignment_status": "not_assessed" if coverage is None else
                ("unmapped_source_leaves" if unmapped else "no_source_leaves_in_supplied_inventory"),
            "unmapped_source_leaves": sorted(unmapped, key=lambda item: item["leaf_sha256"]),
            "unresolved_current_registers": None if coverage is None else
                [item for item in coverage["registers"] if item["status"] != "materialized"],
        },
        "counts": {"workstreams": len(rows), "tasks": len(tasks), "mandatory_targets": len(links),
                   "unmapped_source_leaves": None if coverage is None else len(unmapped)},
    }
    if any(file_digest(_path(root, reference)) != fingerprint for reference, fingerprint in snapshots.items()):
        raise HumanImportError("execution references changed while constructing the report")
    result["report_sha256"] = digest(result)
    return result


def command(arguments: argparse.Namespace) -> int:
    registry = read_json(arguments.registry)
    coverage = None if arguments.coverage is None else read_json(arguments.coverage)
    result = materialize(registry, root=arguments.repository_root, coverage=coverage)
    encoded = canonical_bytes(result) + b"\n"
    if arguments.output is None:
        print(encoded.decode("utf-8"), end="")
    else:
        output = arguments.output
        if output.exists() and output.read_bytes() != encoded:
            raise HumanImportError("execution report is immutable; choose a new output path")
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.exists():
            with output.open("xb") as stream:
                stream.write(encoded)
        print(json.dumps({"report_sha256": result["report_sha256"], **result["counts"],
                          "integrated_qualification": "not_assessed"}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", type=Path, default=ROOT / "config/human-gap-execution.v1.json")
    parser.add_argument("--repository-root", type=Path, default=ROOT,
                        help="root containing the ledger and referenced evidence")
    parser.add_argument("--coverage", type=Path, help="existing validated target-coverage manifest")
    parser.add_argument("--output", type=Path, help="new immutable report; default is stdout")
    parser.set_defaults(handler=command)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return command(arguments)
    except (HumanImportError, OSError) as error:
        parser.exit(2, f"gap-execution: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
