"""Source-frame Open Knee problem capture and fail-closed execution admission.

This is offline authoring, not a solver. No MyoSim registration, constitutive
substitution, contact preparation, or force application happens here. The full
ordered XML tree is compiled without dropping unknown fields. Native execution
remains rejected until Matter implements and validates the source equations.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

SCHEMA = "numi.human.open-knee-source-problem.v1"
PROGRAM_SCHEMA = "numi.human.febio-authored-program.v1"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def xml_record(element: ET.Element) -> dict:
    """Preserve every tag, attribute, text, tail and ordered child, even unknowns."""
    return {"tag": element.tag, "attributes": dict(element.attrib),
            "text": element.text, "tail": element.tail,
            "children": [xml_record(child) for child in element]}


def _walk(root: ET.Element, prefix: str = ""):
    path = prefix + "/" + root.tag
    yield path, root
    counts: Counter = Counter()
    for child in root:
        counts[child.tag] += 1
        # Sibling ordinal stays attached to the parent prefix, then the child tag.
        child_path = path + "/" + child.tag + f"[{counts[child.tag]}]"
        yield from _walk_at(child, child_path)


def _walk_at(element: ET.Element, path: str):
    yield path, element
    counts: Counter = Counter()
    for child in element:
        counts[child.tag] += 1
        yield from _walk_at(child, path + "/" + child.tag + f"[{counts[child.tag]}]")


def _value(element: ET.Element, name: str) -> str:
    value = element.findtext(name)
    if value is None:
        raise ValueError(f"{element.tag} {element.get('name', '')}: missing {name}")
    return value.strip()


def _vec(text: str) -> list[float]:
    result = [float(x) for x in text.split(",")]
    if len(result) != 3 or not all(math.isfinite(x) for x in result):
        raise ValueError("expected a finite source 3-vector")
    return result


def _rank(matrix: list[list[float]], tolerance: float = 1e-10) -> int:
    a = [row[:] for row in matrix]
    row = 0
    for col in range(len(a[0])):
        pivot = max(range(row, len(a)), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) <= tolerance:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        scale = a[row][col]
        a[row] = [v / scale for v in a[row]]
        for i in range(len(a)):
            if i != row:
                scale = a[i][col]
                a[i] = [v - scale * p for v, p in zip(a[i], a[row])]
        row += 1
        if row == len(a):
            break
    return row


def cylindrical_chain_audit(root: ET.Element) -> dict:
    """Infinitesimal neutral-pose source chain, not a flexed native simulation.

    J maps translation/rotation rates to spatial [linear; angular] velocity
    at the source origin. Translation is in mm and rotation in radians. All
    six rows use this stated convention; no MyoSim coordinates are involved.
    """
    joints = [x for x in root.findall(".//constraint")
              if x.get("type") == "rigid cylindrical joint"
              and x.get("name", "").startswith("Patellar_")]
    if not joints:
        return {"status": "absent"}
    # Preserve and order the actual graph rather than assuming XML order.
    outgoing: dict[int, ET.Element] = {}
    ends = set()
    for joint in joints:
        a, b = int(_value(joint, "body_a")), int(_value(joint, "body_b"))
        if a in outgoing:
            raise ValueError("patellar joint graph branches")
        outgoing[a] = joint
        ends.add(b)
    starts = set(outgoing) - ends
    if len(starts) != 1:
        raise ValueError("patellar joint graph is cyclic or disconnected")
    body = next(iter(starts))
    bodies = [body]
    columns, freedoms, axes = [], [], []
    while body in outgoing:
        joint = outgoing.pop(body)
        axis = _vec(_value(joint, "joint_axis"))
        norm = math.sqrt(sum(v * v for v in axis))
        if norm <= 1e-12:
            raise ValueError("zero source joint axis")
        axis = [v / norm for v in axis]
        axes.append(axis)
        origin = _vec(_value(joint, "joint_origin"))
        # v = -omega cross origin for rotation about the authored joint origin.
        v = [origin[1]*axis[2]-origin[2]*axis[1],
             origin[2]*axis[0]-origin[0]*axis[2],
             origin[0]*axis[1]-origin[1]*axis[0]]
        for kind, column in (("translation", axis + [0., 0., 0.]),
                             ("rotation", v + axis)):
            prescribed = _value(joint, "prescribed_" + kind)
            if prescribed not in ("0", "1"):
                raise ValueError("invalid prescribed coordinate flag")
            if prescribed == "0":
                columns.append(column)
                freedoms.append({"joint": joint.get("name"), "coordinate": kind})
        body = int(_value(joint, "body_b"))
        bodies.append(body)
    if outgoing:
        raise ValueError("disconnected patellar joint graph")
    matrix = [list(row) for row in zip(*columns)] if columns else []
    return {"status": "kinematic_calculation_only", "configuration": "authored_neutral",
            "body_material_ids": bodies, "free_coordinates": freedoms,
            "normalized_joint_axes": axes, "spatial_jacobian_mm_radian": matrix,
            "rank": _rank(matrix) if matrix else 0,
            "flexed_transform_and_reaction_validation": "not_run"}


def audit_program(root: ET.Element, geometry: ET.Element | None) -> dict:
    """Check referential closure without assigning defaults or native semantics."""
    errors: list[dict] = []

    def fail(code: str, path: str, reference: str):
        errors.append({"code": code, "path": path, "reference": reference})

    if root.tag != "febio_spec" or root.get("version") != "2.5":
        fail("unsupported_deck_format", "/", str(root.attrib))
    materials = root.findall("./Material/material")
    rigid_ids = {x.get("id") for x in materials if x.get("type") == "rigid body"}
    material_ids = {x.get("id") for x in materials}
    curves = root.findall("./LoadData/loadcurve")
    curve_ids = {x.get("id") for x in curves}
    discrete_ids = {x.get("id") for x in root.findall("./Discrete/discrete_material")}
    for label, values in (("material", [x.get("id") for x in materials]),
                          ("loadcurve", [x.get("id") for x in curves])):
        if None in values or len(set(values)) != len(values):
            fail("duplicate_or_absent_id", "/" + label, str(values))
    tables = {tag: {} for tag in ("NodeSet", "Surface", "SurfacePair", "Elements", "DiscreteSet")}
    if geometry is not None:
        g = geometry if geometry.tag == "Geometry" else geometry.find("Geometry")
        if g is None:
            fail("missing_geometry_section", "/Geometry", "Geometry")
        else:
            for child in g:
                if child.tag in tables:
                    name = child.get("name")
                    if not name or name in tables[child.tag]:
                        fail("duplicate_or_absent_name", "/Geometry/" + child.tag, str(name))
                    tables[child.tag][name] = child
                if child.tag == "Elements" and child.get("mat") not in material_ids:
                    fail("unresolved_material", "/Geometry/Elements", str(child.get("mat")))
    for path, element in _walk(root):
        for attribute, table in (("node_set", "NodeSet"), ("surface_pair", "SurfacePair"),
                                 ("elem_set", "Elements"), ("discrete_set", "DiscreteSet")):
            value = element.get(attribute)
            if value is not None and value not in tables[table]:
                fail("unresolved_" + attribute, path, value)
        for attribute, identifiers in (("lc", curve_ids), ("rb", rigid_ids),
                                       ("dmat", discrete_ids)):
            value = element.get(attribute)
            if value is not None and value not in identifiers:
                fail("unresolved_" + attribute, path, value)
        if element.tag in ("body_a", "body_b") and (element.text or "").strip() not in rigid_ids:
            fail("unresolved_rigid_body", path, (element.text or "").strip())
        if element.tag == "rigid_body" and element.get("mat") not in rigid_ids:
            fail("unresolved_rigid_body", path, str(element.get("mat")))
        if element.tag == "ElementData":
            region = tables["Elements"].get(element.get("elem_set"))
            if region is not None:
                ids = [x.get("lid") for x in element]
                if len(ids) != len(region) or set(ids) != {str(i+1) for i in range(len(region))}:
                    fail("incomplete_element_data", path, str(element.get("elem_set")))
    for name, pair in tables["SurfacePair"].items():
        for side in pair:
            if side.get("surface") not in tables["Surface"]:
                fail("unresolved_surface", "/Geometry/SurfacePair/" + str(name), str(side.attrib))
    active_pairs = [x.get("surface_pair") for x in root.findall(".//contact")]
    constraints = root.findall(".//constraint")
    ties = root.findall("./Boundary/rigid")
    control = root.find("./Step/Control")
    return {
        "reference_errors": errors,
        "inventory": {
            "materials": [dict(x.attrib) for x in materials],
            "rigid_bodies": [dict(x.attrib) for x in materials if x.get("type") == "rigid body"],
            "rigid_ties": [dict(x.attrib) for x in ties],
            "constraints": [xml_record(x) for x in constraints],
            "contacts": [xml_record(x) for x in root.findall(".//contact")],
            "loadcurves": [xml_record(x) for x in curves],
            "control": xml_record(control) if control is not None else None,
            "mesh_data": [{"attributes": dict(x.attrib), "entries": len(x)}
                          for x in root.findall("./MeshData/ElementData")],
            "defined_surface_pair_count": len(tables["SurfacePair"]),
            "active_contact_count": len(active_pairs),
            "unused_surface_pairs": sorted(set(tables["SurfacePair"]) - set(active_pairs)),
            "explicit_load_section_count": len(root.findall(".//Loads")),
        },
        "patellar_chain": cylindrical_chain_audit(root),
        "native_admission": {
            "status": "rejected_unsupported_source_program",
            "executable": False,
            "sections": [{"path": path, "tag": element.tag,
                          "type": element.get("type"), "status": "retained_not_executed",
                          "reason": "No qualified source-equivalent Matter lowering for this section"}
                         for path, element in _walk(root)
                         if path.count("/") == 2],
            "unknown_fields": "retained verbatim; never replaced by runtime defaults",
        },
    }


def freeze_reference_case(directory: Path, output: Path, *, human_revision: str,
                          matter_revision: str, archive: Path | None = None) -> dict:
    """Freeze the pinned legacy inputs and compile their entire source program.

    Missing authored includes are audited against the retained geometry for
    diagnosis ONLY. A different filename is never implicitly rebound for a run.
    The output must be new, so an earlier baseline cannot be overwritten.
    """
    from .open_knee import EXPECTED_HASHES
    directory, output = directory.resolve(), output.resolve()
    for name, expected in EXPECTED_HASHES.items():
        if not (directory / name).is_file() or _sha(directory / name) != expected:
            raise ValueError(f"pinned source identity drift: {name}")
    archive_manifest = None
    archive_files = {}
    if archive is not None:
        archive = archive.resolve()
        archive_manifest = json.loads((Path(__file__).resolve().parents[2] /
            "config/open-knee-oks003-reference-archive.v1.json").read_text())
        archive_files = {**archive_manifest["files"], **{
            "processed-results/" + name: identity
            for name, identity in archive_manifest.get("processed_results", {}).items()}}
        for name, identity in archive_files.items():
            candidate = archive / name
            if not candidate.is_file() or _sha(candidate) != identity["sha256"]:
                raise ValueError(f"pinned reference archive identity drift: {name}")
        plot = archive_manifest.get("plot")
        if plot is not None and (not (archive / plot["file"]).is_file() or
                                 _sha(archive / plot["file"]) != plot["sha256"]):
            raise ValueError("pinned reference archive plot identity drift")
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "source"
    frozen.mkdir()
    for name, expected in EXPECTED_HASHES.items():
        shutil.copyfile(directory / name, frozen / name)
        if _sha(frozen / name) != expected:
            raise ValueError(f"source changed while freezing: {name}")
    if archive_manifest is not None:
        for name, identity in archive_files.items():
            (frozen / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive / name, frozen / name)
            if _sha(frozen / name) != identity["sha256"]:
                raise ValueError(f"reference archive changed while freezing: {name}")
    root = ET.parse(frozen / "FeBio_custom.feb").getroot()
    includes = [(path, e.get("from")) for path, e in _walk(root) if e.get("from")]
    dependencies, dependency_errors = [], []
    selected_geometry = None
    for path, authored in includes:
        basename = PureWindowsPath(authored).name
        candidate = (archive if archive is not None else directory) / basename
        present = candidate.is_file()
        if present and not (frozen / basename).exists():
            shutil.copyfile(candidate, frozen / basename)
            if _sha(frozen / basename) != _sha(candidate):
                raise ValueError(f"dependency changed while freezing: {basename}")
        dependencies.append({"path": path, "authored": authored,
                             "local_filename": basename, "present": present,
                             "sha256": _sha(frozen / basename) if present else None})
        if not present:
            dependency_errors.append({"code": "missing_authored_include", "path": path,
                                      "reference": authored})
        elif path.endswith("/Geometry[1]"):
            selected_geometry = frozen / basename
        else:
            dependency_errors.append({"code": "unsupported_external_include", "path": path,
                                      "reference": authored})
    diagnostic_geometry = selected_geometry or frozen / "Geometry.feb"
    audit = audit_program(root, ET.parse(diagnostic_geometry).getroot())
    audit["reference_errors"] = dependency_errors + audit["reference_errors"]
    # Deck format 2.5 is NOT a solver version. The original executable/settings
    # remain unknown until an independently retained binary/run identifies them.
    audit.update({"schema": SCHEMA, "model_class": "source_frame_reference_candidate",
                  "status": "blocked_reference_dependencies" if audit["reference_errors"]
                            else "awaiting_reference_solver",
                  "revisions": {"human": human_revision, "matter": matter_revision},
                  "coordinate_policy": {"frame": "source_specimen", "scale": 1.0,
                                        "registration": None, "material_reference_reset": False},
                  "geometry_audited": diagnostic_geometry.name,
                  "geometry_binding": "authored_dependency" if selected_geometry else "diagnostic_only",
                  "dependencies": dependencies,
                  "reference_solver": {"status": "not_run", "binary_sha256": None,
                                       "version": None, "settings": "authored_deck",
                                       "deck_format_version": root.get("version")},
                  "qualification": {"source_equivalence": False, "initialized_equilibrium": False,
                                    "extensor_transmission": False, "strict_nonintersection": False,
                                    "whole_body_integration": False}})
    files = set(EXPECTED_HASHES) | {d["local_filename"] for d in dependencies if d["present"]}
    if archive_manifest is not None:
        files |= set(archive_files)
        from .open_knee_febio_log import parse_febio_log
        observations = parse_febio_log((frozen / "FeBio_custom.log").read_text())
        observation_file = output / "archived-observations.json"
        observation_file.write_text(json.dumps(observations, separators=(",", ":")) + "\n")
        audit["archived_reference"] = {
            **{k: v for k, v in observations.items() if k not in ("records", "accepted_times")},
            "evidence_origin": "published DOI archive; not a new local solver run",
            "manifest": archive_manifest,
            "observations": {"file": observation_file.name, "sha256": _sha(observation_file)},
            "local_reproduction": False,
            "pressure_and_tissue_fields": "not contained in text log; require XPLT output",
        }
        if archive_manifest.get("plot") is not None:
            audit["archived_reference"]["plot"] = {**archive_manifest["plot"],
                "path": str(archive / archive_manifest["plot"]["file"])}
            audit["archived_reference"]["pressure_and_tissue_fields"] = "retained in hash-pinned original XPLT"
    audit["source_files"] = {}
    for name in sorted(files):
        digest = _sha(frozen / name)
        audit["source_files"][name] = {"sha256": digest, "bytes": (frozen / name).stat().st_size}
    program = output / "mechanical-program.json"
    program.write_text(json.dumps({"schema": PROGRAM_SCHEMA, "document": xml_record(root)},
                                  separators=(",", ":")) + "\n")
    audit["mechanical_program"] = {"file": program.name, "sha256": _sha(program),
                                   "coverage": "all XML elements, attributes, text, tail and child order"}
    if not audit["reference_errors"]:
        # The only edit to the execution deck is explicit include relocation.
        # The original source bytes remain immutable under source/.
        execution = output / "execution"
        execution.mkdir()
        data = (frozen / "FeBio_custom.feb").read_bytes()
        for dependency in dependencies:
            original = ('from="' + dependency["authored"] + '"').encode("ascii")
            replacement = ('from="../source/' + dependency["local_filename"] + '"').encode("ascii")
            if data.count(original) != 1:
                raise ValueError("cannot uniquely relocate authored include")
            data = data.replace(original, replacement)
        deck = execution / "FeBio_custom.feb"
        deck.write_bytes(data)
        audit["execution_deck"] = {"file": str(deck.relative_to(output)), "sha256": _sha(deck),
                                   "changes": "external include paths only; all mechanical values unchanged"}
    (output / "receipt.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def _translate_febio3_material_frames(root: ET.Element) -> list[dict]:
    records = []
    for material in root.findall("Material/material"):
        if material.get("type") != "uncoupled prestrain elastic":
            continue
        fiber, elastic = material.find("fiber"), material.find("elastic")
        if fiber is None or fiber.get("type") != "vector" or elastic is None or elastic.find("fiber") is not None:
            raise ValueError("unsupported source fibre-frame layout")
        if elastic.get("type") != "trans iso Mooney-Rivlin" or material.find("mat_axis") is not None:
            raise ValueError("unsupported source prestrain material")
        axis = _vec(fiber.text)
        norm = math.sqrt(sum(x*x for x in axis))
        if not .999 <= norm <= 1.001:
            raise ValueError("invalid source fibre axis")
        axis = [x/norm for x in axis]
        # Choose a stable transverse seed; transverse isotropy makes this
        # choice mechanically immaterial. The source direction is preserved.
        seed = [0., 0., 0.]
        seed[min(range(3), key=lambda i: abs(axis[i]))] = 1.
        frame = ET.Element("mat_axis", {"type": "vector"})
        ET.SubElement(frame, "a").text = ",".join(format(x, ".17g") for x in axis)
        ET.SubElement(frame, "d").text = ",".join(format(x, ".17g") for x in seed)
        material.insert(list(material).index(fiber), frame)
        material.remove(fiber)
        ET.SubElement(elastic, "fiber", {"type": "vector"}).text = "1,0,0"
        records.append({"material": material.get("name"), "source_axis": _vec(fiber.text),
                        "normalized_axis": axis, "transverse_seed": seed,
                        "elastic_local_fiber": [1,0,0]})
    if {x["material"] for x in records} != {"QAT", "ACL", "PCL", "MCL", "LCL", "PTL"}:
        raise ValueError("incomplete source prestrain frame conversion")
    return records


def prepare_febio3_comparison(output: Path) -> dict:
    """Explicit legacy fibre-frame translation for the public 3.0 comparator.

    FEBio 2.5 stores a prestrain material's fibre as its material x axis.
    FEBio 3.0 exposes mat_axis on the parent and a local fibre on its elastic
    child. Both the isochoric prestrain and child fibre must share that frame.
    All original source bytes remain immutable. This does not admit Matter.
    """
    output = output.resolve()
    receipt = output / "receipt.json"
    result = json.loads(receipt.read_text())
    if result["reference_solver"]["status"] != "not_run":
        raise ValueError("cannot translate an attempted reference case")
    if result["source_files"]["FeBio_custom.feb"]["sha256"] != "00b6efb53ad7e7330296cbb9569d358d48ed60819e22732e6149db6fb98a158a":
        raise ValueError("FEBio 3 translation is restricted to the pinned source")
    deck = output / result["execution_deck"]["file"]
    if result["execution_deck"].get("required_comparison_version") or _sha(deck) != result["execution_deck"]["sha256"]:
        raise ValueError("execution deck already translated or changed")
    root = ET.parse(deck).getroot()
    records = _translate_febio3_material_frames(root)
    previous = result["execution_deck"]["sha256"]
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    deck.write_bytes(data)
    result["execution_deck"].update(sha256=_sha(deck), before_translation_sha256=previous,
        required_comparison_version="3.0.0", frame_translation=records,
        changes="external include paths and six explicit parent material frames with local elastic fibre; XML formatting normalized")
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_reference_case(output: Path, binary: Path, *, config: Path | None = None,
                       comparison_version: str | None = None) -> dict:
    """Execute the frozen source experiment once; preserve unsuccessful baselines.

    Runtime qualification is deliberately separate from archive import and
    from native source equivalence. Never rerun into a previous output folder.
    """
    from .open_knee_febio_log import parse_febio_log
    output, binary = output.resolve(), binary.resolve()
    if comparison_version is not None:
        import re
        if not re.fullmatch(r"\d+\.\d+\.\d+", comparison_version) or comparison_version == "2.9.1":
            raise ValueError("comparison version must explicitly name a different FEBio version")
    receipt = output / "receipt.json"
    result = json.loads(receipt.read_text())
    if result["reference_errors"] or "execution_deck" not in result:
        raise ValueError("reference dependencies are not closed")
    required_version = result["execution_deck"].get("required_comparison_version")
    if required_version and comparison_version != required_version:
        raise ValueError("translated execution deck requires its explicit comparison version")
    if result["reference_solver"]["status"] != "not_run":
        raise ValueError("reference execution already attempted; freeze a new case")
    for name, identity in result["source_files"].items():
        if _sha(output / "source" / name) != identity["sha256"]:
            raise ValueError(f"frozen source identity drift: {name}")
    deck = output / result["execution_deck"]["file"]
    if _sha(deck) != result["execution_deck"]["sha256"]:
        raise ValueError("execution deck identity drift")
    run = result["reference_solver"]
    run.update({"status": "running", "binary": str(binary), "binary_sha256": _sha(binary),
                "command": [str(binary), "-i", deck.name], "started_unix": time.time()})
    run["comparison_version"] = comparison_version
    if config is not None:
        frozen_config = output / "execution" / "reference-config.xml"
        # Capture the linear backend and thread settings, instead of inheriting
        # an unrecorded febio.xml from the executable's directory.
        with frozen_config.open("xb") as stream:
            stream.write(config.resolve().read_bytes())
        run["config_sha256"] = _sha(frozen_config)
        run["command"] += ["-config", str(frozen_config)]
    else:
        run["command"] += ["-noconfig"]
        run["config"] = "explicit solver defaults; no external febio.xml"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    try:
        with (deck.parent / "stdout.txt").open("x") as stdout:
            process = subprocess.run(run["command"], cwd=deck.parent, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=subprocess.STDOUT, check=False)
        run["returncode"] = process.returncode
        log = deck.with_suffix(".log")
        if not log.exists():
            run["status"] = "failed_missing_log"
        else:
            observations = parse_febio_log(log.read_text(errors="replace"))
            destination = output / "local-observations.json"
            destination.write_text(json.dumps(observations, separators=(",", ":")) + "\n")
            run.update({k: v for k, v in observations.items() if k not in ("records", "accepted_times")})
            run["log_sha256"] = _sha(log)
            run["observations_sha256"] = _sha(destination)
            matched = observations["version"] == (comparison_version or "2.9.1")
            reached = bool(observations["accepted_times"]) and observations["accepted_times"][-1] == 2.0
            completed = "completed_version_comparison" if comparison_version else "completed_source_protocol"
            run["status"] = (completed if process.returncode == 0 and matched and reached
                             and observations["status"] == "normal_termination_with_complete_observations"
                             else "failed_or_nonmatching_reference")
            run["original_binary_identity_match"] = "unknown; archived executable not supplied"
    except BaseException as error:
        run["status"] = "interrupted_or_failed"
        run["error"] = str(error)
        raise
    finally:
        run["finished_unix"] = time.time()
        result["status"] = "reference_" + run["status"]
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def cli(arguments) -> int:
    if arguments.febio3_material_frames and arguments.comparison_version != "3.0.0":
        raise ValueError("FEBio 3 material-frame translation requires --comparison-version 3.0.0")
    def revision(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    human_root = Path(__file__).resolve().parents[2]
    result = freeze_reference_case(arguments.open_knee, arguments.output,
                                   human_revision=revision(human_root),
                                   matter_revision=revision(arguments.matter_root),
                                   archive=arguments.archive)
    result["compiler_sources"] = {
        path: _sha(human_root / path) for path in
        ("src/numilab_human/open_knee.py", "src/numilab_human/open_knee_reference.py",
         "src/numilab_human/open_knee_febio_log.py", "src/numilab_human/cli.py",
         "config/open-knee-oks003-reference-archive.v1.json")}
    result["worktree_status"] = {
        owner: subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True)
        for owner, path in (("human", human_root), ("matter", arguments.matter_root))}
    (arguments.output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if arguments.febio3_material_frames:
        result = prepare_febio3_comparison(arguments.output)
    if arguments.febio is not None:
        result = run_reference_case(arguments.output, arguments.febio,
                                    config=arguments.febio_config,
                                    comparison_version=arguments.comparison_version)
    print(json.dumps({"status": result["status"], "receipt": str(arguments.output / "receipt.json"),
                      "reference_errors": result["reference_errors"],
                      "native_admission": result["native_admission"]["status"]}, indent=2))
    return 2 if result["reference_errors"] or (arguments.febio is not None and
        result["reference_solver"]["status"] not in
        ("completed_source_protocol", "completed_version_comparison")) else 0
