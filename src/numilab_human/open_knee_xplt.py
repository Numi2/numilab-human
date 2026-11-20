"""Read and compare retained fields from the source's uncompressed XPLT v5.

This is an offline reference reader, not a mechanics solver. Nodal, element,
domain, and surface IDs stay those of the archived plot. A partial download is
explicit and cannot qualify completion of the source protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

def inspect_contact_states(path: Path, *, expected_bytes: int = 5345476723,
                           expected_sha256: str | None = None,
                           include_source_fields: bool = False,
                           checkpoint_states: tuple[int, ...] = (),
                           checkpoint_output: Path | None = None) -> dict:
    # Optional authoring dependency; do not require NumPy for other CLI commands.
    import numpy as np

    size = path.stat().st_size  # Snapshot the available prefix during a download.
    with path.open("rb") as stream:
        def read(offset, count):
            stream.seek(offset)
            data = stream.read(count)
            if len(data) != count:
                raise ValueError("truncated XPLT chunk")
            return data

        def chunks(start, end):
            while start < end:
                if start + 8 > end:
                    raise ValueError("truncated XPLT chunk header")
                tag, count = struct.unpack("<II", read(start, 8))
                if start + 8 + count > end:
                    raise ValueError("XPLT chunk exceeds enclosing chunk")
                yield tag, start + 8, start + 8 + count
                start += 8 + count

        def u32(start, end):
            if end-start != 4:
                raise ValueError("invalid XPLT integer")
            return struct.unpack("<I", read(start, 4))[0]

        if size < 12 or read(0, 4) != b"BEF\0":
            raise ValueError("invalid XPLT signature")
        tag, count = struct.unpack("<II", read(4, 8))
        if tag != 0x01000000 or count+12 > size:
            raise ValueError("missing complete XPLT root")
        root_end = count+12
        sections = {t: (l, h) for t, l, h in chunks(12, root_end)}
        mesh_start, mesh_end = sections[0x01040000]
        source_mesh_sha256 = hashlib.sha256(
            read(mesh_start, mesh_end - mesh_start)).hexdigest()
        header = {t: u32(l, h) for t, l, h in chunks(*sections[0x01010000])}
        if header.get(0x01010001) != 5 or header.get(0x01010004) != 0:
            raise ValueError("only uncompressed source XPLT v5 is supported")
        category_tags = {0x01023000: "node", 0x01024000: "element",
                         0x01025000: "surface"}
        dictionaries = {name: {} for name in category_tags.values()}
        for tag, l, h in chunks(*sections[0x01020000]):
            category = category_tags.get(tag)
            if category is None:
                continue
            for identifier, (_, li, hi) in enumerate(chunks(l, h), 1):
                values = {x: read(y, z-y) for x, y, z in chunks(li, hi)}
                dictionaries[category][identifier] = {
                    "name": values[0x01020004].rstrip(b"\0").decode("ascii"),
                    "type": struct.unpack("<I", values[0x01020002])[0],
                    "format": struct.unpack("<I", values[0x01020003])[0]}
        dictionary = dictionaries["surface"]
        surfaces = {}
        for t, l, h in chunks(*sections[0x01040000]):
            if t != 0x01043000:
                continue
            for _, li, hi in chunks(l, h):
                for ts, ls, hs in chunks(li, hi):
                    if ts != 0x01043101:
                        continue
                    values = {x: read(y, z-y) for x, y, z in chunks(ls, hs)}
                    name = values[0x01043104]
                    if struct.unpack("<I", name[:4])[0] != len(name)-4:
                        raise ValueError("invalid XPLT surface name length")
                    identifier = struct.unpack("<I", values[0x01043102])[0]
                    if identifier in surfaces:
                        raise ValueError("duplicate XPLT surface id")
                    surfaces[identifier] = {"name": name[4:].rstrip(b"\0").decode("ascii"),
                        "faces": struct.unpack("<I", values[0x01043103])[0]}
        domains = []
        parts = {}
        node_count = 0
        if include_source_fields:
            mesh_sections = {t: (l, h) for t, l, h in chunks(*sections[0x01040000])}
            if 0x01030000 in sections:
                for tag, lp, hp in chunks(*sections[0x01030000]):
                    if tag != 0x01030001:
                        continue
                    part_fields = {x: (y, z) for x, y, z in chunks(lp, hp)}
                    identifier = u32(*part_fields[0x01030002])
                    raw_name = read(part_fields[0x01030003][0],
                                    part_fields[0x01030003][1] - part_fields[0x01030003][0])
                    parts[identifier] = raw_name.split(b"\0", 1)[0].decode("ascii")
            node_section = next(chunks(*mesh_sections[0x01041000]))
            node_count = (node_section[2] - node_section[1]) // 12
            if node_section[2] - node_section[1] != node_count * 12 or node_count == 0:
                raise ValueError("unsupported source XPLT node coordinate layout")
            domain_section = mesh_sections.get(0x01042000)
            if domain_section is None:
                raise ValueError("source XPLT has no element domains")
            for index, (_, ld, hd) in enumerate(chunks(*domain_section), 1):
                header = next(chunks(ld, hd))
                values = {x: (y, z) for x, y, z in chunks(header[1], header[2])}
                element_count = u32(*values[0x01032104])
                part_id = u32(*values[0x01042103])
                name = None
                if 0x01032105 in values:
                    name_start, name_end = values[0x01032105]
                    raw_name = read(name_start, name_end - name_start)
                    if len(raw_name) < 4 or struct.unpack("<I", raw_name[:4])[0] != len(raw_name) - 4:
                        raise ValueError("invalid source XPLT domain name")
                    name = raw_name[4:].rstrip(b"\0").decode("ascii")
                domains.append({"domain_id": index, "name": name,
                                "part_id": part_id, "part_name": parts.get(part_id),
                                "elements": element_count})
            if len(set(checkpoint_states)) != len(checkpoint_states) or any(
                    index < 0 for index in checkpoint_states):
                raise ValueError("source checkpoint state indices must be unique and nonnegative")
            if checkpoint_states and checkpoint_output is None:
                raise ValueError("source checkpoint output directory is required for array export")

        type_components = {0: 1, 1: 3, 2: 6, 3: 9, 4: 9, 5: 36}
        unit_by_name = {
            "displacement": "mm", "reaction forces": "N", "stress": "MPa",
            "rigid position": "mm", "rigid angular position": "radian",
            "rigid force": "N", "rigid torque": "N mm",
            "prestrain stretch": "dimensionless", "fiber stretch": "dimensionless",
            "contact gap": "mm", "contact pressure": "MPa",
            "contact traction": "MPa"}

        def field_summary(raw, definition, *, expected_count=None):
            import numpy as np
            components = type_components.get(definition["type"])
            if components is None or len(raw) % 4:
                raise ValueError("unsupported or misaligned source XPLT field type")
            values = np.frombuffer(raw, dtype="<f4")
            if values.size == 0 or values.size % components:
                raise ValueError("empty or misaligned source XPLT field values")
            if expected_count is not None and values.size != expected_count * components:
                raise ValueError("source XPLT field count disagrees with its mesh domain")
            if not np.isfinite(values).all():
                raise ValueError("nonfinite source XPLT field value")
            squares = np.square(values, dtype=np.float64)
            return {"value_count": int(values.size), "item_count": int(values.size // components),
                    "components_per_item": components,
                    "minimum": float(values.min()), "maximum": float(values.max()),
                    "mean": float(values.mean(dtype=np.float64)),
                    "rms": float(np.sqrt(squares.mean())),
                    "units": unit_by_name.get(definition["name"], "source-defined"),
                    "float32_array_sha256": hashlib.sha256(raw).hexdigest()}

        states, offset = [], root_end
        while offset + 8 <= size:
            tag, count = struct.unpack("<II", read(offset, 8))
            end = offset + 8 + count
            if end > size:
                break
            if tag != 0x02000000:
                raise ValueError("unexpected top-level XPLT chunk")
            state_index = len(states)
            export_this_state = state_index in checkpoint_states
            exported_arrays = {} if export_this_state else None
            exported_metadata = {} if export_this_state else None
            state = {"continuation_time": None, "contact_fields": []}
            if include_source_fields:
                state["source_fields"] = []
            for ts, ls, hs in chunks(offset+8, end):
                if ts == 0x02010000:
                    for ti, li, hi in chunks(ls, hs):
                        if ti == 0x02010002:
                            if hi-li != 4:
                                raise ValueError("invalid XPLT time")
                            state["continuation_time"] = struct.unpack("<f", read(li, 4))[0]
                elif ts == 0x02020000:
                    class_tags = {0x02020300: "node", 0x02020400: "element",
                                  0x02020500: "surface"}
                    for td, ld, hd in chunks(ls, hs):
                        field_class = class_tags.get(td)
                        if field_class is None:
                            if include_source_fields:
                                raise ValueError(f"unsupported source XPLT state data class {td:#x}")
                            continue
                        field_dictionary = dictionaries[field_class]
                        for tv, lv, hv in chunks(ld, hd):
                            if tv != 0x02020001:
                                raise ValueError("unexpected source XPLT variable record")
                            variable = {x: (y, z) for x, y, z in chunks(lv, hv)}
                            identifier = u32(*variable[0x02020002])
                            definition = field_dictionary.get(identifier)
                            if definition is None:
                                raise ValueError("source XPLT field references an unknown dictionary ID")
                            field_name = definition["name"]
                            if field_class == "node":
                                sets = list(chunks(*variable[0x02020003]))
                                if len(sets) != 1 or sets[0][0] != 0:
                                    raise ValueError("unsupported source XPLT nodal field layout")
                                raw = read(sets[0][1], sets[0][2] - sets[0][1])
                                summary = field_summary(raw, definition,
                                    expected_count=node_count if include_source_fields else None)
                                if include_source_fields:
                                    if export_this_state:
                                        array_key = f"node_{identifier:02d}"
                                        exported_arrays[array_key] = np.frombuffer(
                                            raw, dtype="<f4").copy()
                                        exported_metadata[array_key] = {
                                            "class": field_class, "field": field_name,
                                            "dictionary_id": identifier, "format": definition["format"],
                                            "units": summary["units"]}
                                    state["source_fields"].append({"class": field_class,
                                        "field": field_name, "dictionary_id": identifier,
                                        "format": definition["format"], "domain_id": None, **summary})
                            elif field_class == "element":
                                seen_domains = set()
                                for did, lf, hf in chunks(*variable[0x02020003]):
                                    domain_id = did
                                    if domain_id < 1 or domain_id > len(domains) or domain_id in seen_domains:
                                        raise ValueError("invalid or duplicate source XPLT domain ID")
                                    seen_domains.add(domain_id)
                                    domain = domains[domain_id - 1]
                                    raw = read(lf, hf - lf)
                                    fmt = definition["format"]
                                    expected_count = (domain["elements"] if fmt == 1 else
                                                      1 if fmt == 3 else None)
                                    if include_source_fields and fmt not in (1, 3):
                                        raise ValueError("unsupported source XPLT element format")
                                    summary = field_summary(raw, definition,
                                        expected_count=expected_count if include_source_fields else None)
                                    if include_source_fields:
                                        if export_this_state:
                                            array_key = f"element_{identifier:02d}_domain_{domain_id:02d}"
                                            exported_arrays[array_key] = np.frombuffer(
                                                raw, dtype="<f4").copy()
                                            exported_metadata[array_key] = {
                                                "class": field_class, "field": field_name,
                                                "dictionary_id": identifier, "format": fmt,
                                                "domain_id": domain_id, "part_id": domain["part_id"],
                                                "part_name": domain["part_name"],
                                                "units": summary["units"]}
                                        state["source_fields"].append({"class": field_class,
                                            "field": field_name, "dictionary_id": identifier,
                                            "format": fmt, **domain, **summary})
                            else:
                                for sid, lf, hf in chunks(*variable[0x02020003]):
                                    surface = surfaces.get(sid)
                                    if surface is None:
                                        raise ValueError("source XPLT field references an unknown surface")
                                    raw = read(lf, hf - lf)
                                    summary = field_summary(raw, definition,
                                        expected_count=surface["faces"])
                                    if field_name in ("contact gap", "contact pressure"):
                                        values = np.frombuffer(raw, dtype="<f4")
                                        state["contact_fields"].append({"surface_id": sid, **surface,
                                            "field": field_name, **summary,
                                            "positive_faces": int(np.count_nonzero(values > 0)),
                                            "negative_faces": int(np.count_nonzero(values < 0))})
                                    if include_source_fields:
                                        if export_this_state:
                                            array_key = f"surface_{identifier:02d}_surface_{sid:02d}"
                                            exported_arrays[array_key] = np.frombuffer(
                                                raw, dtype="<f4").copy()
                                            exported_metadata[array_key] = {
                                                "class": field_class, "field": field_name,
                                                "dictionary_id": identifier, "format": definition["format"],
                                                "surface_id": sid, "surface_name": surface["name"],
                                                "units": summary["units"]}
                                        state["source_fields"].append({"class": field_class,
                                            "field": field_name, "dictionary_id": identifier,
                                            "format": definition["format"], "surface_id": sid,
                                            **surface, **summary})
            if state["continuation_time"] is None or not math.isfinite(state["continuation_time"]):
                raise ValueError("missing or nonfinite XPLT state time")
            keys = [(f["surface_id"], f["field"]) for f in state["contact_fields"]]
            expected = {(sid, field) for sid in surfaces for field in ("contact gap", "contact pressure")}
            if not expected or set(keys) != expected or len(keys) != len(expected):
                raise ValueError("incomplete or duplicate contact observations")
            if states and state["continuation_time"] <= states[-1]["continuation_time"]:
                raise ValueError("nonmonotone XPLT state time")
            if export_this_state:
                if not state["source_fields"]:
                    raise ValueError("selected XPLT checkpoint has no source fields")
                checkpoint_output.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_output / f"state-{state_index:05d}.npz"
                metadata = {"state_index": state_index,
                    "continuation_time": state["continuation_time"],
                    "reference_plot_sha256": expected_sha256,
                    "source_mesh_sha256": source_mesh_sha256,
                    "arrays": exported_metadata,
                    "fields": state["source_fields"], "domains": domains,
                    "surfaces": surfaces}
                with checkpoint_path.open("xb") as target:
                    np.savez_compressed(target, metadata_json=json.dumps(metadata, sort_keys=True),
                                        **exported_arrays)
                state["checkpoint_array_archive"] = str(checkpoint_path)
                checkpoint_digest = hashlib.sha256()
                with checkpoint_path.open("rb") as exported:
                    for block in iter(lambda: exported.read(1024 * 1024), b""):
                        checkpoint_digest.update(block)
                state["checkpoint_array_archive_sha256"] = checkpoint_digest.hexdigest()
            states.append(state)
            offset = end
        digest = hashlib.sha256()
        stream.seek(0)
        remaining = offset
        while remaining:
            block = stream.read(min(remaining, 1024*1024))
            if not block:
                raise ValueError("XPLT prefix changed while reading")
            digest.update(block)
            remaining -= len(block)
    complete = size == expected_bytes and offset == size
    if complete and expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError("reference plot identity drift")
    result = {"format": "FEBio XPLT version 5, uncompressed", "available_bytes": size,
            "expected_archive_bytes": expected_bytes, "complete_chunk_prefix_bytes": offset,
            "complete_chunk_prefix_sha256": digest.hexdigest(),
            "archive_complete": complete,
            "archive_identity_verified": complete and expected_sha256 is not None,
            "time_semantics": "quasi-static continuation, not physical seconds",
            "interpretation": "authored FEBio contact fields; no strict nonintersection claim",
            "surface_count": len(surfaces), "states": states}
    if include_source_fields:
        result["node_count"] = node_count
        result["source_mesh_sha256"] = source_mesh_sha256
        result["domains"] = domains
        result["parts"] = [{"part_id": identifier, "name": name}
                            for identifier, name in sorted(parts.items())]
        result["source_field_dictionary"] = {
            category: [{"dictionary_id": identifier, **definition}
                       for identifier, definition in fields.items()]
            for category, fields in dictionaries.items()}
    return result


def compare_source_checkpoint_arrays(reference_path: Path, matter_path: Path,
                                     *, time_tolerance: float = 1.0e-7) -> dict:
    """Compare Matter arrays against one retained FEBio state on identical source indexing.

    The Matter archive must carry the reference plot and source mesh hashes copied
    from the source checkpoint, along with the same field-key map. This function
    reports numerical differences; it deliberately does not choose a pass gate.
    """
    import numpy as np

    def load(path: Path):
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise ValueError(f"checkpoint has no metadata: {path}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {key: np.array(archive[key], copy=True)
                      for key in archive.files if key != "metadata_json"}
        return metadata, arrays

    reference, ref_arrays = load(reference_path)
    matter, sim_arrays = load(matter_path)
    for identity in ("reference_plot_sha256", "source_mesh_sha256"):
        left, right = reference.get(identity), matter.get(identity)
        if not isinstance(left, str) or len(left) != 64 or left != right:
            raise ValueError(f"Matter checkpoint {identity} does not match the FEBio source")
    reference_time = reference.get("continuation_time")
    matter_time = matter.get("continuation_time")
    if (not isinstance(reference_time, (int, float)) or
            not isinstance(matter_time, (int, float)) or
            abs(reference_time - matter_time) > time_tolerance):
        raise ValueError("Matter checkpoint is not at the same FEBio continuation state")
    if reference.get("arrays") != matter.get("arrays"):
        raise ValueError("Matter checkpoint field/domain/surface mapping differs from FEBio")
    if set(ref_arrays) != set(sim_arrays) or set(ref_arrays) != set(reference["arrays"]):
        raise ValueError("Matter checkpoint array keys differ from the FEBio field map")

    comparisons = []
    for key in sorted(ref_arrays):
        expected, actual = ref_arrays[key], sim_arrays[key]
        if expected.shape != actual.shape or expected.size == 0:
            raise ValueError(f"Matter checkpoint shape mismatch for {key}")
        if not np.issubdtype(expected.dtype, np.number) or not np.issubdtype(actual.dtype, np.number):
            raise ValueError(f"non-numeric checkpoint field {key}")
        expected64, actual64 = expected.astype(np.float64), actual.astype(np.float64)
        if not np.isfinite(expected64).all() or not np.isfinite(actual64).all():
            raise ValueError(f"nonfinite checkpoint values for {key}")
        difference = actual64 - expected64
        reference_rms = float(np.sqrt(np.mean(expected64 * expected64)))
        difference_rms = float(np.sqrt(np.mean(difference * difference)))
        comparisons.append({"array": key, **reference["arrays"][key],
            "value_count": int(expected.size), "reference_rms": reference_rms,
            "matter_rms": float(np.sqrt(np.mean(actual64 * actual64))),
            "maximum_absolute_error": float(np.max(np.abs(difference))),
            "rms_error": difference_rms,
            "relative_rms_error": difference_rms / max(reference_rms, 1.0e-30),
            "mean_bias": float(np.mean(difference))})
    return {"status": "compared", "reference_plot_sha256": reference["reference_plot_sha256"],
        "source_mesh_sha256": reference["source_mesh_sha256"],
        "continuation_time": reference_time,
        "qualification": "numerical differences reported without a pass threshold",
        "arrays": comparisons}


def compare_cli(arguments) -> int:
    result = compare_source_checkpoint_arrays(arguments.reference, arguments.matter)
    with arguments.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": result["status"],
        "arrays_compared": len(result["arrays"]), "receipt": str(arguments.output)}, indent=2))
    return 0


def cli(arguments) -> int:
    manifest = json.loads((Path(__file__).resolve().parents[2] /
        "config/open-knee-oks003-reference-archive.v1.json").read_text())
    result = inspect_contact_states(arguments.plot,
        expected_sha256=manifest.get("plot", {}).get("sha256"), include_source_fields=True,
        checkpoint_states=tuple(arguments.checkpoint_state or ()),
        checkpoint_output=arguments.checkpoint_output_dir)
    result["reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with arguments.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"archive_complete": result["archive_complete"],
        "complete_states": len(result["states"]), "receipt": str(arguments.output)}, indent=2))
    return 0 if result["archive_complete"] else 2
