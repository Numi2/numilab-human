"""Read scalar contact observations from the source's uncompressed XPLT v5.

This is an offline reference reader, not a contact model. Surface IDs and face
ordering stay those of the archived plot. A partial download is explicit and
cannot qualify completion of the source protocol.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

def inspect_contact_states(path: Path, *, expected_bytes: int = 5345476723,
                           expected_sha256: str | None = None) -> dict:
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
        header = {t: u32(l, h) for t, l, h in chunks(*sections[0x01010000])}
        if header.get(0x01010001) != 5 or header.get(0x01010004) != 0:
            raise ValueError("only uncompressed source XPLT v5 is supported")
        dictionary = {}
        for t, l, h in chunks(*sections[0x01020000]):
            if t != 0x01025000:
                continue
            for identifier, (_, li, hi) in enumerate(chunks(l, h), 1):
                values = {x: read(y, z-y) for x, y, z in chunks(li, hi)}
                dictionary[identifier] = {
                    "name": values[0x01020004].rstrip(b"\0").decode("ascii"),
                    "type": struct.unpack("<I", values[0x01020002])[0],
                    "format": struct.unpack("<I", values[0x01020003])[0]}
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
        states, offset = [], root_end
        while offset + 8 <= size:
            tag, count = struct.unpack("<II", read(offset, 8))
            end = offset + 8 + count
            if end > size:
                break
            if tag != 0x02000000:
                raise ValueError("unexpected top-level XPLT chunk")
            state = {"continuation_time": None, "contact_fields": []}
            for ts, ls, hs in chunks(offset+8, end):
                if ts == 0x02010000:
                    for ti, li, hi in chunks(ls, hs):
                        if ti == 0x02010002:
                            if hi-li != 4:
                                raise ValueError("invalid XPLT time")
                            state["continuation_time"] = struct.unpack("<f", read(li, 4))[0]
                elif ts == 0x02020000:
                    for td, ld, hd in chunks(ls, hs):
                        if td != 0x02020500:
                            continue
                        for _, lv, hv in chunks(ld, hd):
                            variable = {x: (y, z) for x, y, z in chunks(lv, hv)}
                            definition = dictionary[u32(*variable[0x02020002])]
                            if definition["name"] not in ("contact gap", "contact pressure"):
                                continue
                            if definition["type"] != 0 or definition["format"] != 1:
                                raise ValueError("unsupported scalar contact field layout")
                            for sid, lf, hf in chunks(*variable[0x02020003]):
                                surface = surfaces[sid]
                                if hf-lf != 4*surface["faces"]:
                                    raise ValueError("contact field does not match surface face count")
                                raw = read(lf, hf-lf)
                                values = np.frombuffer(raw, dtype="<f4")
                                if not values.size or not np.isfinite(values).all():
                                    raise ValueError("empty or nonfinite contact field")
                                state["contact_fields"].append({"surface_id": sid, **surface,
                                    "field": definition["name"],
                                    "units": "mm" if definition["name"] == "contact gap" else "MPa",
                                    "minimum": float(values.min()), "maximum": float(values.max()),
                                    "mean": float(values.mean(dtype=np.float64)),
                                    "positive_faces": int(np.count_nonzero(values > 0)),
                                    "negative_faces": int(np.count_nonzero(values < 0)),
                                    "float32_array_sha256": hashlib.sha256(raw).hexdigest()})
            if state["continuation_time"] is None or not math.isfinite(state["continuation_time"]):
                raise ValueError("missing or nonfinite XPLT state time")
            keys = [(f["surface_id"], f["field"]) for f in state["contact_fields"]]
            expected = {(sid, field) for sid in surfaces for field in ("contact gap", "contact pressure")}
            if not expected or set(keys) != expected or len(keys) != len(expected):
                raise ValueError("incomplete or duplicate contact observations")
            if states and state["continuation_time"] <= states[-1]["continuation_time"]:
                raise ValueError("nonmonotone XPLT state time")
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
    return {"format": "FEBio XPLT version 5, uncompressed", "available_bytes": size,
            "expected_archive_bytes": expected_bytes, "complete_chunk_prefix_bytes": offset,
            "complete_chunk_prefix_sha256": digest.hexdigest(),
            "archive_complete": complete,
            "archive_identity_verified": complete and expected_sha256 is not None,
            "time_semantics": "quasi-static continuation, not physical seconds",
            "interpretation": "authored FEBio contact fields; no strict nonintersection claim",
            "surface_count": len(surfaces), "states": states}


def cli(arguments) -> int:
    manifest = json.loads((Path(__file__).resolve().parents[2] /
        "config/open-knee-oks003-reference-archive.v1.json").read_text())
    result = inspect_contact_states(arguments.plot, expected_sha256=manifest.get("plot", {}).get("sha256"))
    result["reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with arguments.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"archive_complete": result["archive_complete"],
        "complete_states": len(result["states"]), "receipt": str(arguments.output)}, indent=2))
    return 0 if result["archive_complete"] else 2
