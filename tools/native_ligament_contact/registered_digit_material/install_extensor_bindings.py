"""Apply the tested donor-frame registrations, retaining the complete NHTISS4 payload and provenance."""

import json
from pathlib import Path
import struct
import sys
import numpy as np
from scipy.spatial.transform import Rotation

from extensor_registration import sha, tissue


def install(control, fit, registry, out):
    out.mkdir(exist_ok=False)
    config = json.loads((control / "launch.json").read_text())
    argv = config["argv"][:]
    index = argv.index("--soft-tissue-payload") + 1
    source = Path(argv[index])
    report = json.loads((fit / "site-fit.json").read_text())
    if (
        sha(source) != config["pins"][str(source)]
        or sha(source) != report["source_sha256"]
        or sha(registry) != report["routes_sha256"]
    ):
        raise ValueError("Mismatched prepared muscle or named force routes")
    anatomical = json.loads(registry.read_text())
    if sha(Path(argv[2])) != anatomical["identities"]["myosim"]:
        raise ValueError("Different named actuator payload")
    data = source.read_bytes()
    new = bytearray(data)
    nr = struct.unpack_from("<I", data, 12)[0]
    records = np.frombuffer(data, "<u4", 8 * nr, 64).reshape(-1, 8)
    metadata = json.loads(source.with_suffix(".manifest.json").read_text())
    if metadata["payload"]["sha256"] != report["source_sha256"]:
        raise ValueError("Changed parent source manifest")
    registrations = {}
    for sid, suffix in ((137, ""), (138, "_l")):
        x, faces, slots, w, old, ids, vo = tissue(data, sid)
        (record,) = records[records[:, 6] == sid]
        candidate = fit / f"{sid}-fitted.npz"
        if sha(candidate) != report["banks"][str(sid)]["candidate_sha256"]:
            raise ValueError("Donor rows differ from the retained fit")
        with np.load(candidate, allow_pickle=False) as saved:
            if int(saved["first_binding"]) != int(record[0]) or not np.array_equal(
                ids, saved["ids"]
            ):
                raise ValueError("Source allocation differs")
            rows = saved["binding"].copy()
        if rows.shape != (len(old), 9):
            raise ValueError("Binding bank length differs")
        # The inverse transforms of all original named route sites must still
        # give the same source material locations after binary32 serialization.
        sites = [
            s
            for digit in range(2, 6)
            for s in anatomical["digits"][f"EDC{digit}{suffix}"]["route"]
        ]
        errors = []
        modified = 0
        for k, (q, row) in enumerate(zip(old, rows)):
            b = int(row[0])
            val = tuple(map(float, row[1:]))
            if (
                b != q[0]
                or val[7] != q[8]
                or not np.all(np.isfinite(val))
                or abs(np.linalg.norm(val[3:7]) - 1) > 1e-5
            ):
                raise ValueError("Donor, scale or rotation admission failed")
            packed = struct.pack("<I8f", b, *val)
            offset = 64 + 32 * nr + 36 * (int(record[0]) + k)
            modified += packed != data[offset : offset + 36]
            new[offset : offset + 36] = packed
            for s in sites:
                if s["body"] != b:
                    continue
                local = np.asarray(s["local_m"])
                src = Rotation.from_quat(q[4:8]).inv().apply(local - q[1:4]) / q[8]
                error = np.linalg.norm(
                    Rotation.from_quat(val[3:7]).apply(src) * val[7] + val[:3] - local
                )
                errors.append(float(error))
        if len(errors) != len(sites) or max(errors) > 1e-6 or not modified:
            raise ValueError("Named sites failed or no repair reached this bank")
        registrations[sid] = dict(
            modified_bindings=modified,
            named_route_sites=len(sites),
            maximum_site_error_m=max(errors),
            candidate_sha256=sha(candidate),
        )
    target = out / source.name
    target.write_bytes(new)
    if (
        len(new) != len(data)
        or new[64 + 32 * nr + 36 * metadata["payload"]["binding_count"] :]
        != data[64 + 32 * nr + 36 * metadata["payload"]["binding_count"] :]
    ):
        raise ValueError("Material coordinates, weights, normals or faces changed")
    metadata["payload"]["sha256"] = sha(target)
    metadata["source"]["supported_extensor_route_registration"] = dict(
        parent_payload_sha256=sha(source),
        paired_pose_sha256=report["frames"],
        source_member_coordinates_preserved=True,
        rotation_pivots="named MyoSim route sites in donor body coordinates",
        target_role="registered passive extensor envelopes",
        regions=registrations,
    )
    target.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    argv[index] = str(target)
    config.update(
        argv=argv,
        pins={
            **config["pins"],
            str(target): sha(target),
            str(target.with_suffix(".manifest.json")): sha(
                target.with_suffix(".manifest.json")
            ),
        },
        parent_launch=sha(control / "launch.json"),
    )
    # Carry the source-owned parent maps only after proving every separately
    # bound tissue retained its exact binding span and vertex/index allocation.
    from stage_bound_assets import stage

    ri = argv.index("--resting-anatomy-receipt") + 1
    old_receipt = Path(argv[ri])
    receipt = json.loads(old_receipt.read_text())
    parent = receipt["passive_tissue_parent_bindings"]
    if parent["child_payload_sha256"] != sha(source):
        raise ValueError("Parent tissue receipt is stale")
    unchanged = []
    for info in parent["surfaces"]:
        sid = info["stable_id"]
        (record,) = records[records[:, 6] == sid]
        begin = 64 + 32 * nr + 36 * int(record[0])
        end = begin + 36 * int(record[1])
        if sid in registrations or new[begin:end] != data[begin:end]:
            raise ValueError(
                "Existing physical parent bindings would require recompilation"
            )
        unchanged.append(sid)
    parent["child_payload_sha256"] = sha(target)
    provenance = receipt["provenance"]["native_muscle_surfaces"]
    if (
        provenance["sha256"] != sha(source)
        or Path(provenance["payload_path"]).resolve() != source.resolve()
    ):
        raise ValueError("Native anatomy refers to a different source muscle program")
    provenance.update(
        sha256=sha(target),
        payload_path=str(target),
        manifest_path=str(target.with_suffix(".manifest.json")),
        manifest_sha256=sha(target.with_suffix(".manifest.json")),
    )
    receipt["provenance"]["supported_extensor_registration"] = dict(
        parent=sha(old_receipt),
        source=sha(source),
        candidate=sha(target),
        unchanged_parent_bound_surfaces=unchanged,
        edited_passive_banks=sorted(registrations),
    )
    dest = out / "resting-anatomy-receipt.json"
    with dest.open("w") as stream:
        json.dump(receipt, stream, separators=(",", ":"))
        stream.write("\n")
    argv[ri] = str(dest)
    config["pins"][str(dest)] = sha(dest)
    for name, digest in stage(old_receipt, dest).items():
        config["pins"][str(out / name)] = digest
    (out / "launch.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "registration.json").write_text(
        json.dumps(
            metadata["source"]["supported_extensor_route_registration"], indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    install(*(Path(v).resolve() for v in sys.argv[1:]))
