"""Compare source startup and the retained terminal state using the same Float32 audit."""
import argparse
import hashlib
import json
import mmap
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
OWNER = WORK / "resting-surface-clearance-001"
REFERENCE = WORK / "resting-clearance-and-atrium-017"
BASE = REFERENCE / "floor-native-live-parent-026"
STARTUP_SHA256 = "b9179f1e87e4bb7990762e20bcf87f209d2303f1d3735c7bda1e463b57a2b91e"
REGISTRATION_SHA256 = "6a66dcfe74b2e9fb266e2f1459dea678ae0a689717f0ef5ec48a31c84c795be8"
REGISTRATION_PREFIX = b"resting_initial_anatomical_registration="


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def load_owner():
    sys.path[:0] = [str(OWNER / "src"), str(WORK / "resting-clearance-and-atrium-005")]
    from fast_skin_intersection_audit import install_for_float32

    install_for_float32()
    owner_path = OWNER / "audit.py"
    source = owner_path.read_text()
    source = replace_once(source, """    body_poses = {b['body_index']:b for b in receipt['accepted_body_poses']}
    assert len(body_poses)==receipt['accepted_body_pose_count']
    with pack.open""", """    body_poses = {b['body_index']:b for b in receipt['accepted_body_poses']}
    assert len(body_poses)==receipt['accepted_body_pose_count']
    audit_surface(pack, output, body_poses, {
        'accepted_step': receipt['accepted_step'],
        'accepted_time_s': receipt['accepted_time_s'],
        'accepted_pack_sha256': receipt['pack_file_sha256'],
        'accepted_root_fingerprint_hex': receipt['accepted_root_fingerprint_hex'],
    })

def audit_surface(pack, output, body_poses, evidence, *, unobserved=None):
    with pack.open""")
    source = replace_once(source, """            'accepted_step':receipt['accepted_step'],
            'accepted_time_s':receipt['accepted_time_s'],
            'accepted_pack_sha256':receipt['pack_file_sha256'],
            'accepted_root_fingerprint_hex':receipt['accepted_root_fingerprint_hex'],""", """            **evidence,""")
    source = replace_once(source, "            body = body_poses[transform[9]]", """            if transform[9] not in body_poses and allow_unobserved:
                unobserved[key] = transform[9]
                continue
            body = body_poses[transform[9]]""")
    source = replace_once(source, "def primitives(mapped, body_poses, *, skin_only=False):",
        "def primitives(mapped, body_poses, *, skin_only=False, unobserved=None):\n    allow_unobserved = unobserved is not None")
    source = replace_once(source, "in primitives(mapped,body_poses):",
        "in primitives(mapped,body_poses,unobserved=unobserved):")
    definitions = {"__name__": "_startup_skin_audit_owner", "__file__": str(owner_path)}
    exec(compile(source, str(owner_path), "exec"), definitions)
    return definitions


def decompressed(archive, destination, sha):
    with archive.open("rb") as archive_input, destination.open("xb") as output:
        subprocess.run(["/opt/homebrew/bin/zstd", "-qdc"], stdin=archive_input,
                       stdout=output, check=True)
    with destination.open("rb") as source:
        assert hashlib.file_digest(source, "sha256").hexdigest() == sha


def topology(pack, body_poses, owner, unobserved=None):
    result = {}
    with pack.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for key, _, positions, faces in owner["primitives"](mapped, body_poses, unobserved=unobserved):
            assert key not in result
            # Compare the same stable bank, local canonical vertex order, and face connectivity.
            result[key] = (len(positions), hashlib.sha256(faces.astype("<u4", copy=False)).hexdigest())
    return result


def pairs(report):
    return { (row["semantic"], row["stable_id"]): set(map(tuple, row["triangle_pairs"]))
             for row in report["surfaces"] if not row["ocular_source_id"] }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    owner = load_owner()
    with (BASE / "stdout.log").open("rb") as native:
        records = [line for line in native if line.startswith(REGISTRATION_PREFIX)]
    assert len(records) == 1
    assert hashlib.sha256(records[0]).hexdigest() == REGISTRATION_SHA256
    registration = json.loads(records[0][len(REGISTRATION_PREFIX):])
    assert registration["coordinate_source"] == "source pack transformed by native initialBodies before physical stepping"
    initial_poses = {b["body_index"]: b for b in registration["body_poses"]}
    assert len(initial_poses) == len(registration["body_poses"])

    terminal_pack = BASE / "native/accepted-geometry/step-16.mrvpack"
    terminal_receipt = json.loads(terminal_pack.with_suffix(".receipt.json").read_text())
    terminal_report = json.loads((BASE / "exact-skin-surface.json").read_text())
    assert terminal_receipt["physical_endpoint"] == "accepted"
    assert terminal_receipt["base_pack_file_sha256"] == STARTUP_SHA256
    assert terminal_receipt["pack_file_sha256"] == terminal_report["accepted_pack_sha256"]
    assert terminal_receipt["accepted_root_fingerprint_hex"] == terminal_report["accepted_root_fingerprint_hex"]
    assert terminal_receipt["accepted_step"] == terminal_report["accepted_step"] == 16
    terminal_poses = {b["body_index"]: b for b in terminal_receipt["accepted_body_poses"]}
    assert len(terminal_poses) == terminal_receipt["accepted_body_pose_count"]

    startup_report = output / "startup-surface.json"
    missing_initial_bodies = {}
    with TemporaryDirectory(prefix="mrvpack-audit-", dir=output) as scratch:
        startup = Path(scratch) / "startup.mrvpack"
        terminal = Path(scratch) / "terminal.mrvpack"
        decompressed(BASE / "native/resting-human.mrvpack.zst", startup, STARTUP_SHA256)
        decompressed(Path(str(terminal_pack)+".zst"), terminal, terminal_receipt["pack_file_sha256"])
        early_topology = topology(startup, initial_poses, owner, missing_initial_bodies)
        late_topology = topology(terminal, terminal_poses, owner)
        owner["audit_surface"](startup, startup_report, initial_poses, {
            "state": "startup_before_physical_stepping",
            "startup_pack_sha256": STARTUP_SHA256,
            "initial_registration_sha256": REGISTRATION_SHA256,
        }, unobserved=missing_initial_bodies)

    startup = json.loads(startup_report.read_text())
    early, late = pairs(startup), pairs(terminal_report)
    counts = {"startup": sum(map(len, early.values())), "terminal": sum(map(len, late.values())),
              "same_pair": 0, "startup_only_pair": 0, "terminal_only_pair": 0,
              "startup_uncorrelated": 0, "terminal_uncorrelated": 0}
    by_bank = []
    for key in sorted(early.keys() | late.keys()):
        x, y = early.get(key, set()), late.get(key, set())
        comparable = (key in early and key in late and early_topology.get(key) == late_topology.get(key)
                      and early_topology.get(owner["SKIN"]) == late_topology.get(owner["SKIN"]))
        row = {"semantic": key[0], "stable_bank": key[1], "startup": len(x), "terminal": len(y),
               "same_local_connectivity": comparable}
        if comparable:
            row.update(same_pair=len(x & y), startup_only_pair=len(x - y), terminal_only_pair=len(y - x))
            for metric in ("same_pair", "startup_only_pair", "terminal_only_pair"):
                counts[metric] += row[metric]
        else:
            counts["startup_uncorrelated"] += len(x)
            counts["terminal_uncorrelated"] += len(y)
        if x or y:
            by_bank.append(row)
    assert counts["startup"] == counts["same_pair"] + counts["startup_only_pair"] + counts["startup_uncorrelated"]
    assert counts["terminal"] == counts["same_pair"] + counts["terminal_only_pair"] + counts["terminal_uncorrelated"]
    report = {"scope": "Exact Float32 nonocular skin surface triangle pairs; count is not penetration depth or physical contact coverage.",
              "accepted_time_s": terminal_report["accepted_time_s"], "counts": counts,
              "startup_skin_self_pairs": startup["self_intersections"]["count"],
              "terminal_skin_self_pairs": terminal_report["self_intersections"]["count"],
              "startup_banks_missing_native_body_pose": [
                  {"semantic": key[0], "stable_bank": key[1], "unobserved_body": body,
                   "terminal_pairs": len(late.get(key, set()))}
                  for key, body in sorted(missing_initial_bodies.items())],
              "banks": by_bank}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
