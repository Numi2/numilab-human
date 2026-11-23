"""Bind the tested detailed surface to the existing compact skin/fat bed quadrature."""

import json, sys, subprocess, runpy
from pathlib import Path
from registered_forearms import read, sha


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def install(base, owner, work, decoder, out, build):
    j = read(base / "launch.json")
    a = j["argv"]
    i = a.index("--skin-payload") + 1
    src = Path(a[i])
    new = work / "constrained.nhskin"
    api = runpy.run_path(str(decoder))
    s = api["decode"](src.read_bytes())
    t = api["decode"](new.read_bytes())
    audit = read(work / "constrained.json")
    last = audit["runs"][-1]
    if (
        sha(src) != audit["source"]
        or sha(new) != last["candidate_sha256"]
        or last["reference_max_m"] > 2e-7
        or last["min_reference_cosine"] < 0.9999
        or any(p["affected_bank_increases"] for p in last["poses"].values())
    ):
        raise ValueError("Unqualified source registration")
    for k in ("binds", "inds"):
        if s[k].tobytes() != t[k].tobytes():
            raise ValueError("Different source identity " + k)
    if s["source_sha256"] != t["source_sha256"] or any(
        s[k] != t[k] for k in ("fingerprint", "nv", "ni", "nb")
    ):
        raise ValueError("Different source indexing")
    cert = dict(
        source=sha(src),
        replacement=sha(new),
        source_registration_sha256=s["source_sha256"].hex(),
        positions=last["reference_max_m"],
        normal_cosine=last["min_reference_cosine"],
        anatomy_check=sha(work / "constrained.json"),
        changed_weight_rows=last["changed_weight_rows"],
    )
    out.mkdir()
    build.mkdir()
    for file in owner.iterdir():
        if file.is_dir():
            subprocess.run(["cp", "-cR", file, build / file.name], check=True)
        elif file.is_file():
            subprocess.run(["cp", "-c", file, build / file.name], check=True)
    header = build / "NumiHumanRestingVisual.hpp"
    h = header.read_text()
    old = f'retainedSourceSkinSHA256 == "{sha(src)}") &&'
    if h.count(old) != 1:
        raise ValueError("Different section admission source")
    header.write_text(
        h.replace(
            old,
            f'retainedSourceSkinSHA256 == "{sha(src)}" ||\n                     retainedSourceSkinSHA256 == "{sha(new)}") &&',
        )
    )
    cmds = [
        [v.replace(str(owner) + "/", str(build) + "/") for v in row]
        for row in read(owner / "build-commands.json")
    ]
    if not all(str(build / "numi-human-native.o") in cmd for cmd in cmds[2:]):
        raise ValueError("Unexpected native build")
    library = Path(j["environment"]["DYLD_LIBRARY_PATH"]) / "libmetalrobo.dylib"
    if sha(library) != j["pins"][str(library)]:
        raise ValueError("Unmatched physics library")
    for cmd in cmds[2:]:
        subprocess.run(cmd, cwd=library.parent.parent, check=True)
    save(build / "build-commands.json", cmds)
    save(
        build / "identities.json",
        {k: sha(build / k) for k in read(owner / "identities.json")},
    )
    a[i] = str(new)
    j["pins"].pop(str(src))
    j["pins"][str(new)] = sha(new)
    if any("INITIAL_PACK_CACHE" in k or "REUSE_SOURCE" in k for k in j["environment"]):
        raise ValueError("Stale geometry reuse")
    cert["owner_sha256"] = sha(build / "numi-human-native")
    save(out / "registration.json", cert)
    j["parent_launch"] = sha(base / "launch.json")
    j["compact_skin_registration"] = cert
    save(out / "launch.json", j)


if __name__ == "__main__":
    install(*(Path(p).resolve() for p in sys.argv[1:]))
