"""Compare actual full-scene accepted steps, physical loads and nonlinear restrictions."""

import json, sys
from pathlib import Path
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def outcome(p):
    env = read(p / "launch.json")["environment"]
    wall = read(p / "execution.json")
    out = {
        "elapsed_s": wall["wall_s"],
        "code": wall["exit"],
        "launch": sha(p / "launch.json"),
        "library": sha(Path(env["NUMI_MATTER_METALLIB_OVERRIDE"])),
    }
    err = (p / "stderr.log").read_text().splitlines()
    std = (p / "stdout.log").read_text().splitlines()
    out["accepted_force_history"] = [
        s for s in std if s.startswith("accepted_fem_body_load_capture=accepted ")
    ]
    out["pleural_coupling"] = any(
        "resting_vascular_external_pressure=accepted_absolute_pleural_gauge" in s
        for s in std
    )
    out["installed_material_area"] = any(
        s.startswith(
            "registered_plane_owner=material_area faces=152261 nodes=76153 corners=456783"
        )
        for s in err
    )
    out["dynamic_library"] = next(
        (
            s[s.find("/Users/") :].strip()
            for s in err
            if "dyld[" in s and s.endswith("/libmetalrobo.dylib")
        ),
        None,
    )

    def records(key):
        return [json.loads(s.split("=", 1)[1]) for s in err if s.startswith(key + "=")]

    if (p / "failure.json").exists():
        j = read(p / "failure.json")
        out.update(
            accepted_steps=j["segment_start"],
            residual=j["matter_diagnostics"][0],
            matter_object=j["matter_object_index"],
            failure=sha(p / "failure.json"),
            common_coordinates_restored=j["accepted_common_coordinates"]
            == j["candidate_common_coordinates"],
        )
    elif (p / "terminal.receipt.json").exists():
        j = read(p / "terminal.receipt.json")
        out.update(
            accepted_steps=j["accepted_step"],
            accepted_s=j["accepted_time_s"],
            receipt=sha(p / "terminal.receipt.json"),
        )
    else:
        raise ValueError("Incomplete integrated run")
    if len(out["accepted_force_history"]) != out["accepted_steps"]:
        raise ValueError("Incomplete physical load history")
    out["iterations"] = []
    for row in records("fem_newton_iterate"):
        if row["stage"] == "before_contact_limits":
            out["iterations"].append(
                dict(
                    i=row["iteration"],
                    root=row["root"],
                    residual=row["fgmres"][0]["nonlinear"][1],
                    fractions={},
                )
            )
        k = out["iterations"][-1]
        if k["i"] != row["iteration"] or k["root"] != row["root"]:
            raise ValueError("Inconsistent iteration")
        k["fractions"][row["stage"]] = min(
            row["alpha"][0], min(a[0] for a in row["object_line_search"])
        )
    return out


def compare(root, base, cases):
    previous = outcome(base / "native-three")
    data = {k: outcome(root / k) for k in cases}
    control = data["control-three"]
    if (
        control["accepted_steps"] != previous["accepted_steps"]
        or control["residual"] != previous["residual"]
        or control["accepted_force_history"] != previous["accepted_force_history"]
    ):
        raise ValueError("Control does not reproduce the retained dynamics")
    for k, r in data.items():
        if not r["dynamic_library"] or not Path(r["dynamic_library"]).is_file():
            raise ValueError("Missing native library evidence for " + k)
        r["dynamic_library_sha256"] = sha(Path(r.pop("dynamic_library")))
        r["force_trace_sha256"] = (
            __import__("hashlib")
            .sha256("\n".join(r.pop("accepted_force_history")).encode())
            .hexdigest()
        )
    return dict(
        parent=sha(base / "native-three/failure.json"), matched_control=True, runs=data
    )


if __name__ == "__main__":
    root, base, out = map(Path, sys.argv[1:4])
    res = compare(root, base, sys.argv[4:])
    out.write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
