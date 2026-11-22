"""Build both native certification paths and exercise the shared fixed-bed arithmetic on Metal."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from material_context import sha


def build(root):
    case = root / "bed-certificate"
    original = root / "limit-witness/build"
    build = case / "build"
    build.mkdir()
    subprocess.run(["cp", "-cR", str(original/"src"), str(build/"src")], check=True)
    mixed = build/"src/metal/mixed_fem.metalinc"
    source = mixed.read_text()
    include = '#include "numi/matter/fixed_convex_contact_step.h"'
    anchor = "kernel void nm_mixed_certify("
    old = """const float floor=max(solver.contactAcceptance.x*dispatch.numericalLimits.x,
                16.0f*kEpsilon*max(maximumAbsoluteCoordinate(sample.pointAndSeparation.xyz),1.0e-12f));"""
    if source.count(old) != 2 or source.count(anchor) != 1:
        raise ValueError("Mixed certificate source changed")
    mixed.unlink()
    mixed.write_text(include + "\n" + source.replace(anchor,(case/"certificate_floor.metalinc").read_text() + "\n" + anchor)
        .replace(old, """const float floor=analyticContactCertificateFloor(contact,dispatch,pair,sample,
                environment,solver.contactAcceptance.x*dispatch.numericalLimits.x);"""))
    compiler = json.loads((root.parent/"velocity-consistent/identities.json").read_text())["commands"][0]
    commands = []
    for src, dst in [(build/"src/matter_kernels.metal", build/"NumiMatter.metallib"),
                     (case/"check_floor.metal", case/"test.metallib")]:
        air = dst.with_suffix(".air")
        cmd = compiler.copy()
        cmd[cmd.index("-c")+1] = str(src)
        cmd[cmd.index("-o")+1] = str(air)
        cmd[7:7] = ["-I", str(build)]
        commands.extend([cmd, ["xcrun", "-sdk", "macosx", "metallib", str(air), "-o", str(dst)]])
    harness = case/"check_floor.mm"
    base = (root/"plane-limit/check_bound.mm").read_text()
    if base.count("test_registered_plane_fraction") != 1:
        raise ValueError("Unexpected compute harness")
    harness.write_text(base.replace("test_registered_plane_fraction", "test_registered_plane_floor"))
    commands.append(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-framework", "Metal", "-framework",
                     "Foundation", str(harness), "-o", str(case/"check-floor")])
    for cmd in commands:
        subprocess.run(cmd,check=True)
    report = test(case)
    config = json.loads((root/"normal-row/parent/launch.json").read_text())
    key = "NUMI_MATTER_METALLIB_OVERRIDE"
    old_lib = config["environment"][key]
    if config["pins"].pop(old_lib) != sha(original/"NumiMatter.metallib"):
        raise ValueError("Unexpected parent Metal library")
    config["environment"][key] = str(build/"NumiMatter.metallib")
    config["pins"][config["environment"][key]] = sha(build/"NumiMatter.metallib")
    parent = case/"parent"
    parent.mkdir()
    (parent/"launch.json").write_text(json.dumps(config,indent=2)+"\n")
    report.update(metallib=sha(build/"NumiMatter.metallib"), parent_metallib=sha(Path(old_lib)), commands=commands)
    (case/"device.json").write_text(json.dumps(report,indent=2)+"\n")


def test(case):
    rng = np.random.default_rng(9487)
    n = 8192
    x = np.zeros((n,4),np.float32)
    x[:, :2] = rng.uniform(-1e3,1e3,(n,2))
    x[:, 2:] = rng.uniform(-1,1,(n,2)) * np.power(10.,rng.uniform(-12,1,(n,2)))
    x[:4] = [[-0.09721837,-0.8239483,1.00974851e-9,0.], [0,0,1.00974851e-9,0.],
             [5000,-5000,0,1.], [0,0,0,1.]]
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)
        x.tofile(f/"input")
        gpu=subprocess.run([str(case/"check-floor"),str(case/"test.metallib"),str(f/"input"),str(f/"out")],
                           check=True,capture_output=True,text=True)
        actual=np.fromfile(f/"out",dtype="<f4")
    eps=np.finfo(np.float32).eps
    expected=np.maximum(np.float32(1.e-9),16*eps*np.maximum(np.float32(1.e-5),abs(x[:,2])+abs(x[:,3])))
    if not np.array_equal(expected,actual):
        raise ValueError("Device does not match the Float32 signed-distance floor")
    if actual[0]!=actual[1] or actual[2]!=actual[3] or actual[2]<=actual[0]:
        raise ValueError("Lateral invariance or bed-height cancellation bound failed")
    # The physical gap must be strictly above the floor, even at the origin.
    gaps=np.array([-1.e-4,0.,actual[0],np.nextafter(actual[0],np.float32(np.inf))],dtype=np.float32)
    if list(gaps>actual[0]) != [False,False,False,True]:
        raise ValueError("Closed-contact boundary predicate changed")
    return dict(device=gpu.stdout.strip(),cases=n,exact_float32=True,
                witness_old_floor_m=float(16*eps*max(abs(x[0,:3]))),
                witness_new_floor_m=float(actual[0]), observed_gap_m=float(x[0,2]),
                translated_z_1m_floor_m=float(actual[2]))


if __name__=="__main__":
    build(Path(sys.argv[1]).resolve())
