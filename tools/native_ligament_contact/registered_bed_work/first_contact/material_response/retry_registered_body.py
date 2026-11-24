"""Replay the authenticated coupled native command with fresh trace destinations."""
import json
import os
from pathlib import Path
import sys
from compile_connected_joint import digest
from run_cavity_motion import execute_native

def run(previous, out, shader_build=None):
    src = previous / "launch.json"
    cfg = json.loads(src.read_bytes())
    pins = cfg["pins"]
    argv = cfg["argv"]
    env = cfg["environment"]
    required = {argv[0], env["NUMI_MATTER_METALLIB_OVERRIDE"]}
    required.update(v for v in argv + list(env.values()) if v in pins)
    for filename in sorted(required):
        if digest(Path(filename)) != pins[filename]:
            raise ValueError(f"Changed input {filename}")
    if shader_build is not None:
        build = json.loads((shader_build / "build.json").read_bytes())
        key = "NUMI_MATTER_METALLIB_OVERRIDE"
        lib = shader_build / "metal/NumiMatter.metallib"
        if build["baseline_metal"] != pins[env[key]] or build["metal"] != digest(lib):
            raise ValueError("Shader control mismatch")
        env[key] = str(lib)
        pins[str(lib)] = build["metal"]
        cfg["direction_build_sha256"] = digest(shader_build / "build.json")
    out.mkdir()
    argv[4] = str(out / "native-traces")
    for key, value in env.items():
        if value.startswith(str(previous) + "/") and value not in pins:
            env[key] = str(out / Path(value).relative_to(previous))
    cfg["parent_launch"] = digest(src)
    (out / "launch.json").write_text(json.dumps(cfg, indent=2) + "\n")
    clean = {k:v for k,v in os.environ.items() if not k.startswith(("NM_", "NUMI_"))}
    print(execute_native(argv,cfg["cwd"],clean|env,out,minimum_free_bytes=4*2**30,timeout_s=560),flush=True)

if __name__ == "__main__":
    run(*(Path(x).resolve() for x in sys.argv[1:]))
