"""Build the exact trial overlay without modifying the live source or build."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
SRC = Path.home() / "Work/skin-fat-native-integration-209"
FLOOR = WORK / "resting-clearance-and-atrium-017"
PRIOR = FLOOR / "pressure-with-abdominal-cavity-build"
OLD_STATIC = WORK / "resting-clearance-and-atrium-007/shared-fascia-pressure-build"
BASE_BUILD = SRC / "build-native"
commands = lambda directory: json.loads((directory / "build-commands.json").read_text())

def relocate(command, prior, replacements):
    return [str(HERE / replacements[Path(p).name])
            if Path(p).parent == prior and Path(p).name in replacements
            else p for p in command]

src = SRC / "matter/src/matter_kernels.metal"
shader = HERE / src.name
shader.write_bytes(src.read_bytes())
shader_compile = [
    "xcrun", "-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-fno-fast-math",
    "-I", str(SRC / "matter/include"), "-I", str(SRC / "include"),
    "-I", str(SRC / "matter/src"), "-c", str(shader), "-o", str(HERE / "NumiMatter.air"),
]
shader_link = ["xcrun", "-sdk", "macosx", "metallib", str(HERE / "NumiMatter.air"),
               "-o", str(HERE / "NumiMatter.metallib")]
shared_compile, shared_link = [
    relocate(next(command for command in commands(PRIOR) if str(PRIOR / key) in command),
             PRIOR, {name: name for name in ("runtime.mm", "runtime.mm.o", "libmetalrobo.dylib")})
    for key in ("runtime.mm", "libmetalrobo.dylib")
]
app_compile = relocate(commands(OLD_STATIC)[0], OLD_STATIC,
                       {name: "app-" + name for name in ("runtime.mm", "runtime.mm.o")})
SCENE = HERE.parents[1] / "last-clearance-029/build"
app_link = relocate(commands(SCENE)[-1], SCENE,
                    {"numi-human-native": "numi-human-native"})
app_link = [str(HERE / "app-runtime.mm.o") if arg.endswith("/app-runtime.mm.o") else arg
            for arg in app_link]
chains = ((shader_compile, shader_link), (shared_compile, shared_link),
          (app_compile, app_link))
(HERE / "build-commands.json").write_text(json.dumps(chains, indent=2) + "\n")
def build(chain):
    for argv in chain:
        subprocess.run(argv, cwd=BASE_BUILD, check=True)
with ThreadPoolExecutor(max_workers=3) as executor:
    for result in executor.map(build, chains[1:] if (HERE / "NumiMatter.metallib").exists() else chains):
        pass
