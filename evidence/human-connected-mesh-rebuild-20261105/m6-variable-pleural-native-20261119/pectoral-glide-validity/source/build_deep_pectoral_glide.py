"""Build an isolated resting candidate with deep pectoral gliding contact."""
import difflib
import json
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
base = here.with_name("resting-clearance-and-atrium-007") / "shared-fascia-rebase-build"
build = here / "deep-pectoral-build"
build.mkdir(exist_ok=True)
source = "numilab_human_myosim_visual_probe.mm"
app = "numi-human-native"
for path in (*base.glob("*.hpp"), base / "runtime.mm.o"):
    os.link(path, build / path.name)
os.symlink(base / "metalrobo", build / "metalrobo", target_is_directory=True)
original = (base / source).read_text()
updated = original

def replace(old, new):
    global updated
    assert updated.count(old) == 1, old
    updated = updated.replace(old, new)

replace("""                if (local >= layerWidth) {
                    normal = femNormalized(normal,
                        "resting fascia projected source normal");
                    point.x -= fascia.header.thicknessMeters * normal.x;
                    point.y -= fascia.header.thicknessMeters * normal.y;
                    point.z -= fascia.header.thicknessMeters * normal.z;
                }""",
"""                // The registered muscle envelope is the deep fascia face.
                // Build the exterior face one sheet thickness outward.
                if (local < layerWidth) {
                    normal = femNormalized(normal,
                        "resting fascia projected source normal");
                    point.x += fascia.header.thicknessMeters * normal.x;
                    point.y += fascia.header.thicknessMeters * normal.y;
                    point.z += fascia.header.thicknessMeters * normal.z;
                }""")

replace("""            const bool outer=std::all_of(face.begin(),face.end(),
                [&](std::uint32_t global) {
                    return global>=region.firstNode &&
                        global<region.firstNode+layerWidth;
                });
            if(!outer) continue;""",
"""            const std::uint32_t interfaceStart = region.firstNode +
                (regionIndex < kPectoralisFasciaRegionCount ? layerWidth : 0u);
            const bool interfaceFace=std::all_of(face.begin(),face.end(),
                [&](std::uint32_t global) {
                    return global>=interfaceStart && global<interfaceStart+layerWidth;
                });
            if(!interfaceFace) continue;""")

replace('                "regional fascia has no outer gliding-contact surface");',
        '                "regional fascia has no gliding-contact surface");')
replace("""            const std::uint32_t sourceNode=region.firstNode+local;
            const std::uint32_t bodyIndex=fasciaNodeDonorBody[sourceNode];""",
"""            const std::uint32_t outerNode=region.firstNode+local;
            const std::uint32_t innerNode=outerNode+layerWidth;
            const std::uint32_t sourceNode=regionIndex < kPectoralisFasciaRegionCount
                ? innerNode : outerNode;
            const std::uint32_t bodyIndex=fasciaNodeDonorBody[sourceNode];""")
replace("""            const mr_float4 thicknessVector=femSubtract(
                points[sourceNode],points[sourceNode+layerWidth]);""",
"""            const mr_float4 thicknessVector=femSubtract(
                points[outerNode],points[innerNode]);""")

(build / source).write_text(updated)
(here / "deep-pectoral-glide.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True), updated.splitlines(keepends=True),
    "a/apps/" + source, "b/apps/" + source)))
paths = {str(base / f): str(build / f) for f in (app, app + ".o", source, "runtime.mm.o")}
commands = [[paths.get(a,a) for a in cmd] for cmd in json.loads(
    (base / "build-commands.json").read_text())]
(build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for command in commands:
    subprocess.run(command, cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",
                   check=True)
