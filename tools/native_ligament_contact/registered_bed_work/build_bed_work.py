"""Build the native supported-scene owner from its pinned parent compile commands."""

import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path


def change(path, old, new):
    data = path.read_text()
    if data.count(old) != 1:
        raise ValueError(f"{path.name}: expected one patch site for {old[:75]!r}")
    path.write_text(data.replace(old, new))


def build(parent, app, out):
    src = Path(__file__).resolve().parent
    out.mkdir(exist_ok=True)
    (out / "metalrobo").mkdir(exist_ok=True)
    for pat in ("*.hpp", "*.mm", "*.metalinc", "human_respiration.metal"):
        for f in parent.glob(pat):
            shutil.copy2(f, out / f.name)
    shp = "NumiHumanRestingSupportGeometry.hpp"
    shutil.copy2(app / "apps" / shp, out / shp)
    for name in ("ThoracicBedAudit.metalinc", "ThoracicBedAudit.hpp"):
        shutil.copy2(src / name, out / name)
    change(
        out / shp,
        "    [[nodiscard]] metalrobo::MetalNumiHumanSupportGeometryProgram program()",
        "    [[nodiscard]] id<MTLBuffer> selectedSkinVertices() const { return selectedVertices_; }\n\n    [[nodiscard]] metalrobo::MetalNumiHumanSupportGeometryProgram program()",
    )
    coupling = out / "NumiHumanRestingCoupling.hpp"
    change(
        coupling,
        "class NumiHumanRestingCoupling {\npublic:",
        '#include "ThoracicBedAudit.hpp"\n\nclass NumiHumanRestingCoupling {\npublic:\n    ThoracicBedAudit bedWorkAudit;',
    )
    change(
        coupling,
        "            bridge(validateBody);r.phase=EncodePhase::postCommit;",
        "            bedWorkAudit.encode(p,presentationCandidateBodies,presentationCandidateRespiration);\n            bridge(validateBody);r.phase=EncodePhase::postCommit;",
    )
    visual = out / "NumiHumanRestingVisual.hpp"
    tag = '        if(bedSurface)std::cout<<"resting_bed_surface=fixed_world_triangulated"'
    change(
        visual,
        tag,
        """        if(std::getenv("NUMI_HUMAN_THORACIC_BED_AUDIT_END_STEP")) {
            require(!bedSurface && !organSupportActive && !ropeConfig,
                "thoracic bed audit requires registered skin on the flat bed");
            coupled.bedWorkAudit.prepare(device,lib,skinSupport->selectedSkinVertices(),
                supportMapBuffer,supportMapOffsetBytes,anatomyParameters,
                static_cast<unsigned>(regions.size()),static_cast<unsigned>(supportMapSpan.size()));
        }
"""
        + tag,
    )
    tag = '        require(vertexCaptureBuffer&&vertexCaptureBuffer.contents,"accepted geometry staging buffer is unavailable");'
    change(
        visual, tag, "        coupled.bedWorkAudit.write(outputDirectory,step);\n" + tag
    )
    metal = out / "human_respiration.metal"
    with metal.open("a") as f:
        f.write('\n#include "ThoracicBedAudit.metalinc"\n')
    base = parent / "metalrobo/MetalRobo.metallib"
    if base.exists() and not (out / "metalrobo" / base.name).exists():
        os.link(base, out / "metalrobo" / base.name)
    commands = [
        [s.replace(str(parent), str(out)) for s in cmd]
        for cmd in json.loads((parent / "build-commands.json").read_text())
    ]
    (out / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    for cmd in commands:
        subprocess.run(cmd, cwd=app / "build-native", check=True)
    names = [
        "numi-human-native",
        "metalrobo/HumanRestingSkin.metallib",
        shp,
        coupling.name,
        visual.name,
        metal.name,
        "ThoracicBedAudit.metalinc",
        "ThoracicBedAudit.hpp",
    ]
    (out / "identities.json").write_text(
        json.dumps(
            {
                n: hashlib.file_digest((out / n).open("rb"), "sha256").hexdigest()
                for n in names
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    build(*(Path(s).resolve() for s in sys.argv[1:]))
