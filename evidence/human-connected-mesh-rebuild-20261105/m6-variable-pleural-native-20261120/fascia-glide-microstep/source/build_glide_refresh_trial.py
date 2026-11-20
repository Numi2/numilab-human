from pathlib import Path
import difflib
import json
import os
import shlex
import subprocess
import sys

root = Path.cwd()
build = root / "glide-refresh-build"
build.mkdir(exist_ok=True)
checkout = Path.home() / "Work/skin-fat-native-integration-209"
baseline = root.parent / "resting-clearance-and-atrium-007"
shader_base = root.parent / "resting-clearance-and-atrium-012/residual-bank-shader"
out_shader = build / "shader"
runtime_base = baseline / "shared-fascia-pressure-build/runtime.mm"
adapter_base = checkout / "matter/src/numi_human.mm"

def apply(source, updates):
    for before, after in updates:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    return source

def publish(base, name, updated, out):
    original = base.read_text()
    (root / (base.stem + "-glide-refresh.patch")).write_text("".join(
        difflib.unified_diff(original.splitlines(keepends=True),updated.splitlines(keepends=True),
                             "a/" + name,"b/" + name)))
    if out.exists(): out.unlink()
    out.write_text(updated)

if sys.argv[1] == "shader":
    out_shader.mkdir(exist_ok=True)
    for path in (shader_base/"src").rglob("*"):
        target = out_shader/path.relative_to(shader_base)
        if path.is_dir(): target.mkdir(exist_ok=True,parents=True)
        elif not target.exists() and path.is_file(): os.link(path.resolve(),target)
    kernel_path = shader_base/"src/metal/numi_human.metalinc"
    kernels = (root/"glide-refresh-kernels.inc").read_text()
    kernels = apply(kernels, [
        ("    uint row [[thread_position_in_grid]]",
         "    device float4* previousForce [[buffer(13)]],\n    uint row [[thread_position_in_grid]]"),
        ("    if (micro.microtick == 0u) rowImpulses[row] = {};",
         "    if (micro.microtick == 0u) {\n        rowImpulses[row] = {};\n        previousForce[row] = float4(0);\n    }"),
        ("    // Admission covers an exact one-row-per-node domain and all other nodal\n"
         "    // loads are absent for these rows. A direct assignment avoids atomics.",
         "    // Update only the plane term, preserving the other loads at this node."),
        ("    externalForces[node] = float4(force,0.0f);",
         "    externalForces[node].xyz += force-previousForce[row].xyz;\n"
         "    previousForce[row] = float4(force,0.0f);"),
    ])
    updated = kernel_path.read_text() + "\nnamespace numi_matter_metal {\n" + kernels + "}\n"
    publish(kernel_path,"matter/src/metal/numi_human.metalinc",updated,
            out_shader/kernel_path.relative_to(shader_base))
    commands = json.loads((shader_base/"build-commands.json").read_text())
    commands = [[x.replace(str(shader_base)+"/",str(out_shader)+"/") for x in cmd] for cmd in commands]
    with (root/"glide-refresh-shader.log").open("wb") as log:
        for cmd in commands: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    Path(commands[0][-1]).unlink()
else:
    assemble = (root/"glide-adapter-setup.inc").read_text()
    adapter = apply(adapter_base.read_text(), [
        ("namespace numi::matter {", '#include "human_glide_refresh.hpp"\n\nnamespace numi::matter {'),
        ("    __strong id<MTLBuffer> femBodyContactWrenchBuffer = nil;",
         "    __strong id<MTLBuffer> femBodyContactWrenchBuffer = nil;\n"
         "    id<MTLComputePipelineState> glideRefreshPipeline = nil, glideWrenchPipeline = nil;\n"
         "    id<MTLBuffer> glidePreviousForce = nil, glideRowImpulses = nil;"),
        ("        if (!(state_->femBodyContactSolverSamplesCount() == 0u) &&\n",
         assemble+"        if (!refreshGlides && state_->femBodyContactSolverSamplesCount() != 0u &&\n"),
        ("        if (captureRequested) {\n            id<MTLBuffer> sourceQ",
         "        const auto encodeSourceSnapshot = [&]() -> bool {\n"
         "            if (!captureRequested) return true;\n            id<MTLBuffer> sourceQ"),
        ("            [snapshotEncoder endEncoding];\n        }\n        if (!state_->articularContactSamples.empty() &&",
         "            [snapshotEncoder endEncoding];\n            return true;\n        };\n"
         "        if (!refreshGlides && !encodeSourceSnapshot()) return false;\n"
         "        if (!state_->articularContactSamples.empty() &&"),
        ("        const auto pre = state_->runtime->encodeWithGravityScale(\n"
         "            request, pass.matterGravityScale);",
         "        humanGlideRefresh = refreshGlides ? &glidePlan : nullptr;\n"
         "        const auto pre = state_->runtime->encodeWithGravityScale(\n"
         "            request, pass.matterGravityScale);\n"
         "        humanGlideRefresh = nullptr;"),
        ("        id<MTLBuffer> reactions = (__bridge id<MTLBuffer>)\n"
         "            state_->runtime->femConstraintReactionBuffer();",
         "        if (refreshGlides && !encodeSourceSnapshot()) return false;\n"
         "        id<MTLBuffer> reactions = (__bridge id<MTLBuffer>)\n"
         "            state_->runtime->femConstraintReactionBuffer();"),
    ])
    runtime = apply(runtime_base.read_text(), [
        ("namespace numi::matter {", '#include "human_glide_refresh.hpp"\n\nnamespace numi::matter {'),
        ("        for (std::uint32_t microtick = 0u;\n             microtick < microtickCount;",
         "        const auto* glidePlan = request.phase == EncodePhase::preDynamics &&\n"
         "            humanGlideRefresh != nullptr && humanGlideRefresh->external == femExternalForces\n"
         "            ? humanGlideRefresh : nullptr;\n"
         "        for (std::uint32_t microtick = 0u;\n             microtick < microtickCount;"),
        ("            if (vascularTotal != 0u) {\n                dispatchThreads(\"nm_vascular_prepare_clock\"",
         (root/"glide-runtime-tick.inc").read_text()+
         "            if (vascularTotal != 0u) {\n                dispatchThreads(\"nm_vascular_prepare_clock\""),
        ("        // ABI-9 gathers the equal-and-opposite spring impulse",
         (root/"glide-runtime-finish.inc").read_text()+"\n"+
         "        // ABI-9 gathers the equal-and-opposite spring impulse"),
    ])
    publish(adapter_base,"matter/src/numi_human.mm",adapter,build/"numi_human.mm")
    publish(runtime_base,"matter/src/runtime.mm",runtime,build/"runtime.mm")
    commands_file = root.parent/"resting-clearance-and-atrium-008/pectoral-alpha-trace-runtime/build-commands.json"
    retained = json.loads(commands_file.read_text())[0]
    compile_db = json.loads((checkout/"build-native/compile_commands.json").read_text())
    owning = next(row for row in compile_db if row["file"] == str(adapter_base))
    link = json.loads((baseline/"shared-fascia-rebase-build/lib/link-command.json").read_text())
    jobs = []
    for name, command in (("runtime.mm",retained),("numi_human.mm",shlex.split(owning["command"]))):
        out = str(build/(name+".o"))
        command[command.index("-c")+1] = str(build/name)
        command[command.index("-o")+1] = out
        command += ["-I"+str(root),"-I"+str(checkout/"matter/src")]
        index, = (i for i,x in enumerate(link["argv"]) if x.endswith("/"+name+".o"))
        link["argv"][index] = out
        jobs.append(command)
    link["argv"][link["argv"].index("-o")+1] = str(build/"libmetalrobo.dylib")
    jobs.append(link["argv"])
    (build/"build-commands.json").write_text(json.dumps(jobs,indent=2)+"\n")
    with (root/"glide-refresh-host.log").open("wb") as log:
        for cmd in jobs:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=link["cwd"],check=True)
    for name in ("runtime.mm","numi_human.mm"): (build/(name+".o")).unlink()
