"""Run the existing vascular fixture against the cached and serial JVPs."""
import difflib
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SRC = Path.home() / "Work/skin-fat-native-integration-209"
BUILD = SRC / "build-native"

source = SRC / "matter/tools/vascular_cavity_check.mm"
trial = HERE / source.name
original = source.read_text()
header = original.index("    const auto cooperativePipeline")
footer = original.index("            ++cooperativeCases;", header)
text = original[header:footer]
text = text.replace("    unsigned cooperativeCases = 0;", '''    const auto cachePipeline = pipeline(device, library,
        @"numi_matter_metal::nm_vascular_prepare_cavity_operator_geometry");
    unsigned cooperativeCases = 0;''')
text = text.replace("        for (unsigned probe = 0; probe < 4; ++probe) {", '''        const auto cache = [device
            newBufferWithLength:envs * faces.size() * 9u * sizeof(nm_float4)
            options:MTLResourceStorageModePrivate];
        need(cache != nil, "cavity geometry cache allocation failed");
        const auto cacheHealth = gpuBuffer(device, std::vector<NMMatterStatusGPU>(envs));
        auto cacheCommand = [queue commandBuffer];
        auto cacheEncoder = [cacheCommand computeCommandEncoder];
        [cacheEncoder setComputePipelineState:cachePipeline];
        [cacheEncoder setBuffer:dispatchBuffer offset:0 atIndex:0];
        [cacheEncoder setBuffer:graphCase offset:0 atIndex:1];
        [cacheEncoder setBuffer:cavitiesCase offset:0 atIndex:2];
        [cacheEncoder setBuffer:facesCase offset:0 atIndex:3];
        [cacheEncoder setBuffer:nodesCase offset:0 atIndex:4];
        [cacheEncoder setBuffer:nodesCase offset:0 atIndex:5];
        [cacheEncoder setBuffer:cache offset:0 atIndex:6];
        [cacheEncoder setBuffer:cacheHealth offset:0 atIndex:7];
        [cacheEncoder dispatchThreadgroups:MTLSizeMake(envs*graph.cavities.x,1,1)
            threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [cacheEncoder endEncoding];
        completed(cacheCommand);
        for (unsigned probe = 0; probe < 4; ++probe) {''')
text = text.replace("results[2], statuses[2]", "results[3], statuses[3]")
text = text.replace("algorithm < 2", "algorithm < 3")
text = text.replace("                if (algorithm == 0u)\n", '''                if (algorithm != 0u) {
                    const unsigned useCache = algorithm == 2u ? 1u : 0u;
                    [encoder setBuffer:cache offset:0 atIndex:28];
                    [encoder setBytes:&useCache length:sizeof(useCache) atIndex:29];
                }
                if (algorithm == 0u)
''')
text = text.replace('''                need(std::memcmp(results[0].contents, results[1].contents,
                    seeds.size()*sizeof(nm_float4)) == 0,
                    "cooperative cavity JVP differs from serial words/canaries");
                need(std::memcmp(statuses[0].contents, statuses[1].contents,
                    envs*sizeof(NMMatterStatusGPU)) == 0,
                    "cooperative cavity status differs on valid input");''', '''                for (unsigned algorithm = 1; algorithm < 3; ++algorithm) {
                    need(std::memcmp(results[0].contents, results[algorithm].contents,
                        seeds.size()*sizeof(nm_float4)) == 0,
                        "cooperative or cached cavity JVP differs from serial words/canaries");
                    need(std::memcmp(statuses[0].contents, statuses[algorithm].contents,
                        envs*sizeof(NMMatterStatusGPU)) == 0,
                        "cooperative or cached cavity status differs on valid input");
                }''')
patched = original[:header] + text + original[footer:]
trial.write_text(patched)
patch = difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                            "a/matter/tools/" + source.name, "b/matter/tools/" + source.name)
(HERE / "vascular-cavity-check.patch").write_text("".join(patch))
commands = subprocess.check_output(["ninja", "-t", "commands",
    "numi-matter-vascular-cavity-check"], cwd=BUILD, text=True).splitlines()
compile_argv = shlex.split(next(line for line in commands
    if "-c " in line and str(source) in line))
link_argv = shlex.split(commands[-1])[2:-2]
def path(arg):
    if arg == str(source):
        return str(trial)
    if arg.endswith("vascular_cavity_check.mm.o") or arg == "matter/numi-matter-vascular-cavity-check":
        return str(HERE / Path(arg).name)
    if arg.startswith("-DNUMI_MATTER_METALLIB="):
        return f'-DNUMI_MATTER_METALLIB="{HERE / "NumiMatter.metallib"}"'
    return arg
chain = ([path(arg) for arg in compile_argv] + ["-iquote", str(source.parent)],
         [path(arg) for arg in link_argv])
(HERE / "fixture-build-commands.json").write_text(json.dumps(chain, indent=2) + "\n")
for argv in chain:
    subprocess.run(argv, cwd=BUILD, check=True)
fixture = HERE / "numi-matter-vascular-cavity-check"
subprocess.run([str(fixture), "--block-preconditioner-only", sys.argv[1]], cwd=BUILD, check=True)
