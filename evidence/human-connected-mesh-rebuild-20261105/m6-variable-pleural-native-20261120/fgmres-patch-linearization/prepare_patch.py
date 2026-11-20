"""Build an exact cached patch preconditioner on the accepted Human/Matter source."""
import difflib
import json
from pathlib import Path
import shutil
import subprocess

here = Path(__file__).resolve().parent
work = here.parent
prior = work / 'resting-clearance-and-atrium-017'
baseline = prior / 'guarded-runtime-admission-build'
metal = prior / 'contact-prefix-certification'
source = here / 'build/src'
source.mkdir(parents=True, exist_ok=True)
build = source.parent
cwd = Path.home() / 'Work/skin-fat-native-integration-209/build-native'

def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one source anchor: {old[:120]} ({text.count(old)} found)')
    return text.replace(old, new, 1)

def write_patch(original, updated, name, label):
    patch = difflib.unified_diff(original.splitlines(keepends=True),updated.splitlines(keepends=True),
                                fromfile='a/'+label,tofile='b/'+label,n=0)
    (here / name).write_text(''.join(patch))

def modify_runtime(original):
    s = original
    field = '    std::uint32_t fgmresParallelBasePreconditionerMask = 7u;'
    s = replace(s, field, field + '''
    bool fgmresPatchLinearizationCacheEnabled = false;
    id<MTLBuffer> fgmresPatchLinearizationCache = nil;''')
    mask_setting = 'candidate->fgmresParallelBasePreconditionerMask=std::uint32_t'
    mask_end = s.index('\n        }\n', s.index(mask_setting)) + len('\n        }\n')
    marker = s[mask_end:s.index('\n', mask_end)]
    s = replace(s, marker, '''        if (const char* value = std::getenv("NM_FGMRES_PATCH_LINEARIZATION_CACHE")) {
            if ((std::strcmp(value,"0") != 0 && std::strcmp(value,"1") != 0) ||
                (std::strcmp(value,"1") == 0 &&
                 (!candidate->fgmresParallelBasePreconditionerEnabled ||
                  (candidate->fgmresParallelBasePreconditionerMask & 1u) == 0u ||
                  world.dispatch.femNodeCount >= (1u << 24u)))) {
                diagnostics.message = "patch linearization cache requires parallel patches and 24-bit node indices";
                return diagnostics;
            }
            candidate->fgmresPatchLinearizationCacheEnabled = std::strcmp(value,"1") == 0;
        }
''' + marker)
    marker = '            "nm_fgmres_precondition_patch_gather",'
    s = replace(s, marker, marker + '''
            "nm_fgmres_build_patch_linearization_cache",
            "nm_fgmres_precondition_cached_patch_elements",
            "nm_fgmres_precondition_cached_patch_gather",''')
    marker = '        for (const char* name : kernelNames) {\n            if (!candidate->fgmresCoarseParityDiagnostic'
    s = replace(s, marker, '''        for (const char* name : kernelNames) {
            if (!candidate->fgmresPatchLinearizationCacheEnabled &&
                (std::strcmp(name, "nm_fgmres_build_patch_linearization_cache") == 0 ||
                 std::strcmp(name, "nm_fgmres_precondition_cached_patch_elements") == 0 ||
                 std::strcmp(name, "nm_fgmres_precondition_cached_patch_gather") == 0)) continue;
            if (!candidate->fgmresCoarseParityDiagnostic''')
    marker = '        candidate->femDirection = privateScratch<nm_float4>('
    s = replace(s, marker, '''        if (candidate->fgmresPatchLinearizationCacheEnabled) {
            // One uint4 of masked node indices and one float4 diagonal per tetrahedron.
            struct PatchLinearizationEntry { nm_uint4 nodes; nm_float4 diagonal; };
            static_assert(sizeof(PatchLinearizationEntry) == 32u);
            const std::size_t entries = multiplied(world.dispatch.tetrahedronCount);
            if (!valid || entries > candidate->device.maxBufferLength / sizeof(PatchLinearizationEntry)) {
                diagnostics.message = "patch linearization cache exceeds the Metal buffer limit";
                return diagnostics;
            }
            candidate->fgmresPatchLinearizationCache = privateScratch<PatchLinearizationEntry>(
                candidate->device, entries, valid, candidate->residentBytes);
            if (!valid || candidate->fgmresPatchLinearizationCache == nil) {
                diagnostics.message = "could not allocate patch linearization cache";
                return diagnostics;
            }
            std::fprintf(stderr, "matter_patch_linearization_cache bytes=%lu\\n",
                static_cast<unsigned long>(candidate->fgmresPatchLinearizationCache.length));
        }
''' + marker)
    marker = '''                    dispatchThreads("nm_fgmres_precondition_patch_elements",tetrahedronTotal,[&] {'''
    s = replace(s, marker, '''                    const bool cachePatchLinearization =
                        state.fgmresPatchLinearizationCache != nil;
                    if (cachePatchLinearization && column == 0u) {
                        // Rebuild from the active tetrahedra and current preconditioner at
                        // restart entry. GPU completion latches guard both build and uses.
                        dispatchThreads("nm_fgmres_build_patch_linearization_cache", tetrahedronTotal, [&] {
                            setDispatch();
                            [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:1u];
                            [encoder setBuffer:state.femPreconditioned offset:0u atIndex:2u];
                            [encoder setBuffer:state.fgmresPatchLinearizationCache offset:0u atIndex:3u];
                            [encoder setBuffer:state.fgmresStates offset:0u atIndex:4u];
                        }, WorkDomain::tetrahedra);
                        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    }
                    dispatchThreads(cachePatchLinearization
                            ? "nm_fgmres_precondition_cached_patch_elements"
                            : "nm_fgmres_precondition_patch_elements", tetrahedronTotal, [&] {''')
    marker = '''                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:1u];
                        [encoder setBuffer:preconditionerInputBasis offset:preconditionerInputOffset atIndex:2u];'''
    s = replace(s, marker, '''                        [encoder setBuffer:cachePatchLinearization
                            ? state.fgmresPatchLinearizationCache : state.femTetrahedraCandidate offset:0u atIndex:1u];
                        [encoder setBuffer:preconditionerInputBasis offset:preconditionerInputOffset atIndex:2u];''')
    marker = '''                    dispatchThreads("nm_fgmres_precondition_patch_gather",femNodeTotal,[&] {
                        setDispatch();
                        [encoder setBuffer:state.femTetrahedraCandidate offset:0u atIndex:1u];'''
    s = replace(s, marker, '''                    dispatchThreads(cachePatchLinearization
                            ? "nm_fgmres_precondition_cached_patch_gather"
                            : "nm_fgmres_precondition_patch_gather", femNodeTotal, [&] {
                        setDispatch();
                        [encoder setBuffer:cachePatchLinearization
                            ? state.fgmresPatchLinearizationCache : state.femTetrahedraCandidate offset:0u atIndex:1u];''')
    return s

for name in ('runtime.mm', 'app-runtime.mm'):
    original = (baseline / name).read_text()
    edited = modify_runtime(original)
    (build / name).write_text(edited)
    write_patch(original, edited, name + '.diff', 'matter/src/runtime.mm')

# Clone only the Metal source tree. The exact compiled include closure is the
# previous production witness build with one modified file.
shutil.copytree(metal / 'src/metal', source / 'metal', dirs_exist_ok=True)
shutil.copy2(metal / 'src/matter_kernels.metal', source / 'matter_kernels.metal')
shader_path = source / 'metal/fgmres.metalinc'
original = shader_path.read_text()
begin = original.index('kernel void nm_fgmres_precondition_patch_gather(')
end = original.index('// In-context diagnostic:', begin)
gather = original[begin:end]
shared = '''struct NMFGMRESPatchLinearization {
    uint4 maskedNodes;
    float4 diagonal;
};
static_assert(sizeof(NMFGMRESPatchLinearization) == 32u);

constant uint kPatchNodeIndexMask = 0x00ffffffu;
constant uint4 kPatchComponentMasks = uint4(0x01000000u, 0x02000000u, 0x04000000u, 0x08000000u);

kernel void nm_fgmres_build_patch_linearization_cache(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device const NMTetrahedronGPU* tetrahedra [[buffer(1)]],
    device const float4* inverseDiagonal [[buffer(2)]],
    device NMFGMRESPatchLinearization* cache [[buffer(3)]],
    device const NMFGMRESStateGPU* states [[buffer(4)]],
    uint global [[thread_position_in_grid]]
) {
    if (dispatch.tetrahedronCount == 0u ||
        global >= dispatch.environmentCount * dispatch.tetrahedronCount) return;
    const uint environment = global / dispatch.tetrahedronCount;
    if (states[environment].diagnostics.z > 0.5f) return;
    const NMTetrahedronGPU tetrahedron = tetrahedra[global];
    if ((tetrahedron.identity.w & NM_OBJECT_ACTIVE) == 0u) {
        cache[global].maskedNodes.x = NM_INVALID_INDEX;
        return;
    }
    const uint nodeBase = environment * dispatch.femNodeCount;
    uint4 maskedNodes(0u);
    float4 diagonalSum(0.0f);
    for (uint local = 0u; local < 4u; ++local) {
        const uint node = tetrahedron.nodes[local];
        if (node >= dispatch.femNodeCount) continue;
        const float4 inverse = inverseDiagonal[nodeBase + node];
        const bool4 active = inverse > float4(0.0f);
        diagonalSum += select(float4(0.0f), 1.0f / max(inverse, float4(1.0e-30f)), active);
        const uint4 mask = select(uint4(0u), kPatchComponentMasks, active);
        maskedNodes[local] = node | mask.x | mask.y | mask.z | mask.w;
    }
    cache[global] = NMFGMRESPatchLinearization{maskedNodes, max(diagonalSum, float4(1.0e-30f))};
}

kernel void nm_fgmres_precondition_cached_patch_elements(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    device const NMFGMRESPatchLinearization* cache [[buffer(1)]],
    device const float4* basis [[buffer(2)]],
    device float4* scratch [[buffer(4)]],
    device const NMFGMRESStateGPU* states [[buffer(5)]],
    uint global [[thread_position_in_grid]]
) {
    if (dispatch.tetrahedronCount == 0u ||
        global >= dispatch.environmentCount * dispatch.tetrahedronCount) return;
    const uint environment = global / dispatch.tetrahedronCount;
    if (states[environment].diagnostics.z > 0.5f) return;
    const NMFGMRESPatchLinearization patch = cache[global];
    if (patch.maskedNodes.x == NM_INVALID_INDEX) return;
    const uint nodeBase = environment * dispatch.femNodeCount;
    float4 residualSum(0.0f);
    for (uint local = 0u; local < 4u; ++local) {
        const uint packed = patch.maskedNodes[local];
        const bool4 active = (uint4(packed) & kPatchComponentMasks) != uint4(0u);
        if (any(active))
            residualSum += select(float4(0.0f), basis[nodeBase + (packed & kPatchNodeIndexMask)], active);
    }
    scratch[global] = residualSum / patch.diagonal;
}

inline bool nmFGMRESPatchIsActive(device const NMTetrahedronGPU& tetrahedron) {
    return (tetrahedron.identity.w & NM_OBJECT_ACTIVE) != 0u;
}

inline bool nmFGMRESPatchIsActive(device const NMFGMRESPatchLinearization& patch) {
    return patch.maskedNodes.x != NM_INVALID_INDEX;
}

'''
generic = gather.replace('kernel void nm_fgmres_precondition_patch_gather(', 'template <typename PatchRow>\nkernel void nm_fgmres_precondition_patch_gather_impl(')
generic = generic.replace('device const NMTetrahedronGPU* tetrahedra', 'device const PatchRow* patches')
generic = generic.replace('(tetrahedra[elementBase + tet].identity.w & NM_OBJECT_ACTIVE) == 0u',
                          '!nmFGMRESPatchIsActive(patches[elementBase + tet])')
signature = gather[:gather.index(' {')].rstrip()
signature = '\n'.join(line.split(' [[')[0] + (',' if line.endswith(',') else '') if '[[' in line
                      else line for line in signature.splitlines())
def instantiate(name, row_type):
    param = signature[signature.index('('):].replace('NMTetrahedronGPU* tetrahedra', row_type+'* patches')
    return f'template [[host_name("numi_matter_metal::{name}")]]\nkernel void nm_fgmres_precondition_patch_gather_impl<{row_type}>{param};\n\n'
edited = original[:begin] + shared + generic + instantiate('nm_fgmres_precondition_patch_gather', 'NMTetrahedronGPU') + instantiate('nm_fgmres_precondition_cached_patch_gather','NMFGMRESPatchLinearization') + original[end:]
shader_path.write_text(edited)
write_patch(original, edited, 'fgmres-patch-linearization.diff', 'matter/src/metal/fgmres.metalinc')

owner_commands = json.loads((prior / 'pressure-with-abdominal-cavity-build/build-commands.json').read_text())
previous = prior / 'pressure-with-abdominal-cavity-build'
dso_compile = next(c for c in owner_commands if str(previous/'runtime.mm') in c)
dso_link = next(c for c in owner_commands if str(previous/'libmetalrobo.dylib') in c)
older_static = work / 'resting-clearance-and-atrium-007/shared-fascia-pressure-build'
app_base = work / 'resting-clearance-and-atrium-015/staged-solver-accuracy-build'
app_compile = json.loads((older_static/'build-commands.json').read_text())[0]
app_link = json.loads((app_base/'build-commands.json').read_text())[-1]

def relocated(command, source_dir, names):
    return [str(build/names[Path(p).name]) if Path(p).parent == source_dir and Path(p).name in names
            else p for p in command]

metal_commands = json.loads((metal/'build-commands.json').read_text())
metal_commands = [[str(source/'matter_kernels.metal') if p.endswith('/matter_kernels.metal')
                   else str(build/'NumiMatter.air') if p.endswith('/failure-owner.air')
                   else str(build/'NumiMatter.metallib') if p.endswith('/NumiMatter.metallib')
                   else p for p in c] for c in metal_commands]
app_link = [str(build/'app-runtime.mm.o') if p.endswith('/runtime.mm.o')
            else str(build/'numi-human-native') if p == str(app_base/'numi-human-native')
            else str(prior/'floor-scene-build'/Path(p).name) if Path(p).parent == app_base else p
            for p in app_link]
commands = [
    *metal_commands,
    relocated(dso_compile, previous, {'runtime.mm':'runtime.mm','runtime.mm.o':'runtime.mm.o'}),
    relocated(dso_link, previous, {'runtime.mm.o':'runtime.mm.o','libmetalrobo.dylib':'libmetalrobo.dylib'}),
    relocated(app_compile, older_static, {'runtime.mm':'app-runtime.mm','runtime.mm.o':'app-runtime.mm.o'}),
    app_link,
]
(here/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for shader in baseline.glob('*.metallib'):
    if shader.name != 'NumiMatter.metallib' and not (build/shader.name).exists():
        (build/shader.name).symlink_to(shader.resolve())
for command in commands:
    print(command[0],command[-1],flush=True)
    subprocess.run(command,cwd=cwd,check=True)
