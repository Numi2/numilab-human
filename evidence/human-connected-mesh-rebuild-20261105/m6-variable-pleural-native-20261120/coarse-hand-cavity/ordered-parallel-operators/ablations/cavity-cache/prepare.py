"""Prepare an opt-in Newton-linearization cavity geometry cache."""
import difflib
from pathlib import Path

TRIAL = Path(__file__).resolve().parent
WORK = TRIAL.parents[2]
SOURCE = Path.home() / "Work/skin-fat-native-integration-209"
RUNTIME = WORK / "resting-clearance-and-atrium-017/guarded-runtime-admission-build"

def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

def write(source, path, label, update):
    before = source.read_text()
    after = update(before)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(after)
    patch = difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                 f"a/{label}", f"b/{label}")
    (TRIAL / (path.name + ".patch")).write_text("".join(patch))

def shader(text):
    marker = "// Shared face algebra for serial and cooperative cavity JVPs."
    text = replace(text, marker, """// Immutable geometry of one face at the current Newton linearization.
struct NMVascularCavityCachedFace {
    MRCompensatedPositionGPU a, b, c, crossBC;
    MRCompensatedScalar volume;
};
static_assert(sizeof(NMVascularCavityCachedFace) == 9u * sizeof(float4));

inline MRCompensatedScalar vascularCavityFaceTangentWords(
    MRCompensatedPositionGPU a, MRCompensatedPositionGPU b,
    MRCompensatedPositionGPU c, MRCompensatedPositionGPU crossBC,
    MRCompensatedPositionGPU da, MRCompensatedPositionGPU db,
    MRCompensatedPositionGPU dc) {
    const auto derivativeCross = mrCompensatedVectorAdd(
        mrCompensatedVectorCross(db, c), mrCompensatedVectorCross(b, dc));
    return mrCompensatedDivide(mrCompensatedAdd(
        mrCompensatedVectorDot(da, crossBC),
        mrCompensatedVectorDot(a, derivativeCross)), {6.0f, 0.0f});
}

""" + marker)
    start = text.index("        if (derivative) {\n", text.index(marker))
    end = text.index("    return {faceVolume, faceTangent};", start)
    text = text[:start] + """        if (derivative) {
            faceTangent = vascularCavityFaceTangentWords(a, b, c, crossBC, da, db, dc);
        }
""" + text[end:]
    marker = "kernel void nm_vascular_operator_cooperative(NM_VASCULAR_EQUATION_ARGUMENTS) {"
    start = text.index("// One SIMD group per equation.", text.index("kernel void nm_vascular_operator("))
    text = text[:start] + """// Fill the cache before encoding the Arnoldi applications for this candidate.
kernel void nm_vascular_prepare_cavity_operator_geometry(
    constant NMMatterDispatchGPU& dispatch [[buffer(0)]],
    constant NMVascularLayoutGPU& graph [[buffer(1)]],
    device const NMVascularCavityGPU* cavities [[buffer(2)]],
    device const NMVascularCavityFaceGPU* faces [[buffer(3)]],
    device const NMFEMNodeStateGPU* femCandidate [[buffer(4)]],
    device const NMFEMNodeStateGPU* femAccepted [[buffer(5)]],
    device NMVascularCavityCachedFace* cache [[buffer(6)]],
    device NMMatterStatusGPU* statuses [[buffer(7)]],
    uint global [[thread_position_in_grid]]) {
    const uint row = global / 32u;
    if (row >= dispatch.environmentCount * graph.cavities.x) return;
    const uint environment = row / graph.cavities.x;
    if (!environmentHealthy(statuses, environment)) return;
    const auto cavity = cavities[row % graph.cavities.x];
    const uint nodeBase = environment * dispatch.femNodeCount;
    const uint cacheBase = environment * graph.cavities.y;
    const auto origin = femAccepted[nodeBase + faces[cavity.faces.x].nodes.x];
    for (uint slot = global % 32u; slot < cavity.faces.y; slot += 32u) {
        const uint index = cavity.faces.x + slot;
        const uint3 n = faces[index].nodes.xyz + nodeBase;
        const auto a = vascularCavityRelativePositionWords(femCandidate[n.x], origin);
        const auto b = vascularCavityRelativePositionWords(femCandidate[n.y], origin);
        const auto c = vascularCavityRelativePositionWords(femCandidate[n.z], origin);
        const auto crossBC = mrCompensatedVectorCross(b, c);
        const auto volume = mrCompensatedDivide(
            mrCompensatedVectorDot(a, crossBC), {6.0f, 0.0f});
        cache[cacheBase + index] = {a, b, c, crossBC, volume};
    }
}

""" + text[start:]
    text = replace(text, "// No geometry, Jacobian, or physical state cache is introduced.\n" + marker,
        """// Direct and certification calls reconstruct face geometry independently.
kernel void nm_vascular_operator_cooperative(
    device const NMVascularCavityCachedFace* cache [[buffer(28)]],
    constant uint& useCache [[buffer(29)]],
    NM_VASCULAR_EQUATION_ARGUMENTS) {""")
    text = replace(text,
        """                const auto face = vascularCavityFaceVolumeWords(
                    cavityFaces[cavity.faces.x + first + lane], origin,
                    femCandidate, direction, nodeBase, microstep.time.x,
                    true, 0.0f);
                contribution = float4(face.volume.high, face.volume.low,
                    face.derivative.high, face.derivative.low);""",
        """                const uint index = cavity.faces.x + first + lane;
                NMVascularCavityGeometryWords face;
                if (useCache != 0u) {
                    const auto geometry = cache[environment * graph.cavities.y + index];
                    const uint3 n = cavityFaces[index].nodes.xyz + nodeBase;
                    const auto da = vascularCavityDirectionWords(direction[n.x], microstep.time.x);
                    const auto db = vascularCavityDirectionWords(direction[n.y], microstep.time.x);
                    const auto dc = vascularCavityDirectionWords(direction[n.z], microstep.time.x);
                    face = {geometry.volume, vascularCavityFaceTangentWords(
                        geometry.a, geometry.b, geometry.c, geometry.crossBC, da, db, dc)};
                } else {
                    face = vascularCavityFaceVolumeWords(cavityFaces[index], origin,
                        femCandidate, direction, nodeBase, microstep.time.x, true, 0.0f);
                }
                contribution = float4(face.volume.high, face.volume.low,
                    face.derivative.high, face.derivative.low);""")
    return text

def runtime(text):
    marker = "    bool vascularCooperativeCavityOperator = false;"
    text = replace(text, marker, marker + """
    bool vascularCacheCavityOperatorGeometry = false;
    id<MTLBuffer> vascularCavityOperatorGeometry = nil;""")
    marker = '        if (const char* policy = std::getenv("NUMI_MATTER_VASCULAR_BLOCK_CAVITY_SCHUR")) {'
    text = replace(text, marker, """        if (const char* policy = std::getenv("NM_VASCULAR_CACHE_CAVITY_OPERATOR_GEOMETRY")) {
            if (std::strcmp(policy, "0") != 0 && std::strcmp(policy, "1") != 0) {
                diagnostics.message = "NM_VASCULAR_CACHE_CAVITY_OPERATOR_GEOMETRY must be 0 or 1";
                return diagnostics;
            }
            candidate->vascularCacheCavityOperatorGeometry = std::strcmp(policy, "1") == 0;
            if (candidate->vascularCacheCavityOperatorGeometry &&
                (!candidate->vascularCooperativeCavityOperator || world.vascular.cavityFaces.empty())) {
                diagnostics.message = "cavity geometry caching requires the cooperative operator and cavity faces";
                return diagnostics;
            }
        }
""" + marker)
    marker = "        if (candidate->contactGeometryZeroStepReuseEnabled) {"
    text = replace(text, marker, """        if (candidate->vascularCacheCavityOperatorGeometry)
            candidate->executionFingerprint = mixFingerprint(
                candidate->executionFingerprint, 0x5641534341434831ull); // VASCACH1
""" + marker)
    marker = '            "nm_vascular_operator_cooperative",'
    text = replace(text, marker, marker + '\n            "nm_vascular_prepare_cavity_operator_geometry",')
    marker = "            // These optional scheduler pipelines have no consumers when"
    text = replace(text, marker, """            if (!candidate->vascularCacheCavityOperatorGeometry &&
                std::strcmp(name, "nm_vascular_prepare_cavity_operator_geometry") == 0)
                continue;
""" + marker)
    marker = "        candidate->vascularCompartmentCavity = uploads.one("
    text = replace(text, marker, """        if (candidate->vascularCooperativeCavityOperator) {
            const std::size_t recordCount = candidate->vascularCacheCavityOperatorGeometry
                ? environments * world.vascular.cavityFaces.size() : 1u;
            candidate->vascularCavityOperatorGeometry = privateScratch<std::array<float, 36>>(
                candidate->device, recordCount, valid, candidate->residentBytes);
        }
""" + marker)
    text = replace(text, """                    dispatchGroups32("nm_vascular_operator_cooperative",
                        vascularTotal, bindEquation);""",
        """                    dispatchGroups32("nm_vascular_operator_cooperative", vascularTotal, [&] {
                        bindEquation();
                        const std::uint32_t useCache = 0u;
                        [encoder setBuffer:state.vascularCavityOperatorGeometry offset:0u atIndex:28u];
                        [encoder setBytes:&useCache length:sizeof(useCache) atIndex:29u];
                    });""")
    marker = "            encodeCavityForces(WorkDomain::newtonAssemblyFemNodes);"
    text = replace(text, marker, marker + """
            // Cache is scoped to the current candidate and its Arnoldi ICB.
            // Direct operator checks and final physical certification stay fresh.
            if (state.vascularCacheCavityOperatorGeometry) {
                dispatchGroups32("nm_vascular_prepare_cavity_operator_geometry",
                    environments * state.vascularValue.layout.cavities.x, [&] {
                        setDispatch();
                        [encoder setBytes:&state.vascularValue.layout
                            length:sizeof(state.vascularValue.layout) atIndex:1u];
                        [encoder setBuffer:state.vascularCavities offset:0u atIndex:2u];
                        [encoder setBuffer:state.vascularCavityFaces offset:0u atIndex:3u];
                        [encoder setBuffer:state.femCandidate offset:0u atIndex:4u];
                        [encoder setBuffer:state.femAccepted offset:0u atIndex:5u];
                        [encoder setBuffer:state.vascularCavityOperatorGeometry offset:0u atIndex:6u];
                        [encoder setBuffer:state.statuses offset:0u atIndex:7u];
                    });
            }""")
    marker = "                        std::uint32_t currentBodyCount;\n                        std::uint32_t bodyStride;"
    text = replace(text, marker, "                        std::uint32_t useVascularGeometryCache;\n" + marker)
    marker = "                        0u, 0u, request.humanImplicitSpringCount, 1u,\n"
    text = replace(text, marker, marker + "                        state.vascularCacheCavityOperatorGeometry ? 1u : 0u,\n")
    marker = """                        if (state.vascularCooperativeCavityOperator) {
                            [cmd setBarrier];"""
    text = replace(text, marker, """                        if (state.vascularCooperativeCavityOperator) {
                            bind(cmd, 28u, state.vascularCavityOperatorGeometry);
                            scalar(cmd, 29u, offsetof(OperatorICBConstants, useVascularGeometryCache));
                            [cmd setBarrier];""")
    return text

write(RUNTIME / "runtime.mm", TRIAL / "runtime.mm", "matter/src/runtime.mm", runtime)
write(RUNTIME / "app-runtime.mm", TRIAL / "app-runtime.mm", "matter/src/runtime.mm", runtime)
write(SOURCE / "matter/src/metal/vascular.metalinc",
      TRIAL / "metal/vascular.metalinc", "matter/src/metal/vascular.metalinc", shader)
