"""Extend Matter's existing stationary FEM/plane sample, residual and matrix product."""

from pathlib import Path
import re


def once(s, a, b):
    if s.count(a) != 1:
        raise ValueError(f"Unexpected source count {s.count(a)}: {a[:100]}")
    return s.replace(a, b)


def metal(root, templates):
    fem = root / "metal/fem.metalinc"
    s = fem.read_text()
    anchor = "device const uint* sampleCapacity [[id(19)]];"
    s = once(
        s,
        anchor,
        anchor
        + """
    device const uint* planeEnabled [[id(20)]];
    device const uint4* planeFaces [[id(21)]];
    device const float4* planeMeasure [[id(22)]];
    device const uint2* planeRanges [[id(23)]];
    device const uint* planeCorners [[id(24)]];
    device float2* planeOffDiagonal [[id(25)]];""",
    )
    fem.unlink()
    fem.write_text(s)
    path = root / "metal/contact.metalinc"
    s = path.read_text()
    anchor = "kernel void nm_contact_evaluate("
    s = once(
        s, anchor, (templates / "plane_response.metalinc").read_text() + "\n" + anchor
    )
    a = s.index(anchor)
    b = s.index("\nkernel void ", a + len(anchor))
    part = s[a:b]
    part = once(
        part,
        "    uint global [[thread_position_in_grid]]",
        """    constant uint& planeLinearization [[buffer(25)]],
    device const NMFEMNodeStateGPU* acceptedPlaneNodes [[buffer(26)]],
    uint global [[thread_position_in_grid]]""",
    )
    key = "    if (separation > slop && predictedSeparation > slop) {"
    part = once(
        part,
        key,
        """    const bool surfacePlane=contact.planeEnabled[0]!=0u;
    const float3 surfaceResponse=surfacePlane
        ? registeredPlaneRow(contact,pairIndex,slop,dt,femNodes,rigid.centerAndRadius.w,
                             acceptedPlaneNodes,planeLinearization!=0) : float3(0);
    if (surfacePlane && (!all(isfinite(surfaceResponse)) || any(surfaceResponse<0.f))) {
        recordFailure(statuses[environment],NM_STATUS_NONFINITE_RESULT,pair.objectIndex,pairIndex,
                      float4(surfaceResponse,0));return;
    }
    if (separation > slop && predictedSeparation > slop && !(surfaceResponse.x>0.f)) {""",
    )
    key = "    if (mpmContact && !all(isfinite(barrier)))"
    part = once(
        part,
        key,
        """    if(surfacePlane) barrier.xy=surfaceResponse.xy;
"""
        + key,
    )
    s = s[:a] + part + s[b:]
    path.unlink()
    path.write_text(s)
    path = root / "metal/fgmres.metalinc"
    s = path.read_text()
    key = "inline float3 coupledRigidPointDirection("
    s = once(
        s,
        key,
        """inline float3 planeNeighborAction(constant NMPrimalContactArguments& contact,uint pair,device const float4* direction) {
    if(!contact.planeEnabled[0]) return float3(0);
    const uint2 range=contact.planeRanges[pair];float z=0;
    for(uint i=range.x;i<range.x+range.y;++i){
        const uint code=contact.planeCorners[i],k=code%3;
        const uint3 nodes=contact.planeFaces[code/3].xyz;
        const float2 off=contact.planeOffDiagonal[i];
        z+=dot(off,float2(direction[nodes[(k+1)%3]].z,direction[nodes[(k+2)%3]].z));
    }
    return float3(0,0,z);
}

"""
        + key,
    )
    m = re.compile(
        r"(coupledBarrierDirectionalForce\(\s*sample, direction\[global\]\.xyz - rigidDirection\) \+)"
    )
    s, n = m.subn(r"\1 planeNeighborAction(contact,pairIndex,direction) +", s)
    if n != 2:
        raise ValueError(f"Unexpected FEM applies {n}")
    path.unlink()
    path.write_text(s)


def host(source, templates):
    s = source.read_text()
    key = "namespace numi::matter {\nnamespace {"
    s = once(
        s, key, key + "\n" + (templates / "compiled_plane_inventory.hpp").read_text()
    )
    key = "    id<MTLBuffer> contactPairs = nil;"
    names = [
        "planeEnabled",
        "planeFaces",
        "planeMeasure",
        "planeRanges",
        "planeCorners",
        "planeOffDiagonal",
    ]
    s = once(
        s,
        key,
        key
        + "\n    bool referenceAreaPlane = false;\n"
        + "".join("    id<MTLBuffer> " + n + " = nil;\n" for n in names),
    )
    # Point-only sparse enumeration misses supported vertices outside the band.
    s = once(
        s,
        "        if (sparseArmijoOptions.requested) {",
        """        const char* referencePlaneRequest = std::getenv("NM_FEM_REFERENCE_AREA_PLANE");
        const bool materialAreaRequested = referencePlaneRequest && !std::strcmp(referencePlaneRequest,"1") && !world.contact.pairs.empty();
        if (sparseArmijoOptions.requested && !materialAreaRequested) {""",
    )
    key = "        candidate->contactPairs = uploads.one("
    s = once(
        s,
        key,
        """        const char* mode=std::getenv("NM_FEM_REFERENCE_AREA_PLANE");
        if(mode && std::strcmp(mode,"0") && std::strcmp(mode,"1")) {
            diagnostics.message="invalid reference-area fixed-plane option";return diagnostics;
        }
        candidate->referenceAreaPlane=mode && !std::strcmp(mode,"1") && !world.contact.pairs.empty();
        const uint32_t enabled=candidate->referenceAreaPlane?1u:0u;
        candidate->planeEnabled=uploads.one(std::span<const uint32_t>(&enabled,1),valid,candidate->residentBytes);
        if(candidate->referenceAreaPlane){
            try{
                const auto face=makePlaneSurfaceInventory(world);
                candidate->planeFaces=uploads.one(std::span<const std::array<uint32_t,4>>(face.faces),valid,candidate->residentBytes);
                candidate->planeMeasure=uploads.one(std::span<const std::array<float,4>>(face.measure),valid,candidate->residentBytes);
                candidate->planeRanges=uploads.one(std::span<const std::array<uint32_t,2>>(face.range),valid,candidate->residentBytes);
                candidate->planeCorners=uploads.one(std::span<const uint32_t>(face.corners),valid,candidate->residentBytes);
                candidate->planeOffDiagonal=privateScratch<std::array<float,2>>(candidate->device,face.corners.size(),valid,candidate->residentBytes);
                std::fprintf(stderr,"registered_plane_owner=material_area faces=%zu nodes=%zu corners=%zu\\n",face.faces.size(),face.range.size(),face.corners.size());
            }catch(const std::exception& e){diagnostics.message=e.what();return diagnostics;}
        }
"""
        + key,
    )
    key = "offset:sizeof(std::uint32_t) atIndex:19u];"
    if s.count(key) != 2:
        raise ValueError("argument layouts")
    s = s.replace(
        key,
        key
        + "\n"
        + "".join(
            f"            [primalContactEncoder setBuffer:(candidate->{name} ?: candidate->dummy) offset:0 atIndex:{i}u];\n"
            for i, name in enumerate(names, 20)
        ),
    )
    reads = (
        "".join(
            f"                if(state.{name}) [encoder useResource:state.{name} usage:MTLResourceUsageRead];\n"
            for name in names[:-1]
        )
        + """                if(state.planeOffDiagonal) [encoder useResource:state.planeOffDiagonal usage:(MTLResourceUsageRead|MTLResourceUsageWrite)];
"""
    )
    for marker in (
        "if (primalResourcesEncoder != encoder) {",
        "const auto bindArmijoPrimalContactArguments =",
    ):
        i = s.index(marker)
        j = s.index("[encoder useResource:state.contactPairs", i)
        s = s[:j] + reads + s[j:]
    key = "bindPrimalContactArguments(24u);"
    if s.count(key) != 2:
        raise ValueError("contact evaluations")
    for flag in (1, 0):
        i = s.index(key)
        s = (
            s[:i]
            + f"""bindPrimalContactArguments(24);
                    const uint32_t writePlaneTangent={flag}u;
                    [encoder setBytes:&writePlaneTangent length:sizeof(writePlaneTangent) atIndex:25u];
                    [encoder setBuffer:state.femAccepted offset:0 atIndex:26u];"""
            + s[i + len(key) :]
        )
    # The native sparse query enumerates point gaps. A loaded triangle can
    # transmit a normal force to a vertex outside the activation band.
    key = "femSparsePrivateContactEnabled =\n                                state.femArmijoSparseContactRequested;"
    s = once(
        s,
        key,
        key.replace(
            "state.femArmijoSparseContactRequested;",
            "state.femArmijoSparseContactRequested && !state.referenceAreaPlane;",
        ),
    )
    return s
