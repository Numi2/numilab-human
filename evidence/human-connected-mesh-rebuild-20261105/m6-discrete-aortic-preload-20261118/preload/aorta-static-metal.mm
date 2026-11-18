#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "NumiHumanVesselWallCoupling.hpp"
#include "numi/matter/human_physiology.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace numi::matter;
void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void report(const char* label, Runtime& runtime,
            id<MTLDevice> device, id<MTLCommandQueue> queue,
            const float gravityScale) {
    const auto source = runtime.snapshotForFrozenFEMStaticLoad();
    check(source.state.available, source.state.message);
    const auto count = source.state.femNodes.size();
    const auto& snap = source.state;
    std::cout << "static_arena nodes=" << count
              << " fields=" << snap.femFields.size()
              << " topology=" << snap.femTopologyTetrahedra.size()
              << " schedulers=" << snap.schedulers.size()
              << " adaptive=" << snap.adaptive.size()
              << " material=" << snap.femMaterialState.size()
              << " material_stride=" << snap.materialStateStride
              << " parameters=" << snap.environmentParameters.size()
              << " statuses=" << snap.statuses.size() << std::endl;

    const auto size = count * sizeof(nm_float4);
    auto force = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    auto reactions = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    auto statuses = [device newBufferWithLength:sizeof(NMMatterStatusGPU)
                                        options:MTLResourceStorageModeShared];
    const auto command = [queue commandBuffer];
    check(force && reactions && statuses && command, "Metal audit allocation failed");

    const FrozenFEMStaticLoadRequest request{
        .commandBuffer = (__bridge void*)command,
        .source = &source,
        .declaredPressurePhase = source.state.vascularClock,
        .nodalResultants = (__bridge void*)force,
        .anchorReactions = (__bridge void*)reactions,
        .statuses = (__bridge void*)statuses,
        .gravityScale = gravityScale,
    };
    FrozenFEMStaticLoadCertificate certificate;
    const auto encoded = runtime.encodeFrozenFEMStaticLoad(request, certificate);
    check(encoded.encoded, encoded.message);
    [command commit];
    [command waitUntilCompleted];
    check(command.status == MTLCommandBufferStatusCompleted, "Metal load command failed");
    const auto result = *static_cast<const NMMatterStatusGPU*>(statuses.contents);
    check(result.code == NM_STATUS_SUCCESS,
          "aorta resultant invalid, status " + std::to_string(result.code));
    const auto* values = static_cast<const nm_float4*>(force.contents);
    const auto* fixed = static_cast<const nm_float4*>(reactions.contents);
    double maxForce = 0, maxAcceleration = 0, norm2 = 0;
    double netX = 0, netY = 0, netZ = 0;
    std::size_t peak = count, free = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& state = source.state.femNodes[i];
        check(state.positionAndMass.w > 0, "nonpositive wall mass");
        const double f = std::hypot(double(values[i].x),double(values[i].y),double(values[i].z));
        check(std::isfinite(f), "nonfinite resultant");
        netX += fixed[i].x;
        netY += fixed[i].y;
        netZ += fixed[i].z;
        if (state.restAndFixed.w != 0) continue;
        ++free;
        norm2 += f*f;
        maxForce = std::max(maxForce, f);
        const double acceleration = f / state.positionAndMass.w;
        if (acceleration > maxAcceleration) {
            maxAcceleration = acceleration;
            peak = i;
        }
    }
    check(free > 0 && peak < count, "missing discrete preload witness");
    const auto& p = source.state.femNodes[peak].positionAndMass;
    std::cout << std::setprecision(12)
              << "aorta_static phase=" << label << " nodes=" << count
              << " free_nodes=" << free << " peak_node=" << peak
              << " peak_x=" << p.x << " peak_y=" << p.y << " peak_z=" << p.z
              << " peak_free_force_N=" << maxForce
              << " free_resultant_L2_N=" << std::sqrt(norm2)
              << " peak_free_acceleration_m_s2=" << maxAcceleration
              << " anchor_reaction_N=" << netX << ',' << netY << ',' << netZ
              << " control_step=" << certificate.controlStep
              << " sealed_phase=" << certificate.pressurePhaseMatchesSourceClock
              << " world=" << certificate.sourcePhysicsFingerprint
              << '\n';
}

void run(const char* vascularSource, const char* metallib) {
    WorldSource source;
    source.environmentCount = 1u;
    source.frameTimestep = 0.000015625;
    source.gravity = {0.,0.,-9.80665};
    std::string error;
    check(readHumanPhysiologyNetwork(vascularSource, source.vascular, &error), error);

    MRBodyStateGPU body{};
    body.orientation.w = 1;
    auto wall = numi::human::vessel_wall_coupling::appendRegisteredAbdominalAortaWall(
        source, body, {}, 7u, body, 20u);
    auto& object = source.objects.at(wall.objectIndex);
    for (const auto& anchor : object.femHumanAttachments)
        object.femFixedNodes.push_back(anchor.node);
    object.femHumanAttachments.clear();
    object.twoWayCoupling = false;
    std::cout << "wall_reference pressure_Pa=" << wall.cavity.initialPressure
              << " length_m=" << wall.segmentLengthM
              << " analytical_axial_residual_N=" << wall.loadedReference.axialEquilibriumResidualN
              << " analytical_outer_traction_residual_Pa=" << wall.loadedReference.outerBoundaryResidualPa
              << '\n';
    const auto compiled = compileWorld(source, {.maximumRateExponent=0});
    if (!compiled.succeeded()) {
        for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
        throw std::runtime_error("could not compile discrete wall audit");
    }
    Runtime runtime;
    RuntimeConfiguration config;
    config.metallib = metallib;
    config.environmentCount = 1;
    config.captureEvents = false;
    const auto initialized = runtime.initialize(compiled.world,config);
    check(initialized.encoded, initialized.message);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    check(device != nil, "Metal unavailable");
    const auto queue = [device newCommandQueue];
    std::cout << "device=" << [device name].UTF8String << " proof_scope=static_wall_without_external_contact\n";
    report("pressure",runtime,device,queue,0);
    report("pressure_and_gravity",runtime,device,queue,1);
}
} // namespace

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        try {
            check(argc == 3, "expected the resting CVSim network and NumiMatter.metallib");
            run(argv[1],argv[2]);
            return 0;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
