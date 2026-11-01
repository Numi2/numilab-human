#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

int main() {
    static const std::array<SpatialMassTransportPoint, 2> parentPoints{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.85, 0.0, 0.0}}};
    static const std::array<double, 2> parentVolumes{0.002, 0.0004};
    static const std::array<double, 2> parentMasses{2.0, 0.4};
    static const std::array<std::uint32_t, 2> parentMaterials{15, 15};
    static const std::array<std::uint32_t, 2> parentDonors{1, 1};
    static const std::array<std::uint32_t, 2> parentRegions{10, 10};
    static const std::array<std::uint32_t, 3> donorOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> donorRows{1, 1};
    static const std::array<double, 2> donorMasses{2.0, 0.4};
    static const std::array<SpatialMassTransportPoint, 6> childPoints{{
        SpatialMassTransportPoint{0.10, 0.0, 0.0},
        SpatialMassTransportPoint{0.20, 0.0, 0.0},
        SpatialMassTransportPoint{0.25, 0.0, 0.0},
        SpatialMassTransportPoint{0.30, 0.0, 0.0},
        SpatialMassTransportPoint{0.80, 0.0, 0.0},
        SpatialMassTransportPoint{0.90, 0.0, 0.0}}};
    static const std::array<double, 6> childVolumes{
        0.0003, 0.0003, 0.0003, 0.0003, 0.0006, 0.0006};
    static const std::array<std::uint32_t, 6> childMaterials{15, 15, 15, 15, 15, 15};
    static const std::array<std::uint32_t, 6> childDonors{1, 1, 1, 1, 1, 1};
    static const std::array<std::uint32_t, 6> childRegions{10, 10, 10, 10, 10, 10};
    static const std::array<double, 16> densityLower=[] {
        std::array<double, 16> out{}; out.fill(999.0); return out;
    }();
    static const std::array<double, 16> densityUpper=[] {
        std::array<double, 16> out{}; out.fill(1001.0); return out;
    }();
    static const std::array<std::uint32_t, 7> childOffsets{0, 1, 2, 3, 4, 5, 6};
    static const std::array<std::uint32_t, 6> slotGroups{0, 0, 0, 0, 0, 0};
    static const std::array<double, 6> slotFractions{1, 1, 1, 1, 1, 1};
    static const std::array<std::uint32_t, 1> groupMaterial{15};
    static const std::array<std::uint32_t, 1> groupDonor{1};
    static const std::array<std::uint32_t, 1> groupRegion{10};
    static const std::array<std::uint64_t, 1> groupIdentity{0x8000000000000010ull};
    static const std::array<std::uint32_t, 2> parentStableFaces{501, 502};
    static const std::array<std::uint32_t, 2> groupParentOffsets{0, 2};
    static const std::array<std::uint32_t, 2> groupParentRows{0, 1};
    static const std::array<std::uint32_t, 2> groupFaceOffsets{0, 2};
    static const std::array<std::uint32_t, 2> groupFaceRows{501, 502};
    const std::string pin(64, 'a');

    ConservativeSpatialMassTransportInput input;
    input.parentCentroidsWorld=parentPoints;
    input.parentReferenceVolumeM3=parentVolumes;
    input.parentMassKg=parentMasses;
    input.parentMaterialIndices=parentMaterials;
    input.parentDonorBodyIndices=parentDonors;
    input.parentAnatomicalRegionIndices=parentRegions;
    input.parentSourceDonorOffsets=donorOffsets;
    input.parentSourceDonorBodyIndices=donorRows;
    input.parentSourceDonorMassKg=donorMasses;
    input.childCentroidsWorld=childPoints;
    input.childReferenceVolumeM3=childVolumes;
    input.childMaterialIndices=childMaterials;
    input.childDonorBodyIndices=childDonors;
    input.childAnatomicalRegionIndices=childRegions;
    input.materialDensityLowerKgM3=densityLower;
    input.materialDensityUpperKgM3=densityUpper;
    input.parentMassPacketSHA256=pin;
    input.parentSourceContributionSHA256=pin;
    input.childMeshSHA256=pin;
    input.regionLabelSHA256=pin;
    input.coordinateFrameSHA256=pin;
    input.maximumTransportDistanceM=1.0;
    input.maximumAllowedEdges=100;
    input.maximumAugmentations=100;

    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;
    support.slotGroupIndices=slotGroups;
    support.volumeFractions=slotFractions;
    support.groupMaterialIndices=groupMaterial;
    support.donorBodyIndices=groupDonor;
    support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=parentStableFaces;
    support.sourceParentTetOffsets=groupParentOffsets;
    support.sourceParentTetRows=groupParentRows;
    support.sourceStableFaceOffsets=groupFaceOffsets;
    support.sourceStableFaceRows=groupFaceRows;
    support.typedSupportSHA256=pin;

    ConservativeSpatialMassTransportResult result;
    std::string error;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(input, support, result, error)) {
        std::cerr << "optimization trial fixture failed: " << error << "\n";
        return 1;
    }
    const double childMass=std::accumulate(result.childMassKg.begin(),result.childMassKg.end(),0.0);
    if (result.semanticCandidateEdgeCount != 10u || result.childMassKg.size()!=6u ||
        std::abs(childMass-2.4)>1e-12 ||
        !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(input, support, result, error)) {
        std::cerr << "optimization trial replay mismatch edges=" << result.semanticCandidateEdgeCount
                  << " error=" << error << "\n";
        return 2;
    }
    std::cout << "optimized parent-K4/cut-top4 exact-capacity fixture PASS\n";
    return 0;
}
