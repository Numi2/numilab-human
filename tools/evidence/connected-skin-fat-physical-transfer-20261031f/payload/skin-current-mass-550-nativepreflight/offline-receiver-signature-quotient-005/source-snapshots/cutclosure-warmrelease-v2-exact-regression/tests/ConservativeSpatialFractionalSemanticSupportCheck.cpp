#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

static ConservativeSpatialMassTransportInput makeInput() {
    static const std::array<SpatialMassTransportPoint, 2> parentCentroids{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.0, 0.0, 0.0}}};
    static const std::array<double, 2> parentVolume{5.0e-4, 5.0e-4};
    static const std::array<double, 2> parentMass{0.5, 0.5};
    static const std::array<std::uint32_t, 2> parentMaterial{15, 15};
    static const std::array<std::uint32_t, 2> parentDonor{1, 2};
    static const std::array<std::uint32_t, 2> parentRegion{10, 20};
    static const std::array<std::uint32_t, 3> sourceOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> sourceDonors{1, 2};
    static const std::array<double, 2> sourceMass{0.5, 0.5};
    static const std::array<SpatialMassTransportPoint, 1> childCentroid{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0}}};
    static const std::array<double, 1> childVolume{1.0e-3};
    static const std::array<std::uint32_t, 1> childMaterial{15};
    static const std::array<std::uint32_t, 1> childDonor{1};
    static const std::array<std::uint32_t, 1> childRegion{10};
    static const std::array<double, 16> densityLower = [] {
        std::array<double, 16> x{}; x.fill(900.0); return x; }();
    static const std::array<double, 16> densityUpper = [] {
        std::array<double, 16> x{}; x.fill(1100.0); return x; }();
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld = parentCentroids;
    in.parentReferenceVolumeM3 = parentVolume;
    in.parentMassKg = parentMass;
    in.parentMaterialIndices = parentMaterial;
    in.parentDonorBodyIndices = parentDonor;
    in.parentAnatomicalRegionIndices = parentRegion;
    in.parentSourceDonorOffsets = sourceOffsets;
    in.parentSourceDonorBodyIndices = sourceDonors;
    in.parentSourceDonorMassKg = sourceMass;
    in.childCentroidsWorld = childCentroid;
    in.childReferenceVolumeM3 = childVolume;
    in.childMaterialIndices = childMaterial;
    in.childDonorBodyIndices = childDonor;
    in.childAnatomicalRegionIndices = childRegion;
    in.materialDensityLowerKgM3 = densityLower;
    in.materialDensityUpperKgM3 = densityUpper;
    static const std::string parentHash(64, 'a'), sourceHash(64, 'b');
    static const std::string childHash(64, 'c'), regionHash(64, 'd'), frameHash(64, 'e');
    in.parentMassPacketSHA256 = parentHash;
    in.parentSourceContributionSHA256 = sourceHash;
    in.childMeshSHA256 = childHash;
    in.regionLabelSHA256 = regionHash;
    in.coordinateFrameSHA256 = frameHash;
    in.maximumTransportDistanceM = 0.1;
    in.maximumAllowedEdges = 100;
    in.maximumAugmentations = 100;
    return in;
}

static ConservativeSpatialChildSemanticSupport makeSupport() {
    static const std::array<std::uint32_t, 2> offsets{0, 2};
    static const std::array<std::uint32_t, 2> slotGroups{0, 1};
    static const std::array<std::uint32_t, 2> materials{15, 15};
    static const std::array<std::uint32_t, 2> donors{1, 2};
    static const std::array<std::uint32_t, 2> regions{10, 20};
    static const std::array<std::uint64_t, 2> identities{0x8000000000000010ull, 0x8000000000000020ull};
    static const std::array<double, 2> fractions{0.5, 0.5};
    static const std::array<std::uint32_t, 2> parentFaces{501, 601};
    static const std::array<std::uint32_t, 3> parentTetOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> parentTets{0, 1};
    static const std::array<std::uint32_t, 3> stableFaceOffsets{0, 1, 2};
    static const std::array<std::uint32_t, 2> stableFaces{501, 601};
    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets = offsets;
    support.slotGroupIndices = slotGroups;
    support.volumeFractions = fractions;
    support.groupMaterialIndices = materials;
    support.donorBodyIndices = donors;
    support.anatomicalRegionIndices = regions;
    support.derivedGroupIdentityIDs = identities;
    support.parentStableSourceFaceIdentities = parentFaces;
    support.sourceParentTetOffsets = parentTetOffsets;
    support.sourceParentTetRows = parentTets;
    support.sourceStableFaceOffsets = stableFaceOffsets;
    support.sourceStableFaceRows = stableFaces;
    static const std::string proofHash(64, 'f');
    support.typedSupportSHA256 = proofHash;
    return support;
}

static bool rejected(ConservativeSpatialMassTransportInput in,
                     ConservativeSpatialChildSemanticSupport support,
                     const char* name) {
    ConservativeSpatialMassTransportResult out;
    std::string error;
    if (assembleConservativeSpatialMassTransportWithSemanticSupport(in, support, out, error)) {
        std::cerr << "unexpected acceptance: " << name << "\n";
        return false;
    }
    return true;
}

static bool receiverSignatureQuotientControl() {
    using Point=SpatialMassTransportPoint;
    const std::vector<Point> parentCentroids{{0.0,0.0,0.0},{0.01,0.0,0.0}};
    const std::vector<double> parentVolumes{6.0e-4,4.0e-4},parentMasses{0.6,0.4};
    const std::vector<std::uint32_t> parentMaterials{15,15},parentDonors{1,1},parentRegions{10,10};
    const std::vector<std::uint32_t> sourceOffsets{0,1,2},sourceDonors{1,1};
    const std::vector<double> sourceMasses{0.6,0.4};
    const std::vector<Point> childCentroids{{0.001,0.0,0.0},{0.009,0.0,0.0},{0.015,0.0,0.0}};
    const std::vector<double> childVolumes{3.0e-4,4.0e-4,3.0e-4};
    const std::vector<std::uint32_t> childMaterials{15,15,15},childDonors{1,1,1},childRegions{10,10,10};
    std::vector<double> lower(16,900.0),upper(16,1100.0);
    const std::vector<std::uint32_t> childOffsets{0,1,2,3},slotGroups{0,0,0},groupMaterials{15},groupDonor{1},groupRegion{10};
    const std::vector<double> preference{1.0,1.0,1.0};
    const std::vector<std::uint64_t> groupIdentities{0x8000000000000abcull};
    const std::vector<std::uint32_t> parentFaces{501,502},groupParentOffsets{0,2},groupParents{0,1};
    const std::vector<std::uint32_t> groupFaceOffsets{0,2},groupFaces{501,502};
    const std::string parentHash(64,'a'),sourceHash(64,'b'),childHash(64,'c'),regionHash(64,'d'),frameHash(64,'e'),supportHash(64,'f');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=parentCentroids;in.parentReferenceVolumeM3=parentVolumes;in.parentMassKg=parentMasses;
    in.parentMaterialIndices=parentMaterials;in.parentDonorBodyIndices=parentDonors;in.parentAnatomicalRegionIndices=parentRegions;
    in.parentSourceDonorOffsets=sourceOffsets;in.parentSourceDonorBodyIndices=sourceDonors;in.parentSourceDonorMassKg=sourceMasses;
    in.childCentroidsWorld=childCentroids;in.childReferenceVolumeM3=childVolumes;in.childMaterialIndices=childMaterials;
    in.childDonorBodyIndices=childDonors;in.childAnatomicalRegionIndices=childRegions;
    in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
    in.parentMassPacketSHA256=parentHash;in.parentSourceContributionSHA256=sourceHash;in.childMeshSHA256=childHash;
    in.regionLabelSHA256=regionHash;in.coordinateFrameSHA256=frameHash;
    in.maximumTransportDistanceM=0.1;in.maximumAllowedEdges=100;in.maximumAugmentations=100;
    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;support.slotGroupIndices=slotGroups;support.volumeFractions=preference;
    support.groupMaterialIndices=groupMaterials;support.donorBodyIndices=groupDonor;support.anatomicalRegionIndices=groupRegion;
    support.derivedGroupIdentityIDs=groupIdentities;support.parentStableSourceFaceIdentities=parentFaces;
    support.sourceParentTetOffsets=groupParentOffsets;support.sourceParentTetRows=groupParents;
    support.sourceStableFaceOffsets=groupFaceOffsets;support.sourceStableFaceRows=groupFaces;
    support.typedSupportSHA256=supportHash;

    // c0 and c1 have exactly {p0,p1}; c2 has only {p1}. The first two have
    // different volumes/lower/upper bounds, so merging must sum intervals.
    const std::array<std::uint64_t,5> candidateEdges{0ull,1ull,(1ull<<32u),(1ull<<32u)|1ull,(2ull<<32u)|1ull};
    const std::array<fractional_semantic_transport_detail::SharedCapacityWarmStartFlow,2> held{{
        {0u,0u,0.3,0u,0u,0u},{1u,1u,0.4,1u,1u,0u}}};
    fractional_semantic_transport_detail::SharedCapacityWarmStart warm;
    warm.flows=held;warm.parentMassPacketSHA256=parentHash;warm.parentSourceContributionSHA256=sourceHash;
    warm.childMeshSHA256=childHash;warm.regionLabelSHA256=regionHash;warm.coordinateFrameSHA256=frameHash;
    warm.typedSupportSHA256=supportHash;warm.useExactReceiverSignatureQuotient=true;
    warm.exactCandidateNeighborEdgeKeys=candidateEdges;
    ConservativeSpatialMassTransportResult result;std::string error;
    fractional_semantic_transport_detail::SharedCapacityFlowTelemetry telemetry;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupport(in,support,result,error,&telemetry,&warm)){
        std::cerr<<"receiver-signature quotient fixture failed: "<<error<<"\n";return false;
    }
    if(telemetry.receiverSignatureGroupCount!=2u||telemetry.receiverSignatureGroupedChildRows!=1u||
       telemetry.receiverSignatureGroupedEdgeCount!=3u||
       result.allocationSolver.find("exact_receiver_signature_quotient")==std::string::npos){
        std::cerr<<"receiver signature failed to merge only the exact duplicate neighbor set\n";return false;
    }
    if(result.parentMassTransferredKg.size()!=2u||
       std::abs(result.parentMassTransferredKg[0]-0.6)>1.0e-12||
       std::abs(result.parentMassTransferredKg[1]-0.4)>1.0e-12){
        std::cerr<<"receiver quotient failed full source exhaustion\n";return false;
    }
    bool p1ToNonidenticalChild=false;double p1ToFormerWarmChild=0.0;
    for(std::size_t e=0;e<result.childParentRows.size();++e){
        const auto c=static_cast<std::uint32_t>(std::upper_bound(result.childParentOffsets.begin(),result.childParentOffsets.end(),
                          static_cast<std::uint32_t>(e))-result.childParentOffsets.begin()-1);
        const auto p=result.childParentRows[e];
        const std::uint64_t key=(static_cast<std::uint64_t>(c)<<32u)|p;
        if(!std::binary_search(candidateEdges.begin(),candidateEdges.end(),key)){
            std::cerr<<"receiver quotient disaggregation invented a source-child edge\n";return false;
        }
        if(c==2u&&p==1u)p1ToNonidenticalChild=true;
        if(c==1u&&p==1u)p1ToFormerWarmChild+=result.childParentTransportMassKg[e];
    }
    if(!p1ToNonidenticalChild||!(p1ToFormerWarmChild<0.4-1.0e-10)){
        std::cerr<<"receiver quotient failed the alternating warm-start reroute\n";return false;
    }
    for(std::size_t c=0;c<3u;++c){
        const auto d=result.childImpliedDensityKgM3[c];
        if(!std::isfinite(d)||d<900.0-1.0e-9||d>1100.0+1.0e-9){
            std::cerr<<"receiver quotient disaggregation violated a child-specific density interval\n";return false;
        }
    }
    // A deliberately incomplete candidate subset needs a residual-cut edge.
    // The exact radius/semantic scan finds p1->c2, which changes c2 from the
    // {p0} signature to {p0,p1}; the quotient must rebuild and preserve the
    // already-sent warm flow while completing all source and child bounds.
    {
        const std::array<double,2> expandedParentVolumes{2.0e-4,8.0e-4};
        const std::array<double,2> expandedParentMasses{0.2,0.8};
        const std::array<double,2> expandedDonorMasses{0.2,0.8};
        in.parentReferenceVolumeM3=expandedParentVolumes;
        in.parentMassKg=expandedParentMasses;
        in.parentSourceDonorMassKg=expandedDonorMasses;
        const std::array<std::uint64_t,3> incompleteEdges{1ull,(1ull<<32u)|1ull,(2ull<<32u)};
        const std::array<fractional_semantic_transport_detail::SharedCapacityWarmStartFlow,2> expandedHeld{{
            {0u,2u,0.15,0u,0u,0u},{1u,0u,0.25,1u,1u,0u}}};
        auto expandingWarm=warm;
        expandingWarm.flows=expandedHeld;
        expandingWarm.exactCandidateNeighborEdgeKeys=incompleteEdges;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry expansionTelemetry;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
               in,support,result,error,&expansionTelemetry,&expandingWarm)){
            std::cerr<<"receiver-signature residual-cut expansion failed: "<<error<<"\n";return false;
        }
        bool expandedEdgeSeen=false;
        for(std::size_t e=0;e<result.childParentRows.size();++e){
            const auto c=static_cast<std::uint32_t>(std::upper_bound(result.childParentOffsets.begin(),result.childParentOffsets.end(),
                              static_cast<std::uint32_t>(e))-result.childParentOffsets.begin()-1);
            if(c==2u&&result.childParentRows[e]==1u)expandedEdgeSeen=true;
        }
        if(!expandedEdgeSeen||expansionTelemetry.receiverSignatureExpansionRounds==0u||
           expansionTelemetry.receiverSignatureExpandedEdges==0u||
           result.parentMassTransferredKg!=std::vector<double>({0.2,0.8})){
            std::cerr<<"receiver-signature cut expansion did not rebuild signatures and exhaust source mass\n";return false;
        }
        if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(in,support,result,expandingWarm,error)){
            std::cerr<<"receiver-signature expanded result failed independent replay: "<<error<<"\n";return false;
        }
        std::cout<<"receiver-signature cut expansion PASS: incomplete exact subset -> one crossing edge -> complete original-row replay\n";
    }
    // Regression for a cut-closure miss: current reached signature A={p,q}
        const std::vector<Point> cutParentCentroids{{0.0,0.0,0.0},{0.01,0.0,0.0}};
        const std::vector<double> cutParentVolumes{2.0e-3,1.0e-3},cutParentMasses{2.0,1.0};
        const std::vector<std::uint32_t> cutParentMaterials{15,15},cutParentDonors{1,1},cutParentRegions{10,10};
        const std::vector<std::uint32_t> cutSourceOffsets{0,1,2},cutSourceDonors{1,1};
        const std::vector<double> cutSourceMasses{2.0,1.0};
        const std::vector<Point> cutChildCentroids{{0.005,0.0,0.0},{0.02,0.0,0.0}};
        const std::vector<double> cutChildVolumes{1.0/900.0,2.0/900.0};
        const std::vector<std::uint32_t> cutChildMaterials{15,15},cutChildDonors{1,1},cutChildRegions{10,10};
        const std::vector<std::uint32_t> cutChildOffsets{0,1,2},cutSlotGroups{0,0};
        const std::vector<double> cutPreferences{1.0,1.0};
        const std::vector<std::uint32_t> cutGroupMaterials{15},cutGroupDonors{1},cutGroupRegions{10};
        const std::vector<std::uint64_t> cutGroupIdentities{0x8000000000000100ull};
        const std::vector<std::uint32_t> cutParentFaces{501,502},cutParentTetOffsets{0,2},cutParentTets{0,1};
        const std::vector<std::uint32_t> cutStableFaceOffsets{0,2},cutStableFaces{501,502};
    // and unreached target B={q}; p is reached with residual source capacity.
    // Adding p->B makes B's signature equal A, but the fact that expanded
    // {p,q} exists says nothing about B's current cut-side membership.
    {
        auto cutIn=in;auto cutSupport=support;
        cutIn.parentCentroidsWorld=cutParentCentroids;
        cutIn.parentReferenceVolumeM3=cutParentVolumes;
        cutIn.parentMassKg=cutParentMasses;
        cutIn.parentMaterialIndices=cutParentMaterials;
        cutIn.parentDonorBodyIndices=cutParentDonors;
        cutIn.parentAnatomicalRegionIndices=cutParentRegions;
        cutIn.parentSourceDonorOffsets=cutSourceOffsets;
        cutIn.parentSourceDonorBodyIndices=cutSourceDonors;
        cutIn.parentSourceDonorMassKg=cutSourceMasses;
        cutIn.childCentroidsWorld=cutChildCentroids;
        cutIn.childReferenceVolumeM3=cutChildVolumes;
        cutIn.childMaterialIndices=cutChildMaterials;
        cutIn.childDonorBodyIndices=cutChildDonors;
        cutIn.childAnatomicalRegionIndices=cutChildRegions;
        cutIn.maximumTransportDistanceM=0.1;
        cutIn.maximumAllowedEdges=100;cutIn.maximumAugmentations=100;
        cutSupport.childOffsets=cutChildOffsets;
        cutSupport.slotGroupIndices=cutSlotGroups;
        cutSupport.volumeFractions=cutPreferences;
        cutSupport.groupMaterialIndices=cutGroupMaterials;
        cutSupport.donorBodyIndices=cutGroupDonors;
        cutSupport.anatomicalRegionIndices=cutGroupRegions;
        cutSupport.derivedGroupIdentityIDs=cutGroupIdentities;
        cutSupport.parentStableSourceFaceIdentities=cutParentFaces;
        cutSupport.sourceParentTetOffsets=cutParentTetOffsets;
        cutSupport.sourceParentTetRows=cutParentTets;
        cutSupport.sourceStableFaceOffsets=cutStableFaceOffsets;
        cutSupport.sourceStableFaceRows=cutStableFaces;
        const std::array<std::uint64_t,3> closureEdges{0ull,1ull,(1ull<<32u)|1ull};
        const std::array<fractional_semantic_transport_detail::SharedCapacityWarmStartFlow,2> closureHeld{{
            {0u,0u,1.0,0u,0u,0u},{1u,1u,1.0,1u,1u,0u}}};
        auto closureWarm=warm;closureWarm.flows=closureHeld;
        closureWarm.exactCandidateNeighborEdgeKeys=closureEdges;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry closureTelemetry;
        ConservativeSpatialMassTransportResult quotientResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
               cutIn,cutSupport,quotientResult,error,&closureTelemetry,&closureWarm)){
            std::cerr<<"current-signature cut closure regression failed: "<<error<<"\n";return false;
        }
        bool omittedCrossingEdgeRepaired=false;
        for(std::size_t e=0;e<quotientResult.childParentRows.size();++e){
            const auto child=static_cast<std::uint32_t>(std::upper_bound(
                quotientResult.childParentOffsets.begin(),quotientResult.childParentOffsets.end(),
                static_cast<std::uint32_t>(e))-quotientResult.childParentOffsets.begin()-1);
            if(child==1u&&quotientResult.childParentRows[e]==0u)omittedCrossingEdgeRepaired=true;
        }
        if(!omittedCrossingEdgeRepaired||closureTelemetry.receiverSignatureExpandedEdges==0u||
           quotientResult.parentMassTransferredKg!=std::vector<double>({2.0,1.0})){
            std::cerr<<"cut closure skipped an edge whose target signature was unreached\n";return false;
        }
        ConservativeSpatialMassTransportResult fullGraphResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(cutIn,cutSupport,fullGraphResult,error)){
            std::cerr<<"full-graph reference control failed: "<<error<<"\n";return false;
        }
        if(fullGraphResult.parentMassTransferredKg!=quotientResult.parentMassTransferredKg||
           std::abs(quotientResult.childImpliedDensityKgM3[0]-900.0)>1.0e-9||
           std::abs(quotientResult.childImpliedDensityKgM3[1]-900.0)>1.0e-9){
            std::cerr<<"cut-expanded quotient did not match full-graph feasibility\n";return false;
        }
        std::cout<<"current-signature cut closure regression PASS: reached {p,q}, unreached {q}, omitted legal p->B restored; full-graph feasibility matched\n";
    }
    // A warm extra on signature {q,r} must be released so the distinct
    // {q} receiver can meet its floor; signature aggregation cannot hide it.
    {
        const std::vector<Point> wParentCentroids{{0.0,0.0,0.0},{10.0,0.0,0.0},{0.2,0.0,0.0}};
        const std::vector<double> wParentVolumes{2.0/700.0,1.0/700.0,0.1/700.0},wParentMasses{2.0,1.0,0.1};
        const std::vector<std::uint32_t> wParentMaterials{15,15,15},wParentDonors{1,1,1},wParentRegions{10,10,10};
        const std::vector<std::uint32_t> wSourceOffsets{0,1,2,3},wSourceDonors{1,1,1};
        const std::vector<double> wSourceMasses{2.0,1.0,0.1};
        const std::vector<Point> wChildCentroids{{0.1,0.0,0.0},{0.0,0.0,0.0},{10.0,0.0,0.0}};
        const std::vector<double> wChildVolumes{1.0/500.0,1.0/500.0,0.5/500.0};
        const std::vector<std::uint32_t> wChildMaterials{15,15,15},wChildDonors{1,1,1},wChildRegions{10,10,10};
        const std::vector<std::uint32_t> wChildOffsets{0,1,2,3},wSlotGroups{0,0,0};
        const std::vector<double> wPreferences{1.0,1.0,1.0};
        const std::vector<std::uint32_t> wGroupMaterials{15},wGroupDonors{1},wGroupRegions{10};
        const std::vector<std::uint64_t> wGroupIdentities{0x8000000000000200ull};
        const std::vector<std::uint32_t> wParentFaces{501,502,503},wParentTetOffsets{0,3},wParentTets{0,1,2};
        const std::vector<std::uint32_t> wStableFaceOffsets{0,3},wStableFaces{501,502,503};
        auto wIn=in;auto wSupport=support;
        wIn.parentCentroidsWorld=wParentCentroids;wIn.parentReferenceVolumeM3=wParentVolumes;wIn.parentMassKg=wParentMasses;
        wIn.parentMaterialIndices=wParentMaterials;wIn.parentDonorBodyIndices=wParentDonors;wIn.parentAnatomicalRegionIndices=wParentRegions;
        wIn.parentSourceDonorOffsets=wSourceOffsets;wIn.parentSourceDonorBodyIndices=wSourceDonors;wIn.parentSourceDonorMassKg=wSourceMasses;
        wIn.childCentroidsWorld=wChildCentroids;wIn.childReferenceVolumeM3=wChildVolumes;
        wIn.childMaterialIndices=wChildMaterials;wIn.childDonorBodyIndices=wChildDonors;wIn.childAnatomicalRegionIndices=wChildRegions;
        std::vector<double> wDensityLower(16,900.0),wDensityUpper(16,1100.0);
        wDensityLower[15]=500.0;wDensityUpper[15]=1000.0;
        wIn.materialDensityLowerKgM3=wDensityLower;wIn.materialDensityUpperKgM3=wDensityUpper;
        wIn.maximumTransportDistanceM=0.11;wIn.maximumAllowedEdges=100;wIn.maximumAugmentations=100;
        wSupport.childOffsets=wChildOffsets;wSupport.slotGroupIndices=wSlotGroups;wSupport.volumeFractions=wPreferences;
        wSupport.groupMaterialIndices=wGroupMaterials;wSupport.donorBodyIndices=wGroupDonors;wSupport.anatomicalRegionIndices=wGroupRegions;
        wSupport.derivedGroupIdentityIDs=wGroupIdentities;wSupport.parentStableSourceFaceIdentities=wParentFaces;
        wSupport.sourceParentTetOffsets=wParentTetOffsets;wSupport.sourceParentTetRows=wParentTets;
        wSupport.sourceStableFaceOffsets=wStableFaceOffsets;wSupport.sourceStableFaceRows=wStableFaces;
        const std::array<std::uint64_t,4> wEdges{0ull,2ull,(1ull<<32u),(2ull<<32u)|1ull};
        const std::array<fractional_semantic_transport_detail::SharedCapacityWarmStartFlow,2> wHeld{{
            {0u,0u,1.5,0u,0u,0u},{1u,2u,0.5,1u,2u,0u}}};
        auto wWarm=warm;wWarm.flows=wHeld;wWarm.exactCandidateNeighborEdgeKeys=wEdges;
        ConservativeSpatialMassTransportResult wResult;std::string wError;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry wTelemetry;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
               wIn,wSupport,wResult,wError,&wTelemetry,&wWarm)){
            std::cerr<<"warm-extra floor-release regression failed: "<<wError<<"\n";return false;
        }
        if(wResult.parentMassTransferredKg.size()!=3u||
           std::abs(wResult.parentMassTransferredKg[0]-2.0)>1.0e-12||
           std::abs(wResult.parentMassTransferredKg[1]-1.0)>1.0e-12||
           std::abs(wResult.parentMassTransferredKg[2]-0.1)>1.0e-12||
           wResult.childImpliedDensityKgM3.size()!=3u){
            std::cerr<<"warm-extra release did not exhaust source and cover every receiver\n";return false;
        }
        for(const auto rho:wResult.childImpliedDensityKgM3)
            if(!std::isfinite(rho)||rho<500.0-1.0e-9||rho>1000.0+1.0e-9){
                std::cerr<<"warm-extra release violated a child density interval\n";return false;
            }
        if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(wIn,wSupport,wResult,wWarm,wError)){
            std::cerr<<"warm-extra release failed independent replay: "<<wError<<"\n";return false;
        }
        ConservativeSpatialMassTransportResult fullWarmResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(wIn,wSupport,fullWarmResult,wError)){
            std::cerr<<"full-graph warm-extra reference control failed: "<<wError<<"\n";return false;
        }
        if(fullWarmResult.parentMassTransferredKg!=wResult.parentMassTransferredKg){
            std::cerr<<"cut-expanded warm-extra result differs from full-graph feasibility\n";return false;
        }
        std::cout<<"warm-extra floor-release regression PASS: q-to-A 1.5 released for q-to-B floor; full-graph feasibility matched\n";
    }
    auto repeated=candidateEdges;repeated[1]=repeated[0];
    warm.exactCandidateNeighborEdgeKeys=repeated;
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(in,support,result,error,&telemetry,&warm)){
        std::cerr<<"receiver quotient accepted repeated candidate graph keys\n";return false;
    }
    std::cout<<"exact receiver-signature quotient PASS: 3 receivers -> 2 signatures, 5 edges -> 3, warm reroute and original child/source replay\n";
    return true;
}

int main() {
    if(!receiverSignatureQuotientControl())return 40;
    auto in = makeInput();
    auto support = makeSupport();
    ConservativeSpatialMassTransportResult result;
    std::string error;
    fractional_semantic_transport_detail::SharedCapacityFlowTelemetry telemetry;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
            in, support, result, error, &telemetry)) {
        std::cerr << "valid 50/50 case failed: " << error << "\n";
        return 1;
    }
    if (telemetry.feasibleFlowCalls == 0u || telemetry.feasibleFlow.bfsPhases == 0u ||
        telemetry.feasibleFlow.augmentations == 0u || telemetry.initialSeedEdges == 0u ||
        telemetry.feasibleFlow.totalPathEdges < telemetry.feasibleFlow.augmentations) {
        std::cerr << "Dinic telemetry did not record its feasibility work\n";
        return 20;
    }
    std::cout << "Dinic telemetry: calls=" << telemetry.feasibleFlowCalls
              << " bfs_phases=" << telemetry.feasibleFlow.bfsPhases
              << " bfs_edge_scans=" << telemetry.feasibleFlow.bfsEdgeScans
              << " path_searches=" << telemetry.feasibleFlow.blockingPathSearches
              << " dfs_edge_scans=" << telemetry.feasibleFlow.dfsEdgeScans
              << " augmentations=" << telemetry.feasibleFlow.augmentations
              << " max_path_edges=" << telemetry.feasibleFlow.maximumPathEdges << "\n";
    {
        ConservativeSpatialMassTransportResult noTelemetryReplay;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(in,support,noTelemetryReplay,error)||
           !fractional_semantic_transport_detail::sameResult(result,noTelemetryReplay)){
            std::cerr<<"default no-deadline flow arrays changed when telemetry is absent: "<<error<<"\n";return 33;
        }
        ConservativeSpatialMassTransportResult replay;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry replayTelemetry;
        if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
                in, support, replay, error, &replayTelemetry) ||
            !fractional_semantic_transport_detail::sameResult(result, replay)) {
            std::cerr << "no-deadline telemetry changed deterministic flow arrays: " << error << "\n";
            return 21;
        }
        auto limited = in;
        limited.maximumAugmentations = 1u;
        ConservativeSpatialMassTransportResult partial;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry partialTelemetry;
        if (assembleConservativeSpatialMassTransportWithSemanticSupport(
                limited, support, partial, error, &partialTelemetry) ||
            !partialTelemetry.partialFlowCaptured ||
            partialTelemetry.partialExpandedParentRows.empty() ||
            partialTelemetry.partialExpandedParentRows.size() != partialTelemetry.partialMassKg.size() ||
            partialTelemetry.partialMassKg.size() != partialTelemetry.partialChildRows.size()) {
            std::cerr << "augmentation-budget stop failed to capture aligned partial flow: " << error << "\n";
            return 22;
        }
        double partialSum = 0.0;
        for (const auto value : partialTelemetry.partialMassKg) partialSum += value;
        if (std::abs(partialSum - partialTelemetry.partialSourceSentKg) > 1.0e-12) {
            std::cerr << "partial edge ledger does not match source-sent capacity\n";
            return 23;
        }
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry budgetTelemetry;
        budgetTelemetry.feasibleFlowBudget=std::chrono::duration<double>(1.0e-12);
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(
                in,support,partial,error,&budgetTelemetry)||
           !budgetTelemetry.deadlineReached||!budgetTelemetry.partialFlowCaptured||
           budgetTelemetry.graphConstructionSeconds<0.0||budgetTelemetry.feasibleSolverSeconds<0.0){
            std::cerr<<"solver-only deadline did not start after graph construction and stop safely: "<<error<<"\n";return 34;
        }
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry expiredTelemetry;
        expiredTelemetry.deadline = std::chrono::steady_clock::now();
        if (assembleConservativeSpatialMassTransportWithSemanticSupport(
                in, support, partial, error, &expiredTelemetry) ||
            !expiredTelemetry.deadlineReached || !expiredTelemetry.partialFlowCaptured ||
            expiredTelemetry.partialExpandedParentRows.size() != expiredTelemetry.partialMassKg.size()) {
            std::cerr << "expired safe-boundary deadline was not retained as partial diagnostic: " << error << "\n";
            return 24;
        }
        std::cout << "deadline controls PASS: no-deadline parity, partial budget capture, expired deadline capture\n";

        // Safe held-flow reuse: record one bounded partial allocation, subtract it
        // from the same original source/child capacities, and solve only the residual.
        std::vector<SpatialMassTransportPoint> rp{{0,0,0},{0,0,0}}, rc{{0,0,0},{0,0,0}};
        std::vector<double> rpv{5.0e-4,5.0e-4}, rpm{0.5,0.5}, rsm{0.5,0.5};
        std::vector<std::uint32_t> rmat{15,15}, rdonor{1,2}, rregion{10,20}, roff{0,1,2}, rsd{1,2};
        std::vector<double> rcv{5.0e-4,5.0e-4};
        std::vector<std::uint32_t> rcmat{15,15}, rcdonor{1,2}, rcregion{10,20};
        auto staged=makeInput();
        staged.parentCentroidsWorld=rp;staged.parentReferenceVolumeM3=rpv;staged.parentMassKg=rpm;
        staged.parentMaterialIndices=rmat;staged.parentDonorBodyIndices=rdonor;staged.parentAnatomicalRegionIndices=rregion;
        staged.parentSourceDonorOffsets=roff;staged.parentSourceDonorBodyIndices=rsd;staged.parentSourceDonorMassKg=rsm;
        staged.childCentroidsWorld=rc;staged.childReferenceVolumeM3=rcv;staged.childMaterialIndices=rcmat;
        staged.childDonorBodyIndices=rcdonor;staged.childAnatomicalRegionIndices=rcregion;
        staged.maximumAugmentations=1u;
        std::vector<std::uint32_t> so{0,1,2}, sg{0,1}, gm{15,15}, gd{1,2}, gr{10,20};
        std::vector<std::uint64_t> gids{0x8000000000001010ull,0x8000000000002020ull};
        std::vector<double> sf{1.0,1.0};
        std::vector<std::uint32_t> psf{501,601}, pto{0,1,2}, ptr{0,1}, sfo{0,1,2}, sfr{501,601};
        auto stagedSupport=makeSupport();
        stagedSupport.childOffsets=so;stagedSupport.slotGroupIndices=sg;stagedSupport.volumeFractions=sf;
        stagedSupport.groupMaterialIndices=gm;stagedSupport.donorBodyIndices=gd;stagedSupport.anatomicalRegionIndices=gr;
        stagedSupport.derivedGroupIdentityIDs=gids;stagedSupport.parentStableSourceFaceIdentities=psf;
        stagedSupport.sourceParentTetOffsets=pto;stagedSupport.sourceParentTetRows=ptr;
        stagedSupport.sourceStableFaceOffsets=sfo;stagedSupport.sourceStableFaceRows=sfr;
        ConservativeSpatialMassTransportResult stagedResult;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry stagedTelemetry;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(staged,stagedSupport,stagedResult,error,&stagedTelemetry)||
           !stagedTelemetry.partialFlowCaptured||stagedTelemetry.partialMassKg.empty()||
           stagedTelemetry.partialExpandedParentRows.size()!=stagedTelemetry.partialChildRows.size()){
            std::cerr<<"staged held-flow run did not capture a bounded partial allocation: "<<error<<"\n";return 25;
        }
        std::vector<double> heldParent(2,0.0),heldChild(2,0.0),heldMeasure(2,0.0),heldDonor(3,0.0);
        for(std::size_t i=0;i<stagedTelemetry.partialMassKg.size();++i){
            const auto pr=stagedTelemetry.partialExpandedParentRows[i],cr=stagedTelemetry.partialChildRows[i];
            if(pr>=2u||cr>=2u||!(stagedTelemetry.partialMassKg[i]>0.0)||
               stagedTelemetry.partialDistanceM[i]>staged.maximumTransportDistanceM){
                std::cerr<<"held-flow row failed source/child/radius accounting\n";return 26;
            }
            heldParent[pr]+=stagedTelemetry.partialMassKg[i];heldChild[cr]+=stagedTelemetry.partialMassKg[i];
            heldMeasure[cr]+=stagedTelemetry.partialMeasureM3[i];heldDonor[rdonor[pr]]+=stagedTelemetry.partialMassKg[i];
        }
        if(std::abs(heldParent[0]+heldParent[1]-stagedTelemetry.partialSourceSentKg)>1e-12||
           !(heldParent[0]>0.0&&heldParent[0]<rpm[0])||!(heldParent[1]<rpm[1])||
           !(heldChild[0]>0.0&&heldChild[0]<0.5)||heldChild[1]!=0.0){
            std::cerr<<"held-flow test did not retain the expected proper partial source/child state\n";return 27;
        }
        std::vector<double> residualMass(2),residualVolume(2),residualDonorMass(2),residualChildVolume(2);
        for(std::size_t p=0;p<2;++p){
            residualMass[p]=rpm[p]-heldParent[p];residualDonorMass[p]=rsm[p]-heldParent[p];
            residualVolume[p]=rpv[p]-heldMeasure[p];
            if(!(residualMass[p]>0.0&&residualVolume[p]>0.0)||residualDonorMass[p]<0.0){
                std::cerr<<"held-flow subtraction produced invalid remaining source capacity\n";return 28;
            }
        }
        for(std::size_t c=0;c<2;++c){residualChildVolume[c]=rcv[c]-heldMeasure[c];if(!(residualChildVolume[c]>0.0)){std::cerr<<"held-flow subtraction produced invalid remaining child volume\n";return 29;}}
        auto residual=staged;residual.parentMassKg=residualMass;residual.parentReferenceVolumeM3=residualVolume;
        residual.parentSourceDonorMassKg=residualDonorMass;residual.childReferenceVolumeM3=residualChildVolume;
        residual.maximumAugmentations=100u;
        ConservativeSpatialMassTransportResult residualResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(residual,stagedSupport,residualResult,error)||
           residualResult.parentMassTransferredKg.size()!=2u||residualResult.childMassKg.size()!=2u){
            std::cerr<<"same-capacity residual solve after held-flow subtraction failed: "<<error<<"\n";return 30;
        }
        for(std::size_t p=0;p<2;++p)
            if(std::abs(heldParent[p]+residualResult.parentMassTransferredKg[p]-rpm[p])>1e-12){
                std::cerr<<"held plus residual source mass does not close original source row\n";return 31;
            }
        for(std::size_t c=0;c<2;++c){
            const double finalMass=heldChild[c]+residualResult.childMassKg[c],density=finalMass/rcv[c];
            if(density<900.0-1e-9||density>1100.0+1e-9){
                std::cerr<<"held plus residual child exceeds original density capacity\n";return 32;
            }
        }
        for(const auto& closure:residualResult.sourceDonorClosure)
            if(closure.sourceDonorBodyIndex<heldDonor.size())heldDonor[closure.sourceDonorBodyIndex]+=closure.childMassKg;
        if(std::abs(heldDonor[1]-0.5)>1e-12||std::abs(heldDonor[2]-0.5)>1e-12){
            std::cerr<<"held plus residual source-donor rows do not close original contribution masses\n";return 35;
        }
        std::cout<<"held-flow reuse PASS: parent rows, donor contributions, child capacities, and original density bounds close after residual solve\n";
    }
    if (result.childMassKg.size() != 1 || result.childMassKg[0] != 1.0 ||
        result.childImpliedDensityKgM3.size() != 1 ||
        result.childImpliedDensityKgM3[0] != 1000.0 ||
        result.childParentRows.size() != 2 || result.groupClosure.size() != 2 ||
        result.childSourceDonorBodyIndices.size() != 2 ||
        result.childSourceDonorMassKg[0] != 0.5 ||
        result.childSourceDonorMassKg[1] != 0.5 ||
        result.parentMassTransferredKg != std::vector<double>({0.5, 0.5})) {
        std::cerr << "50/50 aggregate did not conserve one real-cell mass and two source groups\n";
        return 2;
    }
    if (!verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
            in, support, result, error)) {
        std::cerr << "valid 50/50 receipt failed replay: " << error << "\n";
        return 3;
    }
    auto tampered = result;
    tampered.childParentVolumeFractions[0] += 0.01;
    if (verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
            in, support, tampered, error)) {
        std::cerr << "tampered aggregate flow was accepted\n";
        return 4;
    }

    auto badSum = support;
    static const std::array<double, 2> fractionsBadSum{0.5, 0.25};
    badSum.volumeFractions = fractionsBadSum;
    if (!rejected(in, badSum, "fraction sum")) return 5;

    auto unequalPreference = support;
    static const std::array<double, 2> fractionsDensity{0.9, 0.1};
    unequalPreference.volumeFractions = fractionsDensity;
    ConservativeSpatialMassTransportResult unequalPreferenceResult;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
            in, unequalPreference, unequalPreferenceResult, error) ||
        unequalPreferenceResult.childMassKg != std::vector<double>({1.0}) ||
        unequalPreferenceResult.childImpliedDensityKgM3 != std::vector<double>({1000.0}) ||
        unequalPreferenceResult.groupClosure.size() != 2u ||
        unequalPreferenceResult.groupClosure[0].childMassKg != 0.5 ||
        unequalPreferenceResult.groupClosure[1].childMassKg != 0.5 ||
        !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
            in, unequalPreference, unequalPreferenceResult, error)) {
        std::cerr << "P1 preference changed a shared real-child capacity or source-group conservation: "
                  << error << "\n";
        return 6;
    }
    // Nearest-one seeds deliberately produce an infeasible subgraph. A
    // residual min-cut witness must add only crossing candidates and recover
    // a valid shared-capacity flow without raising the graph budget.
    {
        static const std::array<SpatialMassTransportPoint,3> parentPoints{{
            SpatialMassTransportPoint{0.0,0.0,0.0},
            SpatialMassTransportPoint{0.9,0.0,0.0},
            SpatialMassTransportPoint{1.0,0.0,0.0}}};
        static const std::array<double,3> parentVolumes{2.5e-3,2.5e-4,2.5e-4};
        static const std::array<double,3> parentMasses{2.5,0.25,0.25};
        static const std::array<std::uint32_t,3> parentMaterials{15,15,15};
        static const std::array<std::uint32_t,3> parentDonors{1,1,1};
        static const std::array<std::uint32_t,3> parentRegions{10,10,10};
        static const std::array<std::uint32_t,4> donorOffsets{0,1,2,3};
        static const std::array<std::uint32_t,3> donorIds{1,1,1};
        static const std::array<double,3> donorMasses{2.5,0.25,0.25};
        static const std::array<SpatialMassTransportPoint,3> childPoints{{
            SpatialMassTransportPoint{0.1,0.0,0.0},
            SpatialMassTransportPoint{0.2,0.0,0.0},
            SpatialMassTransportPoint{1.1,0.0,0.0}}};
        static const std::array<double,3> childVolumes{1.0e-3,1.0e-3,1.0e-3};
        static const std::array<std::uint32_t,3> childMaterials{15,15,15};
        static const std::array<std::uint32_t,3> childDonorIds{1,1,1};
        static const std::array<std::uint32_t,3> childRegionIds{10,10,10};
        static const std::array<double,16> exactLower=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
        static const std::array<double,16> exactUpper=[] {std::array<double,16>x{};x.fill(1000.0);return x;}();
        static const std::array<std::uint32_t,4> supportOffsets{0,1,2,3};
        static const std::array<std::uint32_t,3> supportGroups{0,0,0};
        static const std::array<double,3> supportFractions{1.0,1.0,1.0};
        static const std::array<std::uint32_t,1> groupMaterial{15};
        static const std::array<std::uint32_t,1> groupDonor{1};
        static const std::array<std::uint32_t,1> groupRegion{10};
        static const std::array<std::uint64_t,1> groupIdentity{0x8000000000000010ull};
        static const std::array<std::uint32_t,3> parentStable{501,502,503};
        static const std::array<std::uint32_t,2> groupParentOffsets{0,3};
        static const std::array<std::uint32_t,3> groupParentRows{0,1,2};
        static const std::array<std::uint32_t,2> groupFaceOffsets{0,3};
        static const std::array<std::uint32_t,3> groupFaces{501,502,503};
        static const std::string h1(64,'1'),h2(64,'2'),h3(64,'3'),h4(64,'4'),h5(64,'5');
        ConservativeSpatialMassTransportInput sparseInput;
        sparseInput.parentCentroidsWorld=parentPoints;
        sparseInput.parentReferenceVolumeM3=parentVolumes;
        sparseInput.parentMassKg=parentMasses;
        sparseInput.parentMaterialIndices=parentMaterials;
        sparseInput.parentDonorBodyIndices=parentDonors;
        sparseInput.parentAnatomicalRegionIndices=parentRegions;
        sparseInput.parentSourceDonorOffsets=donorOffsets;
        sparseInput.parentSourceDonorBodyIndices=donorIds;
        sparseInput.parentSourceDonorMassKg=donorMasses;
        sparseInput.childCentroidsWorld=childPoints;
        sparseInput.childReferenceVolumeM3=childVolumes;
        sparseInput.childMaterialIndices=childMaterials;
        sparseInput.childDonorBodyIndices=childDonorIds;
        sparseInput.childAnatomicalRegionIndices=childRegionIds;
        sparseInput.materialDensityLowerKgM3=exactLower;
        sparseInput.materialDensityUpperKgM3=exactUpper;
        sparseInput.parentMassPacketSHA256=h1;
        sparseInput.parentSourceContributionSHA256=h2;
        sparseInput.childMeshSHA256=h3;
        sparseInput.regionLabelSHA256=h4;
        sparseInput.coordinateFrameSHA256=h5;
        sparseInput.maximumTransportDistanceM=2.0;
        sparseInput.maximumAllowedEdges=100;
        sparseInput.maximumAugmentations=100;
        ConservativeSpatialChildSemanticSupport sparseSupport;
        sparseSupport.childOffsets=supportOffsets;
        sparseSupport.slotGroupIndices=supportGroups;
        sparseSupport.volumeFractions=supportFractions;
        sparseSupport.groupMaterialIndices=groupMaterial;
        sparseSupport.donorBodyIndices=groupDonor;
        sparseSupport.anatomicalRegionIndices=groupRegion;
        sparseSupport.derivedGroupIdentityIDs=groupIdentity;
        sparseSupport.parentStableSourceFaceIdentities=parentStable;
        sparseSupport.sourceParentTetOffsets=groupParentOffsets;
        sparseSupport.sourceParentTetRows=groupParentRows;
        sparseSupport.sourceStableFaceOffsets=groupFaceOffsets;
        sparseSupport.sourceStableFaceRows=groupFaces;
        sparseSupport.typedSupportSHA256=h5;
        ConservativeSpatialMassTransportResult expanded;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                sparseInput,sparseSupport,expanded,error)||
           expanded.semanticCandidateEdgeCount!=5u||
           expanded.childMassKg!=std::vector<double>({1.0,1.0,1.0})||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                sparseInput,sparseSupport,expanded,error)){
            std::cerr<<"cut-guided sparse expansion failed: "<<error<<"\n";return 17;
        }
        auto cappedInput=sparseInput;
        cappedInput.maximumAllowedEdges=4u;
        ConservativeSpatialMassTransportResult capped;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(
                cappedInput,sparseSupport,capped,error)||
           error.find("INCONCLUSIVE shared-capacity sparse edge budget") == std::string::npos||
           error.find("INFEASIBLE") != std::string::npos){
            std::cerr<<"sparse edge-budget failure was not classified as inconclusive: "<<error<<"\n";return 18;
        }
    }

    // An authenticated one-hop adjacency CSR extends admissibility while the
    // original P1 regions remain unchanged and continue to affect preference.
    {
        auto adjacentInput=in;
        auto adjacentSupport=support;
        static const std::array<std::uint32_t,2> oneRegionOffset{0,1};
        static const std::array<std::uint32_t,1> oneP1Group{0};
        static const std::array<double,1> oneP1Score{1.0};
        static const std::array<std::uint32_t,2> allowedOffsets{0,2};
        static const std::array<std::uint32_t,2> allowedRegions{10,20};
        static const std::string adjacencyHash(64,'9');
        adjacentSupport.childOffsets=oneRegionOffset;
        adjacentSupport.slotGroupIndices=oneP1Group;
        adjacentSupport.volumeFractions=oneP1Score;
        adjacentSupport.cellAdmissibleRegionOffsets=allowedOffsets;
        adjacentSupport.cellAdmissibleRegionIds=allowedRegions;
        adjacentSupport.admissibilitySourceSupportSHA256=adjacencyHash;
        adjacentSupport.admissibilityMaximumSourceRegionHops=1u;
        ConservativeSpatialMassTransportResult adjacentResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,adjacentSupport,adjacentResult,error)||
           adjacentResult.childMassKg!=std::vector<double>({1.0})||
           adjacentResult.groupClosure.size()!=2u||
           adjacentResult.groupClosure[0].childMassKg!=0.5||
           adjacentResult.groupClosure[1].childMassKg!=0.5||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,adjacentSupport,adjacentResult,error)){
            std::cerr<<"authenticated one-hop support failed: "<<error<<"\n";return 19;
        }
        auto changedAdjacency=adjacentSupport;
        static const std::array<std::uint32_t,2> changedAllowedRegions{10,30};
        changedAdjacency.cellAdmissibleRegionIds=changedAllowedRegions;
        if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                adjacentInput,changedAdjacency,adjacentResult,error)){
            std::cerr<<"receipt accepted mutated one-hop support CSR\n";return 23;
        }
        auto missingP1=adjacentSupport;
        static const std::array<std::uint32_t,1> wrongAllowedOffsets{0};
        static const std::array<std::uint32_t,1> wrongAllowedRegions{20};
        missingP1.cellAdmissibleRegionOffsets=wrongAllowedOffsets;
        missingP1.cellAdmissibleRegionIds=wrongAllowedRegions;
        if(!rejected(adjacentInput,missingP1,"adjacency missing original P1 region"))return 20;
        auto badHop=adjacentSupport;
        badHop.admissibilityMaximumSourceRegionHops=2u;
        if(!rejected(adjacentInput,badHop,"non-one-hop adjacency"))return 21;
        auto badOrder=adjacentSupport;
        static const std::array<std::uint32_t,2> unsortedAllowed{20,10};
        badOrder.cellAdmissibleRegionIds=unsortedAllowed;
        if(!rejected(adjacentInput,badOrder,"unsorted adjacency row"))return 22;
    }

    // Tiny positive mass must survive: the production lower-capacity floor can
    // be far below the legacy 1e-18 kg augmentation epsilon.
    {
        static const std::array<SpatialMassTransportPoint,2> tinyParentPoints{{
            SpatialMassTransportPoint{0.0,0.0,0.0},SpatialMassTransportPoint{0.0,0.0,0.0}}};
        static const std::array<double,2> tinyParentVolumes{5.0e-26,5.0e-26};
        static const std::array<double,2> tinyParentMasses{5.0e-23,5.0e-23};
        static const std::array<std::uint32_t,2> tinyMaterials{15,15};
        static const std::array<std::uint32_t,2> tinyDonors{1,2};
        static const std::array<std::uint32_t,2> tinyRegions{10,20};
        static const std::array<std::uint32_t,3> tinyDonorOffsets{0,1,2};
        static const std::array<double,2> tinyDonorMasses{5.0e-23,5.0e-23};
        static const std::array<SpatialMassTransportPoint,1> tinyChildPoints{{
            SpatialMassTransportPoint{0.0,0.0,0.0}}};
        static const std::array<double,1> tinyChildVolumes{1.0e-25};
        static const std::array<std::uint32_t,1> tinyChildMaterial{15};
        static const std::array<std::uint32_t,1> tinyChildDonor{1};
        static const std::array<std::uint32_t,1> tinyChildRegion{10};
        static const std::array<double,16> tinyLower=[] {std::array<double,16>x{};x.fill(900.0);return x;}();
        static const std::array<double,16> tinyUpper=[] {std::array<double,16>x{};x.fill(1100.0);return x;}();
        static const std::string th1(64,'a'),th2(64,'b'),th3(64,'c'),th4(64,'d'),th5(64,'e');
        auto tiny=makeInput();
        tiny.parentCentroidsWorld=tinyParentPoints;tiny.parentReferenceVolumeM3=tinyParentVolumes;
        tiny.parentMassKg=tinyParentMasses;tiny.parentMaterialIndices=tinyMaterials;
        tiny.parentDonorBodyIndices=tinyDonors;tiny.parentAnatomicalRegionIndices=tinyRegions;
        tiny.parentSourceDonorOffsets=tinyDonorOffsets;tiny.parentSourceDonorBodyIndices=tinyDonors;
        tiny.parentSourceDonorMassKg=tinyDonorMasses;tiny.childCentroidsWorld=tinyChildPoints;
        tiny.childReferenceVolumeM3=tinyChildVolumes;tiny.childMaterialIndices=tinyChildMaterial;
        tiny.childDonorBodyIndices=tinyChildDonor;tiny.childAnatomicalRegionIndices=tinyChildRegion;
        tiny.materialDensityLowerKgM3=tinyLower;tiny.materialDensityUpperKgM3=tinyUpper;
        tiny.parentMassPacketSHA256=th1;tiny.parentSourceContributionSHA256=th2;
        tiny.childMeshSHA256=th3;tiny.regionLabelSHA256=th4;tiny.coordinateFrameSHA256=th5;
        tiny.maximumTransportDistanceM=0.1;tiny.maximumAllowedEdges=10;tiny.maximumAugmentations=100;
        ConservativeSpatialMassTransportResult tinyResult;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupport(
                tiny,support,tinyResult,error)||tinyResult.childMassKg.size()!=1u||
           tinyResult.childMassKg[0]<=0.0||
           std::abs(tinyResult.childMassKg[0]-1.0e-22)>1.0e-36||
           std::abs(tinyResult.childImpliedDensityKgM3[0]-1000.0)>1.0e-10||
           !verifyRecordedConservativeSpatialMassTransportWithSemanticSupport(
                tiny,support,tinyResult,error)){
            std::cerr<<"sub-epsilon positive child mass was lost: "<<error<<"\n";return 24;
        }
    }

    auto overDensity = in;
    static const std::array<double,1> tooSmallChildVolume{8.0e-4};
    overDensity.childReferenceVolumeM3=tooSmallChildVolume;
    if (!rejected(overDensity, support, "real child density capacity")) return 16;

    auto missingCSR = support;
    static const std::array<std::uint32_t, 0> emptyOffsets{};
    missingCSR.sourceStableFaceOffsets = emptyOffsets;
    missingCSR.sourceStableFaceRows = emptyOffsets;
    if (!rejected(in, missingCSR, "missing stable-face CSR")) return 7;

    auto wrongDonor = support;
    static const std::array<std::uint32_t, 2> wrongDonors{1, 3};
    wrongDonor.donorBodyIndices = wrongDonors;
    if (!rejected(in, wrongDonor, "unregistered donor group")) return 8;

    auto wrongParent = support;
    static const std::array<std::uint32_t, 2> wrongParents{0, 0};
    wrongParent.sourceParentTetRows = wrongParents;
    if (!rejected(in, wrongParent, "wrong group parent-tet CSR")) return 9;

    auto wrongFace = support;
    static const std::array<std::uint32_t, 2> wrongFaces{501, 999};
    wrongFace.sourceStableFaceRows = wrongFaces;
    if (!rejected(in, wrongFace, "incomplete stable-face union")) return 10;

    auto outOfRangeParent = support;
    static const std::array<std::uint32_t, 2> badParentRows{0, 99};
    outOfRangeParent.sourceParentTetRows = badParentRows;
    if (!rejected(in, outOfRangeParent, "out-of-range source parent row")) return 14;

    auto outOfRangeGroup = support;
    static const std::array<std::uint32_t, 2> badGroupRows{0, 99};
    outOfRangeGroup.slotGroupIndices = badGroupRows;
    if (!rejected(in, outOfRangeGroup, "out-of-range semantic group row")) return 15;

    auto tooFar = in;
    static const std::array<SpatialMassTransportPoint, 2> farParent{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0},
        SpatialMassTransportPoint{0.2, 0.0, 0.0}}};
    tooFar.parentCentroidsWorld = farParent;
    if (!rejected(tooFar, support, "distance bound")) return 11;

    auto missingSupport = support;
    static const std::array<std::uint32_t, 0> noChildren{};
    missingSupport.childOffsets = noChildren;
    if (!rejected(in, missingSupport, "missing child CSR")) return 12;

    // The no-support legacy entrypoint remains usable and verifier-compatible.
    static const std::array<SpatialMassTransportPoint, 1> oneParentCentroid{{
        SpatialMassTransportPoint{0.0, 0.0, 0.0}}};
    static const std::array<double, 1> oneParentVolume{1.0e-3};
    static const std::array<double, 1> oneParentMass{1.0};
    static const std::array<std::uint32_t, 1> oneMaterial{15};
    static const std::array<std::uint32_t, 1> oneDonor{1};
    static const std::array<std::uint32_t, 1> oneRegion{10};
    static const std::array<std::uint32_t, 2> oneSourceOffsets{0, 1};
    static const std::array<double, 1> oneSourceMass{1.0};
    auto legacy = in;
    legacy.parentCentroidsWorld = oneParentCentroid;
    legacy.parentReferenceVolumeM3 = oneParentVolume;
    legacy.parentMassKg = oneParentMass;
    legacy.parentMaterialIndices = oneMaterial;
    legacy.parentDonorBodyIndices = oneDonor;
    legacy.parentAnatomicalRegionIndices = oneRegion;
    legacy.parentSourceDonorOffsets = oneSourceOffsets;
    legacy.parentSourceDonorBodyIndices = oneDonor;
    legacy.parentSourceDonorMassKg = oneSourceMass;
    ConservativeSpatialMassTransportResult legacyResult;
    if (!assembleConservativeSpatialMassTransport(legacy, legacyResult, error) ||
        !verifyRecordedConservativeSpatialMassTransport(legacy, legacyResult, error) ||
        legacyResult.childMassKg.size() != 1 || legacyResult.childMassKg[0] != 1.0) {
        std::cerr << "legacy no-support transport changed: " << error << "\n";
        return 13;
    }
    std::cout << "shared real-child semantic capacity controls PASS\n";
    return 0;
}

