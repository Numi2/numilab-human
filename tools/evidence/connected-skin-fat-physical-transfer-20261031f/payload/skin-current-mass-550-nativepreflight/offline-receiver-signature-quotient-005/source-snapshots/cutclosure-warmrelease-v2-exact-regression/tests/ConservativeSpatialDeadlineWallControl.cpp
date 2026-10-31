#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;

int main() {
    constexpr std::size_t n=8192;
    constexpr double totalMass=1.0;
    const double parentMass=totalMass/static_cast<double>(n);
    std::vector<SpatialMassTransportPoint> parentCentroids(n),childCentroids(n);
    std::vector<double> parentVolumes(n,parentMass/1000.0),parentMasses(n,parentMass);
    std::vector<std::uint32_t> parentMaterials(n,15),parentDonors(n,145),parentRegions(n,145);
    std::vector<std::uint32_t> sourceOffsets(n+1),sourceDonors(n,145);
    std::vector<double> sourceDonorMass(n,parentMass);
    std::vector<double> childVolumes(n,parentMass/1000.0);
    std::vector<std::uint32_t> childMaterials(n,15),childDonors(n,145),childRegions(n,145);
    std::vector<std::uint32_t> childOffsets(n+1),slotGroups(n,0);
    std::vector<double> slotFractions(n,1.0);
    std::vector<std::uint32_t> sourceParentOffsets{0},sourceParentRows;
    sourceParentRows.reserve(n);
    for(std::size_t i=0;i<n;++i){
        const double x=static_cast<double>(i)*0.1;
        parentCentroids[i]={x,0.0,0.0};
        childCentroids[i]={x,0.0,0.0};
        sourceOffsets[i]=static_cast<std::uint32_t>(i);
        childOffsets[i]=static_cast<std::uint32_t>(i);
        sourceParentRows.push_back(static_cast<std::uint32_t>(i));
    }
    sourceOffsets[n]=static_cast<std::uint32_t>(n);
    childOffsets[n]=static_cast<std::uint32_t>(n);
    sourceParentOffsets.push_back(static_cast<std::uint32_t>(n));

    std::array<double,16> densityLower{},densityUpper{};
    densityLower.fill(900.0);densityUpper.fill(1100.0);
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=parentCentroids;in.parentReferenceVolumeM3=parentVolumes;
    in.parentMassKg=parentMasses;in.parentMaterialIndices=parentMaterials;
    in.parentDonorBodyIndices=parentDonors;in.parentAnatomicalRegionIndices=parentRegions;
    in.parentSourceDonorOffsets=sourceOffsets;in.parentSourceDonorBodyIndices=sourceDonors;
    in.parentSourceDonorMassKg=sourceDonorMass;
    in.childCentroidsWorld=childCentroids;in.childReferenceVolumeM3=childVolumes;
    in.childMaterialIndices=childMaterials;in.childDonorBodyIndices=childDonors;
    in.childAnatomicalRegionIndices=childRegions;
    in.materialDensityLowerKgM3=densityLower;in.materialDensityUpperKgM3=densityUpper;
    const std::string parentHash(64,'a'),sourceHash(64,'b'),childHash(64,'c'),regionHash(64,'d'),frameHash(64,'e');
    in.parentMassPacketSHA256=parentHash;in.parentSourceContributionSHA256=sourceHash;
    in.childMeshSHA256=childHash;in.regionLabelSHA256=regionHash;
    in.coordinateFrameSHA256=frameHash;
    in.maximumTransportDistanceM=0.02;in.maximumAllowedEdges=n+16;in.maximumAugmentations=4*n;

    std::vector<std::uint32_t> groupMaterial{15},groupDonor{145},groupRegion{145};
    std::vector<std::uint64_t> groupIdentity{0x8000000000000145ull};
    std::vector<std::uint32_t> stableFace(n,77608),stableOffsets{0,1},stableRows{77608};
    ConservativeSpatialChildSemanticSupport support;
    support.childOffsets=childOffsets;support.slotGroupIndices=slotGroups;support.volumeFractions=slotFractions;
    support.groupMaterialIndices=groupMaterial;support.donorBodyIndices=groupDonor;
    support.anatomicalRegionIndices=groupRegion;support.derivedGroupIdentityIDs=groupIdentity;
    support.parentStableSourceFaceIdentities=stableFace;
    support.sourceParentTetOffsets=sourceParentOffsets;support.sourceParentTetRows=sourceParentRows;
    support.sourceStableFaceOffsets=stableOffsets;support.sourceStableFaceRows=stableRows;
    const std::string supportHash(64,'f');
    support.typedSupportSHA256=supportHash;

    std::string error;
    ConservativeSpatialMassTransportResult complete;
    fractional_semantic_transport_detail::SharedCapacityFlowTelemetry completeTelemetry;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupport(in,support,complete,error,&completeTelemetry)){
        std::cerr<<"unbounded fixture solve failed: "<<error<<"\n";return 1;
    }
    if(complete.parentMassTransferredKg.size()!=n||complete.childMassKg.size()!=n||
       !(completeTelemetry.feasibleSolverSeconds>0.0)){
        std::cerr<<"unbounded fixture did not complete the expected source/child domain\n";return 2;
    }

    ConservativeSpatialMassTransportResult partial;
    fractional_semantic_transport_detail::SharedCapacityFlowTelemetry partialTelemetry;
    const double budgetSeconds=completeTelemetry.feasibleSolverSeconds*0.60;
    if(!(budgetSeconds>0.0)||!std::isfinite(budgetSeconds)){
        std::cerr<<"measured baseline solver duration cannot define a bounded wall control\n";return 3;
    }
    partialTelemetry.feasibleFlowBudget=std::chrono::duration<double>(budgetSeconds);
    if(assembleConservativeSpatialMassTransportWithSemanticSupport(in,support,partial,error,&partialTelemetry)||
       !partialTelemetry.deadlineReached||!partialTelemetry.partialFlowCaptured||
       partialTelemetry.partialMassKg.empty()){
        std::cerr<<"measured wall budget did not stop with a nonempty partial flow: "<<error<<"\n";return 4;
    }
    std::vector<long double> perParent(n,0.0L),perChild(n,0.0L);
    long double massSum=0.0L;
    for(std::size_t i=0;i<partialTelemetry.partialMassKg.size();++i){
        const auto parent=partialTelemetry.partialExpandedParentRows[i];
        const auto child=partialTelemetry.partialChildRows[i];
        const double mass=partialTelemetry.partialMassKg[i];
        if(parent>=n||child>=n||!std::isfinite(mass)||mass<=0.0||
           !std::isfinite(partialTelemetry.partialMeasureM3[i])||
           !std::isfinite(partialTelemetry.partialDistanceM[i])||
           partialTelemetry.partialDistanceM[i]>in.maximumTransportDistanceM){
            std::cerr<<"partial wall-flow has invalid source, child, measure, or radius row\n";return 5;
        }
        perParent[parent]+=mass;perChild[child]+=mass;massSum+=mass;
    }
    const long double scale=std::max(massSum,std::abs(static_cast<long double>(partialTelemetry.partialSourceSentKg)));
    if(!(massSum>0.0L)||scale==0.0L||
       std::abs(massSum-static_cast<long double>(partialTelemetry.partialSourceSentKg))>1.0e-10L*scale){
        std::cerr<<"partial wall-flow row sum does not match source sent mass\n";return 6;
    }
    std::size_t touchedChildren=0,underfloorChildren=0,sourceRowsWithRemainder=0;
    long double floorDeficit=0.0L,childUpperHeadroom=0.0L,totalSourceMass=0.0L;
    for(double m:parentMasses)totalSourceMass+=static_cast<long double>(m);
    for(std::size_t p=0;p<n;++p){
        if(perParent[p]>static_cast<long double>(parentMasses[p])*(1.0L+1.0e-10L)+1.0e-16L){
            std::cerr<<"partial wall-flow overspent a source row\n";return 7;
        }
        if(perParent[p]<static_cast<long double>(parentMasses[p]))++sourceRowsWithRemainder;
    }
    for(std::size_t c=0;c<n;++c){
        const long double cap=static_cast<long double>(childVolumes[c])*1100.0L;
        if(perChild[c]>cap*(1.0L+1.0e-10L)+1.0e-16L){
            std::cerr<<"partial wall-flow overspent a real-child upper capacity\n";return 8;
        }
        childUpperHeadroom+=std::max(0.0L,cap-perChild[c]);
        if(perChild[c]>0.0L)++touchedChildren;
        const long double low=static_cast<long double>(childVolumes[c])*900.0L;
        if(perChild[c]<low){++underfloorChildren;floorDeficit+=low-perChild[c];}
    }
    std::cout<<std::setprecision(17)
             <<"{\"schema\":\"numi.human.dinic-deadline-wall-control.v1\","
             <<"\"status\":\"INCONCLUSIVE_PARTIAL_FLOW_DIAGNOSTIC\","
             <<"\"physical_admission\":false,\"native_steps\":0,"
             <<"\"physical_infeasibility_claim\":false,"
             <<"\"source_children\":"<<n<<","
             <<"\"baseline_solver_seconds\":"<<completeTelemetry.feasibleSolverSeconds<<","
             <<"\"deadline_budget_seconds\":"<<budgetSeconds<<","
             <<"\"deadline_reached\":true,"
             <<"\"partial_flow_rows\":"<<partialTelemetry.partialMassKg.size()<<","
             <<"\"partial_source_sent_kg\":"<<partialTelemetry.partialSourceSentKg<<","
             <<"\"partial_rows_sum_kg\":"<<static_cast<double>(massSum)<<","
             <<"\"source_total_mass_kg\":"<<static_cast<double>(totalSourceMass)<<","
             <<"\"remaining_source_mass_kg\":"<<static_cast<double>(std::max(0.0L,totalSourceMass-massSum))<<","
             <<"\"source_rows_with_remainder\":"<<sourceRowsWithRemainder<<","
             <<"\"child_upper_headroom_kg\":"<<static_cast<double>(childUpperHeadroom)<<","
             <<"\"touched_children\":"<<touchedChildren<<","
             <<"\"underfloor_children\":"<<underfloorChildren<<","
             <<"\"remaining_floor_deficit_kg\":"<<static_cast<double>(floorDeficit)<<","
             <<"\"residual_reachable_parent_rows\":"<<partialTelemetry.residualReachableExpandedParentRows.size()<<","
             <<"\"residual_reachable_child_rows\":"<<partialTelemetry.residualReachableChildRows.size()
             <<"}\n";
    return 0;
}
