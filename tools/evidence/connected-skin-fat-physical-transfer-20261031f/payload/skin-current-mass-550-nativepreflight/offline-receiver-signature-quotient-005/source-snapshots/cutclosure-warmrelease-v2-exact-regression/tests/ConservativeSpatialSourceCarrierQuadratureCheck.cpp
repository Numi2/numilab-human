#include "ConservativeSpatialSourceCarrierQuadrature.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using P = SpatialMassTransportPoint;
using B = std::array<double,4>;
using Tet = std::array<P,4>;

static P point(const Tet& t,const B& b) {
    P p{}; for (std::size_t i=0;i<4;++i) for(std::size_t a=0;a<3;++a) p[a]+=b[i]*t[i][a]; return p;
}
static double dist(const P&a,const P&b) {
    double q=0; for(int i=0;i<3;++i){double d=a[i]-b[i];q+=d*d;} return std::sqrt(q);
}
struct Fixture {
    double s=std::cbrt(0.006);
    Tet tet{{P{0,0,0},P{s,0,0},P{0,s,0},P{0,0,s}}};
    std::vector<P> pc{P{s/4,s/4,s/4}}, cc;
    std::vector<double> pv{1.0e-3}, pm{1.0}, cv{2.5e-4,2.5e-4,2.5e-4,2.5e-4};
    std::vector<std::uint32_t> pmat{15}, pdonor{145}, pregion{145}, poff{0,1}, pdids{145};
    std::vector<double> pdmass{1.0};
    std::vector<std::uint32_t> cmat{15,15,15,15}, cdonor{145,145,145,145}, cregion{145,145,145,145};
    std::vector<double> lower=std::vector<double>(16,900.0), upper=std::vector<double>(16,1100.0);
    std::vector<std::uint32_t> childOff{0,1,2,3,4}, slotGroup{0,0,0,0}, gm{15}, gd{145}, gr{145};
    std::vector<double> preference{1,1,1,1};
    std::vector<std::uint64_t> gids{0x8000000000000100ull};
    std::vector<std::uint32_t> stable{77608}, gpoff{0,1}, gprows{0}, gfoff{0,1}, gfrows{77608};
    std::vector<std::uint32_t> carrierOff{0,4}, tetRows{9908}, tetNodes{0,1,2,3}, witness{0,1,2,3};
    std::vector<double> weights{.25,.25,.25,.25}, witnessDists{0,0,0,0};
    std::vector<B> bary;
    std::vector<Tet> parentTets{tet};
    std::string parentHash=std::string(64,'a'), sourceHash=std::string(64,'b'), childHash=std::string(64,'c'), regionHash=std::string(64,'d'), frameHash=std::string(64,'e'), supportHash=std::string(64,'f'), carrierHash=std::string(64,'1'), tetHash=std::string(64,'2'), nodeHash=std::string(64,'3');
    ConservativeSpatialMassTransportInput in;
    ConservativeSpatialChildSemanticSupport support;
    SourceCellCarrierQuadratureInput carriers;

    Fixture() {
        bary.resize(4);
        for(std::size_t r=0;r<4;++r) {
            for(std::size_t i=0;i<4;++i) bary[r][i]=0.25+0.3*(i==r?0.75:-0.25);
            cc.push_back(point(tet,bary[r]));
        }
        std::array<std::uint32_t,4> tr{0,1,2,3};
        sourceTetNodeRows.push_back(tr);
        witnessDists.resize(4);
        for(std::size_t i=0;i<4;++i) witnessDists[i]=dist(cc[i],point(tet,bary[i]));
        in.parentCentroidsWorld=pc; in.parentReferenceVolumeM3=pv; in.parentMassKg=pm;
        in.parentMaterialIndices=pmat; in.parentDonorBodyIndices=pdonor; in.parentAnatomicalRegionIndices=pregion;
        in.parentSourceDonorOffsets=poff; in.parentSourceDonorBodyIndices=pdids; in.parentSourceDonorMassKg=pdmass;
        in.childCentroidsWorld=cc; in.childReferenceVolumeM3=cv; in.childMaterialIndices=cmat;
        in.childDonorBodyIndices=cdonor; in.childAnatomicalRegionIndices=cregion;
        in.materialDensityLowerKgM3=lower; in.materialDensityUpperKgM3=upper;
        in.parentMassPacketSHA256=parentHash; in.parentSourceContributionSHA256=sourceHash;
        in.childMeshSHA256=childHash; in.regionLabelSHA256=regionHash;
        in.coordinateFrameSHA256=frameHash; in.maximumTransportDistanceM=0.02;
        in.maximumAllowedEdges=100; in.maximumAugmentations=100;
        support.childOffsets=childOff; support.slotGroupIndices=slotGroup; support.volumeFractions=preference;
        support.groupMaterialIndices=gm; support.donorBodyIndices=gd; support.anatomicalRegionIndices=gr;
        support.derivedGroupIdentityIDs=gids; support.parentStableSourceFaceIdentities=stable;
        support.sourceParentTetOffsets=gpoff; support.sourceParentTetRows=gprows;
        support.sourceStableFaceOffsets=gfoff; support.sourceStableFaceRows=gfrows;
        support.typedSupportSHA256=supportHash;
        carriers.parentOffsets=carrierOff; carriers.globalSourceTetRows=tetRows;
        carriers.sourceTetNodeRows=sourceTetNodeRows; carriers.parentReferenceTetWorldVertices=parentTets;
        carriers.carrierWeights=weights; carriers.carrierBarycentric=bary;
        carriers.witnessChildRows=witness; carriers.witnessChildCentroidDistancesM=witnessDists;
        carriers.sourceCarrierPacketSHA256=carrierHash;
        carriers.sourceTetTableSHA256=tetHash; carriers.sourceNodeTableSHA256=nodeHash;
    }
    std::vector<std::array<std::uint32_t,4>> sourceTetNodeRows;
};

static bool alternatingWarmStartControl() {
    const double side=std::cbrt(0.006);
    const auto makeTet=[&](double cx){
        const double q=side/4.0;
        return Tet{{P{cx-q,-q,-q},P{cx+3*q,-q,-q},P{cx-q,3*q,-q},P{cx-q,-q,3*q}}};
    };
    const Tet t0=makeTet(0.0),t1=makeTet(0.01);
    const P c0{0.0,0.0,0.0},c1{0.01,0.0,0.0};
    std::vector<P> parentCentroids{c0,c1};
    std::vector<double> parentVolumes{1.0e-3,1.0e-3},parentMasses{1.0,1.0};
    std::vector<std::uint32_t> parentMaterials{15,15},parentDonors{145,145},parentRegions{145,145};
    std::vector<std::uint32_t> donorOffsets{0,1,2},donorIDs{145,145};
    std::vector<double> donorMasses{1.0,1.0};
    std::vector<P> children{P{0.009,0.0,0.0},P{-0.015,0.0,0.0}};
    std::vector<double> childVolumes{1.0/900.0,1.0/900.0};
    std::vector<std::uint32_t> childMaterials{15,15},childDonors{145,145},childRegions{145,145};
    std::vector<double> lower(16,900.0),upper(16,1100.0);
    std::vector<std::uint32_t> childOffsets{0,1,2},slotGroups{0,0},groupMaterials{15},groupDonors{145},groupRegions{145};
    std::vector<double> preference{1.0,1.0};
    std::vector<std::uint64_t> identities{0x8000000000000771ull};
    std::vector<std::uint32_t> stable{77608,77608},groupParentOffsets{0,2},groupParentRows{0,1};
    std::vector<std::uint32_t> groupFaceOffsets{0,1},groupFaceRows{77608};
    std::vector<std::uint32_t> carrierOffsets{0,1,2},globalTetRows{100,200};
    std::vector<std::array<std::uint32_t,4>> sourceTetNodes{{0,1,2,3},{4,5,6,7}};
    std::vector<Tet> parentTets{t0,t1};
    std::vector<double> carrierWeights{1.0,1.0};
    std::vector<B> carrierBary{{B{0.25,0.25,0.25,0.25}}, {B{0.25,0.25,0.25,0.25}}};
    std::vector<std::uint32_t> witnessRows{0,0};
    std::vector<double> witnessDistances{0.009,0.001};
    const std::string parentHash(64,'a'),sourceHash(64,'b'),childHash(64,'c'),regionHash(64,'d'),frameHash(64,'e'),supportHash(64,'f'),packetHash(64,'1'),tetHash(64,'2'),nodeHash(64,'3');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=parentCentroids;in.parentReferenceVolumeM3=parentVolumes;in.parentMassKg=parentMasses;
    in.parentMaterialIndices=parentMaterials;in.parentDonorBodyIndices=parentDonors;in.parentAnatomicalRegionIndices=parentRegions;
    in.parentSourceDonorOffsets=donorOffsets;in.parentSourceDonorBodyIndices=donorIDs;in.parentSourceDonorMassKg=donorMasses;
    in.childCentroidsWorld=children;in.childReferenceVolumeM3=childVolumes;in.childMaterialIndices=childMaterials;
    in.childDonorBodyIndices=childDonors;in.childAnatomicalRegionIndices=childRegions;
    in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
    in.parentMassPacketSHA256=parentHash;in.parentSourceContributionSHA256=sourceHash;
    in.childMeshSHA256=childHash;in.regionLabelSHA256=regionHash;in.coordinateFrameSHA256=frameHash;
    in.maximumTransportDistanceM=0.02;in.relativeClosureTolerance=1.0e-10;in.absoluteMassToleranceKg=1.0e-13;
    in.maximumAllowedEdges=32;in.maximumAugmentations=32;
    ConservativeSpatialChildSemanticSupportData supportData;
    supportData.childOffsets=childOffsets;supportData.slotGroupIndices=slotGroups;supportData.volumeFractions=preference;
    supportData.groupMaterialIndices=groupMaterials;supportData.donorBodyIndices=groupDonors;supportData.anatomicalRegionIndices=groupRegions;
    supportData.derivedGroupIdentityIDs=identities;supportData.parentStableSourceFaceIdentities=stable;
    supportData.sourceParentTetOffsets=groupParentOffsets;supportData.sourceParentTetRows=groupParentRows;
    supportData.sourceStableFaceOffsets=groupFaceOffsets;supportData.sourceStableFaceRows=groupFaceRows;
    supportData.typedSupportSHA256=supportHash;
    SourceCellCarrierQuadratureInput carrier;
    carrier.parentOffsets=carrierOffsets;carrier.globalSourceTetRows=globalTetRows;carrier.sourceTetNodeRows=sourceTetNodes;
    carrier.parentReferenceTetWorldVertices=parentTets;carrier.carrierWeights=carrierWeights;
    carrier.carrierBarycentric=carrierBary;carrier.witnessChildRows=witnessRows;
    carrier.witnessChildCentroidDistancesM=witnessDistances;carrier.sourceCarrierPacketSHA256=packetHash;
    carrier.sourceTetTableSHA256=tetHash;carrier.sourceNodeTableSHA256=nodeHash;
    std::vector<fractional_semantic_transport_detail::SharedCapacityWarmStartFlow> held{{0u,0u,1.0,0u,100u,0u}};
    fractional_semantic_transport_detail::SharedCapacityWarmStart warm;
    warm.flows=held;warm.parentMassPacketSHA256=parentHash;warm.parentSourceContributionSHA256=sourceHash;
    warm.childMeshSHA256=childHash;warm.regionLabelSHA256=regionHash;warm.coordinateFrameSHA256=frameHash;
    warm.typedSupportSHA256=supportHash;warm.sourceCarrierPacketSHA256=packetHash;
    const std::string carrierArrayHash=source_carrier_transport_detail::carrierArrayDigest(in,carrier);
    warm.sourceCarrierArraySHA256=carrierArrayHash;
    ConservativeSpatialMassTransportResult result;SourceCellCarrierTransportTrace trace;std::string error;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
          in,supportData.view(),carrier,result,trace,error,nullptr,&warm)){
        std::cerr<<"alternating warm-start solve failed: "<<error<<"\\n";return false;
    }
    if(result.childMassKg.size()!=2||result.parentMassTransferredKg!=std::vector<double>({1.0,1.0})||
       std::abs(result.childMassKg[0]-1.0)>1e-12||std::abs(result.childMassKg[1]-1.0)>1e-12||
       trace.expandedParentToCarrierRows!=std::vector<std::uint32_t>({0,1})||
       trace.childParentCarrierOffsets.size()!=3||trace.childParentCarrierRows.size()!=2||
       trace.childParentCarrierRows[trace.childParentCarrierOffsets[0]]!=1u||
       trace.childParentCarrierRows[trace.childParentCarrierOffsets[1]]!=0u||
       result.childImpliedDensityKgM3.size()!=2||
       std::abs(result.childImpliedDensityKgM3[0]-900.0)>1e-8||
       std::abs(result.childImpliedDensityKgM3[1]-900.0)>1e-8||
       result.childImpliedDensityKgM3[0]>1100.0||result.childImpliedDensityKgM3[1]>1100.0){
        std::cerr<<"alternating warm-start did not reassign the occupied carrier through the reverse residual edge\\n";return false;
    }
    auto pushWarm=warm;
    pushWarm.useHighestLabelPushRelabel=true;
    ConservativeSpatialMassTransportResult pushResult;
    SourceCellCarrierTransportTrace pushTrace;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
          in,supportData.view(),carrier,pushResult,pushTrace,error,nullptr,&pushWarm)){
        std::cerr<<"highest-label alternating warm-start solve failed: "<<error<<"\n";return false;
    }
    if(pushResult.allocationSolver!="highest_label_push_relabel_feasibility"||
       pushResult.parentMassTransferredKg.size()!=2||
       std::abs(pushResult.parentMassTransferredKg[0]-1.0)>1e-12||
       std::abs(pushResult.parentMassTransferredKg[1]-1.0)>1e-12||
       pushResult.childMassKg.size()!=2||
       std::abs(pushResult.childMassKg[0]-1.0)>1e-12||
       std::abs(pushResult.childMassKg[1]-1.0)>1e-12||
       pushResult.childImpliedDensityKgM3.size()!=2||
       std::abs(pushResult.childImpliedDensityKgM3[0]-900.0)>1e-8||
       std::abs(pushResult.childImpliedDensityKgM3[1]-900.0)>1e-8||
       pushResult.maximumParentMassClosureErrorKg>1e-12||
       pushResult.maximumTransportDistanceUsedM>in.maximumTransportDistanceM){
        std::cerr<<"highest-label flow failed source, receiver-density, radius, or exact algorithm reporting checks\n";return false;
    }
    using FlowNetwork=spatial_mass_transport_detail::MinCostFlow;
    FlowNetwork interrupted(3u);
    const auto sourceEdge=interrupted.addEdge(0u,1u,1.0,0.0);
    const auto sinkEdge=interrupted.addEdge(1u,2u,1.0,0.0);
    if(!interrupted.seedExistingFlow(0u,sourceEdge,0.25,error)||
       !interrupted.seedExistingFlow(1u,sinkEdge,0.25,error)){
        std::cerr<<"push-relabel rollback fixture could not seed its feasible flow: "<<error<<"\n";return false;
    }
    std::size_t stoppedPushes=0u;double stoppedSent=0.0;
    spatial_mass_transport_detail::FeasibleFlowTelemetry stoppedTelemetry;
    const auto immediateDeadline=std::chrono::steady_clock::now();
    if(interrupted.runFeasibleHighestLabel(0u,2u,8u,stoppedPushes,stoppedSent,error,
                                           &stoppedTelemetry,immediateDeadline)||
       error.find("INCONCLUSIVE")==std::string::npos||
       std::abs(interrupted.sentFromSource(0u)-0.25)>1e-15||
       std::abs(interrupted.remainingCapacity(0u,sourceEdge)-0.75)>1e-15||
       std::abs(interrupted.remainingCapacity(1u,sinkEdge)-0.75)>1e-15){
        std::cerr<<"bounded push-relabel interruption did not restore the conserved seeded flow\n";return false;
    }
    ConservativeSpatialMassTransportResult defaultA,defaultB;
    SourceCellCarrierTransportTrace defaultTraceA,defaultTraceB;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(in,supportData.view(),carrier,defaultA,defaultTraceA,error)||
       !assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(in,supportData.view(),carrier,defaultB,defaultTraceB,error)||
       !fractional_semantic_transport_detail::sameResult(defaultA,defaultB)){
        std::cerr<<"warm-start extension changed/default route parity failed: "<<error<<"\\n";return false;
    }
    auto stale=warm;const std::string staleFrameHash(64,'9');stale.coordinateFrameSHA256=staleFrameHash;
    ConservativeSpatialMassTransportResult rejectedResult;SourceCellCarrierTransportTrace rejectedTrace;
    if(assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
          in,supportData.view(),carrier,rejectedResult,rejectedTrace,error,nullptr,&stale)){
        std::cerr<<"stale warm-start input hash was accepted\\n";return false;
    }
    auto staleMap=held;staleMap[0].sourceRoleParentRow=1u;stale.flows=staleMap;stale.coordinateFrameSHA256=frameHash;
    if(assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
          in,supportData.view(),carrier,rejectedResult,rejectedTrace,error,nullptr,&stale)){
        std::cerr<<"stale warm-start source/carrier row map was accepted\\n";return false;
    }
    std::cout<<"PASS alternating carrier warm start: Dinic and highest-label push-relabel reallocated occupied edges through reverse residuals; source/child density/radius checks passed; bounded preflow rollback and stale-input rejection passed\\n";
    return true;
}

static bool rejects(Fixture f,const char* label) {
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    std::string error;
    if(assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"unexpected acceptance: "<<label<<"\n"; return false;
    }
    return true;
}
int main() {
    if(!alternatingWarmStartControl())return 30;
    Fixture f;
    double nearest=1.0;
    for(const auto& c:f.cc) nearest=std::min(nearest,dist(f.pc[0],c));
    if(!(nearest>f.in.maximumTransportDistanceM)) {std::cerr<<"fixture does not exceed strict centroid bound\n";return 1;}
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    std::string error;
    if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"carrier transport failed: "<<error<<"\n";return 2;
    }
    if(result.childMassKg.size()!=4||result.parentMassTransferredKg!=std::vector<double>{1.0}||
       std::abs(result.parentMeasureTransferredM3[0]-1.0e-3)>1e-14||
       result.childParentRows.size()!=4||trace.carrierSourceParentRows!=std::vector<std::uint32_t>{0}||
       trace.expandedParentToSourceParentRows!=std::vector<std::uint32_t>({0,0,0,0})||
       trace.expandedParentToCarrierRows!=std::vector<std::uint32_t>({0,1,2,3})||
       trace.childParentCarrierRows.size()!=4||trace.childParentCarrierOffsets.size()!=5||
       trace.sourceCarrierPacketSHA256!=std::string(64,'1')||trace.sourceCarrierArraySHA256.empty()||
       trace.originalCentroidNearestAdmissibleDistanceM.size()!=1||
       !(trace.originalCentroidNearestAdmissibleDistanceM[0]>0.02)||
       result.groupClosure.size()!=1||std::abs(result.groupClosure[0].childMassKg-1.0)>1e-12) {
        std::cerr<<"carrier flow/aggregate mapping invariant failed\n";return 3;
    }
    for(std::size_t c=0;c<4;++c)
        if(std::abs(result.childMassKg[c]-0.25)>1e-10||
           std::abs(result.childImpliedDensityKgM3[c]-1000.0)>1e-8||
           trace.childParentCarrierOffsets[c+1]-trace.childParentCarrierOffsets[c]!=1u) {
            std::cerr<<"real child density or carrier edge failed at "<<c<<"\n";return 4;
        }
    if(!verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,trace,error)) {
        std::cerr<<"carrier deterministic replay failed: "<<error<<"\n";return 5;
    }
    {
        auto bounded=f.in;bounded.maximumAugmentations=1u;
        ConservativeSpatialMassTransportResult partialResult;
        SourceCellCarrierTransportTrace partialTrace;
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry partialTelemetry;
        if(assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
                bounded,f.support,f.carriers,partialResult,partialTrace,error,&partialTelemetry)||
           !partialTelemetry.partialFlowCaptured||partialTelemetry.partialMassKg.empty()||
           partialTelemetry.partialExpandedParentRoleRows.size()!=partialTelemetry.partialExpandedParentCarrierRows.size()||
           partialTelemetry.partialExpandedParentRows.size()!=partialTelemetry.partialChildRows.size()||
           partialTelemetry.partialMassKg.size()!=partialTelemetry.partialMeasureM3.size()||
           partialTelemetry.partialMassKg.size()!=partialTelemetry.partialDistanceM.size()){
            std::cerr<<"bounded carrier run did not expose aligned source/carrier/child partial accounting: "<<error<<"\n";return 16;
        }
        std::vector<double> perCarrier(f.weights.size(),0.0),perChild(f.cc.size(),0.0);
        double sourceSent=0.0;
        for(std::size_t i=0;i<partialTelemetry.partialMassKg.size();++i){
            const auto ep=partialTelemetry.partialExpandedParentRows[i],child=partialTelemetry.partialChildRows[i];
            if(ep>=partialTelemetry.partialExpandedParentRoleRows.size()||child>=f.cc.size()){
                std::cerr<<"partial carrier flow row has invalid expanded-parent or child index\n";return 17;
            }
            const auto role=partialTelemetry.partialExpandedParentRoleRows[ep];
            const auto carrier=partialTelemetry.partialExpandedParentCarrierRows[ep];
            if(role!=0u||role>=f.tetRows.size()||f.tetRows[role]!=9908u||
               role+1u>=f.poff.size()||f.poff[role+1u]-f.poff[role]!=1u||
               f.pdids[f.poff[role]]!=145u||carrier>=f.weights.size()||!(partialTelemetry.partialMassKg[i]>0.0)||
               partialTelemetry.partialDistanceM[i]>f.in.maximumTransportDistanceM||
               std::abs(partialTelemetry.partialMeasureM3[i]-partialTelemetry.partialMassKg[i]/1000.0)>1e-14){
                std::cerr<<"partial carrier source mapping/distance/measure failed replay\n";return 18;
            }
            perCarrier[carrier]+=partialTelemetry.partialMassKg[i];perChild[child]+=partialTelemetry.partialMassKg[i];
            sourceSent+=partialTelemetry.partialMassKg[i];
        }
        if(std::abs(sourceSent-partialTelemetry.partialSourceSentKg)>1e-12){
            std::cerr<<"partial carrier rows do not sum to source sent mass\n";return 19;
        }
        for(std::size_t k=0;k<perCarrier.size();++k)
            if(perCarrier[k]>f.weights[k]*f.pm[0]+1e-12){std::cerr<<"partial flow overspent carrier capacity\n";return 20;}
        for(std::size_t c=0;c<perChild.size();++c)
            if(perChild[c]>f.cv[c]*f.upper[f.cmat[c]]+1e-12){std::cerr<<"partial flow overspent child capacity\n";return 21;}
        std::cout<<"bounded carrier partial PASS: source row, carrier sites, child rows, measures, and capacities accounted\n";
    }
    auto badTrace=trace; badTrace.childParentCarrierMassKg[0]+=0.01;
    if(verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        f.in,f.support,f.carriers,result,badTrace,error)) {std::cerr<<"tampered carrier flow accepted\n";return 6;}
    {
        ConservativeSpatialMassTransportResult legacy;
        if(assembleConservativeSpatialMassTransportWithSemanticSupport(f.in,f.support,legacy,error)) {
            std::cerr<<"legacy centroid route unexpectedly accepted the out-of-bound parent\n"; return 7;
        }
    }
    {
        Fixture x; x.weights[0]=0.30;
        if(!rejects(x,"weight sum"))return 8;
    }
    {
        Fixture x; x.bary[0][0]=1.1;
        if(!rejects(x,"outside tetrahedron"))return 9;
    }
    {
        Fixture x; x.bary[0][0]+=0.01;x.bary[0][1]-=0.01;
        if(!rejects(x,"first moment"))return 10;
    }
    {
        Fixture x; x.witness[0]=99;
        if(!rejects(x,"witness child range"))return 11;
    }
    {
        Fixture x; x.witnessDists[0]=0.001;
        if(!rejects(x,"false distance"))return 12;
    }
    {
        Fixture x; x.in.maximumTransportDistanceM=0.03;
        ConservativeSpatialMassTransportResult expanded;
        SourceCellCarrierTransportTrace expandedTrace;
        if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
            x.in,x.support,x.carriers,expanded,expandedTrace,error)||
           expandedTrace.originalCentroidNearestAdmissibleDistanceM.size()!=1u||
           !(expandedTrace.originalCentroidNearestAdmissibleDistanceM[0]<=0.03)){
            std::cerr<<"explicit source quadrature was rejected solely because the old centroid route was reachable: "<<error<<"\n";return 13;
        }
    }
    {
        Fixture x; x.parentTets[0][0][0]+=0.01;
        if(!rejects(x,"tet centroid mismatch"))return 14;
    }
    {
        Fixture x; x.carrierOff[1]=0;
        if(!rejects(x,"no fallback rows"))return 15;
    }
    std::cout<<"PASS source carrier quadrature: 1 parent, 4 carriers, 4 real cells; aggregate mass/density/measure and replay verified; legacy route unchanged; 7 tamper/eligibility controls rejected; explicit quadrature may also split a centroid-reachable source cell\n";
    return 0;
}
