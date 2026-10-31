#include "NumiHumanConservativeSpatialMassTransport.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
using namespace metalrobo::human::coarse_tissue;
int main(){
    using Point=SpatialMassTransportPoint;
    std::vector<Point> pp{{0,0,0},{10,0,0},{0.2,0,0}};
    std::vector<double> pv{2.0/700.0,1.0/700.0,0.1/700.0},pm{2.0,1.0,0.1};
    std::vector<std::uint32_t> pmat{15,15,15},pdon{1,1,1},preg{10,10,10};
    std::vector<std::uint32_t> pso{0,1,2,3},psd{1,1,1};std::vector<double> psm{2.0,1.0,0.1};
    std::vector<Point> cp{{0.1,0,0},{0,0,0},{10,0,0}};
    std::vector<double> cv{1.0/500.0,1.0/500.0,0.5/500.0};
    std::vector<std::uint32_t> cmat{15,15,15},cdon{1,1,1},creg{10,10,10};
    std::vector<double> dlo(16,900.0),dhi(16,1100.0);dlo[15]=500.0;dhi[15]=1000.0;
    const std::string ph(64,'a'),sh(64,'b'),ch(64,'c'),rh(64,'d'),fh(64,'e'),th(64,'f'),packet(64,'1'),arrays(64,'2');
    ConservativeSpatialMassTransportInput in;
    in.parentCentroidsWorld=pp;in.parentReferenceVolumeM3=pv;in.parentMassKg=pm;in.parentMaterialIndices=pmat;
    in.parentDonorBodyIndices=pdon;in.parentAnatomicalRegionIndices=preg;in.parentSourceDonorOffsets=pso;
    in.parentSourceDonorBodyIndices=psd;in.parentSourceDonorMassKg=psm;in.childCentroidsWorld=cp;
    in.childReferenceVolumeM3=cv;in.childMaterialIndices=cmat;in.childDonorBodyIndices=cdon;in.childAnatomicalRegionIndices=creg;
    in.materialDensityLowerKgM3=dlo;in.materialDensityUpperKgM3=dhi;
    in.parentMassPacketSHA256=ph;in.parentSourceContributionSHA256=sh;in.childMeshSHA256=ch;
    in.regionLabelSHA256=rh;in.coordinateFrameSHA256=fh;in.maximumTransportDistanceM=0.11;
    in.maximumAllowedEdges=100;in.maximumAugmentations=100;
    std::vector<std::uint32_t> co{0,1,2,3},sg{0,0,0},gm{15},gd{1},gr{10},pf{501,502,503},pto{0,3},ptr{0,1,2},sfo{0,3},sfr{501,502,503};
    std::vector<double> pref{1,1,1};std::vector<std::uint64_t> gid{0x8000000000000200ull};
    ConservativeSpatialChildSemanticSupport sup;
    sup.childOffsets=co;sup.slotGroupIndices=sg;sup.volumeFractions=pref;sup.groupMaterialIndices=gm;
    sup.donorBodyIndices=gd;sup.anatomicalRegionIndices=gr;sup.derivedGroupIdentityIDs=gid;
    sup.parentStableSourceFaceIdentities=pf;sup.sourceParentTetOffsets=pto;sup.sourceParentTetRows=ptr;
    sup.sourceStableFaceOffsets=sfo;sup.sourceStableFaceRows=sfr;sup.typedSupportSHA256=th;
    std::vector<fractional_semantic_transport_detail::SharedCapacityWarmStartFlow> wf{
        {0,0,1.5,0,0,0},{1,2,0.5,1,1,0}};
    fractional_semantic_transport_detail::SharedCapacityWarmStart warm;
    warm.flows=wf;warm.parentMassPacketSHA256=ph;warm.parentSourceContributionSHA256=sh;
    warm.childMeshSHA256=ch;warm.regionLabelSHA256=rh;warm.coordinateFrameSHA256=fh;
    warm.typedSupportSHA256=th;warm.sourceCarrierPacketSHA256=packet;warm.sourceCarrierArraySHA256=arrays;
    ConservativeSpatialMassTransportResult out;std::string error;
    fractional_semantic_transport_detail::SharedCapacityFlowTelemetry telemetry;
    const bool ok=assembleConservativeSpatialMassTransportWithSemanticSupport(in,sup,out,error,&telemetry,&warm);
    if(ok){std::cerr<<"legacy warm-extra policy unexpectedly passed complete source transfer\n";return 2;}
    std::cout<<"legacy warm-extra policy rejected the feasible control as expected: "<<error<<"\n";
    return error.find("INFEASIBLE")!=std::string::npos?0:3;
}
