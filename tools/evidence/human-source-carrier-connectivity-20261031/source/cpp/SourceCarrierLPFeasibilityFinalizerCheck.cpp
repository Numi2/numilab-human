#include "SourceCarrierLPFeasibilityFinalizer.hpp"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
using P=SpatialMassTransportPoint;
using B=std::array<double,4>;
using Tet=std::array<P,4>;

static P baryPoint(const Tet& tet,const B& bary) {
    P out{};
    for(std::size_t k=0;k<4;++k)for(std::size_t a=0;a<3;++a)out[a]+=bary[k]*tet[k][a];
    return out;
}
static double dist(const P&a,const P&b) {
    double s=0;for(std::size_t aidx=0;aidx<3;++aidx){const double d=a[aidx]-b[aidx];s+=d*d;}return std::sqrt(s);
}
struct Fixture {
    std::uint32_t material;
    double s=std::cbrt(0.006);
    Tet tet{{P{0,0,0},P{s,0,0},P{0,s,0},P{0,0,s}}};
    std::vector<P> parentCentroids;
    std::vector<P> childCentroids;
    std::vector<double> parentVol{1.0e-3},parentMass{1.0};
    std::vector<std::uint32_t> parentMaterial,parentDonor{145},parentRegion{145};
    std::vector<std::uint32_t> parentDonorOffsets{0,1},parentSourceDonors{145};
    std::vector<double> parentSourceMass{1.0};
    std::vector<double> childVol{2.5e-4,2.5e-4,2.5e-4,2.5e-4};
    std::vector<std::uint32_t> childMaterial,childDonor,childRegion;
    std::vector<double> lower=std::vector<double>(17,850.0),upper=std::vector<double>(17,1100.0);
    std::vector<std::uint32_t> childOffsets{0,1,2,3,4},slotGroup{0,0,0,0};
    std::vector<double> slotVolumeFractions{1,1,1,1};
    std::vector<std::uint32_t> groupMaterial,groupDonor{145},groupRegion{145};
    std::vector<std::uint64_t> groupIdentity{0x8000000000000100ull};
    std::vector<std::uint32_t> parentStable{77608},sourceParentOffsets{0,1},sourceParentRows{0};
    std::vector<std::uint32_t> sourceStableOffsets{0,1},sourceStableRows{77608};
    std::array<std::uint32_t,4> sourceNodes{0,1,2,3};
    std::vector<std::uint32_t> sourceRows{9908},oldOffsets{0,4};
    std::vector<std::array<std::uint32_t,4>> sourceNodeRows{{0,1,2,3}};
    std::vector<Tet> sourceTets;
    std::vector<double> oldWeights{.25,.25,.25,.25};
    std::vector<B> carrierBary;
    std::vector<std::uint32_t> oldWitness{0,1,2,3};
    std::vector<double> oldWitnessDistance{0,0,0,0};
    std::string pMassSHA=std::string(64,'a'),pSourceSHA=std::string(64,'b'),childSHA=std::string(64,'c'),regionSHA=std::string(64,'d'),frameSHA=std::string(64,'e'),supportSHA=std::string(64,'f'),baseCarrierSHA=std::string(64,'1'),tetSHA=std::string(64,'2'),nodeSHA=std::string(64,'3');

    ConservativeSpatialMassTransportInput input;
    ConservativeSpatialChildSemanticSupport support;
    SourceCellCarrierQuadratureInput originalCarrier;
    std::vector<double> scale{1.0};
    std::vector<SourceCarrierLPHeldFlow> held;
    std::vector<SourceCarrierLPPositiveVariable> lp;
    SourceCarrierLPProvenance provenance;

    explicit Fixture(std::uint32_t m):material(m) {
        parentMaterial={m};childMaterial={m,m,m,m};groupMaterial={m};
        childDonor={145,145,145,145};childRegion={145,145,145,145};
        parentCentroids={P{s/4,s/4,s/4}};sourceTets={tet};
        std::vector<B> bary(4);
        for(std::size_t r=0;r<4;++r) {
            for(std::size_t k=0;k<4;++k)bary[r][k]=0.25+0.3*(k==r?0.75:-0.25);
            childCentroids.push_back(baryPoint(tet,bary[r]));
        }
        carrierBary=bary;
        input.parentCentroidsWorld=parentCentroids;input.parentReferenceVolumeM3=parentVol;input.parentMassKg=parentMass;
        input.parentMaterialIndices=parentMaterial;input.parentDonorBodyIndices=parentDonor;input.parentAnatomicalRegionIndices=parentRegion;
        input.parentSourceDonorOffsets=parentDonorOffsets;input.parentSourceDonorBodyIndices=parentSourceDonors;input.parentSourceDonorMassKg=parentSourceMass;
        input.childCentroidsWorld=childCentroids;input.childReferenceVolumeM3=childVol;input.childMaterialIndices=childMaterial;
        input.childDonorBodyIndices=childDonor;input.childAnatomicalRegionIndices=childRegion;
        lower[m]=850.0;upper[m]=1100.0;
        input.materialDensityLowerKgM3=lower;input.materialDensityUpperKgM3=upper;
        input.parentMassPacketSHA256=pMassSHA;input.parentSourceContributionSHA256=pSourceSHA;
        input.childMeshSHA256=childSHA;input.regionLabelSHA256=regionSHA;input.coordinateFrameSHA256=frameSHA;
        input.maximumTransportDistanceM=.019353;input.relativeClosureTolerance=1e-10;
        input.absoluteMassToleranceKg=1e-13;input.absoluteMeasureToleranceM3=1e-15;
        input.maximumAllowedEdges=100;input.maximumAugmentations=100;
        support.childOffsets=childOffsets;support.slotGroupIndices=slotGroup;support.volumeFractions=slotVolumeFractions;
        support.groupMaterialIndices=groupMaterial;support.donorBodyIndices=groupDonor;support.anatomicalRegionIndices=groupRegion;
        support.derivedGroupIdentityIDs=groupIdentity;support.parentStableSourceFaceIdentities=parentStable;
        support.sourceParentTetOffsets=sourceParentOffsets;support.sourceParentTetRows=sourceParentRows;
        support.sourceStableFaceOffsets=sourceStableOffsets;support.sourceStableFaceRows=sourceStableRows;
        support.typedSupportSHA256=supportSHA;
        originalCarrier.parentOffsets=oldOffsets;originalCarrier.globalSourceTetRows=sourceRows;
        originalCarrier.sourceTetNodeRows=sourceNodeRows;originalCarrier.parentReferenceTetWorldVertices=sourceTets;
        originalCarrier.carrierWeights=oldWeights;originalCarrier.carrierBarycentric=carrierBary;
        originalCarrier.witnessChildRows=oldWitness;originalCarrier.witnessChildCentroidDistancesM=oldWitnessDistance;
        originalCarrier.sourceCarrierPacketSHA256=baseCarrierSHA;originalCarrier.sourceTetTableSHA256=tetSHA;originalCarrier.sourceNodeTableSHA256=nodeSHA;

        held={{.childRow=0,.parentRow=0,.oldExpandedParentRow=0,.oldCarrierRow=0,
               .massKg=.25,.sourceMeasureM3=.00025,.recordedDistanceM=0.0}};
        for(std::uint32_t r=1;r<4;++r)lp.push_back({.positiveIndex=r-1,.childRow=r,.parentRow=0,
            .flowMassKg=.25,.barycentric=bary[r],.claimedDistanceM=0.0});
        provenance.heldPartialManifestSHA256=std::string(64,'4');
        provenance.heldCarrierFlowCSV_SHA256=std::string(64,'5');
        provenance.lpProposalSHA256=std::string(64,'6');
        provenance.lpInputManifestSHA256=std::string(64,'7');
        provenance.feasibilitySourceSHA256=std::string(64,'9');
        provenance.heldDinicAugmentationCount=1;provenance.lpPositiveVariableCount=lp.size();
        provenance.heldPositiveFlowRows=held.size();provenance.lpCandidateEdgeCount=3;
        provenance.heldCandidateEdgeCount=1;provenance.semanticCandidateEdgeCount=4;
    }
};

static bool passRole(std::uint32_t material) {
    Fixture f(material);
    SourceCarrierLPFinalizedFlow out;std::string error;
    if(!source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
         f.held,f.lp,f.provenance,out,error)) {
        std::cerr<<"role "<<material<<" failed: "<<error<<"\n";return false;
    }
    if(out.result.childMassKg.size()!=4||out.trace.childParentCarrierRows.size()!=4||
       out.trace.childParentCarrierOffsets.size()!=5||out.result.childParentRows.size()!=4||
       out.result.transportEdgeCount!=4||out.result.augmentationCount!=1||
       out.lpPositiveVariableCount!=3||out.heldPositiveFlowRows!=1) {
        std::cerr<<"role "<<material<<" output domain/count mismatch\n";return false;
    }
    for(std::size_t c=0;c<4;++c)
        if(std::abs(out.result.childMassKg[c]-.25)>1e-12||
           std::abs(out.result.childImpliedDensityKgM3[c]-1000.0)>1e-9) {
            std::cerr<<"role "<<material<<" child mass/density mismatch at "<<c<<"\n";return false;
        }
    return true;
}
static bool exportContract(std::uint32_t material) {
    Fixture f(material);
    SourceCarrierLPFinalizedFlow flow;std::string error;
    if(!source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
         f.held,f.lp,f.provenance,flow,error)) {
        std::cerr<<"export fixture build failed: "<<error<<"\n";return false;
    }
    const auto nonce=std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root=std::filesystem::temp_directory_path()/
        ("numi-source-carrier-lp-export-"+std::to_string(nonce));
    source_carrier_lp_detail::SourceCarrierLPExportMetadata metadata;
    metadata.feasibilitySourceSHA256=f.provenance.feasibilitySourceSHA256;
    metadata.inputSHA256={{"input_child_node_sha256",std::string(64,'a')},
                          {"input_child_element_sha256",std::string(64,'b')},
                          {"input_parent_mass_csv_sha256",std::string(64,'c')},
                          {"input_parent_label_csv_sha256",std::string(64,'d')},
                          {"support_binary_sha256",std::string(64,'e')},
                          {"input_region_label_sha256",f.regionSHA}};
    if(!writeExistingSourceCarrierCSVContract(root,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          flow.result,flow.trace,f.provenance,metadata,error)) {
        std::cerr<<"export fixture failed: "<<error<<"\n";
        std::filesystem::remove_all(root);return false;
    }
    const std::array<std::pair<const char*,const char*>,6> expected={{
        {"transport-child-mass.csv","child_row,reference_volume_m3,mass_kg,density_kg_m3"},
        {"transport-flow.csv","child_row,parent_row,mass_kg,source_measure_m3,distance_m"},
        {"source-carrier-flow.csv","child_row,parent_row,expanded_parent_row,carrier_row,mass_kg,source_measure_m3,distance_m"},
        {"transport-child-source-donor.csv","child_row,source_donor_body,mass_kg"},
        {"transport-source-parent.csv","parent_row,global_source330_row,material,geometric_donor,region,stable_source_face,mass_kg,reference_volume_m3,source_centroid_x_m,source_centroid_y_m,source_centroid_z_m"},
        {"transport-source-parent-donor.csv","parent_row,source_donor_body,mass_kg"}}};
    bool ok=true;
    for(const auto& [name,header]:expected) {
        std::ifstream file(root/name,std::ios::binary);std::string line;
        if(!file||!std::getline(file,line)||line!=header) {
            std::cerr<<"export fixture header mismatch: "<<name<<"\n";ok=false;
        }
    }
    std::ifstream receiptFile(root/"result.json",std::ios::binary);
    std::ostringstream buffer;buffer<<receiptFile.rdbuf();const auto receipt=buffer.str();
    const auto role=material==15u?"SKIN":"FAT";
    const std::vector<std::string> tokens={std::string("\"status\": \"PASS_SOURCE_ONLY_")+role+"_SOURCE_CARRIER_FEASIBLE_ALLOCATION\"",
                           "\"complete_allocation\": true","\"physical_admission\": false",
                           "\"native_steps\": 0","\"recorded_flow_feasibility\": \"PASS_SOLVER_FREE\"",
                           "\"deterministic_optimizer_replay\": \"NOT_RUN\"",
                           "\"held_partial_manifest_sha256\": \""+std::string(64,'4')+"\"",
                           "\"lp_proposal_sha256\": \""+std::string(64,'6')+"\"",
                           "\"semantic_candidate_edge_count_basis\": \"pinned held+LP proposal graph count; feasibility does not reconstruct optimization graph\"",
                           "\"held_dinic_augmentation_count\": 1",
                           "\"lp_positive_variable_count\": 3",
                           "\"candidate_edges\": 4",
                           "\"child_mesh_sha256\": \""+f.childSHA+"\"",
                           "\"input_child_node_sha256\": \""+std::string(64,'a')+"\"",
                           "\"input_parent_mass_csv_sha256\": \""+std::string(64,'c')+"\"",
                           "\"support_binary_sha256\": \""+std::string(64,'e')+"\"",
                           "\"input_region_label_sha256\": \""+f.regionSHA+"\"",
                           "\"input_parent_mass_sha256\": \""+f.pMassSHA+"\"",
                           "\"semantic_support_sha256\": \""+f.supportSHA+"\"",
                           "\"transport_receipt_sha256\": \""+flow.result.receiptSHA256+"\"",
                           "\"input_parent_source_sha256\": \""+f.pSourceSHA+"\"",
                           "\"source_carrierized_parents\": 1","\"source_carriers\": 4"};
    for(const auto& token:tokens)
        if(receipt.find(token)==std::string::npos) {
            std::cerr<<"export fixture receipt missing: "<<token<<"\n";ok=false;
        }
    if(receipt.find("PASS_DETERMINISTIC")!=std::string::npos)ok=false;

    const auto rejectPath=root.parent_path()/(root.filename().string()+"-reject");
    source_carrier_lp_detail::SourceCarrierLPExportMetadata missingPinMetadata=metadata;
    missingPinMetadata.inputSHA256.erase("input_parent_label_csv_sha256");
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          flow.result,flow.trace,f.provenance,missingPinMetadata,error)||std::filesystem::exists(rejectPath)) {
        std::cerr<<"missing required file pin was not rejected before creating output\\n";ok=false;
    }
    source_carrier_lp_detail::SourceCarrierLPExportMetadata missingRegionMetadata=metadata;
    missingRegionMetadata.inputSHA256.erase("input_region_label_sha256");
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          flow.result,flow.trace,f.provenance,missingRegionMetadata,error)||std::filesystem::exists(rejectPath)) {
        std::cerr<<"missing region-label pin was not rejected before creating output\n";ok=false;
    }
    source_carrier_lp_detail::SourceCarrierLPExportMetadata invalidRegionMetadata=metadata;
    invalidRegionMetadata.inputSHA256["input_region_label_sha256"]=std::string(64,'0');
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          flow.result,flow.trace,f.provenance,invalidRegionMetadata,error)||std::filesystem::exists(rejectPath)) {
        std::cerr<<"mismatched region-label pin was not rejected before creating output\\n";ok=false;
    }
    source_carrier_lp_detail::SourceCarrierLPExportMetadata invalidMetadata=metadata;invalidMetadata.physicalAdmission=true;
    SourceCarrierLPFinalizedFlow malformed=flow;malformed.trace.childParentCarrierOffsets.pop_back();
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          malformed.result,malformed.trace,f.provenance,metadata,error)||std::filesystem::exists(rejectPath)) {
        std::cerr<<"malformed exporter input was not rejected before creating output\n";ok=false;
    }
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          flow.result,flow.trace,f.provenance,invalidMetadata,error)||std::filesystem::exists(rejectPath)) {
        std::cerr<<"physical-admission metadata was not rejected before creating output\n";ok=false;
    }
    SourceCarrierLPFinalizedFlow badAggregate=flow;
    badAggregate.result.childParentTransportMassKg[0]*=0.5;
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          badAggregate.result,badAggregate.trace,f.provenance,metadata,error)||std::filesystem::exists(rejectPath)) {
        std::cerr<<"aggregate/carrier flow mismatch was not rejected before creating output\\n";ok=false;
    }
    std::filesystem::create_directories(rejectPath);
    {std::ofstream sentinel(rejectPath/"keep.txt");sentinel<<"preserve\n";}
    if(writeExistingSourceCarrierCSVContract(rejectPath,f.input,f.support,f.originalCarrier.globalSourceTetRows,
          flow.result,flow.trace,f.provenance,metadata,error)||!std::filesystem::exists(rejectPath/"keep.txt")) {
        std::cerr<<"nonempty output directory was overwritten\n";ok=false;
    }
    std::filesystem::remove_all(root);std::filesystem::remove_all(rejectPath);
    return ok;
}

static bool rejectBadDistance() {
    Fixture f(16);f.lp[0].claimedDistanceM=.001;
    SourceCarrierLPFinalizedFlow out;std::string error;
    return !source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
        f.held,f.lp,f.provenance,out,error);
}
static bool rejectOldExpandedMismatch() {
    Fixture f(15);f.held[0].oldExpandedParentRow=1;
    SourceCarrierLPFinalizedFlow out;std::string error;
    return !source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
        f.held,f.lp,f.provenance,out,error);
}
static bool rejectCarrierMoment() {
    Fixture f(16);f.lp[0].barycentric[0]+=.01;f.lp[0].barycentric[1]-=.01;
    f.lp[0].claimedDistanceM=dist(baryPoint(f.tet,f.lp[0].barycentric),f.childCentroids[1]);
    SourceCarrierLPFinalizedFlow out;std::string error;
    return !source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
        f.held,f.lp,f.provenance,out,error);
}
static bool rejectPureLPWithoutHeldDinicEvidence() {
    Fixture f(15);f.held.clear();f.provenance.heldPositiveFlowRows=0;
    f.provenance.heldDinicAugmentationCount=0;f.provenance.heldCandidateEdgeCount=0;
    SourceCarrierLPFinalizedFlow out;std::string error;
    return !source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
        f.held,f.lp,f.provenance,out,error);
}
static bool rejectWrongMaterial() {
    Fixture f(16);f.childMaterial[2]=15;
    SourceCarrierLPFinalizedFlow out;std::string error;
    return !source_carrier_lp_detail::buildAndVerify(f.input,f.support,f.originalCarrier,f.scale,
        f.held,f.lp,f.provenance,out,error);
}
int main() {
    if(!passRole(15)||!passRole(16)||!exportContract(15)||!exportContract(16)||
       !rejectBadDistance()||!rejectOldExpandedMismatch()||!rejectCarrierMoment()||!rejectWrongMaterial()||
       !rejectPureLPWithoutHeldDinicEvidence())return 1;
    std::cout<<"PASS source-carrier LP feasibility/export: roles 15/16 accepted; six CSVs and truthful source-only receipt verified; malformed outputs, invalid registration and region-label pins, wrong radius, stale expanded row, altered source moment, cross-material support, and pure-LP/no-held-evidence path rejected; optimizer not invoked\n";
    return 0;
}

