#include "NumiHumanCurrentFractionalSupportNPZ_029.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace metalrobo::human::coarse_tissue;
namespace fs=std::filesystem;

static const std::string kHeader=
    "record_kind,role,carrier_construction,packet_carrier_row,local_parent_row,"
    "global_source330_tet_row,child_local_row,child_global_row,material,donor,region,"
    "provenance_mode,p1_group_id,projection_csv_row,bary0,bary1,bary2,bary3,weight,"
    "source_x_m,source_y_m,source_z_m,child_centroid_x_m,child_centroid_y_m,"
    "child_centroid_z_m,distance_m";
static constexpr std::array<std::array<double,4>,11> kSites{{
    {{1.0,0.0,0.0,0.0}},{{0.0,1.0,0.0,0.0}},{{0.0,0.0,1.0,0.0}},{{0.0,0.0,0.0,1.0}},
    {{0.5,0.5,0.0,0.0}},{{0.5,0.0,0.5,0.0}},{{0.5,0.0,0.0,0.5}},
    {{0.0,0.5,0.5,0.0}},{{0.0,0.5,0.0,0.5}},{{0.0,0.0,0.5,0.5}},
    {{0.25,0.25,0.25,0.25}}
}};
static constexpr std::array<std::array<double,3>,4> kTet{{
    {{0.250,0.250,0.250}},{{0.251,0.250,0.250}},
    {{0.250,0.251,0.250}},{{0.250,0.250,0.251}}
}};
static constexpr std::array<double,3> kChild{{0.25025,0.25025,0.25025}};

static std::array<double,3> pointAt(const std::array<double,4>& b){
    std::array<double,3> p{};
    for(std::size_t i=0;i<4u;++i)for(std::size_t a=0;a<3u;++a)p[a]+=b[i]*kTet[i][a];
    return p;
}
static double distance(const std::array<double,3>& a,const std::array<double,3>& b){
    double s=0.0;for(std::size_t i=0;i<3u;++i){const auto d=a[i]-b[i];s+=d*d;}return std::sqrt(s);
}
struct CsvVariant { int row=-1; std::string mode="source_tet_canonical_quadrature";
    std::string construction="source_tet_canonical_degree1_quadrature_1046e";
    int p1=-1,projection=-1; bool noncanonical=false,childSlot=false; };
static fs::path writeRoleCSV(const std::string& role,std::uint32_t material,
        std::uint32_t donor,std::uint32_t region,std::uint32_t globalParent,
        std::uint32_t child,const CsvVariant& v={}){
    const auto path=fs::temp_directory_path()/("h706-029-"+role+"-canonical.csv");
    std::ofstream out(path,std::ios::binary|std::ios::trunc);out<<kHeader<<"\n"<<std::setprecision(17);
    for(std::size_t i=0;i<kSites.size();++i){
        auto b=kSites[i]; if(v.noncanonical&&static_cast<int>(i)==v.row)b={0.30,0.20,0.25,0.25};
        const auto p=pointAt(b);const auto d=distance(p,kChild);
        const auto rowMode=(static_cast<int>(i)==v.row?v.mode:std::string("source_tet_canonical_quadrature"));
        const auto construction=(static_cast<int>(i)==v.row?v.construction:std::string("source_tet_canonical_degree1_quadrature_1046e"));
        const int p1=(static_cast<int>(i)==v.row?v.p1:-1);
        const int proj=(static_cast<int>(i)==v.row?v.projection:-1);
        const bool childSlot=(static_cast<int>(i)==v.row&&v.childSlot);
        out<<(childSlot?"carrier_child_slot":"carrier")<<','<<role<<','<<construction<<','<<i<<",0,"<<globalParent<<",0,"<<child<<','
           <<material<<','<<donor<<','<<region<<','<<rowMode<<','<<p1<<','<<proj<<','
           <<b[0]<<','<<b[1]<<','<<b[2]<<','<<b[3]<<','<<(1.0/11.0)<<','
           <<p[0]<<','<<p[1]<<','<<p[2]<<','<<kChild[0]<<','<<kChild[1]<<','<<kChild[2]<<','<<d<<"\n";
    }
    return path;
}
static bool load(const fs::path& path,std::uint32_t role,std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
        CurrentCarrierWitnessCSVSummary& summary,std::string& sha,std::string& error){
    return physical_skin_child_mass_detail::sha256File(path,sha,error)&&
        loadCurrentCarrierWitnessCSV(path,sha,role,rows,summary,error);
}
static bool rejectedCSV(const CsvVariant& variant,const char* label){
    auto p=writeRoleCSV("Skin",15u,43u,43u,100u,0u,variant);std::vector<CurrentCarrierWitnessEvidenceRow> rows;
    CurrentCarrierWitnessCSVSummary summary;std::string sha,error;
    const bool parsed=load(p,15u,rows,summary,sha,error);fs::remove(p);
    if(parsed){std::cerr<<"unexpected CSV acceptance: "<<label<<"\n";return false;}return true;
}
static ConservativeSpatialMassTransportInput makeInput(){
    static const std::array<SpatialMassTransportPoint,4> parentPoints{{
        {0.25025,0.25025,0.25025},{0.25025,0.25025,0.25025},
        {0.25025,0.25025,0.25025},{0.25025,0.25025,0.25025}}};
    static const std::array<double,4> parentVolumes{0.001,0.001,0.0001,0.0001};
    static const std::array<double,4> parentMass{1.0,1.0,0.1,0.1};
    static const std::array<std::uint32_t,4> parentMaterial{15u,16u,15u,16u};
    static const std::array<std::uint32_t,4> parentDonor{43u,42u,99u,99u};
    static const std::array<std::uint32_t,4> parentRegion{43u,42u,99u,99u};
    static const std::array<std::uint32_t,5> donorOffsets{0u,1u,2u,3u,4u};
    static const std::array<std::uint32_t,4> donorRows{43u,42u,99u,99u};
    static const std::array<double,4> donorMass{1.0,1.0,0.1,0.1};
    static const std::array<SpatialMassTransportPoint,2> childPoints{{
        {0.25025,0.25025,0.25025},{0.25025,0.25025,0.25025}}};
    static const std::array<double,2> childVolumes{0.0011,0.0011};
    static const std::array<std::uint32_t,2> childMaterial{15u,16u},childDonor{99u,99u},childRegion{99u,99u};
    static const std::array<double,17> lower=[] {std::array<double,17>x{};x.fill(900.0);return x;}();
    static const std::array<double,17> upper=[] {std::array<double,17>x{};x.fill(1100.0);return x;}();
    static const std::string massSHA(64u,'1'),sourceSHA(64u,'2'),meshSHA(64u,'3'),regionSHA(64u,'4'),frameSHA(64u,'5');
    ConservativeSpatialMassTransportInput in{};in.parentCentroidsWorld=parentPoints;in.parentReferenceVolumeM3=parentVolumes;
    in.parentMassKg=parentMass;in.parentMaterialIndices=parentMaterial;in.parentDonorBodyIndices=parentDonor;in.parentAnatomicalRegionIndices=parentRegion;
    in.parentSourceDonorOffsets=donorOffsets;in.parentSourceDonorBodyIndices=donorRows;in.parentSourceDonorMassKg=donorMass;
    in.childCentroidsWorld=childPoints;in.childReferenceVolumeM3=childVolumes;in.childMaterialIndices=childMaterial;
    in.childDonorBodyIndices=childDonor;in.childAnatomicalRegionIndices=childRegion;in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
    in.parentMassPacketSHA256=massSHA;in.parentSourceContributionSHA256=sourceSHA;in.childMeshSHA256=meshSHA;in.regionLabelSHA256=regionSHA;in.coordinateFrameSHA256=frameSHA;
    in.maximumTransportDistanceM=0.019353;in.maximumAllowedEdges=1000u;in.maximumAugmentations=1000u;return in;
}
static ConservativeSpatialChildSemanticSupportData makeSupport(){
    ConservativeSpatialChildSemanticSupportData s;
    s.childOffsets={0u,1u,2u};s.slotGroupIndices={0u,1u};s.volumeFractions={1.0,1.0};
    s.childAdmissibleGroupOffsets={0u,1u,2u};s.childAdmissibleGroupIndices={0u,1u};
    s.groupMaterialIndices={15u,16u,15u,16u};s.donorBodyIndices={99u,99u,43u,42u};
    s.anatomicalRegionIndices={99u,99u,43u,42u};s.derivedGroupIdentityIDs={1u,2u,3u,4u};
    s.parentStableSourceFaceIdentities={501u,502u,503u,504u};s.sourceParentTetOffsets={0u,1u,2u,3u,4u};
    s.sourceParentTetRows={2u,3u,0u,1u};s.sourceStableFaceOffsets={0u,1u,2u,3u,4u};
    s.sourceStableFaceRows={503u,504u,501u,502u};s.cellAdmissibleRegionOffsets={0u,2u,4u};
    s.cellAdmissibleRegionIds={43u,99u,42u,99u};s.admissibilitySourceSupportSHA256=std::string(64u,'a');
    s.admissibleGroupEvidenceSHA256=std::string(64u,'b');s.admissibilityMaximumSourceRegionHops=1u;s.typedSupportSHA256=std::string(64u,'c');return s;
}
static CurrentSourceCarrierQuadratureData makeCarrier(){
    CurrentSourceCarrierQuadratureData c;c.parentOffsets={0u,11u,22u,22u,22u};c.globalSourceTetRows={100u,200u,300u,400u};
    c.sourceTetNodeRows={{{0u,1u,2u,3u}},{{4u,5u,6u,7u}},{{8u,9u,10u,11u}},{{12u,13u,14u,15u}}};
    c.parentReferenceTetWorldVertices={kTet,kTet,kTet,kTet};
    for(std::uint32_t role=0;role<2u;++role)for(std::uint32_t i=0;i<11u;++i){
        c.carrierWeights.push_back(1.0/11.0);c.carrierBarycentric.push_back(kSites[i]);
        c.witnessChildRows.push_back(role);c.witnessChildCentroidDistancesM.push_back(distance(pointAt(kSites[i]),kChild));
        c.carrierRoleLocalRows.push_back(i);c.carrierRoleMaterialIndices.push_back(role==0u?15u:16u);
        c.carrierRoleLocalChildRows.push_back(0u);c.carrierRoleLocalParentRows.push_back(0u);
    }
    c.sourceCarrierPacketSHA256=std::string(64u,'d');c.sourceTetTableSHA256=std::string(64u,'e');c.sourceNodeTableSHA256=std::string(64u,'f');return c;
}
static bool bindCanonical(std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
        ConservativeSpatialChildSemanticSupportData& support,const ConservativeSpatialMassTransportInput& input,
        const CurrentSourceCarrierQuadratureData& carrier,std::span<const std::uint32_t> skinRows,
        std::span<const std::uint32_t> fatRows,std::string& error){
    return bindCurrentCarrierWitnessAllowlist(support,input,carrier,rows,{},std::string(64u,'e'),std::string(64u,'f'),error,skinRows,fatRows);
}
static constexpr std::array<std::array<double,4>,4> kProjectionSites{{
    {{0.4,0.2,0.2,0.2}},{{0.2,0.4,0.2,0.2}},
    {{0.2,0.2,0.4,0.2}},{{0.2,0.2,0.2,0.4}}
}};
static fs::path writeProjectionCSV(const std::string& role,std::uint32_t material,
        std::uint32_t donor,std::uint32_t region,std::uint32_t globalParent,
        std::uint32_t child,const std::array<std::array<double,4>,4>& sites,
        const std::string& tag){
    const auto path=fs::temp_directory_path()/("h706-031-"+tag+"-"+role+".csv");
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    out<<kHeader<<"\n"<<std::setprecision(17);
    for(std::size_t i=0;i<sites.size();++i){
        const auto& b=sites[i];const auto p=pointAt(b);const auto d=distance(p,kChild);
        out<<"carrier,"<<role<<",source_tetra_anchor_refinement_1046d,"<<i<<",0,"<<globalParent<<",0,"<<child<<','
           <<material<<','<<donor<<','<<region<<",CurrentCarrierPointProjection,-1,-1,"
           <<b[0]<<','<<b[1]<<','<<b[2]<<','<<b[3]<<",0.25,"
           <<p[0]<<','<<p[1]<<','<<p[2]<<','<<kChild[0]<<','<<kChild[1]<<','<<kChild[2]<<','<<d<<"\n";
    }
    return path;
}
static CurrentSourceCarrierQuadratureData makeProjectionCarrier(
        const std::array<std::array<double,4>,4>& skin,
        const std::array<std::array<double,4>,4>& fat){
    CurrentSourceCarrierQuadratureData c;c.parentOffsets={0u,4u,8u,8u,8u};
    c.globalSourceTetRows={100u,200u,300u,400u};
    c.sourceTetNodeRows={{{0u,1u,2u,3u}},{{4u,5u,6u,7u}},{{8u,9u,10u,11u}},{{12u,13u,14u,15u}}};
    c.parentReferenceTetWorldVertices={kTet,kTet,kTet,kTet};
    for(std::uint32_t i=0;i<4u;++i){
        c.carrierWeights.push_back(0.25);c.carrierBarycentric.push_back(skin[i]);
        c.witnessChildRows.push_back(0u);c.witnessChildCentroidDistancesM.push_back(distance(pointAt(skin[i]),kChild));
    }
    for(std::uint32_t i=0;i<4u;++i){
        c.carrierWeights.push_back(0.25);c.carrierBarycentric.push_back(fat[i]);
        c.witnessChildRows.push_back(1u);c.witnessChildCentroidDistancesM.push_back(distance(pointAt(fat[i]),kChild));
    }
    for(std::uint32_t i=0;i<4u;++i){
        c.carrierRoleLocalRows.push_back(i);c.carrierRoleMaterialIndices.push_back(15u);
        c.carrierRoleLocalChildRows.push_back(0u);c.carrierRoleLocalParentRows.push_back(0u);
    }
    for(std::uint32_t i=0;i<4u;++i){
        c.carrierRoleLocalRows.push_back(i);c.carrierRoleMaterialIndices.push_back(16u);
        c.carrierRoleLocalChildRows.push_back(0u);c.carrierRoleLocalParentRows.push_back(0u);
    }
    c.sourceCarrierPacketSHA256=std::string(64u,'d');c.sourceTetTableSHA256=std::string(64u,'e');c.sourceNodeTableSHA256=std::string(64u,'f');return c;
}
static bool loadProjectionRows(const std::string& tag,
        const std::array<std::array<double,4>,4>& skinSites,
        const std::array<std::array<double,4>,4>& fatSites,
        std::uint32_t skinDonor,std::vector<CurrentCarrierWitnessEvidenceRow>& rows,
        fs::path& skinPath,fs::path& fatPath,std::string& error){
    skinPath=writeProjectionCSV("Skin",15u,skinDonor,43u,100u,0u,skinSites,tag);
    fatPath=writeProjectionCSV("Fat",16u,42u,42u,200u,1u,fatSites,tag);
    std::vector<CurrentCarrierWitnessEvidenceRow> skinRows,fatRows;
    CurrentCarrierWitnessCSVSummary skinSummary,fatSummary;std::string skinSHA,fatSHA;
    if(!load(skinPath,15u,skinRows,skinSummary,skinSHA,error)||
       !load(fatPath,16u,fatRows,fatSummary,fatSHA,error))return false;
    if(skinSummary.carrierRecordCount!=4u||fatSummary.carrierRecordCount!=4u||
       skinRows.size()!=4u||fatRows.size()!=4u){error="projection sidecar did not cover four primary points per role";return false;}
    rows=std::move(skinRows);rows.insert(rows.end(),fatRows.begin(),fatRows.end());return true;
}
static bool runProjectionControls(){
    auto skinSites=kProjectionSites,fatSites=kProjectionSites;
    std::array<double,4> mean{};for(const auto& b:kProjectionSites)
        for(std::size_t j=0;j<4u;++j)mean[j]+=0.25*b[j];
    if(mean!=std::array<double,4>{0.25,0.25,0.25,0.25}){
        std::cerr<<"projection fixture does not reproduce exact parent first moments\n";return false;
    }
    fs::path skinPath,fatPath;std::vector<CurrentCarrierWitnessEvidenceRow> rows;std::string error;
    if(!loadProjectionRows("positive",skinSites,fatSites,43u,rows,skinPath,fatPath,error)){
        std::cerr<<"projection sidecar parse failed: "<<error<<"\n";return false;
    }
    auto input=makeInput();auto carrier=makeProjectionCarrier(skinSites,fatSites);auto support=makeSupport();
    const std::array<std::uint32_t,1> skinMap{0u},fatMap{1u};
    if(!bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){
        std::cerr<<"arbitrary CurrentCarrierPointProjection binding failed: "<<error<<"\n";return false;
    }
    std::vector<std::uint32_t> parentGroup,carrierRows;std::vector<std::map<std::uint32_t,double>> regionPreference;
    std::vector<std::vector<std::uint32_t>> childGroups;std::vector<SpatialMassTransportPoint> carrierPoints;
    std::vector<double> strictNearest;
    if(!source_carrier_transport_detail::validateCarrierInput(input,support.view(),carrier.view(),
        parentGroup,regionPreference,childGroups,carrierPoints,carrierRows,strictNearest,error)||
        carrierRows!=std::vector<std::uint32_t>{0u,1u}||carrierPoints.size()!=8u){
        std::cerr<<"native arbitrary projection validation failed: "<<error<<"\n";return false;
    }
    fs::remove(skinPath);fs::remove(fatPath);

    auto badMomentSites=skinSites;badMomentSites[0]={0.50,0.10,0.20,0.20};
    if(!loadProjectionRows("bad-moment",badMomentSites,fatSites,43u,rows,skinPath,fatPath,error))return false;
    input=makeInput();carrier=makeProjectionCarrier(badMomentSites,fatSites);support=makeSupport();
    if(!bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){
        std::cerr<<"bad-moment witness failed too early at point binding: "<<error<<"\n";return false;
    }
    parentGroup.clear();carrierRows.clear();regionPreference.clear();childGroups.clear();carrierPoints.clear();strictNearest.clear();error.clear();
    if(source_carrier_transport_detail::validateCarrierInput(input,support.view(),carrier.view(),
        parentGroup,regionPreference,childGroups,carrierPoints,carrierRows,strictNearest,error)){
        std::cerr<<"native owner accepted a non-centroid-preserving arbitrary quadrature\n";return false;
    }
    fs::remove(skinPath);fs::remove(fatPath);

    auto outsideSites=skinSites;outsideSites[0]={-0.01,0.41,0.30,0.30};
    if(!loadProjectionRows("outside-tet",outsideSites,fatSites,43u,rows,skinPath,fatPath,error))return false;
    input=makeInput();carrier=makeProjectionCarrier(outsideSites,fatSites);support=makeSupport();error.clear();
    if(bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){
        std::cerr<<"binder accepted a carrier point outside its source tetrahedron\n";return false;
    }
    fs::remove(skinPath);fs::remove(fatPath);

    if(!loadProjectionRows("wrong-tuple",skinSites,fatSites,44u,rows,skinPath,fatPath,error))return false;
    input=makeInput();carrier=makeProjectionCarrier(skinSites,fatSites);support=makeSupport();error.clear();
    if(bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){
        std::cerr<<"binder accepted an arbitrary point with the wrong exact source tuple\n";return false;
    }
    fs::remove(skinPath);fs::remove(fatPath);

    if(!loadProjectionRows("radius",skinSites,fatSites,43u,rows,skinPath,fatPath,error))return false;
    input=makeInput();input.maximumTransportDistanceM=1.0e-8;carrier=makeProjectionCarrier(skinSites,fatSites);support=makeSupport();error.clear();
    if(bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){
        std::cerr<<"binder accepted arbitrary source points beyond the unchanged radius\n";return false;
    }
    fs::remove(skinPath);fs::remove(fatPath);
    return true;
}

int main(){
    for(const auto& b:kSites)if(!isCanonicalSourceTetDegree1Barycentric(b)){std::cerr<<"canonical site rejected\n";return 1;}
    if(isCanonicalSourceTetDegree1Barycentric({0.30,0.20,0.25,0.25})){std::cerr<<"noncanonical point accepted\n";return 2;}
    double total=0.0;std::array<double,4> moment{};for(const auto& b:kSites){total+=1.0/11.0;for(std::size_t j=0;j<4u;++j)moment[j]+=b[j]/11.0;}
    if(std::abs(total-1.0)>1e-15||std::any_of(moment.begin(),moment.end(),[](double x){return std::abs(x-0.25)>1e-15;})){
        std::cerr<<"canonical equal-positive weights do not reproduce degree-one first moments\n";return 3;
    }
    CsvVariant v;v.row=0;v.noncanonical=true;if(!rejectedCSV(v,"noncanonical bary"))return 4;
    v={};v.row=0;v.p1=0;if(!rejectedCSV(v,"false P1 group claim"))return 5;
    v={};v.row=0;v.projection=0;if(!rejectedCSV(v,"false projection claim"))return 6;
    v={};v.row=0;v.mode="CurrentCarrierPointProjection";if(!rejectedCSV(v,"mode/construction mismatch"))return 7;
    v={};v.row=0;v.childSlot=true;if(!rejectedCSV(v,"canonical extra child slot"))return 8;

    auto skinPath=writeRoleCSV("Skin",15u,43u,43u,100u,0u),fatPath=writeRoleCSV("Fat",16u,42u,42u,200u,1u);
    std::vector<CurrentCarrierWitnessEvidenceRow> rows,part;CurrentCarrierWitnessCSVSummary ss,fsum;
    std::string skinSHA,fatSHA,error;
    if(!load(skinPath,15u,rows,ss,skinSHA,error)||!load(fatPath,16u,part,fsum,fatSHA,error)){
        std::cerr<<"canonical CSV parse failed: "<<error<<"\n";return 9;
    }
    rows.insert(rows.end(),part.begin(),part.end());
    if(rows.size()!=22u||ss.carrierRecordCount!=11u||fsum.carrierRecordCount!=11u){std::cerr<<"canonical sidecar coverage is incomplete\n";return 10;}
    auto input=makeInput();auto carrier=makeCarrier();auto initial=makeSupport();auto support=initial;
    const std::array<std::uint32_t,1> skinMap{0u},fatMap{1u};
    if(!bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){std::cerr<<"canonical sidecar/binder failed: "<<error<<"\n";return 11;}
    if(support.childAdmissibleGroupOffsets!=std::vector<std::uint32_t>{0u,2u,4u}||
       support.childAdmissibleGroupIndices!=std::vector<std::uint32_t>{0u,2u,1u,3u}||
       support.volumeFractions!=std::vector<double>{1.0,1.0}){
        std::cerr<<"canonical mode changed P1 preferences or failed exact group insertion\n";return 12;
    }
    std::vector<std::uint32_t> parentGroup,carrierRows;
    std::vector<std::map<std::uint32_t,double>> regionPreference;
    std::vector<std::vector<std::uint32_t>> childGroups;
    std::vector<SpatialMassTransportPoint> carrierPoints;
    std::vector<double> strictNearest;
    if(!source_carrier_transport_detail::validateCarrierInput(input,support.view(),carrier.view(),
          parentGroup,regionPreference,childGroups,carrierPoints,carrierRows,strictNearest,error)||
       carrierRows!=std::vector<std::uint32_t>{0u,1u}||carrierPoints.size()!=22u){
        std::cerr<<"native positive degree-one carrier validation failed: "<<error<<"\n";return 13;
    }
    support=initial;if(bindCanonical(rows,support,input,carrier,{},fatMap,error)){std::cerr<<"canonical mode accepted without role-local child crosswalk\n";return 14;}
    support=initial;const std::array<std::uint32_t,1> wrongSkinMap{1u};if(bindCanonical(rows,support,input,carrier,wrongSkinMap,fatMap,error)){std::cerr<<"canonical mode accepted wrong role child crosswalk\n";return 14;}
    support=initial;auto badRows=rows;badRows[0].claimedSourcePointWorld[0]+=1.0e-4;
    if(bindCanonical(badRows,support,input,carrier,skinMap,fatMap,error)){std::cerr<<"canonical mode accepted a false source coordinate\n";return 15;}
    support=initial;support.cellAdmissibleRegionIds={99u,100u,42u,99u};
    if(bindCanonical(rows,support,input,carrier,skinMap,fatMap,error)){std::cerr<<"canonical mode accepted a source region outside the authenticated one-hop list\n";return 16;}
    support=initial;auto tooFar=input;tooFar.maximumTransportDistanceM=1.0e-8;
    if(bindCanonical(rows,support,tooFar,carrier,skinMap,fatMap,error)){std::cerr<<"canonical mode accepted a source point outside unchanged R\n";return 17;}
    fs::remove(skinPath);fs::remove(fatPath);
    if(!runProjectionControls())return 18;
    std::cout<<"PASS 031 arbitrary CurrentCarrierPointProjection: noncanonical barycentrics, parent first moments, tuple/one-hop/R replay; bad moments, outside tet, tuple and radius rejected; canonical restrictions retained\n";
    std::cout<<"PASS 029 canonical source-tet quadrature: 11 positive degree-one sites, owner/packet bary replay, exact child map, tuple group, one-hop and unchanged R; malformed provenance rejected\n";
    return 0;
}
