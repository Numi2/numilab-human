#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "ConservativeSpatialFractionalSemanticSupport.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace metalrobo::human::coarse_tissue;
namespace fs=std::filesystem;
static bool tokens(std::istream& in,std::vector<std::string>& out){
 std::string line;
 while(std::getline(in,line)){
  if(line.size()>16384)return false;
  auto p=line.find('#');if(p!=std::string::npos)line.resize(p);
  std::istringstream ss(line);out.clear();std::string s;
  while(ss>>s)out.push_back(s);
  if(!out.empty())return true;
 }
 return false;
}
template<class T> static bool number(const std::string& s,T& v){
 if(s.empty())return false;auto p=std::from_chars(s.data(),s.data()+s.size(),v);
 return p.ec==std::errc{}&&p.ptr==s.data()+s.size();
}
static bool readU32(std::ifstream& f,std::uint32_t& x){return bool(f.read(reinterpret_cast<char*>(&x),4));}
static bool readU64(std::ifstream& f,std::uint64_t& x){return bool(f.read(reinterpret_cast<char*>(&x),8));}
static bool readD(std::ifstream& f,double& x){return bool(f.read(reinterpret_cast<char*>(&x),8));}
static double volume(const V3&a,const V3&b,const V3&c,const V3&d){
 return physical_skin_child_mass_detail::signedTetVolume(a,b,c,d);
}
// Same FSSUP090 driver; Fat is material16 with 850–950 kg/m3 bounds.
int main(int argc,char**argv){
 if(argc!=8){std::cerr<<"usage: driver parent-mass.csv parent-labels.csv node ele support.bin coord-frame-sha result.json\n";return 64;}
 const auto t0=std::chrono::steady_clock::now();
 const fs::path massPath=argv[1],labelsPath=argv[2],nodePath=argv[3],elePath=argv[4],supportPath=argv[5],resultPath=argv[7];
 const std::string frame=argv[6];
 auto writeFailure=[&](const std::string& message,const std::size_t parentCount,const std::size_t childCount){
  fs::create_directories(resultPath.parent_path());std::ofstream o(resultPath);
  const bool inconclusive=message.find("INCONCLUSIVE")!=std::string::npos;
  o<<"{\n  \"status\": \""<<(inconclusive?"INCONCLUSIVE_SOURCE_ONLY_FAT_MASS_TRANSFER":"FAIL_SOURCE_ONLY_FAT_MASS_TRANSFER")<<"\",\n  \"physical_admission\": false,\n  \"native_steps\": 0,\n  \"parent_fat_tets\": "<<parentCount<<",\n  \"child_fat_tets\": "<<childCount<<",\n  \"error\": "<<std::quoted(message)<<",\n  \"elapsed_seconds\": "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"\n}\n";
 };
 PhysicalSkinParentMassPacket all;std::string error;
 if(!loadCompiledSkinFatParentMassPacketCSV(massPath,all,error)){writeFailure("parent packet: "+error,0,0);return 2;}
 if(!loadCompiledSkinFatParentLabelsCSV(labelsPath,all,error)){writeFailure("parent labels: "+error,0,0);return 2;}
 std::cerr<<"PARENT_PACKET_PASS nodes="<<all.referenceWorldPoints.size()<<" cells="<<all.tetrahedra.size()<<std::endl;
 std::string parentMassFileHash,parentLabelsFileHash,supportBinaryHash;
 if(!physical_skin_child_mass_detail::sha256File(massPath,parentMassFileHash,error)||
    !physical_skin_child_mass_detail::sha256File(labelsPath,parentLabelsFileHash,error)||
    !physical_skin_child_mass_detail::sha256File(supportPath,supportBinaryHash,error)){
  writeFailure("input file hash: "+error,0,0);return 2;
 }
 std::ifstream sf(supportPath,std::ios::binary);char magic[8]{};
 if(!sf.read(magic,8)||std::string(magic,8)!="FSSUP090"){writeFailure("support binary magic mismatch",0,0);return 2;}
 std::uint64_t nc64=0,ns64=0,ng64=0,np64=0,npr64=0,nfr64=0;
 if(!readU64(sf,nc64)||!readU64(sf,ns64)||!readU64(sf,ng64)||!readU64(sf,np64)||!readU64(sf,npr64)||!readU64(sf,nfr64)||
    nc64>2000000||ns64>4000000||ng64>10000||np64>200000||npr64>200000||nfr64>200000)
  {writeFailure("support binary count header invalid",0,0);return 2;}
 const auto nc=std::size_t(nc64),ns=std::size_t(ns64),ng=std::size_t(ng64),np=std::size_t(np64),npr=std::size_t(npr64),nfr=std::size_t(nfr64);
 ConservativeSpatialChildSemanticSupportData support;
 auto readVec32=[&](std::vector<std::uint32_t>&v,std::size_t count){
  v.resize(count);return count==0||bool(sf.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*4)));
 };
 auto readVec64=[&](std::vector<std::uint64_t>&v,std::size_t count){
  v.resize(count);return count==0||bool(sf.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*8)));
 };
 auto readVecD=[&](std::vector<double>&v,std::size_t count){
  v.resize(count);return count==0||bool(sf.read(reinterpret_cast<char*>(v.data()),std::streamsize(count*8)));
 };
 std::vector<std::uint32_t> childMaterial,childDonor,childRegion;
 if(!readVec32(support.childOffsets,nc+1)||!readVec32(support.slotGroupIndices,ns)||
    !readVecD(support.volumeFractions,ns)||
    !readVec32(support.groupMaterialIndices,ng)||!readVec32(support.donorBodyIndices,ng)||
    !readVec32(support.anatomicalRegionIndices,ng)||!readVec64(support.derivedGroupIdentityIDs,ng)||
    !readVec32(support.parentStableSourceFaceIdentities,np)||
    !readVec32(support.sourceParentTetOffsets,ng+1)||!readVec32(support.sourceParentTetRows,npr)||
    !readVec32(support.sourceStableFaceOffsets,ng+1)||!readVec32(support.sourceStableFaceRows,nfr)||
    !readVec32(childMaterial,nc)||!readVec32(childDonor,nc)||!readVec32(childRegion,nc))
  {writeFailure("support binary array truncation",0,nc);return 2;}
 char supportDigest[64];
 if(!sf.read(supportDigest,64)){writeFailure("support binary digest truncated",0,nc);return 2;}
 support.typedSupportSHA256.assign(supportDigest,64);
 char adjacencyMagic[8]{};std::uint64_t adjacencyCount64=0;std::uint32_t adjacencyHops=0;
 char adjacencySourceDigest[64]{};
 if(!sf.read(adjacencyMagic,8)||std::string(adjacencyMagic,8)!="FSSADJ01"||
    !readU64(sf,adjacencyCount64)||!readU32(sf,adjacencyHops)||
    adjacencyCount64==0||adjacencyCount64>50000000ull||adjacencyHops!=1u||
    !sf.read(adjacencySourceDigest,64)){
  writeFailure("one-hop admissibility extension header invalid",0,nc);return 2;
 }
 support.admissibilitySourceSupportSHA256.assign(adjacencySourceDigest,64);
 support.admissibilityMaximumSourceRegionHops=adjacencyHops;
 if(!readVec32(support.cellAdmissibleRegionOffsets,nc+1u)||
    !readVec32(support.cellAdmissibleRegionIds,static_cast<std::size_t>(adjacencyCount64))||sf.peek()!=EOF){
  writeFailure("one-hop admissibility extension arrays/trailing data invalid",0,nc);return 2;
 }
 auto validSHA=[](const std::string& value){
  if(value.size()!=64)return false;
  for(const auto c:value)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
  return true;
 };
 if(!validSHA(support.typedSupportSHA256)||!validSHA(support.admissibilitySourceSupportSHA256)||
    support.cellAdmissibleRegionOffsets.front()!=0u||
    support.cellAdmissibleRegionOffsets.back()!=support.cellAdmissibleRegionIds.size()){
  writeFailure("one-hop admissibility digest/CSR endpoint invalid",0,nc);return 2;
 }
 std::cerr<<"SUPPORT_PASS children="<<nc<<" slots="<<ns<<" groups="<<ng<<" parent_fat="<<np<<std::endl;
 if(np==0||np>all.tetrahedra.size()||nc==0||support.childOffsets.size()!=nc+1||
    childMaterial.size()!=nc||childDonor.size()!=nc||childRegion.size()!=nc){
  writeFailure("support count mismatch",0,nc);return 2;
 }
 std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> sourceRows;
 sourceRows.reserve(all.sourceMassContributions.size());
 for(const auto&r:all.sourceMassContributions)
  sourceRows.push_back({.parentNodeIndex=r.parentNodeIndex,.materialIndex=r.materialIndex,
    .sourceDonorBodyIndex=r.donorBodyIndex,.massKg=r.massKg});
 std::vector<std::uint32_t> allOffsets,allDonors;std::vector<double> allDonorMass;
 if(!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
      all.tetrahedronMassKg,all.materialIndices,all.tetrahedra,sourceRows,1.0e-10,5.0e-4,
      allOffsets,allDonors,allDonorMass,error)){
  writeFailure("source donor split: "+error,0,nc);return 2;
 }
 std::vector<std::uint32_t> origToSkin(all.tetrahedra.size(),std::numeric_limits<std::uint32_t>::max());
 std::size_t nParent=0;for(auto m:all.materialIndices)if(m==16u)++nParent;
 if(nParent!=np){writeFailure("filtered source Fat parent count differs from support binary",nParent,nc);return 2;}
 std::vector<SpatialMassTransportPoint> parentCentroids;std::vector<double> parentVolumes,parentMass;
 std::vector<std::uint32_t> parentMaterial,parentDonor,parentRegion,parentStable;
 std::vector<std::uint32_t> parentSourceOffsets{0},parentSourceDonors;std::vector<double> parentSourceMass;
 parentCentroids.reserve(np);parentVolumes.reserve(np);parentMass.reserve(np);
 parentMaterial.reserve(np);parentDonor.reserve(np);parentRegion.reserve(np);parentStable.reserve(np);
 for(std::size_t r=0;r<all.tetrahedra.size();++r)if(all.materialIndices[r]==16u){
  const auto out=static_cast<std::uint32_t>(parentMass.size());origToSkin[r]=out;
  const auto tet=all.tetrahedra[r];SpatialMassTransportPoint c{};
  for(auto node:tet)for(unsigned axis=0;axis<3;++axis)c[axis]+=all.referenceWorldPoints[node][axis]*0.25;
  parentCentroids.push_back(c);parentVolumes.push_back(all.referenceTetVolumeM3[r]);parentMass.push_back(all.tetrahedronMassKg[r]);
  parentMaterial.push_back(all.materialIndices[r]);parentDonor.push_back(all.donorBodyIndices[r]);
  parentRegion.push_back(all.anatomicalRegionIndices[r]);parentStable.push_back(all.stableSourceFaceIdentities[r]);
  for(std::uint32_t q=allOffsets[r];q<allOffsets[r+1];++q){parentSourceDonors.push_back(allDonors[q]);parentSourceMass.push_back(allDonorMass[q]);}
  parentSourceOffsets.push_back(static_cast<std::uint32_t>(parentSourceDonors.size()));
 }
 if(parentStable!=support.parentStableSourceFaceIdentities){writeFailure("filtered source stable-face identity differs from support binary",np,nc);return 2;}
 // The binary producer already expresses source-parent CSR rows in the filtered Skin input domain.
 // Validate complete unique coverage without reinterpreting those row IDs as full-330 rows.
 std::vector<std::uint8_t> parentRowSeen(np,0u);
 for(const auto r:support.sourceParentTetRows){
  if(r>=np||parentRowSeen[r]++){writeFailure("source group CSR has a duplicate or out-of-range filtered Fat parent row",np,nc);return 2;}
 }
 if(support.sourceParentTetRows.size()!=np||
    std::any_of(parentRowSeen.begin(),parentRowSeen.end(),[](std::uint8_t x){return x!=1u;})){
  writeFailure("source group CSR does not cover every filtered Fat parent row exactly once",np,nc);return 2;
 }
 // Read the exact source node/element inputs and use the same F32 widening as the native child importer.
 std::string nodeHash,eleHash;
 if(!physical_skin_child_mass_detail::sha256File(nodePath,nodeHash,error)||
    !physical_skin_child_mass_detail::sha256File(elePath,eleHash,error)){
  writeFailure("child mesh hash: "+error,np,nc);return 2;
 }
 std::ifstream nodes(nodePath),elements(elePath);
 std::vector<std::string> row;std::size_t nodeCount=0,dim=0,attrs=0,markers=0;
 if(!tokens(nodes,row)||row.size()!=4||!number(row[0],nodeCount)||!number(row[1],dim)||
    !number(row[2],attrs)||!number(row[3],markers)||nodeCount==0||dim!=3||attrs!=0||markers>1||
    nodeCount>500000){writeFailure("child node header invalid",np,nc);return 2;}
 std::unordered_map<std::uint32_t,std::uint32_t> nodeMap;nodeMap.reserve(nodeCount*2);
 std::vector<SpatialMassTransportPoint> childPoints;childPoints.reserve(nodeCount);
 for(std::size_t i=0;i<nodeCount;++i){
  if(!tokens(nodes,row)||row.size()!=4+markers){writeFailure("child node row width invalid",np,nc);return 2;}
  std::uint32_t id=0;if(!number(row[0],id)||!nodeMap.emplace(id,static_cast<std::uint32_t>(i)).second){writeFailure("child node identity invalid",np,nc);return 2;}
  SpatialMassTransportPoint p{};
  for(unsigned a=0;a<3;++a){double x=0;if(!number(row[a+1],x)||!std::isfinite(x)){writeFailure("child node coordinate invalid",np,nc);return 2;}p[a]=double(static_cast<float>(x));if(!std::isfinite(p[a])){writeFailure("F32 child node coordinate invalid",np,nc);return 2;}}
  childPoints.push_back(p);
 }
 if(tokens(nodes,row)||!nodes.eof()){writeFailure("child node file has extra data",np,nc);return 2;}
 std::size_t cellCount=0,corners=0,nattrs=0;
 if(!tokens(elements,row)||row.size()!=3||!number(row[0],cellCount)||!number(row[1],corners)||!number(row[2],nattrs)||
    cellCount!=nc||corners!=4||nattrs!=1||cellCount>2000000){writeFailure("child element header mismatch",np,nc);return 2;}
 std::vector<SpatialMassTransportPoint> childCentroids;childCentroids.reserve(nc);
 std::vector<double> childVolumes;childVolumes.reserve(nc);
 std::vector<std::array<std::uint32_t,4>> childTets;childTets.reserve(nc);
 std::vector<std::uint32_t> cellMaterial;cellMaterial.reserve(nc);
 std::set<std::uint32_t> eids;std::set<std::array<std::uint32_t,4>> unique;
 for(std::size_t i=0;i<nc;++i){
  if(!tokens(elements,row)||row.size()!=6){writeFailure("child element row width invalid",np,nc);return 2;}
  std::uint32_t eid=0;double mv=0;
  if(!number(row[0],eid)||!eids.insert(eid).second||!number(row[5],mv)||!std::isfinite(mv)||mv<0||std::floor(mv)!=mv){writeFailure("child element ID/material invalid",np,nc);return 2;}
  std::array<std::uint32_t,4> tet{};
  for(unsigned k=0;k<4;++k){std::uint32_t id=0;if(!number(row[k+1],id)||!nodeMap.contains(id)){writeFailure("child element node ID invalid",np,nc);return 2;}tet[k]=nodeMap.at(id);}
  auto sorted=tet;std::sort(sorted.begin(),sorted.end());if(std::adjacent_find(sorted.begin(),sorted.end())!=sorted.end()||!unique.insert(sorted).second){writeFailure("child tet duplicates node/cell",np,nc);return 2;}
  auto v=volume(childPoints[tet[0]],childPoints[tet[1]],childPoints[tet[2]],childPoints[tet[3]]);
  if(!std::isfinite(v)||v<=0){writeFailure("child tet not positive after F32",np,nc);return 2;}
  SpatialMassTransportPoint c{};for(auto id:tet)for(unsigned a=0;a<3;++a)c[a]+=childPoints[id][a]*0.25;
  childTets.push_back(tet);childVolumes.push_back(v);childCentroids.push_back(c);cellMaterial.push_back(static_cast<std::uint32_t>(mv));
  if(cellMaterial.back()!=childMaterial[i]){writeFailure("TetGen material differs from semantic support material",np,nc);return 2;}
 }
 if(tokens(elements,row)||!elements.eof()){writeFailure("child element file has extra data",np,nc);return 2;}
 spatial_mass_transport_detail::SHA256Builder meshHash;
 meshHash.string("existing-tetgen-reference-child.v1");meshHash.string(nodeHash);meshHash.string(eleHash);
 meshHash.u64(childPoints.size());for(const auto&p:childPoints)for(double x:p)meshHash.real(x);
 meshHash.u64(childTets.size());for(const auto&t:childTets)for(auto x:t)meshHash.u32(x);
 const auto childHash=meshHash.finish();
 ConservativeSpatialMassTransportInput in;
 in.parentCentroidsWorld=parentCentroids;in.parentReferenceVolumeM3=parentVolumes;in.parentMassKg=parentMass;
 in.parentMaterialIndices=parentMaterial;in.parentDonorBodyIndices=parentDonor;in.parentAnatomicalRegionIndices=parentRegion;
 in.parentSourceDonorOffsets=parentSourceOffsets;in.parentSourceDonorBodyIndices=parentSourceDonors;in.parentSourceDonorMassKg=parentSourceMass;
 in.childCentroidsWorld=childCentroids;in.childReferenceVolumeM3=childVolumes;in.childMaterialIndices=cellMaterial;
 in.childDonorBodyIndices=childDonor;in.childAnatomicalRegionIndices=childRegion;
 std::vector<double> lower(17,std::numeric_limits<double>::quiet_NaN()),upper=lower;lower[16]=850.;upper[16]=950.;
 in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
 in.parentMassPacketSHA256=all.parentMassPacketSHA256;in.parentSourceContributionSHA256=all.parentSourceContributionSHA256;
 in.childMeshSHA256=childHash;in.regionLabelSHA256=support.typedSupportSHA256;in.coordinateFrameSHA256=frame;
 in.maximumTransportDistanceM=0.019353;in.relativeClosureTolerance=1e-10;in.absoluteMassToleranceKg=1e-13;in.absoluteMeasureToleranceM3=1e-15;
 in.maximumAllowedEdges=12000000;in.maximumAugmentations=4000000;
 std::cerr<<"INPUT_READY parent_fat="<<np<<" child_fat="<<nc<<" groups="<<ng<<" slots="<<ns<<" child_volume_m3="<<std::setprecision(17);
 double totalVol=0;for(double v:childVolumes)totalVol+=v;std::cerr<<totalVol<<std::endl;
 ConservativeSpatialMassTransportResult result;
 const auto begin=std::chrono::steady_clock::now();
 std::cerr<<"SHARED_CAPACITY_ASSEMBLY_BEGIN"<<std::endl;
 if(!assembleConservativeSpatialMassTransportWithSemanticSupport(in,support.view(),result,error)){
  writeFailure("shared-capacity allocation: "+error,np,nc);std::cerr<<"ALLOC_FAIL "<<error<<std::endl;return 3;
 }
 const double solveSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 std::cerr<<"SHARED_CAPACITY_ASSEMBLY_PASS edges="<<result.semanticCandidateEdgeCount<<" augmentations="<<result.augmentationCount<<" seconds="<<solveSeconds<<std::endl;
 fs::create_directories(resultPath.parent_path());std::ofstream out(resultPath);out<<std::setprecision(17);
 out<<"{\n  \"status\": \"PASS_SOURCE_ONLY_FAT_PARTIAL_SHARED_CAPACITY_ALLOCATION\",\n  \"physical_admission\": false,\n  \"native_steps\": 0,\n";
 out<<"  \"scope\": \"Fat-only; no Fat allocation or paired Fat/Fat closure\",\n  \"parent_fat_tets\": "<<np<<",\n  \"child_fat_tets\": "<<nc<<",\n  \"support_groups\": "<<ng<<",\n  \"p1_support_slots\": "<<ns<<",\n";
 out<<"  \"source_mass_kg\": ";double sm=0;for(auto x:parentMass)sm+=x;out<<sm<<",\n";
 out<<"  \"child_reference_volume_m3\": "<<totalVol<<",\n  \"minimum_child_density_kg_m3\": "<<result.minimumChildImpliedDensityKgM3<<",\n  \"maximum_child_density_kg_m3\": "<<result.maximumChildImpliedDensityKgM3<<",\n";
 out<<"  \"density_bounds_kg_m3\": [850, 950],\n  \"maximum_transport_distance_allowed_m\": 0.019353,\n  \"maximum_transport_distance_used_m\": "<<result.maximumTransportDistanceUsedM<<",\n";
 out<<"  \"mass_weighted_mean_transport_distance_m\": "<<result.massWeightedMeanTransportDistanceM<<",\n  \"transport_edges\": "<<result.transportEdgeCount<<",\n  \"candidate_edges\": "<<result.semanticCandidateEdgeCount<<",\n  \"augmentations\": "<<result.augmentationCount<<",\n  \"allocation_solver\": \"deterministic_dinic_feasibility\",\n  \"cost_optimality_certified\": false,\n";
 out<<"  \"maximum_parent_mass_closure_error_kg\": "<<result.maximumParentMassClosureErrorKg<<",\n  \"maximum_parent_measure_closure_error_m3\": "<<result.maximumParentMeasureClosureErrorM3<<",\n";
 out<<"  \"maximum_source_donor_closure_error_kg\": "<<result.maximumSourceDonorMassClosureErrorKg<<",\n  \"maximum_child_fraction_sum_error\": "<<result.maximumChildFractionSumError<<",\n";
 out<<"  \"child_mesh_sha256\": "<<std::quoted(childHash)<<",\n  \"input_child_node_sha256\": "<<std::quoted(nodeHash)<<",\n  \"input_child_element_sha256\": "<<std::quoted(eleHash)<<",\n  \"input_parent_mass_csv_sha256\": "<<std::quoted(parentMassFileHash)<<",\n  \"input_parent_label_csv_sha256\": "<<std::quoted(parentLabelsFileHash)<<",\n  \"support_binary_sha256\": "<<std::quoted(supportBinaryHash)<<",\n  \"semantic_support_sha256\": "<<std::quoted(result.semanticSupportSHA256)<<",\n  \"transport_receipt_sha256\": "<<std::quoted(result.receiptSHA256)<<",\n";
 out<<"  \"admissibility_source_support_sha256\": "<<std::quoted(support.admissibilitySourceSupportSHA256)<<",\n  \"admissibility_max_source_region_hops\": "<<support.admissibilityMaximumSourceRegionHops<<",\n  \"admissible_region_slots\": "<<support.cellAdmissibleRegionIds.size()<<",\n";
 out<<"  \"input_parent_mass_sha256\": "<<std::quoted(all.parentMassPacketSHA256)<<",\n  \"input_parent_source_sha256\": "<<std::quoted(all.parentSourceContributionSHA256)<<",\n  \"input_coordinate_frame_sha256\": "<<std::quoted(frame)<<",\n";
 out<<"  \"group_closure\": [";
 for(std::size_t i=0;i<result.groupClosure.size();++i){const auto&g=result.groupClosure[i];if(i)out<<",";out<<"{\"material\":"<<g.materialIndex<<",\"geometric_donor\":"<<g.donorBodyIndex<<",\"region\":"<<g.anatomicalRegionIndex<<",\"parent_mass_kg\":"<<g.parentMassKg<<",\"child_mass_kg\":"<<g.childMassKg<<",\"reachable_capacity_kg\":"<<g.childMaximumCapacityKg<<"}";}
 out<<"],\n  \"source_donor_closure\": [";
 for(std::size_t i=0;i<result.sourceDonorClosure.size();++i){const auto&g=result.sourceDonorClosure[i];if(i)out<<",";out<<"{\"material\":"<<g.materialIndex<<",\"source_donor\":"<<g.sourceDonorBodyIndex<<",\"parent_mass_kg\":"<<g.parentMassKg<<",\"child_mass_kg\":"<<g.childMassKg<<"}";}
 out<<"],\n  \"solve_seconds\": "<<solveSeconds<<",\n  \"elapsed_seconds\": "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"\n}\n";
 return 0;
}
