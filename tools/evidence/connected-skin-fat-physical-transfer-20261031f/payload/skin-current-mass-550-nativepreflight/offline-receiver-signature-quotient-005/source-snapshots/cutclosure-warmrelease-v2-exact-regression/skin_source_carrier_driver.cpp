#include "NumiHumanPhysicalSkinChildMassMapping.hpp"
#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "ConservativeSpatialSourceCarrierQuadrature.hpp"
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
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
static bool readKeyValueManifest(const fs::path& path,std::map<std::string,std::string>& values,std::string& error){
 std::ifstream in(path);if(!in){error="cannot open warm-start manifest";return false;}
 std::string line;std::size_t rows=0;
 while(std::getline(in,line)){
  if(line.empty()||line.size()>4096){error="warm-start manifest has an empty or oversized line";return false;}
  const auto eq=line.find('=');
  if(eq==std::string::npos||eq==0||eq+1u>=line.size()||line.find('=',eq+1u)!=std::string::npos){error="warm-start manifest line is not a strict key=value pair";return false;}
  if(!values.emplace(line.substr(0,eq),line.substr(eq+1u)).second){error="warm-start manifest repeats a key";return false;}
  ++rows;
 }
 if(!in.eof()||rows==0){error="warm-start manifest read failed or is empty";return false;}
 return true;
}
static bool readBoundedText(const fs::path& path,std::string& value,std::string& error){
 std::ifstream in(path,std::ios::binary);if(!in){error="cannot open pinned source receipt";return false;}
 value.assign(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
 if(in.bad()||value.size()>4u*1024u*1024u){error="pinned source receipt read failed or exceeds size bound";return false;}
 return true;
}
static bool csvFields(const std::string& line,std::vector<std::string>& fields){
 fields.clear();std::size_t begin=0;
 for(;;){const auto end=line.find(',',begin);if(end==std::string::npos){fields.push_back(line.substr(begin));break;}fields.push_back(line.substr(begin,end-begin));begin=end+1u;}
 return fields.size()==8u&&std::all_of(fields.begin(),fields.end(),[](const std::string& x){return !x.empty();});
}
static bool readU32(std::ifstream& f,std::uint32_t& x){return bool(f.read(reinterpret_cast<char*>(&x),4));}
static bool readU64(std::ifstream& f,std::uint64_t& x){return bool(f.read(reinterpret_cast<char*>(&x),8));}
static bool readD(std::ifstream& f,double& x){return bool(f.read(reinterpret_cast<char*>(&x),8));}
template<class T> static bool writeVector(std::ofstream& f,const std::vector<T>& v){
 return v.empty()||bool(f.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(T))));
}
static double volume(const V3&a,const V3&b,const V3&c,const V3&d){
 return physical_skin_child_mass_detail::signedTetVolume(a,b,c,d);
}
int main(int argc,char**argv){
 if(argc!=9&&argc!=10&&argc!=13&&argc!=14&&argc!=16){std::cerr<<"usage: driver parent-mass.csv parent-labels.csv node ele support.bin source-carriers.bin coord-frame-sha result.json [solver-deadline-seconds [warm-flow.csv warm-manifest.kv warm-manifest-sha256 [highest-label-push-relabel|candidate-neighbor-census] | exact-receiver-signature-quotient candidate-edges.u64le candidate-edges-sha256]]\n";return 64;}
 const auto t0=std::chrono::steady_clock::now();
 const fs::path massPath=argv[1],labelsPath=argv[2],nodePath=argv[3],elePath=argv[4],supportPath=argv[5],carrierPath=argv[6],resultPath=argv[8];
 const std::string frame=argv[7];
 double solverDeadlineSeconds=0.0;
 if(argc>=10&&(!number(argv[9],solverDeadlineSeconds)||!std::isfinite(solverDeadlineSeconds)||solverDeadlineSeconds<=0.0)){std::cerr<<"solver deadline must be positive and finite\n";return 64;}
 const bool useWarmStart=argc==13||argc==14||argc==16;
 const bool useHighestLabelPushRelabel=argc==14&&std::string(argv[13])=="highest-label-push-relabel";
 const bool captureCandidateGraphCensus=argc==14&&std::string(argv[13])=="candidate-neighbor-census";
 const bool useReceiverSignatureQuotient=argc==16&&std::string(argv[13])=="exact-receiver-signature-quotient";
 if((argc==14&&!useHighestLabelPushRelabel&&!captureCandidateGraphCensus)||
    (argc==16&&!useReceiverSignatureQuotient)){std::cerr<<"unknown optional feasible-flow diagnostic/solver\n";return 64;}
 const fs::path warmFlowPath=useWarmStart?fs::path(argv[10]):fs::path{};
 const fs::path warmManifestPath=useWarmStart?fs::path(argv[11]):fs::path{};
 const std::string expectedWarmManifestSHA=(argc==13||argc==14||argc==16)?std::string(argv[12]):std::string{};
 const fs::path candidateGraphPath=useReceiverSignatureQuotient?fs::path(argv[14]):fs::path{};
 const std::string expectedCandidateGraphSHA=useReceiverSignatureQuotient?std::string(argv[15]):std::string{};
 std::string candidateGraphSHA;
 std::vector<std::uint64_t> exactCandidateNeighborEdgeKeys;
 if(useWarmStart&&solverDeadlineSeconds<=0.0){std::cerr<<"warm-start replay requires a bounded solver deadline\n";return 64;}
 auto writeFailure=[&](const std::string& message,const std::size_t parentCount,const std::size_t childCount){
  fs::create_directories(resultPath.parent_path());std::ofstream o(resultPath);
  const bool inconclusive=message.find("INCONCLUSIVE")!=std::string::npos;
  const bool deadlineInterrupted=message.find("deadline reached")!=std::string::npos;
  o<<"{\n  \"status\": \""<<(inconclusive?"INCONCLUSIVE_SOURCE_ONLY_SKIN_MASS_TRANSFER":"FAIL_SOURCE_ONLY_SKIN_MASS_TRANSFER")<<"\",\n"
   <<"  \"solver_outcome\": \""<<(deadlineInterrupted?"INTERRUPTED_PARTIAL":(inconclusive?"INCONCLUSIVE_PARTIAL_OR_BUDGET_STOP":"FAILED_SOURCE_ONLY"))<<"\",\n"
   <<"  \"complete_allocation\": false,\n  \"allocation_feasibility\": \"NOT_DETERMINED\",\n  \"physical_infeasibility_claim\": false,\n"
   <<"  \"physical_admission\": false,\n  \"native_steps\": 0,\n"
   <<"  \"parent_skin_tets\": "<<parentCount<<",\n  \"child_skin_tets\": "<<childCount<<",\n"
   <<"  \"error\": "<<std::quoted(message)<<",\n  \"elapsed_seconds\": "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"\n}\n";
 };
 if(useReceiverSignatureQuotient){
  std::string hashError;
  const auto bytes=fs::file_size(candidateGraphPath);
  if(bytes==0u||bytes%sizeof(std::uint64_t)!=0u||bytes>512ull*1024ull*1024ull){
   writeFailure("exact candidate-neighbor graph size is empty, unaligned or above the 512 MiB input bound",0,0);return 2;
  }
  if(!physical_skin_child_mass_detail::sha256File(candidateGraphPath,candidateGraphSHA,hashError)||
     candidateGraphSHA!=expectedCandidateGraphSHA){
   writeFailure("exact candidate-neighbor graph SHA-256 pin mismatch: "+hashError,0,0);return 2;
  }
  std::ifstream graph(candidateGraphPath,std::ios::binary);
  exactCandidateNeighborEdgeKeys.resize(static_cast<std::size_t>(bytes/sizeof(std::uint64_t)));
  for(auto& key:exactCandidateNeighborEdgeKeys)if(!readU64(graph,key)){
   writeFailure("exact candidate-neighbor graph has a truncated uint64 record",0,0);return 2;
  }
  char trailing=0;
  if(graph.read(&trailing,1)){
   writeFailure("exact candidate-neighbor graph has trailing bytes",0,0);return 2;
  }
 }
 PhysicalSkinParentMassPacket all;std::string error;
 if(!loadCompiledSkinFatParentMassPacketCSV(massPath,all,error)){writeFailure("parent packet: "+error,0,0);return 2;}
 if(!loadCompiledSkinFatParentLabelsCSV(labelsPath,all,error)){writeFailure("parent labels: "+error,0,0);return 2;}
 std::cerr<<"PARENT_PACKET_PASS nodes="<<all.referenceWorldPoints.size()<<" cells="<<all.tetrahedra.size()<<std::endl;
 std::string parentMassFileHash,parentLabelsFileHash,supportBinaryHash,carrierBinaryHash,sourceTetTableHash,sourceNodeTableHash;
 fs::path sourceTetPath=massPath;sourceTetPath += ".tets.csv";
 fs::path sourceNodePath=massPath;sourceNodePath += ".nodes.csv";
 if(!physical_skin_child_mass_detail::sha256File(massPath,parentMassFileHash,error)||
    !physical_skin_child_mass_detail::sha256File(labelsPath,parentLabelsFileHash,error)||
    !physical_skin_child_mass_detail::sha256File(supportPath,supportBinaryHash,error)||
    !physical_skin_child_mass_detail::sha256File(carrierPath,carrierBinaryHash,error)||
    !physical_skin_child_mass_detail::sha256File(sourceTetPath,sourceTetTableHash,error)||
    !physical_skin_child_mass_detail::sha256File(sourceNodePath,sourceNodeTableHash,error)){
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
 // FSSCARR01 uses the canonical complete role-parent row map followed by a
 // sparse nonempty carrier CSR. Source330 geometry is rebuilt from the pinned
 // parent mass packet below and compared row-for-row before use.
 std::ifstream cf(carrierPath,std::ios::binary);
 char carrierMagic[9]{};std::uint64_t carrierParentCount64=0,carrierCount64=0;
 if(!cf.read(carrierMagic,9)||std::string(carrierMagic,9)!="FSSCARR01"||
    !readU64(cf,carrierParentCount64)||!readU64(cf,carrierCount64)||
    carrierParentCount64!=np||carrierCount64==0||carrierCount64>100000){
  writeFailure("source carrier binary header/counts invalid",np,0);return 2;
 }
 const auto nCarrier=static_cast<std::size_t>(carrierCount64);
 std::vector<std::uint32_t> carrierRoleRows(np),carrierGlobalTetRows(np),carrierOffsets(np+1u),carrierWitnessRows(nCarrier);
 std::vector<double> carrierWeights(nCarrier),carrierDistances(nCarrier);
 std::vector<std::array<double,4>> carrierBary(nCarrier);
 auto readCarrierU32=[&](std::vector<std::uint32_t>&v){
  return v.empty()||bool(cf.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(std::uint32_t))));
 };
 auto readCarrierD=[&](std::vector<double>&v){
  return v.empty()||bool(cf.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(double))));
 };
 if(!readCarrierU32(carrierRoleRows)||!readCarrierU32(carrierGlobalTetRows)||!readCarrierU32(carrierOffsets)||
    !readCarrierD(carrierWeights)){
  writeFailure("source carrier binary arrays truncated",np,0);return 2;
 }
 for(std::size_t row=0;row<np;++row)if(carrierRoleRows[row]!=row){
  writeFailure("source carrier role-parent row map is not the complete ordered role domain",np,0);return 2;
 }
 for(auto& bary:carrierBary)for(auto& value:bary)if(!readD(cf,value)){
  writeFailure("source carrier barycentric array truncated",np,0);return 2;
 }
 if(!readCarrierU32(carrierWitnessRows)||!readCarrierD(carrierDistances)||cf.peek()!=EOF){
  writeFailure("source carrier witness arrays truncated or trailing bytes present",np,0);return 2;
 }
 std::size_t carrierizedParentRows=0;
 for(std::size_t row=1u;row<carrierOffsets.size();++row){
  if(carrierOffsets[row]<carrierOffsets[row-1u]||carrierOffsets[row]>nCarrier){
   writeFailure("source carrier parent offsets are not monotone/in-range",np,0);return 2;
  }
  if(carrierOffsets[row]!=carrierOffsets[row-1u])++carrierizedParentRows;
 }
 if(carrierOffsets.front()!=0u||carrierOffsets.back()!=nCarrier||carrierizedParentRows==0){
  writeFailure("source carrier sparse CSR endpoints or nonempty rows invalid",np,0);return 2;
 }
 SourceCellCarrierQuadratureInput carrier;
 carrier.parentOffsets=carrierOffsets;
 carrier.globalSourceTetRows=carrierGlobalTetRows;
 carrier.carrierWeights=carrierWeights;
 carrier.carrierBarycentric=carrierBary;
 carrier.witnessChildRows=carrierWitnessRows;
 carrier.witnessChildCentroidDistancesM=carrierDistances;
 carrier.sourceCarrierPacketSHA256=carrierBinaryHash;
 carrier.sourceTetTableSHA256=sourceTetTableHash;
 carrier.sourceNodeTableSHA256=sourceNodeTableHash;

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
 std::cerr<<"SUPPORT_PASS children="<<nc<<" slots="<<ns<<" groups="<<ng<<" parent_skin="<<np<<std::endl;
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
 std::size_t nParent=0;for(auto m:all.materialIndices)if(m==15u)++nParent;
 if(nParent!=np){writeFailure("filtered source Skin parent count differs from support binary",nParent,nc);return 2;}
 std::vector<SpatialMassTransportPoint> parentCentroids;std::vector<double> parentVolumes,parentMass;
 std::vector<std::uint32_t> parentMaterial,parentDonor,parentRegion,parentStable,parentGlobalTetRows;
 std::vector<std::array<std::uint32_t,4>> parentTetNodeRows;
 std::vector<std::array<std::array<double,3>,4>> parentTetReferenceWorldVertices;
 std::vector<std::uint32_t> parentSourceOffsets{0},parentSourceDonors;std::vector<double> parentSourceMass;
 parentCentroids.reserve(np);parentVolumes.reserve(np);parentMass.reserve(np);
 parentMaterial.reserve(np);parentDonor.reserve(np);parentRegion.reserve(np);parentStable.reserve(np);
 for(std::size_t r=0;r<all.tetrahedra.size();++r)if(all.materialIndices[r]==15u){
  const auto out=static_cast<std::uint32_t>(parentMass.size());origToSkin[r]=out;
  const auto tet=all.tetrahedra[r];SpatialMassTransportPoint c{};
  for(auto node:tet)for(unsigned axis=0;axis<3;++axis)c[axis]+=all.referenceWorldPoints[node][axis]*0.25;
  parentCentroids.push_back(c);parentVolumes.push_back(all.referenceTetVolumeM3[r]);parentMass.push_back(all.tetrahedronMassKg[r]);
  parentGlobalTetRows.push_back(static_cast<std::uint32_t>(r));
  std::array<std::uint32_t,4> sourceNodeRows{};
  std::array<std::array<double,3>,4> sourceReferenceVertices{};
  for(std::size_t corner=0;corner<4u;++corner){sourceNodeRows[corner]=tet[corner];for(std::size_t axis=0;axis<3u;++axis)sourceReferenceVertices[corner][axis]=all.referenceWorldPoints[tet[corner]][axis];}
  parentTetNodeRows.push_back(sourceNodeRows);parentTetReferenceWorldVertices.push_back(sourceReferenceVertices);
  parentMaterial.push_back(all.materialIndices[r]);parentDonor.push_back(all.donorBodyIndices[r]);
  parentRegion.push_back(all.anatomicalRegionIndices[r]);parentStable.push_back(all.stableSourceFaceIdentities[r]);
  for(std::uint32_t q=allOffsets[r];q<allOffsets[r+1];++q){parentSourceDonors.push_back(allDonors[q]);parentSourceMass.push_back(allDonorMass[q]);}
  parentSourceOffsets.push_back(static_cast<std::uint32_t>(parentSourceDonors.size()));
 }
 if(parentStable!=support.parentStableSourceFaceIdentities){writeFailure("filtered source stable-face identity differs from support binary",np,nc);return 2;}
 if(parentGlobalTetRows!=carrierGlobalTetRows){writeFailure("source carrier global tet rows differ from the exact filtered source330 parent row map",np,nc);return 2;}
 carrier.sourceTetNodeRows=parentTetNodeRows;
 carrier.parentReferenceTetWorldVertices=parentTetReferenceWorldVertices;
 // The binary producer already expresses source-parent CSR rows in the filtered Skin input domain.
 // Validate complete unique coverage without reinterpreting those row IDs as full-330 rows.
 std::vector<std::uint8_t> parentRowSeen(np,0u);
 for(const auto r:support.sourceParentTetRows){
  if(r>=np||parentRowSeen[r]++){writeFailure("source group CSR has a duplicate or out-of-range filtered Skin parent row",np,nc);return 2;}
 }
 if(support.sourceParentTetRows.size()!=np||
    std::any_of(parentRowSeen.begin(),parentRowSeen.end(),[](std::uint8_t x){return x!=1u;})){
  writeFailure("source group CSR does not cover every filtered Skin parent row exactly once",np,nc);return 2;
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
 std::vector<double> lower(16,std::numeric_limits<double>::quiet_NaN()),upper=lower;lower[15]=900.;upper[15]=1100.;
 in.materialDensityLowerKgM3=lower;in.materialDensityUpperKgM3=upper;
 in.parentMassPacketSHA256=all.parentMassPacketSHA256;in.parentSourceContributionSHA256=all.parentSourceContributionSHA256;
 in.childMeshSHA256=childHash;in.regionLabelSHA256=support.typedSupportSHA256;in.coordinateFrameSHA256=frame;
 in.maximumTransportDistanceM=0.019353;in.relativeClosureTolerance=1e-10;in.absoluteMassToleranceKg=1e-13;in.absoluteMeasureToleranceM3=1e-15;
 in.maximumAllowedEdges=12000000;in.maximumAugmentations=4000000;
 using WarmFlow=fractional_semantic_transport_detail::SharedCapacityWarmStartFlow;
 fractional_semantic_transport_detail::SharedCapacityWarmStart warmStart;
 std::vector<WarmFlow> warmFlows;
 std::map<std::string,std::string> warmManifest;
 std::string warmManifestSHA,warmFlowSHA,warmCarrierArraySHA;
 double warmSeedMassKg=0.0,warmStartRoundoffReleasedKg=0.0;std::size_t warmStartRoundoffCorrectedParents=0u,warmStartRoundoffCorrectedChildren=0u;
 if(useWarmStart){
  std::string hashError;
  if(!physical_skin_child_mass_detail::sha256File(warmManifestPath,warmManifestSHA,hashError)||
     warmManifestSHA!=expectedWarmManifestSHA){writeFailure("warm-start manifest hash mismatch: "+hashError,np,nc);return 2;}
  const std::vector<std::string> required{
   "schema","status","allocation_feasibility","physical_infeasibility_claim","complete_allocation","physical_admission","native_steps","frontier_semantics","solver_deadline_seconds",
   "source_result_path","source_result_sha256","partial_receipt_path","partial_receipt_sha256","partial_flow_path","partial_flow_sha256",
   "parent_mass_path","parent_labels_path","child_node_path","child_ele_path","semantic_support_path","carrier_packet_path",
   "frozen_input_manifest_path","frozen_input_manifest_sha256","geometry_path","geometry_sha256",
   "parent_mass_sha256","parent_labels_sha256","child_node_sha256","child_ele_sha256","child_mesh_sha256","semantic_support_sha256","carrier_packet_sha256","source_tets_sha256","source_nodes_sha256","coordinate_frame_sha256"
  };
  if(!readKeyValueManifest(warmManifestPath,warmManifest,hashError)||warmManifest.size()!=required.size()){
   writeFailure("warm-start manifest schema: "+hashError,np,nc);return 2;
  }
  for(const auto& key:required)if(!warmManifest.contains(key)){writeFailure("warm-start manifest missing key "+key,np,nc);return 2;}
  const auto requireValue=[&](const std::string& key,const std::string& expected){
   return warmManifest.at(key)==expected;
  };
  if(!requireValue("schema","numi.human.skin538-partial-flow-warm-start.v1")||
     !requireValue("status","INCONCLUSIVE_PARTIAL_FLOW_DIAGNOSTIC")||
     !requireValue("allocation_feasibility","NOT_DETERMINED")||
     !requireValue("physical_infeasibility_claim","false")||
     !requireValue("complete_allocation","false")||!requireValue("physical_admission","false")||
     !requireValue("native_steps","0")||
     !requireValue("frontier_semantics","RESIDUAL_REACHABILITY_ONLY_NOT_MIN_CUT")){
   writeFailure("warm-start manifest does not describe a frozen incomplete diagnostic",np,nc);return 2;
  }
  double pinnedDeadline=0.0;
  if(!number(warmManifest.at("solver_deadline_seconds"),pinnedDeadline)||pinnedDeadline!=solverDeadlineSeconds){
   writeFailure("warm-start manifest deadline does not match the requested bounded replay",np,nc);return 2;
  }
  const auto expectedPath=[&](const std::string& key,const fs::path& path){return warmManifest.at(key)==path.string();};
  if(!expectedPath("partial_flow_path",warmFlowPath)||
     !expectedPath("parent_mass_path",massPath)||!expectedPath("parent_labels_path",labelsPath)||
     !expectedPath("child_node_path",nodePath)||!expectedPath("child_ele_path",elePath)||
     !expectedPath("semantic_support_path",supportPath)||!expectedPath("carrier_packet_path",carrierPath)){
   writeFailure("warm-start manifest paths do not match the requested source/child inputs",np,nc);return 2;
  }
  // Required path keys are exact and the corresponding contents are hash pinned.
  const std::vector<std::pair<std::string,std::string>> filePins{
   {"source_result_path","source_result_sha256"},{"partial_receipt_path","partial_receipt_sha256"},
   {"partial_flow_path","partial_flow_sha256"},{"frozen_input_manifest_path","frozen_input_manifest_sha256"},
   {"geometry_path","geometry_sha256"}
  };
  for(const auto& [pathKey,shaKey]:filePins){
   std::string actual;
   if(!physical_skin_child_mass_detail::sha256File(fs::path(warmManifest.at(pathKey)),actual,hashError)||
      actual!=warmManifest.at(shaKey)){writeFailure("warm-start provenance artifact hash mismatch for "+pathKey,np,nc);return 2;}
  }
  const std::vector<std::pair<std::string,std::string>> inputPins{
   {"parent_mass_sha256",parentMassFileHash},{"parent_labels_sha256",parentLabelsFileHash},
   {"child_node_sha256",nodeHash},{"child_ele_sha256",eleHash},{"child_mesh_sha256",childHash},
   {"semantic_support_sha256",supportBinaryHash},{"carrier_packet_sha256",carrierBinaryHash},
   {"source_tets_sha256",sourceTetTableHash},{"source_nodes_sha256",sourceNodeTableHash},
   {"coordinate_frame_sha256",frame}
  };
  for(const auto& [key,actual]:inputPins)if(!requireValue(key,actual)){
   writeFailure("warm-start manifest does not bind current input pin "+key,np,nc);return 2;
  }
  std::string sourceResultText,sourceReceiptText;
  if(!readBoundedText(fs::path(warmManifest.at("source_result_path")),sourceResultText,hashError)||
     !readBoundedText(fs::path(warmManifest.at("partial_receipt_path")),sourceReceiptText,hashError)||
     sourceResultText.find("\"solver_outcome\": \"INTERRUPTED_PARTIAL\"")==std::string::npos||
     sourceReceiptText.find("\"status\": \"INCONCLUSIVE_PARTIAL_FLOW_DIAGNOSTIC\"")==std::string::npos||
     sourceReceiptText.find("\"flow_source_carrier_child_accounting_valid\": true")==std::string::npos||
     sourceReceiptText.find("\"frontier_semantics\": \"residual-reachability snapshot only; not a proven min-cut\"")==std::string::npos){
   writeFailure("warm-start source receipts do not attest an exported interrupted partial diagnostic",np,nc);return 2;
  }
  if(!physical_skin_child_mass_detail::sha256File(warmFlowPath,warmFlowSHA,hashError)||
     warmFlowSHA!=warmManifest.at("partial_flow_sha256")){writeFailure("warm-start partial flow hash mismatch",np,nc);return 2;}
  warmCarrierArraySHA=source_carrier_transport_detail::carrierArrayDigest(in,carrier);
  std::vector<std::uint32_t> expandedRole,expandedCarrier;
  std::vector<SpatialMassTransportPoint> expandedCentroids;
  const auto invalid=std::numeric_limits<std::uint32_t>::max();
  for(std::size_t role=0;role<np;++role){
   const auto begin=carrierOffsets[role],end=carrierOffsets[role+1u];
   if(begin==end){expandedRole.push_back(static_cast<std::uint32_t>(role));expandedCarrier.push_back(invalid);expandedCentroids.push_back(parentCentroids[role]);}
   else for(std::uint32_t cr=begin;cr<end;++cr){
    SpatialMassTransportPoint point{};
    for(std::size_t corner=0;corner<4u;++corner)for(std::size_t axis=0;axis<3u;++axis)
     point[axis]+=carrierBary[cr][corner]*parentTetReferenceWorldVertices[role][corner][axis];
    expandedRole.push_back(static_cast<std::uint32_t>(role));expandedCarrier.push_back(cr);expandedCentroids.push_back(point);
   }
  }
  std::ifstream flow(warmFlowPath);std::string line;
  const std::string flowHeader="expanded_parent_row,source_role_parent_row,global_source330_tet_row,carrier_row,child_row,mass_kg,source_measure_m3,distance_m";
  if(!flow||!std::getline(flow,line)||line!=flowHeader){writeFailure("warm-start CSV header is invalid",np,nc);return 2;}
  std::vector<std::string> fields;std::set<std::pair<std::uint32_t,std::uint32_t>> seenEdges;
  std::size_t lineNumber=1u;
  while(std::getline(flow,line)){
   ++lineNumber;if(line.empty()||(!line.empty()&&line.back()=='\r')){writeFailure("warm-start CSV contains an empty/CR row",np,nc);return 2;}
   std::uint32_t ep=0u,role=0u,globalTet=0u,cr=0u,child=0u;double mass=0.0,measure=0.0,reportedDistance=0.0;
   if(!csvFields(line,fields)||!number(fields[0],ep)||!number(fields[1],role)||!number(fields[2],globalTet)||
      !number(fields[3],cr)||!number(fields[4],child)||!number(fields[5],mass)||!number(fields[6],measure)||!number(fields[7],reportedDistance)||
      ep>=expandedRole.size()||child>=nc||role!=expandedRole[ep]||cr!=expandedCarrier[ep]||
      globalTet!=carrierGlobalTetRows[role]||!std::isfinite(mass)||mass<=0.0||!std::isfinite(measure)||measure<=0.0||
      !std::isfinite(reportedDistance)||reportedDistance<0.0||!seenEdges.emplace(ep,child).second){
    writeFailure("warm-start CSV row is malformed, duplicated, or outside the current row maps at line "+std::to_string(lineNumber),np,nc);return 2;
   }
   double distanceSquared=0.0;
   for(std::size_t axis=0;axis<3u;++axis){const double d=expandedCentroids[ep][axis]-childCentroids[child][axis];distanceSquared+=d*d;}
   const double distanceNow=std::sqrt(distanceSquared);
   const double density=parentMass[role]/parentVolumes[role];
   const double measureNow=mass/density;
   const double distanceTolerance=64.0*std::numeric_limits<double>::epsilon()*std::max(1.0,distanceNow);
   if(!std::isfinite(distanceNow)||distanceNow>in.maximumTransportDistanceM||
      std::abs(reportedDistance-distanceNow)>distanceTolerance||
      !std::isfinite(measureNow)||std::abs(measure-measureNow)>1.0e-10*measureNow){
    writeFailure("warm-start CSV distance/measure does not replay against the current source/child geometry",np,nc);return 2;
   }
   warmSeedMassKg+=mass;
   warmFlows.push_back({ep,child,mass,role,globalTet,cr});
  }
  if(!flow.eof()||warmFlows.empty()||!std::isfinite(warmSeedMassKg)||warmSeedMassKg<=0.0){
   writeFailure("warm-start CSV is incomplete or contains no positive flows",np,nc);return 2;
  }
  // Round-trip CSV rows are individually exact doubles, but sequentially
  // summing thousands of source-child edges can exceed a parent/child arc by
  // a few representational ULPs. Release only this measured summation residue
  // into the unchanged residual network; never scale a parent or alter source
  // mass. All released values are explicit in the output receipt.
  std::sort(warmFlows.begin(),warmFlows.end(),[](const WarmFlow& a,const WarmFlow& b){
   return a.parentRow!=b.parentRow?a.parentRow<b.parentRow:a.childRow<b.childRow;
  });
  std::vector<double> originalWarmMasses;originalWarmMasses.reserve(warmFlows.size());
  for(const auto& row:warmFlows)originalWarmMasses.push_back(row.massKg);
  const double maximumRoundoffCorrectionRelative=64.0*std::numeric_limits<double>::epsilon();
  const auto correctOverCapacity=[&](const std::vector<std::size_t>& indices,const double capacity,
                                     const std::string& label)->bool{
   double sum=0.0;spatial_mass_transport_detail::ExactDoubleSum exact;
   for(const auto index:indices){sum+=warmFlows[index].massKg;exact.add(warmFlows[index].massKg);}
   if(!std::isfinite(sum)||!std::isfinite(capacity)||capacity<=0.0)return false;
   if(sum<=capacity)return true;
   exact.add(-capacity);
   if((sum-capacity)/capacity>maximumRoundoffCorrectionRelative||
      (exact.sign()>0&&exact.rounded()/capacity>maximumRoundoffCorrectionRelative)){
    hashError="warm-start "+label+" row exceeds capacity beyond the declared floating-sum correction envelope";return false;
   }
   std::size_t largest=indices.front();
   for(const auto index:indices)if(warmFlows[index].massKg>warmFlows[largest].massKg)largest=index;
   for(std::size_t attempt=0;attempt<8u&&sum>capacity;++attempt){
    const double excess=sum-capacity;
    const double target=warmFlows[largest].massKg-excess-2.0*std::numeric_limits<double>::epsilon()*capacity;
    const double next=std::nextafter(std::max(0.0,target),0.0);
    if(!(next<warmFlows[largest].massKg)){hashError="warm-start "+label+" roundoff correction made no progress";return false;}
    warmFlows[largest].massKg=next;sum=0.0;for(const auto index:indices)sum+=warmFlows[index].massKg;
   }
   if(sum>capacity){hashError="warm-start "+label+" bounded roundoff correction could not restore exact arc capacity";return false;}
   if(label=="source")++warmStartRoundoffCorrectedParents;else ++warmStartRoundoffCorrectedChildren;
   return true;
  };
  for(std::size_t begin=0;begin<warmFlows.size();){
   std::size_t end=begin+1u;while(end<warmFlows.size()&&warmFlows[end].parentRow==warmFlows[begin].parentRow)++end;
   const auto ep=warmFlows[begin].parentRow,role=expandedRole[ep],cr=expandedCarrier[ep];
   const double capacity=cr==invalid?parentMass[role]:parentMass[role]*carrierWeights[cr];
   std::vector<std::size_t> indices(end-begin);for(std::size_t i=begin;i<end;++i)indices[i-begin]=i;
   if(!correctOverCapacity(indices,capacity,"source")){writeFailure(hashError,np,nc);return 2;}
   begin=end;
  }
  std::vector<std::vector<std::size_t>> warmByChild(nc);
  for(std::size_t i=0;i<warmFlows.size();++i)warmByChild[warmFlows[i].childRow].push_back(i);
  for(std::size_t child=0;child<nc;++child)if(!warmByChild[child].empty()){
   const auto material=cellMaterial[child];const double capacity=childVolumes[child]*upper[material];
   if(!correctOverCapacity(warmByChild[child],capacity,"child")){writeFailure(hashError,np,nc);return 2;}
  }
  warmSeedMassKg=0.0;for(std::size_t i=0;i<warmFlows.size();++i){
   warmSeedMassKg+=warmFlows[i].massKg;warmStartRoundoffReleasedKg+=originalWarmMasses[i]-warmFlows[i].massKg;
  }
  if(!std::isfinite(warmStartRoundoffReleasedKg)||!std::isfinite(warmSeedMassKg)){
   writeFailure("warm-start roundoff replay produced a nonfinite mass",np,nc);return 2;
  }
  warmStart.flows=warmFlows;warmStart.parentMassPacketSHA256=all.parentMassPacketSHA256;
  warmStart.parentSourceContributionSHA256=all.parentSourceContributionSHA256;warmStart.childMeshSHA256=childHash;
  warmStart.regionLabelSHA256=support.typedSupportSHA256;warmStart.coordinateFrameSHA256=frame;
  warmStart.typedSupportSHA256=support.typedSupportSHA256;warmStart.sourceCarrierPacketSHA256=carrierBinaryHash;
  warmStart.sourceCarrierArraySHA256=warmCarrierArraySHA;warmStart.useHighestLabelPushRelabel=useHighestLabelPushRelabel;
  warmStart.captureCandidateNeighborEdges=captureCandidateGraphCensus;
  warmStart.useExactReceiverSignatureQuotient=useReceiverSignatureQuotient;
  warmStart.exactCandidateNeighborEdgeKeys=exactCandidateNeighborEdgeKeys;
 }
 std::cerr<<"INPUT_READY parent_skin="<<np<<" child_skin="<<nc<<" groups="<<ng<<" slots="<<ns<<" child_volume_m3="<<std::setprecision(17);
 double totalVol=0;for(double v:childVolumes)totalVol+=v;std::cerr<<totalVol<<std::endl;
 ConservativeSpatialMassTransportResult result;
 const auto begin=std::chrono::steady_clock::now();
 std::cerr<<"SHARED_CAPACITY_ASSEMBLY_BEGIN deadline_seconds="<<solverDeadlineSeconds<<std::endl;
 SourceCellCarrierTransportTrace carrierTrace;
 fractional_semantic_transport_detail::SharedCapacityFlowTelemetry telemetry;
 if(solverDeadlineSeconds>0.0)
  telemetry.feasibleFlowBudget=std::chrono::duration<double>(solverDeadlineSeconds);
 auto writePartialDiagnostic=[&](const fractional_semantic_transport_detail::SharedCapacityFlowTelemetry& t){
  fs::create_directories(resultPath.parent_path());
  fs::path flowPath=resultPath;flowPath += ".partial.flow.csv";
  fs::path frontierPath=resultPath;frontierPath += ".partial.frontier.csv";
  fs::path candidateNeighborEdgesPath=resultPath;candidateNeighborEdgesPath += ".candidate-neighbor-edge-keys.u64le";
  bool candidateNeighborEdgesOK=false;
  if(t.candidateNeighborEdgeKeysCaptured){
   std::ofstream candidateEdges(candidateNeighborEdgesPath,std::ios::binary|std::ios::trunc);
   if(candidateEdges){
    for(const auto key:t.candidateNeighborEdgeKeys)candidateEdges.write(reinterpret_cast<const char*>(&key),sizeof(key));
    candidateEdges.close();candidateNeighborEdgesOK=bool(candidateEdges);
   }
  }
  const auto invalidRow=std::numeric_limits<std::uint32_t>::max();
  bool mapsValid=t.partialExpandedParentRoleRows.size()==t.partialExpandedParentCarrierRows.size()&&
      carrierOffsets.size()==parentMass.size()+1u;
  std::vector<std::uint32_t> carrierRole(carrierWeights.size(),invalidRow);
  if(carrierOffsets.size()==parentMass.size()+1u)
   for(std::size_t role=0;role<parentMass.size();++role){
    if(carrierOffsets[role]>carrierOffsets[role+1u]||carrierOffsets[role+1u]>carrierWeights.size()){mapsValid=false;continue;}
    for(std::uint32_t cr=carrierOffsets[role];cr<carrierOffsets[role+1u];++cr)carrierRole[cr]=static_cast<std::uint32_t>(role);
   }
  for(std::size_t ep=0;ep<t.partialExpandedParentRoleRows.size();++ep){
   const auto role=t.partialExpandedParentRoleRows[ep],carrierRow=t.partialExpandedParentCarrierRows[ep];
   mapsValid=mapsValid&&role<carrierGlobalTetRows.size()&&
       (carrierRow==invalidRow||(carrierRow<carrierWeights.size()&&carrierRole[carrierRow]==role));
  }
  const std::size_t nFlow=t.partialMassKg.size();
  mapsValid=mapsValid&&t.partialExpandedParentRows.size()==nFlow&&t.partialChildRows.size()==nFlow&&
      t.partialMeasureM3.size()==nFlow&&t.partialDistanceM.size()==nFlow;
  std::vector<long double> partialByRole(parentMass.size(),0.0L),partialByCarrier(carrierWeights.size(),0.0L),
      partialByChild(childVolumes.size(),0.0L);
  long double partialMassSum=0.0L;
  for(std::size_t i=0;i<nFlow;++i){
   const auto ep=t.partialExpandedParentRows[i],child=t.partialChildRows[i];
   const bool rowDomainOK=ep<t.partialExpandedParentRoleRows.size()&&child<childVolumes.size();
   mapsValid=mapsValid&&rowDomainOK&&std::isfinite(t.partialMassKg[i])&&t.partialMassKg[i]>0.0&&
       std::isfinite(t.partialMeasureM3[i])&&t.partialMeasureM3[i]>0.0&&
       std::isfinite(t.partialDistanceM[i])&&t.partialDistanceM[i]>=0.0&&t.partialDistanceM[i]<=0.019353;
   if(!rowDomainOK){partialMassSum+=t.partialMassKg[i];continue;}
   const auto role=t.partialExpandedParentRoleRows[ep],cr=t.partialExpandedParentCarrierRows[ep];
   const double mass=t.partialMassKg[i];
   if(role>=parentMass.size()||!(parentVolumes[role]>0.0)||!(parentMass[role]>0.0)){
    mapsValid=false;partialMassSum+=mass;continue;
   }
   const double expectedMeasure=mass/(parentMass[role]/parentVolumes[role]);
   mapsValid=mapsValid&&std::isfinite(expectedMeasure)&&expectedMeasure>0.0&&
       std::abs(t.partialMeasureM3[i]-expectedMeasure)<=1.0e-10*expectedMeasure;
   partialByRole[role]+=mass;
   if(cr!=invalidRow){
    if(cr>=partialByCarrier.size()||carrierRole[cr]!=role)mapsValid=false;
    else partialByCarrier[cr]+=mass;
   }
   partialByChild[child]+=mass;
   partialMassSum+=mass;
  }
  constexpr long double capacityRelativeTolerance=1.0e-10L;
  std::size_t partialSourceRoles=0u,partialCarrierSites=0u,partialChildren=0u;
  for(std::size_t role=0;role<partialByRole.size();++role){
   if(partialByRole[role]>0.0L)++partialSourceRoles;
   mapsValid=mapsValid&&partialByRole[role]<=static_cast<long double>(parentMass[role])*(1.0L+capacityRelativeTolerance);
  }
  for(std::size_t cr=0;cr<partialByCarrier.size();++cr){
   if(partialByCarrier[cr]>0.0L)++partialCarrierSites;
   const auto role=carrierRole[cr];
   mapsValid=mapsValid&&role<parentMass.size()&&
       partialByCarrier[cr]<=static_cast<long double>(parentMass[role])*carrierWeights[cr]*(1.0L+capacityRelativeTolerance);
  }
  for(std::size_t child=0;child<partialByChild.size();++child){
   if(partialByChild[child]>0.0L)++partialChildren;
   const auto material=cellMaterial[child];
   mapsValid=mapsValid&&material<upper.size()&&
       partialByChild[child]<=static_cast<long double>(childVolumes[child])*upper[material]*(1.0L+capacityRelativeTolerance);
  }
  const long double partialScale=std::max(std::abs(partialMassSum),std::abs(static_cast<long double>(t.partialSourceSentKg)));
  mapsValid=mapsValid&&std::isfinite(t.partialSourceSentKg)&&t.partialSourceSentKg>=0.0&&
      (partialScale==0.0L||std::abs(partialMassSum-static_cast<long double>(t.partialSourceSentKg))<=1.0e-10L*partialScale);
  std::ofstream flow(flowPath);bool flowOK=bool(flow);
  if(flowOK){
   flow<<"expanded_parent_row,source_role_parent_row,global_source330_tet_row,carrier_row,child_row,mass_kg,source_measure_m3,distance_m\n";
   flow<<std::setprecision(17);
   if(mapsValid)for(std::size_t i=0;i<nFlow;++i){
    const auto ep=t.partialExpandedParentRows[i],role=t.partialExpandedParentRoleRows[ep],cr=t.partialExpandedParentCarrierRows[ep];
    if(role>=carrierGlobalTetRows.size()){flowOK=false;break;}
    flow<<ep<<","<<role<<","<<carrierGlobalTetRows[role]<<","<<cr<<","<<t.partialChildRows[i]<<","<<
      t.partialMassKg[i]<<","<<t.partialMeasureM3[i]<<","<<t.partialDistanceM[i]<<"\n";
   }
   flow.close();flowOK=flowOK&&bool(flow);
  }
  std::ofstream frontier(frontierPath);bool frontierOK=bool(frontier);
  if(frontierOK){
   frontier<<"kind,expanded_parent_row,source_role_parent_row,global_source330_tet_row,carrier_row,child_row\n";
   if(mapsValid){
    for(const auto ep:t.residualReachableExpandedParentRows){
     if(ep>=t.partialExpandedParentRoleRows.size())continue;
     const auto role=t.partialExpandedParentRoleRows[ep],cr=t.partialExpandedParentCarrierRows[ep];
     if(role<carrierGlobalTetRows.size())frontier<<"parent,"<<ep<<","<<role<<","<<carrierGlobalTetRows[role]<<","<<cr<<",\n";
    }
    for(const auto c:t.residualReachableChildRows)frontier<<"child,,,,,"<<c<<"\n";
   }
   frontier.close();frontierOK=frontierOK&&bool(frontier);
  }
  auto shaFile=[](const fs::path& path){std::string h,e;physical_skin_child_mass_detail::sha256File(path,h,e);return h;};
  fs::path receiptPath=resultPath;receiptPath += ".partial.receipt.json";
  std::ofstream receipt(receiptPath);
  receipt<<"{\n  \"schema\": \"numi.human.source-carrier-partial-flow-diagnostic.v1\",\n"
         <<"  \"status\": \"INCONCLUSIVE_PARTIAL_FLOW_DIAGNOSTIC\",\n"
         <<"  \"outcome_kind\": \"INTERRUPTED_OR_BUDGET_LIMITED_PARTIAL\",\n"
         <<"  \"allocation_feasibility\": \"NOT_DETERMINED\",\n  \"physical_infeasibility_claim\": false,\n"
         <<"  \"complete_allocation\": false,\n  \"physical_admission\": false,\n  \"native_steps\": 0,\n"
         <<"  \"solver_deadline_seconds_after_network_build\": "<<solverDeadlineSeconds<<",\n  \"deadline_reached\": "<<(t.deadlineReached?"true":"false")<<",\n"
         <<"  \"graph_construction_seconds\": "<<std::setprecision(17)<<t.graphConstructionSeconds<<",\n  \"feasible_solver_seconds\": "<<t.feasibleSolverSeconds<<",\n"
         <<"  \"total_elapsed_seconds\": "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<",\n"
         <<"  \"warm_start_used\": "<<(useWarmStart?"true":"false")<<",\n  \"warm_start_flow_rows\": "<<warmFlows.size()<<",\n"
         <<"  \"warm_start_seed_mass_kg\": "<<std::setprecision(17)<<warmSeedMassKg<<",\n"
         <<"  \"warm_start_roundoff_corrected_source_rows\": "<<warmStartRoundoffCorrectedParents<<",\n"
         <<"  \"warm_start_roundoff_corrected_child_rows\": "<<warmStartRoundoffCorrectedChildren<<",\n"
         <<"  \"warm_start_roundoff_released_kg\": "<<std::setprecision(17)<<warmStartRoundoffReleasedKg<<",\n"
         <<"  \"warm_start_roundoff_policy\": \"at most eight aggregate corrections per parent or child when ordered floating summation exceeds its exact capacity by at most 64 epsilon; source capacities and total masses unchanged\",\n"
         <<"  \"warm_start_manifest_sha256\": \""<<(useWarmStart?warmManifestSHA:std::string{})<<"\",\n"
         <<"  \"warm_start_flow_sha256\": \""<<(useWarmStart?warmFlowSHA:std::string{})<<"\",\n"
         <<"  \"partial_phase\": "<<t.partialPhase<<",\n  \"partial_source_sent_kg\": "<<std::setprecision(17)<<t.partialSourceSentKg<<",\n"
         <<"  \"partial_flow_rows\": "<<nFlow<<",\n  \"partial_source_roles_with_flow\": "<<partialSourceRoles<<",\n"
         <<"  \"partial_carrier_sites_with_flow\": "<<partialCarrierSites<<",\n  \"partial_children_with_flow\": "<<partialChildren<<",\n"
         <<"  \"flow_source_carrier_child_accounting_valid\": "<<(mapsValid?"true":"false")<<",\n"
         <<"  \"flow_export_valid\": "<<(flowOK?"true":"false")<<",\n  \"frontier_export_valid\": "<<(frontierOK?"true":"false")<<",\n"
         <<"  \"allocation_solver\": "<<std::quoted(t.allocationSolver)<<",\n"
         <<"  \"dinic_bfs_phases\": "<<t.feasibleFlow.bfsPhases<<",\n  \"feasible_solver_augmentations\": "<<t.feasibleFlow.augmentations<<",\n"
         <<"  \"push_relabel_pushes\": "<<t.feasibleFlow.pushRelabelPushes<<",\n  \"push_relabel_relabels\": "<<t.feasibleFlow.pushRelabelRelabels<<",\n"
         <<"  \"push_relabel_global_relabels\": "<<t.feasibleFlow.pushRelabelGlobalRelabels<<",\n  \"push_relabel_gap_events\": "<<t.feasibleFlow.pushRelabelGapEvents<<",\n"
         <<"  \"push_relabel_edge_scans\": "<<t.feasibleFlow.pushRelabelEdgeScans<<",\n"
         <<"  \"candidate_edges\": "<<t.initialSeedEdges+t.residualCutEdgesAdded<<",\n"
         <<"  \"receiver_signature_expansion_rounds\": "<<t.receiverSignatureExpansionRounds<<",\n"
         <<"  \"receiver_signature_expanded_edges\": "<<t.receiverSignatureExpandedEdges<<",\n"
         <<"  \"receiver_signature_cut_candidates_scanned\": "<<t.receiverSignatureCutCandidatesScanned<<",\n"
         <<"  \"receiver_signature_cut_enumeration_complete\": "<<(t.receiverSignatureCutEnumerationComplete?"true":"false")<<",\n"
         <<"  \"receiver_signature_warm_roundoff_corrections\": "<<t.receiverSignatureWarmRoundoffCorrections<<",\n"
         <<"  \"receiver_signature_warm_roundoff_released_kg\": "<<std::setprecision(17)<<t.receiverSignatureWarmRoundoffReleasedKg<<",\n"
         <<"  \"candidate_neighbor_edge_keys_captured\": "<<(t.candidateNeighborEdgeKeysCaptured?"true":"false")<<",\n"
         <<"  \"candidate_neighbor_edge_keys_count\": "<<t.candidateNeighborEdgeKeys.size()<<",\n"
         <<"  \"candidate_neighbor_edge_keys_export_valid\": "<<(candidateNeighborEdgesOK?"true":"false")<<",\n"
         <<"  \"candidate_neighbor_edge_keys_path\": "<<std::quoted(candidateNeighborEdgesOK?candidateNeighborEdgesPath.string():std::string{})<<",\n"
         <<"  \"candidate_neighbor_edge_keys_sha256\": \""<<(candidateNeighborEdgesOK?shaFile(candidateNeighborEdgesPath):"")<<"\",\n"
         <<"  \"candidate_neighbor_edge_keys_bytes\": "<<(candidateNeighborEdgesOK?fs::file_size(candidateNeighborEdgesPath):0u)<<",\n"
         <<"  \"candidate_neighbor_edge_key_encoding\": \"uint64 little-endian: high32 child row, low32 expanded source/carrier parent row\",\n"
         <<"  \"residual_reachable_expanded_parents\": "<<t.residualReachableExpandedParentRows.size()<<",\n"
         <<"  \"residual_reachable_children\": "<<t.residualReachableChildRows.size()<<",\n"
         <<"  \"input_sha256\": {\"parent_mass\": \""<<parentMassFileHash<<"\", \"parent_labels\": \""<<parentLabelsFileHash
         <<"\", \"child_mesh\": \""<<childHash<<"\", \"semantic_support\": \""<<supportBinaryHash
         <<"\", \"carrier_packet\": \""<<carrierBinaryHash<<"\", \"source_tets\": \""<<sourceTetTableHash
         <<"\", \"source_nodes\": \""<<sourceNodeTableHash<<"\", \"coordinate_frame\": \""<<frame<<"\"},\n"
         <<"  \"partial_flow_csv\": {\"path\": "<<std::quoted(flowPath.string())<<", \"sha256\": \""<<(flowOK?shaFile(flowPath):"")<<"\"},\n"
         <<"  \"partial_frontier_csv\": {\"path\": "<<std::quoted(frontierPath.string())<<", \"sha256\": \""<<(frontierOK?shaFile(frontierPath):"")<<"\"},\n"
         <<"  \"frontier_semantics\": \"residual-reachability snapshot only; not a proven min-cut\",\n"
         <<"  \"reuse_policy\": "<<std::quoted(useWarmStart?"this output is the cumulative warm-seeded residual-network flow; independently replay before any subsequent reuse; network state itself is not persisted":"diagnostic only; any later warm start must independently replay every edge against unchanged source and child capacities")<<"\n}\n";
  return receiptPath;
 };
 if(!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
      in,support.view(),carrier,result,carrierTrace,error,&telemetry,useWarmStart?&warmStart:nullptr)){
  if(telemetry.partialFlowCaptured){const auto partialReceipt=writePartialDiagnostic(telemetry);std::cerr<<"PARTIAL_FLOW_DIAGNOSTIC "<<partialReceipt.string()<<std::endl;}
  writeFailure("shared-capacity allocation: "+error,np,nc);std::cerr<<"ALLOC_FAIL "<<error<<std::endl;return 3;
 }
 const double solveSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
 std::cerr<<"SHARED_CAPACITY_ASSEMBLY_PASS edges="<<result.semanticCandidateEdgeCount<<" augmentations="<<result.augmentationCount<<" seconds="<<solveSeconds<<std::endl;
 // Versioned per-run receipt sidecar. It serializes the exact existing
 // aggregate transport arrays plus explicit carrier-expanded edge identities.
 // All scalars are the host's little-endian IEEE representations on Numi.
 static_assert(std::endian::native==std::endian::little);
 fs::path flowPath=resultPath;flowPath += ".flow.bin";
 std::ofstream flow(flowPath,std::ios::binary|std::ios::trunc);
 const char flowMagic[9]={'F','S','S','C','F','L','O','W','1'};
 const std::uint32_t flowVersion=1u;
 const std::array<std::uint64_t,10> flowCounts{
   static_cast<std::uint64_t>(parentMass.size()),static_cast<std::uint64_t>(childVolumes.size()),
   static_cast<std::uint64_t>(result.childParentRows.size()),static_cast<std::uint64_t>(carrierTrace.expandedParentToSourceParentRows.size()),
   static_cast<std::uint64_t>(carrierTrace.childParentCarrierRows.size()),static_cast<std::uint64_t>(carrierTrace.carrierSourceParentRows.size()),
   static_cast<std::uint64_t>(result.childSourceDonorBodyIndices.size()),static_cast<std::uint64_t>(result.groupClosure.size()),
   static_cast<std::uint64_t>(carrierOffsets.size()),static_cast<std::uint64_t>(carrierWeights.size())};
 bool flowOK=bool(flow.write(flowMagic,sizeof(flowMagic)))&&bool(flow.write(reinterpret_cast<const char*>(&flowVersion),sizeof(flowVersion)))&&
   bool(flow.write(reinterpret_cast<const char*>(flowCounts.data()),static_cast<std::streamsize>(flowCounts.size()*sizeof(std::uint64_t))));
 const auto writeS=[&](const std::string& value){return value.size()==64u&&bool(flow.write(value.data(),64));};
 flowOK=flowOK&&writeS(result.parentDataSHA256)&&writeS(result.childDataSHA256)&&
   writeS(result.semanticSupportSHA256)&&writeS(result.receiptSHA256)&&writeS(carrierTrace.sourceCarrierPacketSHA256)&&
   writeS(carrierTrace.sourceCarrierArraySHA256)&&writeS(carrierTrace.receiptSHA256)&&
   writeS(carrierTrace.expandedSolverReceiptSHA256)&&writeS(sourceTetTableHash)&&writeS(sourceNodeTableHash);
 flowOK=flowOK&&writeVector(flow,parentGlobalTetRows)&&writeVector(flow,result.parentMassTransferredKg)&&
   writeVector(flow,result.parentMeasureTransferredM3)&&writeVector(flow,result.childParentOffsets)&&
   writeVector(flow,result.childParentRows)&&writeVector(flow,result.childParentTransportMassKg)&&
   writeVector(flow,result.childParentTransportMeasureM3)&&writeVector(flow,result.childParentTransportDistanceM)&&
   writeVector(flow,result.childParentVolumeFractions)&&writeVector(flow,result.childMassKg)&&
   writeVector(flow,result.childImpliedDensityKgM3)&&writeVector(flow,result.childDominantParentRows)&&
   writeVector(flow,result.childSourceDonorOffsets)&&writeVector(flow,result.childSourceDonorBodyIndices)&&
   writeVector(flow,result.childSourceDonorMassKg)&&writeVector(flow,carrierTrace.expandedParentToSourceParentRows)&&
   writeVector(flow,carrierTrace.expandedParentToCarrierRows)&&writeVector(flow,carrierTrace.carrierSourceParentRows)&&
   writeVector(flow,carrierTrace.originalCentroidNearestAdmissibleDistanceM)&&writeVector(flow,carrierTrace.childParentCarrierOffsets)&&
   writeVector(flow,carrierTrace.childParentCarrierRows)&&writeVector(flow,carrierTrace.childParentCarrierMassKg)&&
   writeVector(flow,carrierTrace.childParentCarrierMeasureM3)&&writeVector(flow,carrierTrace.childParentCarrierDistanceM);
 flow.close();
 std::string flowSHA;
 if(!flowOK||!physical_skin_child_mass_detail::sha256File(flowPath,flowSHA,error)){
  writeFailure("carrier transport flow artifact write/hash failed: "+error,np,nc);return 4;
 }
 fs::create_directories(resultPath.parent_path());std::ofstream out(resultPath);out<<std::setprecision(17);
 out<<"{\n  \"status\": \"PASS_SOURCE_ONLY_SKIN_PARTIAL_SHARED_CAPACITY_ALLOCATION\",\n  \"physical_admission\": false,\n  \"native_steps\": 0,\n";
 out<<"  \"scope\": \"Skin-only; no Fat allocation or paired Skin/Fat closure\",\n  \"parent_skin_tets\": "<<np<<",\n  \"child_skin_tets\": "<<nc<<",\n  \"support_groups\": "<<ng<<",\n  \"p1_support_slots\": "<<ns<<",\n";
 out<<"  \"source_mass_kg\": ";double sm=0;for(auto x:parentMass)sm+=x;out<<sm<<",\n";
 out<<"  \"child_reference_volume_m3\": "<<totalVol<<",\n  \"minimum_child_density_kg_m3\": "<<result.minimumChildImpliedDensityKgM3<<",\n  \"maximum_child_density_kg_m3\": "<<result.maximumChildImpliedDensityKgM3<<",\n";
 out<<"  \"density_bounds_kg_m3\": [900, 1100],\n  \"maximum_transport_distance_allowed_m\": 0.019353,\n  \"maximum_transport_distance_used_m\": "<<result.maximumTransportDistanceUsedM<<",\n";
 out<<"  \"mass_weighted_mean_transport_distance_m\": "<<result.massWeightedMeanTransportDistanceM<<",\n  \"transport_edges\": "<<result.transportEdgeCount<<",\n  \"candidate_edges\": "<<result.semanticCandidateEdgeCount<<",\n  \"receiver_signature_group_count\": "<<telemetry.receiverSignatureGroupCount<<",\n  \"receiver_signature_grouped_child_rows\": "<<telemetry.receiverSignatureGroupedChildRows<<",\n  \"receiver_signature_grouped_edge_count\": "<<telemetry.receiverSignatureGroupedEdgeCount<<",\n  \"receiver_signature_expansion_rounds\": "<<telemetry.receiverSignatureExpansionRounds<<",\n  \"receiver_signature_expanded_edges\": "<<telemetry.receiverSignatureExpandedEdges<<",\n  \"receiver_signature_cut_candidates_scanned\": "<<telemetry.receiverSignatureCutCandidatesScanned<<",\n  \"receiver_signature_cut_enumeration_complete\": "<<(telemetry.receiverSignatureCutEnumerationComplete?"true":"false")<<",\n  \"augmentations\": "<<result.augmentationCount<<",\n  \"allocation_solver\": "<<std::quoted(result.allocationSolver)<<",\n  \"cost_optimality_certified\": false,\n";
 out<<"  \"maximum_parent_mass_closure_error_kg\": "<<result.maximumParentMassClosureErrorKg<<",\n  \"maximum_parent_measure_closure_error_m3\": "<<result.maximumParentMeasureClosureErrorM3<<",\n";
 out<<"  \"maximum_source_donor_closure_error_kg\": "<<result.maximumSourceDonorMassClosureErrorKg<<",\n  \"maximum_child_fraction_sum_error\": "<<result.maximumChildFractionSumError<<",\n";
 out<<"  \"child_mesh_sha256\": "<<std::quoted(childHash)<<",\n  \"input_child_node_sha256\": "<<std::quoted(nodeHash)<<",\n  \"input_child_element_sha256\": "<<std::quoted(eleHash)<<",\n  \"input_parent_mass_csv_sha256\": "<<std::quoted(parentMassFileHash)<<",\n  \"input_parent_label_csv_sha256\": "<<std::quoted(parentLabelsFileHash)<<",\n  \"support_binary_sha256\": "<<std::quoted(supportBinaryHash)<<",\n  \"semantic_support_sha256\": "<<std::quoted(result.semanticSupportSHA256)<<",\n  \"transport_receipt_sha256\": "<<std::quoted(result.receiptSHA256)<<",\n";
 out<<"  \"admissibility_source_support_sha256\": "<<std::quoted(support.admissibilitySourceSupportSHA256)<<",\n  \"admissibility_max_source_region_hops\": "<<support.admissibilityMaximumSourceRegionHops<<",\n  \"admissible_region_slots\": "<<support.cellAdmissibleRegionIds.size()<<",\n";
 out<<"  \"source_carrier_packet_sha256\": "<<std::quoted(carrierTrace.sourceCarrierPacketSHA256)<<",\n  \"source_carrier_array_sha256\": "<<std::quoted(carrierTrace.sourceCarrierArraySHA256)<<",\n  \"carrier_trace_sha256\": "<<std::quoted(carrierTrace.receiptSHA256)<<",\n  \"expanded_solver_receipt_sha256\": "<<std::quoted(carrierTrace.expandedSolverReceiptSHA256)<<",\n";
 out<<"  \"source_carrier_parent_rows\": "<<carrierTrace.carrierSourceParentRows.size()<<",\n  \"source_carrier_count\": "<<carrierTrace.expandedParentToCarrierRows.size()-parentMass.size()<<",\n  \"carrier_flow_edge_count\": "<<carrierTrace.childParentCarrierRows.size()<<",\n  \"expanded_source_parent_count\": "<<carrierTrace.expandedParentToSourceParentRows.size()<<",\n";
 out<<"  \"flow_artifact_path\": "<<std::quoted(flowPath.string())<<",\n  \"flow_artifact_sha256\": "<<std::quoted(flowSHA)<<",\n  \"flow_artifact_schema\": \"FSSCFLOW1-v1 little-endian arrays; see header counts and 10 SHA fields\",\n";
 out<<"  \"input_parent_mass_sha256\": "<<std::quoted(all.parentMassPacketSHA256)<<",\n  \"input_parent_source_sha256\": "<<std::quoted(all.parentSourceContributionSHA256)<<",\n  \"input_coordinate_frame_sha256\": "<<std::quoted(frame)<<",\n";
 out<<"  \"group_closure\": [";
 for(std::size_t i=0;i<result.groupClosure.size();++i){const auto&g=result.groupClosure[i];if(i)out<<",";out<<"{\"material\":"<<g.materialIndex<<",\"geometric_donor\":"<<g.donorBodyIndex<<",\"region\":"<<g.anatomicalRegionIndex<<",\"parent_mass_kg\":"<<g.parentMassKg<<",\"child_mass_kg\":"<<g.childMassKg<<",\"reachable_capacity_kg\":"<<g.childMaximumCapacityKg<<"}";}
 out<<"],\n  \"source_donor_closure\": [";
 for(std::size_t i=0;i<result.sourceDonorClosure.size();++i){const auto&g=result.sourceDonorClosure[i];if(i)out<<",";out<<"{\"material\":"<<g.materialIndex<<",\"source_donor\":"<<g.sourceDonorBodyIndex<<",\"parent_mass_kg\":"<<g.parentMassKg<<",\"child_mass_kg\":"<<g.childMassKg<<"}";}
 out<<"],\n  \"solve_seconds\": "<<solveSeconds<<",\n  \"elapsed_seconds\": "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"\n}\n";
 return 0;
}
