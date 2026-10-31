#pragma once

#include "ConservativeSpatialSourceCarrierQuadrature.hpp"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

inline constexpr std::uint32_t kSourceCarrierLPInvalidRow =
    std::numeric_limits<std::uint32_t>::max();

struct SourceCarrierLPHeldFlow {
    std::uint32_t childRow=0;
    std::uint32_t parentRow=0;
    std::uint32_t oldExpandedParentRow=kSourceCarrierLPInvalidRow;
    std::uint32_t oldCarrierRow=kSourceCarrierLPInvalidRow;
    double massKg=0.0;
    double sourceMeasureM3=0.0;
    double recordedDistanceM=0.0;
};

struct SourceCarrierLPPositiveVariable {
    std::uint32_t positiveIndex=0;
    std::uint32_t childRow=0;
    std::uint32_t parentRow=0;
    double flowMassKg=0.0;
    std::array<double,4> barycentric{};
    double claimedDistanceM=0.0;
};

struct SourceCarrierLPProvenance {
    std::string heldPartialManifestSHA256;
    std::string heldCarrierFlowCSV_SHA256;
    std::string lpProposalSHA256;
    std::string lpInputManifestSHA256;
    std::string feasibilitySourceSHA256;
    std::size_t heldDinicAugmentationCount=0;
    std::size_t lpPositiveVariableCount=0;
    std::size_t heldPositiveFlowRows=0;
    std::size_t lpCandidateEdgeCount=0;
    std::size_t heldCandidateEdgeCount=0;
    std::size_t semanticCandidateEdgeCount=0;
};

struct SourceCarrierLPOwnedArrays {
    std::vector<std::uint32_t> parentOffsets;
    std::vector<double> weights;
    std::vector<std::array<double,4>> barycentric;
    std::vector<std::uint32_t> witnessChildRows;
    std::vector<double> witnessDistancesM;
    SourceCellCarrierQuadratureInput view(
        const SourceCellCarrierQuadratureInput& geometry,
        const std::string& outputCarrierPacketSHA256) const {
        return {parentOffsets,geometry.globalSourceTetRows,geometry.sourceTetNodeRows,
            geometry.parentReferenceTetWorldVertices,weights,barycentric,witnessChildRows,
            witnessDistancesM,outputCarrierPacketSHA256,geometry.sourceTetTableSHA256,
            geometry.sourceNodeTableSHA256};
    }
};

struct SourceCarrierLPFinalizedFlow {
    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    SourceCarrierLPOwnedArrays carriers;
    std::vector<std::uint32_t> expandedParentToSourceParentRows;
    std::vector<std::uint32_t> expandedParentToCarrierRows;
    std::size_t heldPositiveFlowRows=0;
    std::size_t lpPositiveVariableCount=0;
    std::string sourceCarrierPacketSHA256;
};

namespace source_carrier_lp_detail {
using Key=std::pair<std::uint32_t,std::uint32_t>;
using ParentChild=std::pair<std::uint32_t,std::uint32_t>;
using ChildParentVirtual=std::tuple<std::uint32_t,std::uint32_t,std::uint32_t>;

struct CarrierRecord {
    std::uint32_t parent=0;
    std::uint32_t oldCarrier=kSourceCarrierLPInvalidRow;
    std::uint32_t lpPositiveIndex=kSourceCarrierLPInvalidRow;
    std::uint32_t witnessChild=0;
    double weight=0.0;
    std::array<double,4> bary{};
    double witnessDistance=0.0;
    bool isLP=false;
};

inline double distance(const SpatialMassTransportPoint& a,const SpatialMassTransportPoint& b) {
    const double x=a[0]-b[0],y=a[1]-b[1],z=a[2]-b[2];
    return std::sqrt(x*x+y*y+z*z);
}
inline SpatialMassTransportPoint point(const std::array<SpatialMassTransportPoint,4>& tet,
                                       const std::array<double,4>& bary) {
    SpatialMassTransportPoint p{};
    for(std::size_t k=0;k<4;++k)for(std::size_t a=0;a<3;++a)p[a]+=bary[k]*tet[k][a];
    return p;
}
inline bool nearRecorded(const double actual,const double recorded) {
    if(!std::isfinite(actual)||!std::isfinite(recorded))return false;
    return std::abs(actual-recorded)<=std::max(1.0e-12,
        64.0*std::numeric_limits<double>::epsilon()*std::max({1.0,std::abs(actual),std::abs(recorded)}));
}
inline bool validDigest(const std::string& s) {
    if(s.size()!=64)return false;
    for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
}
inline bool sha256File(const std::filesystem::path& path,std::string& out,std::string& error) {
    std::ifstream f(path,std::ios::binary);if(!f){error="cannot open output for SHA-256";return false;}
    CC_SHA256_CTX ctx{};CC_SHA256_Init(&ctx);std::array<unsigned char,65536> buffer{};
    while(f){f.read(reinterpret_cast<char*>(buffer.data()),static_cast<std::streamsize>(buffer.size()));
        const auto n=f.gcount();if(n>0)CC_SHA256_Update(&ctx,buffer.data(),static_cast<CC_LONG>(n));}
    if(!f.eof()){error="failed reading output for SHA-256";return false;}
    std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};CC_SHA256_Final(digest.data(),&ctx);
    constexpr char hex[]="0123456789abcdef";out.resize(digest.size()*2u);
    for(std::size_t i=0;i<digest.size();++i){out[2u*i]=hex[digest[i]>>4u];out[2u*i+1u]=hex[digest[i]&15u];}
    return true;
}

// Convert held terminal Dinic rows plus strictly positive LP variables into one
// complete source-cell quadrature and recorded carrier-flow trace. This function
// does not call an optimizer. The owning feasibility API rechecks all constraints.
[[nodiscard]] inline bool buildAndVerify(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& originalCarrier,
        const std::span<const double> heldParentScale,
        const std::span<const SourceCarrierLPHeldFlow> heldRows,
        const std::span<const SourceCarrierLPPositiveVariable> lpRows,
        const SourceCarrierLPProvenance& provenance,
        SourceCarrierLPFinalizedFlow& output,
        std::string& error) {
    using namespace source_carrier_transport_detail;
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    const std::size_t np=in.parentMassKg.size(),nc=in.childReferenceVolumeM3.size();
    if(np==0||nc==0||heldParentScale.size()!=np||heldRows.empty()||
       originalCarrier.parentOffsets.size()!=np+1u||originalCarrier.parentOffsets.empty()||
       originalCarrier.sourceTetNodeRows.size()!=np||originalCarrier.globalSourceTetRows.size()!=np||
       originalCarrier.parentReferenceTetWorldVertices.size()!=np||
       originalCarrier.parentOffsets.front()!=0u||
       originalCarrier.parentOffsets.back()!=originalCarrier.carrierWeights.size()||
       originalCarrier.carrierBarycentric.size()!=originalCarrier.carrierWeights.size()||
       originalCarrier.witnessChildRows.size()!=originalCarrier.carrierWeights.size()||
       originalCarrier.witnessChildCentroidDistancesM.size()!=originalCarrier.carrierWeights.size()||
       !validDigest(provenance.heldPartialManifestSHA256)||
       !validDigest(provenance.heldCarrierFlowCSV_SHA256)||
       !validDigest(provenance.lpProposalSHA256)||!validDigest(provenance.lpInputManifestSHA256)||
       !validDigest(provenance.feasibilitySourceSHA256)||
       provenance.heldDinicAugmentationCount==0u||
       provenance.heldDinicAugmentationCount>in.maximumAugmentations||
       provenance.lpPositiveVariableCount!=lpRows.size()||provenance.heldPositiveFlowRows!=heldRows.size()||
       provenance.heldCandidateEdgeCount==0u||provenance.semanticCandidateEdgeCount==0u||
       provenance.semanticCandidateEdgeCount>in.maximumAllowedEdges||
       provenance.heldCandidateEdgeCount>in.maximumAllowedEdges||
       provenance.lpCandidateEdgeCount>in.maximumAllowedEdges-provenance.heldCandidateEdgeCount)
        return fail("LP finalizer input domains or provenance pins are incomplete");

    // Reconstruct the source packet's exact expanded-row numbering before
    // interpreting the held partial-flow CSV.
    std::vector<std::uint32_t> oldCarrierToExpanded(originalCarrier.carrierWeights.size(),kSourceCarrierLPInvalidRow);
    std::vector<std::uint32_t> oldDefaultExpanded(np,kSourceCarrierLPInvalidRow);
    std::uint32_t oldExpandedCount=0;
    for(std::size_t p=0;p<np;++p) {
        const auto b=originalCarrier.parentOffsets[p],e=originalCarrier.parentOffsets[p+1u];
        if(e<b||e>originalCarrier.carrierWeights.size())return fail("old quadrature parent offsets are malformed");
        if(b==e)oldDefaultExpanded[p]=oldExpandedCount++;
        else for(std::uint32_t k=b;k<e;++k)oldCarrierToExpanded[k]=oldExpandedCount++;
    }
    for(std::size_t p=0;p<np;++p) {
        const double scale=heldParentScale[p];
        if(!std::isfinite(scale)||scale<0.0||scale>1.0)
            return fail("held-parent scale is outside [0,1]");
    }

    std::map<Key,std::vector<const SourceCarrierLPHeldFlow*>> heldBySite;
    for(const auto& row:heldRows) {
        if(row.parentRow>=np||row.childRow>=nc||!std::isfinite(row.massKg)||row.massKg<=0.0||
           !std::isfinite(row.sourceMeasureM3)||row.sourceMeasureM3<=0.0||
           !std::isfinite(row.recordedDistanceM)||row.recordedDistanceM<0.0)
            return fail("held partial flow contains an invalid row or nonpositive flow");
        const auto p=row.parentRow;
        const auto oldCarrier=row.oldCarrierRow;
        std::uint32_t expectedExpanded=kSourceCarrierLPInvalidRow;
        if(oldCarrier==kSourceCarrierLPInvalidRow) {
            if(originalCarrier.parentOffsets[p]!=originalCarrier.parentOffsets[p+1u])
                return fail("held centroid flow points at a parent that had explicit old carrier rows");
            expectedExpanded=oldDefaultExpanded[p];
        } else {
            if(oldCarrier>=oldCarrierToExpanded.size()||
               oldCarrier<originalCarrier.parentOffsets[p]||
               oldCarrier>=originalCarrier.parentOffsets[p+1u])
                return fail("held flow old carrier row is outside its parent source packet row");
            expectedExpanded=oldCarrierToExpanded[oldCarrier];
        }
        if(row.oldExpandedParentRow!=expectedExpanded)
            return fail("held partial-flow expanded-parent row does not match the pinned old carrier packet");
        const auto oldPoint=oldCarrier==kSourceCarrierLPInvalidRow?
            in.parentCentroidsWorld[p]:
            point(originalCarrier.parentReferenceTetWorldVertices[p],originalCarrier.carrierBarycentric[oldCarrier]);
        const double measured=distance(oldPoint,in.childCentroidsWorld[row.childRow]);
        const double density=in.parentMassKg[p]/in.parentReferenceVolumeM3[p];
        const double sourceMeasure=row.massKg/density;
        if(!nearRecorded(measured,row.recordedDistanceM)||!nearRecorded(sourceMeasure,row.sourceMeasureM3)||
           measured>in.maximumTransportDistanceM)
            return fail("held partial flow distance or source measure differs from current canonical input");
        if(heldParentScale[p]>0.0)heldBySite[{p,oldCarrier}].push_back(&row);
    }

    std::set<std::uint32_t> lpIndices;
    std::vector<std::vector<CarrierRecord>> records(np);
    std::map<Key,std::size_t> heldRecordIndex;
    std::map<std::uint32_t,std::pair<std::uint32_t,std::size_t>> lpRecordIndex;
    for(const auto& [key,rows]:heldBySite) {
        const auto p=key.first,oldCarrier=key.second;
        spatial_mass_transport_detail::ExactDoubleSum massSum;
        std::uint32_t witness=rows.front()->childRow;
        for(const auto* row:rows)massSum.add(row->massKg*heldParentScale[p]);
        const double heldMass=massSum.rounded();
        if(!std::isfinite(heldMass)||heldMass<=0.0)return fail("adjusted held carrier mass is nonpositive");
        CarrierRecord rec;rec.parent=p;rec.oldCarrier=oldCarrier;rec.witnessChild=oldCarrier==kSourceCarrierLPInvalidRow?witness:originalCarrier.witnessChildRows[oldCarrier];
        rec.weight=heldMass/in.parentMassKg[p];
        rec.bary=oldCarrier==kSourceCarrierLPInvalidRow?
            std::array<double,4>{0.25,0.25,0.25,0.25}:originalCarrier.carrierBarycentric[oldCarrier];
        records[p].push_back(rec);
    }
    std::vector<const SourceCarrierLPPositiveVariable*> lpSorted;
    lpSorted.reserve(lpRows.size());
    for(const auto& row:lpRows) {
        if(row.parentRow>=np||row.childRow>=nc||!std::isfinite(row.flowMassKg)||row.flowMassKg<=0.0||
           !std::isfinite(row.claimedDistanceM)||row.claimedDistanceM<0.0||
           !lpIndices.insert(row.positiveIndex).second)
            return fail("LP positive-variable row is malformed, duplicated, or nonpositive");
        double sum=0.0;
        for(double b:row.barycentric) {
            if(!std::isfinite(b)||b<-1.0e-13||b>1.0+1.0e-13)
                return fail("LP carrier barycentric lies outside its source tetrahedron");
            sum+=b;
        }
        if(std::abs(sum-1.0)>2.0e-12)return fail("LP carrier barycentric does not sum to one");
        const double measured=distance(point(originalCarrier.parentReferenceTetWorldVertices[row.parentRow],
                                              row.barycentric),in.childCentroidsWorld[row.childRow]);
        if(!nearRecorded(measured,row.claimedDistanceM)||measured>in.maximumTransportDistanceM)
            return fail("LP carrier distance fails the unchanged role radius when recomputed in C++");
        lpSorted.push_back(&row);
    }
    std::sort(lpSorted.begin(),lpSorted.end(),[](auto a,auto b){
        if(a->parentRow!=b->parentRow)return a->parentRow<b->parentRow;
        return a->positiveIndex<b->positiveIndex;
    });
    for(const auto* row:lpSorted) {
        CarrierRecord rec;rec.parent=row->parentRow;rec.lpPositiveIndex=row->positiveIndex;
        rec.witnessChild=row->childRow;rec.weight=row->flowMassKg/in.parentMassKg[row->parentRow];
        rec.bary=row->barycentric;rec.isLP=true;
        records[rec.parent].push_back(rec);
    }

    SourceCarrierLPFinalizedFlow next;
    next.heldPositiveFlowRows=heldRows.size();next.lpPositiveVariableCount=lpRows.size();
    next.carriers.parentOffsets.assign(np+1u,0u);
    for(std::size_t p=0;p<np;++p) {
        auto& rows=records[p];
        std::stable_sort(rows.begin(),rows.end(),[](const CarrierRecord&a,const CarrierRecord&b){
            if(a.isLP!=b.isLP)return !a.isLP;
            if(a.isLP)return a.lpPositiveIndex<b.lpPositiveIndex;
            return a.oldCarrier<b.oldCarrier;
        });
        next.carriers.parentOffsets[p]=static_cast<std::uint32_t>(next.carriers.weights.size());
        for(const auto& rec:rows) {
            if(rec.parent!=p||rec.witnessChild>=nc||!std::isfinite(rec.weight)||rec.weight<=0.0)
                return fail("assembled source carrier row is malformed");
            const auto carrierRow=static_cast<std::uint32_t>(next.carriers.weights.size());
            next.carriers.weights.push_back(rec.weight);
            next.carriers.barycentric.push_back(rec.bary);
            next.carriers.witnessChildRows.push_back(rec.witnessChild);
            const auto world=point(originalCarrier.parentReferenceTetWorldVertices[p],rec.bary);
            const auto measured=distance(world,in.childCentroidsWorld[rec.witnessChild]);
            next.carriers.witnessDistancesM.push_back(measured);
            if(rec.isLP)lpRecordIndex.emplace(rec.lpPositiveIndex,std::pair{p,static_cast<std::size_t>(carrierRow)});
            else heldRecordIndex.emplace(Key{p,rec.oldCarrier},static_cast<std::size_t>(carrierRow));
        }
    }
    next.carriers.parentOffsets[np]=static_cast<std::uint32_t>(next.carriers.weights.size());
    spatial_mass_transport_detail::SHA256Builder packetHash;
    packetHash.string("numi.human.source-carrier-held-plus-lp-packet.v1");
    packetHash.string(originalCarrier.sourceCarrierPacketSHA256);
    packetHash.string(provenance.heldPartialManifestSHA256);
    packetHash.string(provenance.heldCarrierFlowCSV_SHA256);
    packetHash.string(provenance.lpProposalSHA256);
    packetHash.string(provenance.lpInputManifestSHA256);
    packetHash.u64(next.carriers.parentOffsets.size());for(auto x:next.carriers.parentOffsets)packetHash.u32(x);
    packetHash.u64(originalCarrier.globalSourceTetRows.size());for(auto x:originalCarrier.globalSourceTetRows)packetHash.u32(x);
    packetHash.u64(next.carriers.weights.size());for(auto x:next.carriers.weights)packetHash.real(x);
    for(const auto& row:next.carriers.barycentric)for(auto x:row)packetHash.real(x);
    for(auto x:next.carriers.witnessChildRows)packetHash.u32(x);
    for(auto x:next.carriers.witnessDistancesM)packetHash.real(x);
    next.sourceCarrierPacketSHA256=packetHash.finish();
    auto carriers=next.carriers.view(originalCarrier,next.sourceCarrierPacketSHA256);
    std::vector<std::uint32_t> parentGroup,carrierParentRows;
    std::vector<std::map<std::uint32_t,double>> preferences;
    std::vector<std::vector<std::uint32_t>> admissibleRegions;
    std::vector<SpatialMassTransportPoint> carrierPoints;
    std::vector<double> strictNearest;
    if(!source_carrier_transport_detail::validateCarrierInput(in,support,carriers,parentGroup,
            preferences,admissibleRegions,carrierPoints,carrierParentRows,strictNearest,error))
        return false;

    std::vector<std::uint32_t> carrierToVirtual(next.carriers.weights.size(),kSourceCarrierLPInvalidRow);
    std::vector<std::uint32_t> defaultVirtual(np,kSourceCarrierLPInvalidRow);
    std::uint32_t virtualCount=0;
    for(std::size_t p=0;p<np;++p) {
        const auto b=next.carriers.parentOffsets[p],e=next.carriers.parentOffsets[p+1u];
        if(b==e)defaultVirtual[p]=virtualCount++;
        else for(std::uint32_t k=b;k<e;++k)carrierToVirtual[k]=virtualCount++;
    }
    auto heldVirtual=[&](std::uint32_t p,std::uint32_t old)->std::uint32_t {
        if(next.carriers.parentOffsets[p]==next.carriers.parentOffsets[p+1u])return defaultVirtual[p];
        auto found=heldRecordIndex.find({p,old});
        if(found==heldRecordIndex.end())return kSourceCarrierLPInvalidRow;
        return carrierToVirtual[found->second];
    };
    auto lpVirtual=[&](std::uint32_t index)->std::uint32_t {
        auto found=lpRecordIndex.find(index);
        if(found==lpRecordIndex.end())return kSourceCarrierLPInvalidRow;
        return carrierToVirtual[found->second.second];
    };

    std::map<ChildParentVirtual,spatial_mass_transport_detail::ExactDoubleSum> flowMassSums;
    std::map<ChildParentVirtual,double> flowDistances;
    for(const auto& row:heldRows) {
        const double mass=row.massKg*heldParentScale[row.parentRow];
        if(mass==0.0)continue;
        const auto v=heldVirtual(row.parentRow,row.oldCarrierRow);
        if(v==kSourceCarrierLPInvalidRow)return fail("held partial-flow row lost its assembled carrier mapping");
        const auto carrierRow=next.carriers.parentOffsets[row.parentRow]==next.carriers.parentOffsets[row.parentRow+1u]?
            kSourceCarrierLPInvalidRow: [&](){
                const auto found=heldRecordIndex.find({row.parentRow,row.oldCarrierRow});
                return found==heldRecordIndex.end()?kSourceCarrierLPInvalidRow:static_cast<std::uint32_t>(found->second);
            }();
        const auto from=carrierRow==kSourceCarrierLPInvalidRow?in.parentCentroidsWorld[row.parentRow]:
            point(originalCarrier.parentReferenceTetWorldVertices[row.parentRow],next.carriers.barycentric[carrierRow]);
        const double d=distance(from,in.childCentroidsWorld[row.childRow]);
        auto key=ChildParentVirtual{row.childRow,row.parentRow,v};
        flowMassSums[key].add(mass);
        auto [it,inserted]=flowDistances.emplace(key,d);
        if(!inserted&&it->second!=d)return fail("same held carrier-child route recomputed to inconsistent distances");
    }
    for(const auto* row:lpSorted) {
        const auto v=lpVirtual(row->positiveIndex);
        if(v==kSourceCarrierLPInvalidRow)return fail("LP positive variable lost its assembled carrier mapping");
        const auto index=lpRecordIndex.at(row->positiveIndex).second;
        const auto d=distance(point(originalCarrier.parentReferenceTetWorldVertices[row->parentRow],
                                    next.carriers.barycentric[index]),in.childCentroidsWorld[row->childRow]);
        auto key=ChildParentVirtual{row->childRow,row->parentRow,v};
        flowMassSums[key].add(row->flowMassKg);
        auto [it,inserted]=flowDistances.emplace(key,d);
        if(!inserted&&it->second!=d)return fail("same LP carrier-child route recomputed to inconsistent distances");
    }
    std::map<std::pair<std::uint32_t,std::uint32_t>,
        std::vector<std::pair<std::uint32_t,double>>> edgePieces;
    for(const auto& [key,sum]:flowMassSums) {
        const auto [child,parent,virt]=key;const double mass=sum.rounded();
        if(!std::isfinite(mass)||mass<=0.0)return fail("canonical flow mass is invalid");
        edgePieces[{child,parent}].push_back({virt,mass});
    }
    next.expandedParentToSourceParentRows.reserve(virtualCount);
    next.expandedParentToCarrierRows.reserve(virtualCount);
    std::vector<double> virtualMass(virtualCount),virtualVolume(virtualCount);
    std::uint32_t v=0;
    for(std::size_t p=0;p<np;++p) {
        const auto b=next.carriers.parentOffsets[p],e=next.carriers.parentOffsets[p+1u];
        if(b==e) {
            next.expandedParentToSourceParentRows.push_back(static_cast<std::uint32_t>(p));
            next.expandedParentToCarrierRows.push_back(kSourceCarrierLPInvalidRow);
            virtualMass[v]=in.parentMassKg[p];virtualVolume[v]=in.parentReferenceVolumeM3[p];++v;
        } else for(std::uint32_t k=b;k<e;++k) {
            next.expandedParentToSourceParentRows.push_back(static_cast<std::uint32_t>(p));
            next.expandedParentToCarrierRows.push_back(k);
            virtualMass[v]=in.parentMassKg[p]*next.carriers.weights[k];
            virtualVolume[v]=in.parentReferenceVolumeM3[p]*next.carriers.weights[k];++v;
        }
    }
    if(v!=virtualCount)return fail("expanded carrier row count does not close");
    auto& result=next.result;auto& trace=next.trace;
    result.method=UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    result.usesFractionalSemanticSupport=true;
    result.parentMassPacketSHA256=std::string(in.parentMassPacketSHA256);
    result.parentSourceContributionSHA256=std::string(in.parentSourceContributionSHA256);
    result.childMeshSHA256=std::string(in.childMeshSHA256);
    result.regionLabelSHA256=std::string(in.regionLabelSHA256);
    result.coordinateFrameSHA256=std::string(in.coordinateFrameSHA256);
    result.maximumTransportDistanceM=in.maximumTransportDistanceM;
    result.relativeClosureTolerance=in.relativeClosureTolerance;
    result.absoluteMassToleranceKg=in.absoluteMassToleranceKg;
    result.absoluteMeasureToleranceM3=in.absoluteMeasureToleranceM3;
    result.maximumAllowedEdges=in.maximumAllowedEdges;
    result.maximumAugmentations=in.maximumAugmentations;
    result.augmentationCount=provenance.heldDinicAugmentationCount;
    result.semanticCandidateEdgeCount=provenance.semanticCandidateEdgeCount;
    result.parentMassTransferredKg.assign(np,0.0);
    result.parentMeasureTransferredM3.assign(np,0.0);
    result.childParentOffsets.assign(nc+1u,0u);
    result.childMassKg.assign(nc,0.0);
    result.childAncestryMeasureVolumeM3.assign(nc,0.0);
    result.childImpliedDensityKgM3.assign(nc,0.0);
    result.childDominantParentRows.assign(nc,kSourceCarrierLPInvalidRow);
    result.childSourceDonorOffsets.assign(nc+1u,0u);
    result.transportEdgeCount=0;
    for(const auto& [edge,pieces]:edgePieces) {
        (void)edge;(void)pieces;
    }
    trace.childParentCarrierOffsets.assign(1u,0u);
    for(std::uint32_t child=0;child<nc;++child) {
        result.childParentOffsets[child]=static_cast<std::uint32_t>(result.childParentRows.size());
        const auto firstChildEdge=result.childParentRows.size();
        spatial_mass_transport_detail::ExactDoubleSum childMass,childMeasure;
        double bestMeasure=-1.0;std::uint32_t dominant=kSourceCarrierLPInvalidRow;
        for(auto it=edgePieces.lower_bound({child,0u});it!=edgePieces.end()&&it->first.first==child;++it) {
            const auto parent=it->first.second;
            auto pieces=it->second;
            std::sort(pieces.begin(),pieces.end(),[](auto a,auto b){return a.first<b.first;});
            spatial_mass_transport_detail::ExactDoubleSum edgeMass,edgeMeasure;
            long double weightedDistance=0.0L;
            for(const auto& piece:pieces) {
                const auto virt=piece.first;const double mass=piece.second;
                if(virt>=virtualCount||next.expandedParentToSourceParentRows[virt]!=parent||
                   !std::isfinite(mass)||mass<=0.0)return fail("flow edge uses an invalid expanded source row");
                const double density=virtualMass[virt]/virtualVolume[virt];
                const double measure=mass/density;
                const auto key=ChildParentVirtual{child,parent,virt};
                const double d=flowDistances.at(key);
                if(!std::isfinite(measure)||measure<=0.0||!std::isfinite(d)||
                   d>in.maximumTransportDistanceM)
                    return fail("recomputed LP/held edge measure or distance is invalid");
                edgeMass.add(mass);edgeMeasure.add(measure);
                weightedDistance+=static_cast<long double>(mass)*d;
                trace.childParentCarrierRows.push_back(virt);
                trace.childParentCarrierMassKg.push_back(mass);
                trace.childParentCarrierMeasureM3.push_back(measure);
                trace.childParentCarrierDistanceM.push_back(d);
                result.parentMassTransferredKg[parent]+=mass;
                result.parentMeasureTransferredM3[parent]+=measure;
            }
            const double mass=edgeMass.rounded(),measure=edgeMeasure.rounded();
            if(!(mass>0.0&&measure>0.0))return fail("aggregate LP/held child-parent flow is nonpositive");
            result.childParentRows.push_back(parent);
            result.childParentTransportMassKg.push_back(mass);
            result.childParentTransportMeasureM3.push_back(measure);
            result.childParentTransportDistanceM.push_back(static_cast<double>(weightedDistance/mass));
            trace.childParentCarrierOffsets.push_back(static_cast<std::uint32_t>(trace.childParentCarrierRows.size()));
            childMass.add(mass);childMeasure.add(measure);
            if(measure>bestMeasure||(measure==bestMeasure&&parent<dominant)){bestMeasure=measure;dominant=parent;}
        }
        result.childParentOffsets[child+1u]=static_cast<std::uint32_t>(result.childParentRows.size());
        const double cm=childMass.rounded(),cv=childMeasure.rounded();
        if(!(cm>0.0&&cv>0.0)||dominant==kSourceCarrierLPInvalidRow)
            return fail("LP/held flow leaves a child without positive mass or reference measure");
        result.childMassKg[child]=cm;result.childAncestryMeasureVolumeM3[child]=cv;
        result.childImpliedDensityKgM3[child]=cm/in.childReferenceVolumeM3[child];
        for(std::size_t edge=firstChildEdge;edge<result.childParentRows.size();++edge)
            result.childParentVolumeFractions.push_back(result.childParentTransportMeasureM3[edge]/cv);
        result.childDominantParentRows[child]=dominant;
    }
    result.transportEdgeCount=result.childParentRows.size();
    trace.expandedParentToSourceParentRows=next.expandedParentToSourceParentRows;
    trace.expandedParentToCarrierRows=next.expandedParentToCarrierRows;
    trace.carrierSourceParentRows=carrierParentRows;
    trace.originalCentroidNearestAdmissibleDistanceM=strictNearest;
    trace.sourceCarrierPacketSHA256=std::string(carriers.sourceCarrierPacketSHA256);
    trace.sourceCarrierArraySHA256=source_carrier_transport_detail::carrierArrayDigest(in,carriers);
    trace.expandedSolverReceiptSHA256=provenance.feasibilitySourceSHA256;

    std::string feasibilityError;
    if(!canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        in,support,carriers,result,trace,feasibilityError))
        return fail("solver-free source-carrier feasibility rejected LP/held flow: "+feasibilityError);
    output=std::move(next);
    error.clear();return true;
}


struct SourceCarrierLPExportMetadata {
    std::map<std::string,std::string> inputSHA256;
    std::string method="held_dinic_partial_plus_lp_positive_carrier_flows";
    std::string feasibilitySourceSHA256;
    bool physicalAdmission=false;
    std::uint32_t nativeSteps=0;
};

[[nodiscard]] inline bool writeExistingSourceCarrierCSVContract(
        const std::filesystem::path& outputDirectory,
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        const std::span<const std::uint32_t> globalSourceTetRows,
        const ConservativeSpatialMassTransportResult& result,
        const SourceCellCarrierTransportTrace& trace,
        const SourceCarrierLPProvenance& provenance,
        const SourceCarrierLPExportMetadata& metadata,
        std::string& error) {
    using namespace source_carrier_lp_detail;
    const auto fail=[&](std::string m){error=std::move(m);return false;};
    const auto np=in.parentMassKg.size(),nc=in.childReferenceVolumeM3.size();
    const auto monotoneOffsets=[](const auto& offsets,std::size_t rows,std::size_t expected) {
        if(offsets.size()!=expected||offsets.empty()||offsets.front()!=0u||offsets.back()!=rows)return false;
        for(std::size_t i=1;i<offsets.size();++i)if(offsets[i]<offsets[i-1u])return false;
        return true;
    };
    if(outputDirectory.empty()||np==0||nc==0||globalSourceTetRows.size()!=np||
       in.parentCentroidsWorld.size()!=np||in.parentReferenceVolumeM3.size()!=np||
       in.parentMaterialIndices.size()!=np||in.parentDonorBodyIndices.size()!=np||
       in.parentAnatomicalRegionIndices.size()!=np||
       in.parentSourceDonorOffsets.size()!=np+1u||
       in.parentSourceDonorBodyIndices.size()!=in.parentSourceDonorMassKg.size()||
       !monotoneOffsets(in.parentSourceDonorOffsets,in.parentSourceDonorBodyIndices.size(),np+1u)||
       in.childCentroidsWorld.size()!=nc||in.childMaterialIndices.size()!=nc||
       in.childDonorBodyIndices.size()!=nc||in.childAnatomicalRegionIndices.size()!=nc||
       support.parentStableSourceFaceIdentities.size()!=np||
       !validDigest(metadata.feasibilitySourceSHA256)||
       metadata.feasibilitySourceSHA256!=provenance.feasibilitySourceSHA256||
       !validDigest(provenance.heldPartialManifestSHA256)||
       !validDigest(provenance.heldCarrierFlowCSV_SHA256)||
       !validDigest(provenance.lpProposalSHA256)||!validDigest(provenance.lpInputManifestSHA256)||
       !validDigest(provenance.feasibilitySourceSHA256)||
       provenance.heldDinicAugmentationCount==0u||provenance.heldPositiveFlowRows==0u||
       provenance.lpPositiveVariableCount==0u||provenance.lpCandidateEdgeCount==0u||
       provenance.heldCandidateEdgeCount==0u||provenance.semanticCandidateEdgeCount==0u||
       provenance.semanticCandidateEdgeCount>in.maximumAllowedEdges||
       provenance.heldCandidateEdgeCount>in.maximumAllowedEdges||
       provenance.lpCandidateEdgeCount>in.maximumAllowedEdges-provenance.heldCandidateEdgeCount||
       metadata.method!="held_dinic_partial_plus_lp_positive_carrier_flows"||
       metadata.physicalAdmission||metadata.nativeSteps!=0u||
       !result.usesFractionalSemanticSupport||
       result.parentMassTransferredKg.size()!=np||result.parentMeasureTransferredM3.size()!=np||
       result.childMassKg.size()!=nc||result.childAncestryMeasureVolumeM3.size()!=nc||
       result.childImpliedDensityKgM3.size()!=nc||result.childParentRows.size()!=result.childParentTransportMassKg.size()||
       result.childParentRows.size()!=result.childParentTransportMeasureM3.size()||
       result.childParentRows.size()!=result.childParentTransportDistanceM.size()||
       result.childParentRows.size()!=result.childParentVolumeFractions.size()||
       result.transportEdgeCount!=result.childParentRows.size()||
       result.augmentationCount!=provenance.heldDinicAugmentationCount||
       result.semanticCandidateEdgeCount!=provenance.semanticCandidateEdgeCount||
       !monotoneOffsets(result.childParentOffsets,result.childParentRows.size(),nc+1u)||
       result.childSourceDonorBodyIndices.size()!=result.childSourceDonorMassKg.size()||
       !monotoneOffsets(result.childSourceDonorOffsets,result.childSourceDonorBodyIndices.size(),nc+1u)||
       trace.expandedParentToSourceParentRows.size()!=trace.expandedParentToCarrierRows.size()||
       trace.childParentCarrierMassKg.size()!=trace.childParentCarrierRows.size()||
       trace.childParentCarrierMeasureM3.size()!=trace.childParentCarrierRows.size()||
       trace.childParentCarrierDistanceM.size()!=trace.childParentCarrierRows.size()||
       !monotoneOffsets(trace.childParentCarrierOffsets,trace.childParentCarrierRows.size(),result.childParentRows.size()+1u)||
       !validDigest(trace.sourceCarrierPacketSHA256)||!validDigest(trace.sourceCarrierArraySHA256)||
       !validDigest(trace.receiptSHA256)||!validDigest(result.receiptSHA256)||
       trace.expandedSolverReceiptSHA256!=provenance.feasibilitySourceSHA256||
       in.materialDensityLowerKgM3.size()<=16u||in.materialDensityUpperKgM3.size()<=16u)
        return fail("LP CSV exporter received incomplete or malformed source-only feasibility results");
    for(std::size_t p=0;p<np;++p) {
        if(!std::isfinite(in.parentMassKg[p])||in.parentMassKg[p]<=0.0||
           !std::isfinite(in.parentReferenceVolumeM3[p])||in.parentReferenceVolumeM3[p]<=0.0)
            return fail("LP CSV exporter received invalid parent mass or measure");
    }
    for(std::size_t c=0;c<nc;++c) {
        if(!std::isfinite(in.childReferenceVolumeM3[c])||in.childReferenceVolumeM3[c]<=0.0||
           !std::isfinite(result.childMassKg[c])||result.childMassKg[c]<=0.0||
           !std::isfinite(result.childImpliedDensityKgM3[c])||result.childImpliedDensityKgM3[c]<=0.0)
            return fail("LP CSV exporter received invalid child mass, measure, or density");
    }
    for(std::size_t e=0;e<result.childParentRows.size();++e) {
        if(result.childParentRows[e]>=np||!std::isfinite(result.childParentTransportMassKg[e])||
           result.childParentTransportMassKg[e]<=0.0||!std::isfinite(result.childParentTransportMeasureM3[e])||
           result.childParentTransportMeasureM3[e]<=0.0||!std::isfinite(result.childParentTransportDistanceM[e])||
           result.childParentTransportDistanceM[e]<0.0)
            return fail("LP CSV exporter received an invalid aggregate parent-child edge");
    }
    const auto carrierCount=static_cast<std::size_t>(std::count_if(
        trace.expandedParentToCarrierRows.begin(),trace.expandedParentToCarrierRows.end(),
        [](std::uint32_t row){return row!=kSourceCarrierLPInvalidRow;}));
    std::vector<bool> seenCarrier(carrierCount,false);
    std::set<std::uint32_t> carrierizedParents;
    for(std::size_t v=0;v<trace.expandedParentToSourceParentRows.size();++v) {
        const auto sourceParent=trace.expandedParentToSourceParentRows[v];
        const auto carrier=trace.expandedParentToCarrierRows[v];
        if(sourceParent>=np||(carrier!=kSourceCarrierLPInvalidRow&&carrier>=carrierCount))
            return fail("LP CSV exporter received an invalid expanded source/carrier row map");
        if(carrier!=kSourceCarrierLPInvalidRow) {
            if(seenCarrier[carrier])return fail("LP CSV exporter received duplicate expanded carrier-row mapping");
            seenCarrier[carrier]=true;carrierizedParents.insert(sourceParent);
        }
    }
    if(std::find(seenCarrier.begin(),seenCarrier.end(),false)!=seenCarrier.end()||
       carrierizedParents.size()!=trace.carrierSourceParentRows.size())
        return fail("LP CSV exporter received incomplete carrier-row or carrierized-parent mapping");
    for(auto p:trace.carrierSourceParentRows)
        if(p>=np||!carrierizedParents.contains(p))
            return fail("LP CSV exporter received invalid carrierized source-parent rows");
    for(std::size_t k=0;k<trace.childParentCarrierRows.size();++k) {
        if(trace.childParentCarrierRows[k]>=trace.expandedParentToSourceParentRows.size()||
           !std::isfinite(trace.childParentCarrierMassKg[k])||trace.childParentCarrierMassKg[k]<=0.0||
           !std::isfinite(trace.childParentCarrierMeasureM3[k])||trace.childParentCarrierMeasureM3[k]<=0.0||
           !std::isfinite(trace.childParentCarrierDistanceM[k])||trace.childParentCarrierDistanceM[k]<0.0)
            return fail("LP CSV exporter received an invalid carrier-level flow record");
    }
    const auto closeWithin=[&](double a,double b,double absoluteTolerance) {
        return std::isfinite(a)&&std::isfinite(b)&&
            std::abs(a-b)<=std::max(absoluteTolerance,in.relativeClosureTolerance*
                std::max(std::abs(a),std::abs(b)));
    };
    std::vector<spatial_mass_transport_detail::ExactDoubleSum> parentMassSums(np),parentMeasureSums(np);
    for(std::size_t c=0;c<nc;++c) {
        spatial_mass_transport_detail::ExactDoubleSum childMassSum,childMeasureSum;
        for(std::uint32_t e=result.childParentOffsets[c];e<result.childParentOffsets[c+1u];++e) {
            const auto parent=result.childParentRows[e];
            const auto begin=trace.childParentCarrierOffsets[e],end=trace.childParentCarrierOffsets[e+1u];
            if(begin==end)return fail("LP CSV exporter found an aggregate edge without carrier-flow rows");
            spatial_mass_transport_detail::ExactDoubleSum edgeMassSum,edgeMeasureSum;
            long double weightedDistance=0.0L;
            for(std::uint32_t k=begin;k<end;++k) {
                const auto expanded=trace.childParentCarrierRows[k];
                if(trace.expandedParentToSourceParentRows[expanded]!=parent)
                    return fail("LP CSV carrier row does not map to its aggregate source parent");
                edgeMassSum.add(trace.childParentCarrierMassKg[k]);
                edgeMeasureSum.add(trace.childParentCarrierMeasureM3[k]);
                weightedDistance+=static_cast<long double>(trace.childParentCarrierMassKg[k])*
                                  trace.childParentCarrierDistanceM[k];
            }
            const double edgeMass=edgeMassSum.rounded(),edgeMeasure=edgeMeasureSum.rounded();
            const double edgeDistance=static_cast<double>(weightedDistance/edgeMass);
            if(!closeWithin(edgeMass,result.childParentTransportMassKg[e],in.absoluteMassToleranceKg)||
               !closeWithin(edgeMeasure,result.childParentTransportMeasureM3[e],in.absoluteMeasureToleranceM3)||
               !nearRecorded(edgeDistance,result.childParentTransportDistanceM[e]))
                return fail("LP CSV carrier rows do not close to their aggregate parent-child edge");
            childMassSum.add(edgeMass);childMeasureSum.add(edgeMeasure);
            parentMassSums[parent].add(edgeMass);parentMeasureSums[parent].add(edgeMeasure);
            const double expectedFraction=edgeMeasure/result.childAncestryMeasureVolumeM3[c];
            if(!closeWithin(expectedFraction,result.childParentVolumeFractions[e],in.relativeClosureTolerance))
                return fail("LP CSV aggregate edge volume fraction does not close to the child ancestry measure");
        }
        if(!closeWithin(childMassSum.rounded(),result.childMassKg[c],in.absoluteMassToleranceKg)||
           !closeWithin(childMeasureSum.rounded(),result.childAncestryMeasureVolumeM3[c],in.absoluteMeasureToleranceM3))
            return fail("LP CSV aggregate edges do not close to child mass or ancestry measure");
    }
    for(std::size_t p=0;p<np;++p) {
        if(!closeWithin(parentMassSums[p].rounded(),result.parentMassTransferredKg[p],in.absoluteMassToleranceKg)||
           !closeWithin(parentMeasureSums[p].rounded(),result.parentMeasureTransferredM3[p],in.absoluteMeasureToleranceM3))
            return fail("LP CSV aggregate edges do not close to transferred parent mass or measure");
    }
    for(const auto& [name,sha]:metadata.inputSHA256) {
        const bool safeName=!name.empty()&&std::all_of(name.begin(),name.end(),[](unsigned char c){
            return (c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_'||c=='-'||c=='.';
        });
        if(!safeName||!validDigest(sha))return fail("LP receipt input pin is not a safe named SHA-256");
    }
    const std::array<const char*,5> requiredFilePins={
        "input_child_node_sha256","input_child_element_sha256","input_parent_mass_csv_sha256",
        "input_parent_label_csv_sha256","support_binary_sha256"};
    for(const auto* key:requiredFilePins)
        if(!metadata.inputSHA256.contains(key))
            return fail(std::string("LP receipt is missing required legacy-compatible input pin: ")+key);
    if(!validDigest(std::string(in.regionLabelSHA256))||
       !metadata.inputSHA256.contains("input_region_label_sha256")||
       metadata.inputSHA256.at("input_region_label_sha256")!=in.regionLabelSHA256)
        return fail("LP receipt region-label input pin does not match the authenticated transport input");
    const auto material=in.parentMaterialIndices.front();
    if(material!=15u&&material!=16u)return fail("LP CSV exporter role is outside Skin/Fat material indices");
    for(auto x:in.parentMaterialIndices)if(x!=material)return fail("LP CSV exporter parent rows mix material roles");
    for(auto x:in.childMaterialIndices)if(x!=material)return fail("LP CSV exporter child rows mix material roles");
    if(!std::filesystem::exists(outputDirectory)) {
        std::error_code ec;std::filesystem::create_directories(outputDirectory,ec);
        if(ec)return fail("cannot create LP CSV output directory");
    } else if(!std::filesystem::is_directory(outputDirectory)||!std::filesystem::is_empty(outputDirectory)) {
        return fail("LP CSV output directory already exists and is not empty");
    }
    const auto makeCSV=[&](const char* name)->std::ofstream {
        std::ofstream f(outputDirectory/name,std::ios::binary|std::ios::trunc);
        f<<std::setprecision(17);return f;
    };
    {
        auto f=makeCSV("transport-child-mass.csv");
        if(!f)return fail("cannot open child mass CSV");
        f<<"child_row,reference_volume_m3,mass_kg,density_kg_m3\n";
        for(std::size_t c=0;c<nc;++c)
            f<<c<<","<<in.childReferenceVolumeM3[c]<<","<<result.childMassKg[c]<<","<<result.childImpliedDensityKgM3[c]<<"\n";
        f.close();if(!f)return fail("failed writing child mass CSV");
    }
    {
        auto f=makeCSV("transport-flow.csv");
        if(!f)return fail("cannot open aggregate transport CSV");
        f<<"child_row,parent_row,mass_kg,source_measure_m3,distance_m\n";
        for(std::size_t c=0;c<nc;++c)
            for(std::uint32_t e=result.childParentOffsets[c];e<result.childParentOffsets[c+1u];++e)
                f<<c<<","<<result.childParentRows[e]<<","<<result.childParentTransportMassKg[e]<<","
                 <<result.childParentTransportMeasureM3[e]<<","<<result.childParentTransportDistanceM[e]<<"\n";
        f.close();if(!f)return fail("failed writing aggregate transport CSV");
    }
    {
        auto f=makeCSV("source-carrier-flow.csv");
        if(!f)return fail("cannot open source carrier flow CSV");
        f<<"child_row,parent_row,expanded_parent_row,carrier_row,mass_kg,source_measure_m3,distance_m\n";
        for(std::size_t c=0;c<nc;++c)
            for(std::uint32_t e=result.childParentOffsets[c];e<result.childParentOffsets[c+1u];++e)
                for(std::uint32_t k=trace.childParentCarrierOffsets[e];k<trace.childParentCarrierOffsets[e+1u];++k) {
                    const auto v=trace.childParentCarrierRows[k];
                    if(v>=trace.expandedParentToSourceParentRows.size())
                        return fail("carrier-flow trace references an invalid expanded parent row");
                    f<<c<<","<<trace.expandedParentToSourceParentRows[v]<<","<<v<<","
                     <<trace.expandedParentToCarrierRows[v]<<","<<trace.childParentCarrierMassKg[k]<<","
                     <<trace.childParentCarrierMeasureM3[k]<<","<<trace.childParentCarrierDistanceM[k]<<"\n";
                }
        f.close();if(!f)return fail("failed writing source carrier flow CSV");
    }
    {
        auto f=makeCSV("transport-child-source-donor.csv");
        if(!f)return fail("cannot open child donor CSV");
        f<<"child_row,source_donor_body,mass_kg\n";
        for(std::size_t c=0;c<nc;++c)
            for(std::uint32_t q=result.childSourceDonorOffsets[c];q<result.childSourceDonorOffsets[c+1u];++q)
                f<<c<<","<<result.childSourceDonorBodyIndices[q]<<","<<result.childSourceDonorMassKg[q]<<"\n";
        f.close();if(!f)return fail("failed writing child donor CSV");
    }
    {
        auto f=makeCSV("transport-source-parent.csv");
        if(!f)return fail("cannot open source parent CSV");
        f<<"parent_row,global_source330_row,material,geometric_donor,region,stable_source_face,mass_kg,reference_volume_m3,source_centroid_x_m,source_centroid_y_m,source_centroid_z_m\n";
        for(std::size_t p=0;p<np;++p)
            f<<p<<","<<globalSourceTetRows[p]<<","<<in.parentMaterialIndices[p]<<","
             <<in.parentDonorBodyIndices[p]<<","<<in.parentAnatomicalRegionIndices[p]<<","
             <<support.parentStableSourceFaceIdentities[p]<<","<<in.parentMassKg[p]<<","
             <<in.parentReferenceVolumeM3[p]<<","<<in.parentCentroidsWorld[p][0]<<","
             <<in.parentCentroidsWorld[p][1]<<","<<in.parentCentroidsWorld[p][2]<<"\n";
        f.close();if(!f)return fail("failed writing source parent CSV");
    }
    {
        auto f=makeCSV("transport-source-parent-donor.csv");
        if(!f)return fail("cannot open source parent donor CSV");
        f<<"parent_row,source_donor_body,mass_kg\n";
        if(in.parentSourceDonorOffsets.size()!=np+1u||
           in.parentSourceDonorOffsets.back()!=in.parentSourceDonorBodyIndices.size()||
           in.parentSourceDonorBodyIndices.size()!=in.parentSourceDonorMassKg.size())
            return fail("source parent donor input CSR is malformed");
        for(std::size_t p=0;p<np;++p)
            for(std::uint32_t q=in.parentSourceDonorOffsets[p];q<in.parentSourceDonorOffsets[p+1u];++q)
                f<<p<<","<<in.parentSourceDonorBodyIndices[q]<<","<<in.parentSourceDonorMassKg[q]<<"\n";
        f.close();if(!f)return fail("failed writing source parent donor CSV");
    }

    const std::array<const char*,6> names={"transport-child-mass.csv","transport-flow.csv",
        "source-carrier-flow.csv","transport-child-source-donor.csv","transport-source-parent.csv",
        "transport-source-parent-donor.csv"};
    std::array<std::string,6> hashes{};
    std::array<std::uintmax_t,6> sizes{};
    for(std::size_t i=0;i<names.size();++i) {
        const auto path=outputDirectory/names[i];
        if(!source_carrier_lp_detail::sha256File(path,hashes[i],error))
            return fail("cannot hash LP CSV output: "+error);
        std::error_code ec;sizes[i]=std::filesystem::file_size(path,ec);
        if(ec)return fail("cannot stat LP CSV output");
    }
    const auto receiptPath=outputDirectory/"result.json";
    std::ofstream out(receiptPath,std::ios::binary|std::ios::trunc);
    if(!out)return fail("cannot create LP finalization receipt");
    out<<std::setprecision(17);
    const std::string role=material==15u?"SKIN":"FAT";
    out<<"{\n"
       <<"  \"schema\": \"numi.human.source-carrier-lp-feasibility-allocation.v1\",\n"
       <<"  \"status\": \"PASS_SOURCE_ONLY_"<<role<<"_SOURCE_CARRIER_FEASIBLE_ALLOCATION\",\n"
       <<"  \"complete_allocation\": true,\n"
       <<"  \"physical_admission\": false,\n"
       <<"  \"native_steps\": 0,\n"
       <<"  \"recorded_flow_feasibility\": \"PASS_SOLVER_FREE\",\n"
       <<"  \"deterministic_optimizer_replay\": \"NOT_RUN\",\n"
       <<"  \"allocation_construction\": "<<std::quoted(metadata.method)<<",\n"
       <<"  \"feasibility_source_sha256\": "<<std::quoted(metadata.feasibilitySourceSHA256)<<",\n"
       <<"  \"held_partial_manifest_sha256\": "<<std::quoted(provenance.heldPartialManifestSHA256)<<",\n"
       <<"  \"held_carrier_flow_csv_sha256\": "<<std::quoted(provenance.heldCarrierFlowCSV_SHA256)<<",\n"
       <<"  \"lp_proposal_sha256\": "<<std::quoted(provenance.lpProposalSHA256)<<",\n"
       <<"  \"lp_input_manifest_sha256\": "<<std::quoted(provenance.lpInputManifestSHA256)<<",\n"
       <<"  \"allocation_solver\": \"held_dinic_partial_plus_sparse_lp_positive_carrier_flow\",\n"
       <<"  \"cost_optimality_certified\": false,\n"
       <<"  \"held_dinic_augmentation_count\": "<<provenance.heldDinicAugmentationCount<<",\n"
       <<"  \"lp_positive_variable_count\": "<<provenance.lpPositiveVariableCount<<",\n"
       <<"  \"held_positive_flow_rows\": "<<provenance.heldPositiveFlowRows<<",\n"
       <<"  \"held_candidate_edges\": "<<provenance.heldCandidateEdgeCount<<",\n"
       <<"  \"lp_candidate_edges\": "<<provenance.lpCandidateEdgeCount<<",\n"
       <<"  \"semantic_candidate_edge_count_basis\": \"pinned held+LP proposal graph count; feasibility does not reconstruct optimization graph\",\n"
       <<"  \"maximum_allowed_candidate_edges\": "<<in.maximumAllowedEdges<<",\n"
       <<"  \"parent_"<<(material==15u?"skin":"fat")<<"_tets\": "<<np<<",\n"
       <<"  \"child_"<<(material==15u?"skin":"fat")<<"_tets\": "<<nc<<",\n"
       <<"  \"source_carrier_packet_sha256\": "<<std::quoted(trace.sourceCarrierPacketSHA256)<<",\n"
       <<"  \"source_carrier_array_sha256\": "<<std::quoted(trace.sourceCarrierArraySHA256)<<",\n"
       <<"  \"source_carrier_flow_trace_sha256\": "<<std::quoted(trace.receiptSHA256)<<",\n"
       <<"  \"source_carrierized_parents\": "<<trace.carrierSourceParentRows.size()<<",\n"
       <<"  \"source_carriers\": "<<carrierCount<<",\n"
       <<"  \"transport_receipt_sha256\": "<<std::quoted(result.receiptSHA256)<<",\n"
       <<"  \"transport_edges\": "<<result.transportEdgeCount<<",\n"
       <<"  \"candidate_edges\": "<<result.semanticCandidateEdgeCount<<",\n"
       <<"  \"augmentations\": "<<provenance.heldDinicAugmentationCount<<",\n"
       <<"  \"augmentation_count_basis\": \"held partial Dinic subflow only; LP construction is counted separately\",\n"
       <<"  \"maximum_transport_distance_allowed_m\": "<<in.maximumTransportDistanceM<<",\n"
       <<"  \"maximum_transport_distance_used_m\": "<<result.maximumTransportDistanceUsedM<<",\n"
       <<"  \"mass_weighted_mean_transport_distance_m\": "<<result.massWeightedMeanTransportDistanceM<<",\n"
       <<"  \"minimum_child_density_kg_m3\": "<<result.minimumChildImpliedDensityKgM3<<",\n"
       <<"  \"maximum_child_density_kg_m3\": "<<result.maximumChildImpliedDensityKgM3<<",\n"
       <<"  \"density_bounds_kg_m3\": ["<<in.materialDensityLowerKgM3[material]<<","<<in.materialDensityUpperKgM3[material]<<"],\n"
       <<"  \"maximum_parent_mass_closure_error_kg\": "<<result.maximumParentMassClosureErrorKg<<",\n"
       <<"  \"maximum_parent_measure_closure_error_m3\": "<<result.maximumParentMeasureClosureErrorM3<<",\n"
       <<"  \"maximum_source_donor_closure_error_kg\": "<<result.maximumSourceDonorMassClosureErrorKg<<",\n"
       <<"  \"maximum_child_fraction_sum_error\": "<<result.maximumChildFractionSumError<<",\n"
       <<"  \"source_mass_kg\": ";
    spatial_mass_transport_detail::ExactDoubleSum parentMassSum,childVolumeSum;
    for(double x:in.parentMassKg)parentMassSum.add(x);
    for(double x:in.childReferenceVolumeM3)childVolumeSum.add(x);
    out<<parentMassSum.rounded()<<",\n  \"child_reference_volume_m3\": "<<childVolumeSum.rounded()<<",\n";
    out<<"  \"child_mesh_sha256\": "<<std::quoted(in.childMeshSHA256)<<",\n"
       <<"  \"input_child_node_sha256\": "<<std::quoted(metadata.inputSHA256.at("input_child_node_sha256"))<<",\n"
       <<"  \"input_child_element_sha256\": "<<std::quoted(metadata.inputSHA256.at("input_child_element_sha256"))<<",\n"
       <<"  \"input_parent_mass_csv_sha256\": "<<std::quoted(metadata.inputSHA256.at("input_parent_mass_csv_sha256"))<<",\n"
       <<"  \"input_parent_label_csv_sha256\": "<<std::quoted(metadata.inputSHA256.at("input_parent_label_csv_sha256"))<<",\n"
       <<"  \"support_binary_sha256\": "<<std::quoted(metadata.inputSHA256.at("support_binary_sha256"))<<",\n"
       <<"  \"input_region_label_sha256\": "<<std::quoted(in.regionLabelSHA256)<<",\n"
       <<"  \"input_region_label_sha256\": "<<std::quoted(in.regionLabelSHA256)<<",\n"
       <<"  \"semantic_support_sha256\": "<<std::quoted(support.typedSupportSHA256)<<",\n"
       <<"  \"transport_receipt_sha256\": "<<std::quoted(result.receiptSHA256)<<",\n"
       <<"  \"admissibility_source_support_sha256\": "<<std::quoted(support.admissibilitySourceSupportSHA256)<<",\n"
       <<"  \"admissibility_max_source_region_hops\": "<<support.admissibilityMaximumSourceRegionHops<<",\n"
       <<"  \"admissible_region_slots\": "<<support.cellAdmissibleRegionIds.size()<<",\n"
       <<"  \"input_parent_mass_sha256\": "<<std::quoted(in.parentMassPacketSHA256)<<",\n"
       <<"  \"input_parent_source_sha256\": "<<std::quoted(in.parentSourceContributionSHA256)<<",\n"
       <<"  \"input_coordinate_frame_sha256\": "<<std::quoted(in.coordinateFrameSHA256)<<",\n";
    out<<"  \"input_sha256\": {";
    std::size_t pinIndex=0;
    for(const auto& [name,sha]:metadata.inputSHA256) {
        if(!validDigest(sha)){out.close();return fail("LP receipt input pin is not SHA-256");}
        if(pinIndex++)out<<",";
        out<<"\n    "<<std::quoted(name)<<": "<<std::quoted(sha);
    }
    out<<"\n  },\n  \"flow_csv\": {";
    for(std::size_t i=0;i<names.size();++i) {
        if(i)out<<",";
        out<<"\n    "<<std::quoted(names[i])<<": {\"sha256\": "<<std::quoted(hashes[i])
           <<", \"size_bytes\": "<<sizes[i]<<"}";
    }
    out<<"\n  },\n  \"group_closure\": [";
    for(std::size_t i=0;i<result.groupClosure.size();++i) {
        const auto& g=result.groupClosure[i];if(i)out<<",";
        out<<"{\"material\":"<<g.materialIndex<<",\"geometric_donor\":"<<g.donorBodyIndex
           <<",\"region\":"<<g.anatomicalRegionIndex<<",\"parent_mass_kg\":"<<g.parentMassKg
           <<",\"child_mass_kg\":"<<g.childMassKg<<",\"reachable_capacity_kg\":"<<g.childMaximumCapacityKg<<"}";
    }
    out<<"],\n  \"source_donor_closure\": [";
    for(std::size_t i=0;i<result.sourceDonorClosure.size();++i) {
        const auto& g=result.sourceDonorClosure[i];if(i)out<<",";
        out<<"{\"material\":"<<g.materialIndex<<",\"source_donor\":"<<g.sourceDonorBodyIndex
           <<",\"parent_mass_kg\":"<<g.parentMassKg<<",\"child_mass_kg\":"<<g.childMassKg<<"}";
    }
    out<<"]\n}\n";
    out.close();if(!out)return fail("failed writing LP finalization receipt");
    error.clear();return true;
}


} // namespace source_carrier_lp_detail
} // namespace metalrobo::human::coarse_tissue

