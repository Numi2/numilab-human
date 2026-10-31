#pragma once

#include "ConservativeSpatialFractionalSemanticSupport.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <span>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace metalrobo::human::coarse_tissue {

// Optional 090 source-tetra carrier quadrature. The legacy entry point remains
// unchanged when this API is not selected. A nonempty row is an explicitly
// witnessed source-cell quadrature: every carrier has a pinned admissible child
// and remains under the original transport radius, while positive weights
// preserve the source cell's first moment, mass, measure, and donor split.
// Barycentric coordinates are in the ordered source parent tet node domain;
// positive weights partition, rather than replace, the original cell mass,
// source measure, and source-donor contribution rows.
struct SourceCellCarrierQuadratureInput {
    std::span<const std::uint32_t> parentOffsets; // nParent + 1; empty rows use equal offsets
    std::span<const std::uint32_t> globalSourceTetRows; // one original source330 tet row per parent input row
    std::span<const std::array<std::uint32_t,4>> sourceTetNodeRows; // parent input row -> ordered source330 node rows
    std::span<const std::array<std::array<double,3>,4>> parentReferenceTetWorldVertices;
    std::span<const double> carrierWeights;
    std::span<const std::array<double,4>> carrierBarycentric;
    std::span<const std::uint32_t> witnessChildRows;
    std::span<const double> witnessChildCentroidDistancesM;
    std::string_view sourceCarrierPacketSHA256;
    std::string_view sourceTetTableSHA256;
    std::string_view sourceNodeTableSHA256;
};

// Exact carrier-level flows remain available beside the existing aggregate
// parent/child CSR. This preserves the original source parent IDs at the public
// boundary while making every virtual carrier allocation auditable.
struct SourceCellCarrierTransportTrace {
    std::string sourceCarrierPacketSHA256;
    std::string sourceCarrierArraySHA256;
    std::string expandedSolverReceiptSHA256;
    std::vector<std::uint32_t> expandedParentToSourceParentRows;
    std::vector<std::uint32_t> expandedParentToCarrierRows; // kInvalid means unchanged centroid route
    std::vector<std::uint32_t> carrierSourceParentRows;
    std::vector<double> originalCentroidNearestAdmissibleDistanceM;
    // CSR aligned with the aggregate result's childParentRows edges.
    std::vector<std::uint32_t> childParentCarrierOffsets;
    std::vector<std::uint32_t> childParentCarrierRows; // expanded solver parent rows
    std::vector<double> childParentCarrierMassKg;
    std::vector<double> childParentCarrierMeasureM3;
    std::vector<double> childParentCarrierDistanceM;
    std::string receiptSHA256;
};

namespace source_carrier_transport_detail {

using Point = SpatialMassTransportPoint;
using TetVertices = std::array<Point,4>;
using Bary = std::array<double,4>;
constexpr std::uint32_t kInvalidCarrierRow = std::numeric_limits<std::uint32_t>::max();

inline Point barycentricPoint(const TetVertices& tet, const Bary& bary) {
    Point point{};
    for (std::size_t node=0; node<4u; ++node)
        for (std::size_t axis=0; axis<3u; ++axis)
            point[axis] += bary[node]*tet[node][axis];
    return point;
}

inline double pointDistance(const Point& a, const Point& b) {
    const double dx=a[0]-b[0], dy=a[1]-b[1], dz=a[2]-b[2];
    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

inline std::string carrierArrayDigest(const ConservativeSpatialMassTransportInput& in,
                                      const SourceCellCarrierQuadratureInput& carrier) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("numi.human.source-tetra-centroid-carrier-quadrature.arrays.v1");
    hash.string(carrier.sourceCarrierPacketSHA256);
    hash.string(carrier.sourceTetTableSHA256);
    hash.string(carrier.sourceNodeTableSHA256);
    hash.u64(carrier.parentOffsets.size());
    for (const auto value : carrier.parentOffsets) hash.u32(value);
    hash.u64(carrier.globalSourceTetRows.size());
    for (const auto value : carrier.globalSourceTetRows) hash.u32(value);
    for (const auto& tet : carrier.sourceTetNodeRows)
        for (const auto value : tet) hash.u32(value);
    for (const auto& tet : carrier.parentReferenceTetWorldVertices)
        for (const auto& point : tet)
            for (const auto value : point) hash.real(value);
    hash.u64(carrier.carrierWeights.size());
    for (const auto value : carrier.carrierWeights) hash.real(value);
    for (const auto& bary : carrier.carrierBarycentric)
        for (const auto value : bary) hash.real(value);
    hash.u64(carrier.witnessChildRows.size());
    for (const auto value : carrier.witnessChildRows) hash.u32(value);
    for (const auto value : carrier.witnessChildCentroidDistancesM) hash.real(value);
    hash.u64(in.parentMassKg.size());
    hash.string(in.parentMassPacketSHA256);
    hash.string(in.parentSourceContributionSHA256);
    hash.string(in.childMeshSHA256);
    hash.string(in.regionLabelSHA256);
    hash.string(in.coordinateFrameSHA256);
    return hash.finish();
}

inline std::string combinedSupportDigest(const ConservativeSpatialChildSemanticSupport& support,
                                         const std::string_view carrierDigest,
                                         const std::string_view expandedSupportDigest) {
    spatial_mass_transport_detail::SHA256Builder hash;
    hash.string("conservativeSpatialMassTransport.sourceCarrierQuadratureSupport.v1");
    hash.string(fractional_semantic_transport_detail::supportDigest(support));
    hash.string(carrierDigest);
    hash.string(expandedSupportDigest);
    return hash.finish();
}

inline bool validateCarrierInput(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        std::vector<std::uint32_t>& parentGroup,
        std::vector<std::map<std::uint32_t,double>>& regionPreference,
        std::vector<std::vector<std::uint32_t>>& admissibleRegions,
        std::vector<Point>& carrierWorldPoints,
        std::vector<std::uint32_t>& carrierRows,
        std::vector<double>& strictCentroidNearest,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=in.parentMassKg.size(), nc=in.childReferenceVolumeM3.size();
    if (!fractional_semantic_transport_detail::validateSharedCapacitySupport(
            in,support,parentGroup,regionPreference,admissibleRegions,error)) return false;
    if (carrier.parentOffsets.size()!=np+1u || carrier.parentOffsets.empty() ||
        carrier.parentOffsets.front()!=0u ||
        carrier.parentOffsets.back()!=carrier.carrierWeights.size() ||
        carrier.carrierBarycentric.size()!=carrier.carrierWeights.size() ||
        carrier.witnessChildRows.size()!=carrier.carrierWeights.size() ||
        carrier.witnessChildCentroidDistancesM.size()!=carrier.carrierWeights.size() ||
        carrier.globalSourceTetRows.size()!=np || carrier.sourceTetNodeRows.size()!=np ||
        carrier.parentReferenceTetWorldVertices.size()!=np ||
        !validSHA256(carrier.sourceCarrierPacketSHA256) ||
        !validSHA256(carrier.sourceTetTableSHA256) || !validSHA256(carrier.sourceNodeTableSHA256))
        return fail("source carrier packet arrays, source geometry maps or SHA-256 pins are incomplete");
    for (std::size_t row=1u; row<carrier.parentOffsets.size(); ++row)
        if (carrier.parentOffsets[row]<carrier.parentOffsets[row-1u] ||
            carrier.parentOffsets[row]>carrier.carrierWeights.size())
            return fail("source carrier parent CSR offsets are reversed or out of range");
    std::set<std::uint32_t> uniqueTetRows;
    for (std::size_t parent=0; parent<np; ++parent) {
        if (!uniqueTetRows.insert(carrier.globalSourceTetRows[parent]).second)
            return fail("source carrier source330 tet-row map is not one-to-one");
        std::set<std::uint32_t> nodes;
        Point centroid{};
        for (std::size_t node=0; node<4u; ++node) {
            if (!nodes.insert(carrier.sourceTetNodeRows[parent][node]).second)
                return fail("source carrier source tetrahedron repeats a source node row");
            for (std::size_t axis=0; axis<3u; ++axis) {
                const double x=carrier.parentReferenceTetWorldVertices[parent][node][axis];
                if (!std::isfinite(x)) return fail("source carrier source tetrahedron has a nonfinite reference coordinate");
                centroid[axis]+=0.25*x;
            }
        }
        if (pointDistance(centroid,in.parentCentroidsWorld[parent])>1.0e-11)
            return fail("source carrier source-tet centroid does not equal the authenticated parent reference centroid");
        const auto begin=carrier.parentOffsets[parent], end=carrier.parentOffsets[parent+1u];
        if (begin==end) continue;
        if (in.parentMaterialIndices[parent]==kInvalidCarrierRow ||
            in.parentDonorBodyIndices[parent]==kInvalidCarrierRow ||
            in.parentAnatomicalRegionIndices[parent]==kInvalidCarrierRow)
            return fail("source carrier parent has invalid material/donor/region labels");
        carrierRows.push_back(static_cast<std::uint32_t>(parent));
        double weightSum=0.0;
        Bary baryMoment{};
        Point worldMoment{};
        double strictNearest=std::numeric_limits<double>::infinity();
        const auto material=in.parentMaterialIndices[parent];
        const auto region=in.parentAnatomicalRegionIndices[parent];
        for (std::size_t child=0; child<nc; ++child) {
            if (in.childMaterialIndices[child]!=material ||
                !std::binary_search(admissibleRegions[child].begin(),admissibleRegions[child].end(),region))
                continue;
            strictNearest=std::min(strictNearest,pointDistance(in.parentCentroidsWorld[parent],
                                                               in.childCentroidsWorld[child]));
        }
        strictCentroidNearest.push_back(strictNearest);
        for (std::uint32_t row=begin; row<end; ++row) {
            const double weight=carrier.carrierWeights[row];
            const auto& bary=carrier.carrierBarycentric[row];
            if (!std::isfinite(weight) || weight<=0.0 ||
                carrier.witnessChildRows[row]>=nc ||
                !std::isfinite(carrier.witnessChildCentroidDistancesM[row]))
                return fail("source carrier weight or child witness row is invalid");
            double barySum=0.0;
            for (std::size_t node=0; node<4u; ++node) {
                if (!std::isfinite(bary[node]) || bary[node]<-1.0e-13 || bary[node]>1.0+1.0e-13)
                    return fail("source carrier barycentric coordinate lies outside its source tetrahedron");
                barySum+=bary[node];
                baryMoment[node]+=weight*bary[node];
            }
            if (std::abs(barySum-1.0)>2.0e-12)
                return fail("source carrier barycentric coordinates do not sum to one");
            const auto point=barycentricPoint(carrier.parentReferenceTetWorldVertices[parent],bary);
            const auto child=carrier.witnessChildRows[row];
            if (in.childMaterialIndices[child]!=material ||
                !std::binary_search(admissibleRegions[child].begin(),admissibleRegions[child].end(),region))
                return fail("source carrier witness child has the wrong material or is not semantically admissible for the source region");
            const double distance=pointDistance(point,in.childCentroidsWorld[child]);
            const double tolerance=std::max(1.0e-12,64.0*std::numeric_limits<double>::epsilon()*
                std::max({1.0,distance,carrier.witnessChildCentroidDistancesM[row]}));
            if (std::abs(distance-carrier.witnessChildCentroidDistancesM[row])>tolerance ||
                distance>in.maximumTransportDistanceM+tolerance)
                return fail("source carrier witness distance does not recompute within the unchanged centroid bound");
            weightSum+=weight;
            for (std::size_t axis=0; axis<3u; ++axis) worldMoment[axis]+=weight*point[axis];
            carrierWorldPoints.push_back(point);
        }
        if (std::abs(weightSum-1.0)>2.0e-12)
            return fail("source carrier weights do not conserve the original parent mass and reference measure");
        for (std::size_t node=0; node<4u; ++node)
            if (std::abs(baryMoment[node]-0.25)>2.0e-10)
                return fail("source carrier weighted barycentric moment does not preserve the parent centroid");
        if (pointDistance(worldMoment,in.parentCentroidsWorld[parent])>1.0e-11)
            return fail("source carrier weighted world moment does not preserve the parent centroid");
    }
    if (carrierRows.empty() || carrier.carrierWeights.empty())
        return fail("source carrier packet is present but contains no explicitly witnessed quadrature source rows");
    error.clear(); return true;
}

inline bool expandParentRows(
        const ConservativeSpatialMassTransportInput& in,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        std::vector<std::uint32_t>& virtualToParent,
        std::vector<std::uint32_t>& virtualToCarrier,
        std::vector<std::vector<std::uint32_t>>& parentToVirtual,
        std::vector<Point>& centroids,
        std::vector<double>& volumes,
        std::vector<double>& masses,
        std::vector<std::uint32_t>& materials,
        std::vector<std::uint32_t>& donors,
        std::vector<std::uint32_t>& regions,
        std::vector<std::uint32_t>& donorOffsets,
        std::vector<std::uint32_t>& donorIDs,
        std::vector<double>& donorMass,
        ConservativeSpatialMassTransportInput& expandedInput,
        ConservativeSpatialChildSemanticSupportData& expandedSupport,
        const std::string_view carrierArraySHA,
        std::string& error) {
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=in.parentMassKg.size();
    constexpr std::uint32_t invalid=std::numeric_limits<std::uint32_t>::max();
    for (std::size_t parent=0; parent<np; ++parent) {
        const auto begin=carrier.parentOffsets[parent],end=carrier.parentOffsets[parent+1u];
        if (begin==end) {
            virtualToParent.push_back(static_cast<std::uint32_t>(parent));
            virtualToCarrier.push_back(invalid);
            parentToVirtual[parent].push_back(static_cast<std::uint32_t>(virtualToParent.size()-1u));
            centroids.push_back(in.parentCentroidsWorld[parent]);
            volumes.push_back(in.parentReferenceVolumeM3[parent]); masses.push_back(in.parentMassKg[parent]);
            materials.push_back(in.parentMaterialIndices[parent]); donors.push_back(in.parentDonorBodyIndices[parent]);
            regions.push_back(in.parentAnatomicalRegionIndices[parent]);
            const auto sb=in.parentSourceDonorOffsets[parent],se=in.parentSourceDonorOffsets[parent+1u];
            for (std::uint32_t r=sb;r<se;++r) { donorIDs.push_back(in.parentSourceDonorBodyIndices[r]); donorMass.push_back(in.parentSourceDonorMassKg[r]); }
            donorOffsets.push_back(static_cast<std::uint32_t>(donorIDs.size()));
            continue;
        }
        for (std::uint32_t carrierRow=begin; carrierRow<end; ++carrierRow) {
            const double weight=carrier.carrierWeights[carrierRow];
            const Bary& bary=carrier.carrierBarycentric[carrierRow];
            virtualToParent.push_back(static_cast<std::uint32_t>(parent));
            virtualToCarrier.push_back(carrierRow);
            parentToVirtual[parent].push_back(static_cast<std::uint32_t>(virtualToParent.size()-1u));
            centroids.push_back(barycentricPoint(carrier.parentReferenceTetWorldVertices[parent],bary));
            volumes.push_back(in.parentReferenceVolumeM3[parent]*weight);
            masses.push_back(in.parentMassKg[parent]*weight);
            materials.push_back(in.parentMaterialIndices[parent]); donors.push_back(in.parentDonorBodyIndices[parent]);
            regions.push_back(in.parentAnatomicalRegionIndices[parent]);
            const auto sb=in.parentSourceDonorOffsets[parent],se=in.parentSourceDonorOffsets[parent+1u];
            double priorMass=0.0;
            for (std::uint32_t r=sb;r<se;++r) {
                const double scaled=(r+1u==se)?(masses.back()-priorMass):in.parentSourceDonorMassKg[r]*weight;
                if (!std::isfinite(scaled) || scaled<=0.0)
                    return fail("source carrier scaling produced a nonpositive donor contribution");
                donorIDs.push_back(in.parentSourceDonorBodyIndices[r]); donorMass.push_back(scaled); priorMass+=scaled;
            }
            donorOffsets.push_back(static_cast<std::uint32_t>(donorIDs.size()));
        }
    }
    if (virtualToParent.empty() || virtualToParent.size()>invalid)
        return fail("expanded source carrier row count exceeds the bounded index range");
    expandedInput=in;
    expandedInput.parentCentroidsWorld=centroids;
    expandedInput.parentReferenceVolumeM3=volumes;
    expandedInput.parentMassKg=masses;
    expandedInput.parentMaterialIndices=materials;
    expandedInput.parentDonorBodyIndices=donors;
    expandedInput.parentAnatomicalRegionIndices=regions;
    expandedInput.parentSourceDonorOffsets=donorOffsets;
    expandedInput.parentSourceDonorBodyIndices=donorIDs;
    expandedInput.parentSourceDonorMassKg=donorMass;

    expandedSupport.childOffsets.assign(support.childOffsets.begin(),support.childOffsets.end());
    expandedSupport.slotGroupIndices.assign(support.slotGroupIndices.begin(),support.slotGroupIndices.end());
    expandedSupport.volumeFractions.assign(support.volumeFractions.begin(),support.volumeFractions.end());
    expandedSupport.groupMaterialIndices.assign(support.groupMaterialIndices.begin(),support.groupMaterialIndices.end());
    expandedSupport.donorBodyIndices.assign(support.donorBodyIndices.begin(),support.donorBodyIndices.end());
    expandedSupport.anatomicalRegionIndices.assign(support.anatomicalRegionIndices.begin(),support.anatomicalRegionIndices.end());
    expandedSupport.derivedGroupIdentityIDs.assign(support.derivedGroupIdentityIDs.begin(),support.derivedGroupIdentityIDs.end());
    expandedSupport.sourceStableFaceOffsets.assign(support.sourceStableFaceOffsets.begin(),support.sourceStableFaceOffsets.end());
    expandedSupport.sourceStableFaceRows.assign(support.sourceStableFaceRows.begin(),support.sourceStableFaceRows.end());
    expandedSupport.typedSupportSHA256.assign(support.typedSupportSHA256);
    expandedSupport.cellAdmissibleRegionOffsets.assign(support.cellAdmissibleRegionOffsets.begin(),support.cellAdmissibleRegionOffsets.end());
    expandedSupport.cellAdmissibleRegionIds.assign(support.cellAdmissibleRegionIds.begin(),support.cellAdmissibleRegionIds.end());
    expandedSupport.admissibilitySourceSupportSHA256.assign(support.admissibilitySourceSupportSHA256);
    expandedSupport.admissibilityMaximumSourceRegionHops=support.admissibilityMaximumSourceRegionHops;
    expandedSupport.parentStableSourceFaceIdentities.reserve(virtualToParent.size());
    for (const auto parent : virtualToParent)
        expandedSupport.parentStableSourceFaceIdentities.push_back(support.parentStableSourceFaceIdentities[parent]);
    expandedSupport.sourceParentTetOffsets.assign(support.groupMaterialIndices.size()+1u,0u);
    for (std::size_t group=0;group<support.groupMaterialIndices.size();++group) {
        const auto begin=support.sourceParentTetOffsets[group],end=support.sourceParentTetOffsets[group+1u];
        for (std::uint32_t r=begin;r<end;++r) {
            const auto parent=support.sourceParentTetRows[r];
            if (parent>=parentToVirtual.size() || parentToVirtual[parent].empty())
                return fail("source carrier expansion lost a source parent in semantic group ancestry");
            for (const auto virtualRow:parentToVirtual[parent])
                expandedSupport.sourceParentTetRows.push_back(virtualRow);
        }
        expandedSupport.sourceParentTetOffsets[group+1u]=
            static_cast<std::uint32_t>(expandedSupport.sourceParentTetRows.size());
    }
    spatial_mass_transport_detail::SHA256Builder bound;
    bound.string("numi.human.source-tetra-centroid-carrier-expanded-typed-support.v1");
    bound.string(fractional_semantic_transport_detail::supportDigest(support));
    bound.string(carrierArraySHA);
    bound.string(carrier.sourceCarrierPacketSHA256);
    expandedSupport.typedSupportSHA256=bound.finish();
    return true;
}

inline bool aggregateCarrierResult(
        const ConservativeSpatialMassTransportInput& originalInput,
        const ConservativeSpatialChildSemanticSupport& originalSupport,
        const SourceCellCarrierQuadratureInput& carrier,
        const std::vector<std::uint32_t>& virtualToParent,
        const std::vector<std::uint32_t>& virtualToCarrier,
        const ConservativeSpatialMassTransportResult& expandedResult,
        ConservativeSpatialMassTransportResult& out,
        SourceCellCarrierTransportTrace& trace,
        std::string& error) {
    using namespace spatial_mass_transport_detail;
    const auto fail=[&](std::string message){error=std::move(message);return false;};
    const std::size_t np=originalInput.parentMassKg.size(),nc=originalInput.childReferenceVolumeM3.size();
    if (expandedResult.childParentOffsets.size()!=nc+1u ||
        expandedResult.childParentOffsets.empty() ||
        expandedResult.childParentOffsets.back()!=expandedResult.childParentRows.size() ||
        expandedResult.childParentTransportMassKg.size()!=expandedResult.childParentRows.size() ||
        expandedResult.childParentTransportMeasureM3.size()!=expandedResult.childParentRows.size() ||
        expandedResult.childParentTransportDistanceM.size()!=expandedResult.childParentRows.size() ||
        expandedResult.parentMassTransferredKg.size()!=virtualToParent.size() ||
        expandedResult.parentMeasureTransferredM3.size()!=virtualToParent.size())
        return fail("expanded source carrier solver result has incomplete flow arrays");

    struct Piece { std::uint32_t virtualParent; double mass,measure,distance; };
    out=expandedResult;
    out.parentMassTransferredKg.assign(np,0.0);
    out.parentMeasureTransferredM3.assign(np,0.0);
    std::vector<ExactDoubleSum> parentMassSum(np),parentMeasureSum(np);
    for (std::size_t v=0;v<virtualToParent.size();++v) {
        const auto p=virtualToParent[v];
        if (p>=np) return fail("expanded source carrier map references an invalid source parent row");
        parentMassSum[p].add(expandedResult.parentMassTransferredKg[v]);
        parentMeasureSum[p].add(expandedResult.parentMeasureTransferredM3[v]);
    }
    for (std::size_t p=0;p<np;++p) {
        out.parentMassTransferredKg[p]=parentMassSum[p].rounded();
        out.parentMeasureTransferredM3[p]=parentMeasureSum[p].rounded();
        const double massError=std::abs(out.parentMassTransferredKg[p]-originalInput.parentMassKg[p]);
        const double measureError=std::abs(out.parentMeasureTransferredM3[p]-originalInput.parentReferenceVolumeM3[p]);
        out.maximumParentMassClosureErrorKg=std::max(out.maximumParentMassClosureErrorKg,massError);
        out.maximumParentMeasureClosureErrorM3=std::max(out.maximumParentMeasureClosureErrorM3,measureError);
        if (massError>std::max(originalInput.absoluteMassToleranceKg,
                originalInput.relativeClosureTolerance*originalInput.parentMassKg[p]) ||
            measureError>std::max(originalInput.absoluteMeasureToleranceM3,
                originalInput.relativeClosureTolerance*originalInput.parentReferenceVolumeM3[p]))
            return fail("collapsed source carrier flows do not close original source parent mass and measure");
    }

    out.childParentOffsets.assign(nc+1u,0u);
    out.childParentRows.clear(); out.childParentTransportMassKg.clear();
    out.childParentTransportMeasureM3.clear(); out.childParentTransportDistanceM.clear();
    out.childParentVolumeFractions.clear();
    out.childDominantParentRows.assign(nc,kInvalid);
    trace.childParentCarrierOffsets.assign(1u,0u);
    trace.childParentCarrierRows.clear(); trace.childParentCarrierMassKg.clear();
    trace.childParentCarrierMeasureM3.clear(); trace.childParentCarrierDistanceM.clear();
    trace.expandedParentToSourceParentRows=virtualToParent;
    trace.expandedParentToCarrierRows=virtualToCarrier;
    trace.sourceCarrierPacketSHA256=std::string(carrier.sourceCarrierPacketSHA256);
    trace.sourceCarrierArraySHA256=carrierArrayDigest(originalInput,carrier);
    long double weightedDistance=0.0L,totalMass=0.0L;
    for(std::size_t child=0;child<nc;++child){
        out.childParentOffsets[child]=static_cast<std::uint32_t>(out.childParentRows.size());
        std::map<std::uint32_t,std::vector<Piece>> grouped;
        const auto begin=expandedResult.childParentOffsets[child],end=expandedResult.childParentOffsets[child+1u];
        if(begin>=end)return fail("source carrier transport left a real child without a parent flow");
        for(std::uint32_t edge=begin;edge<end;++edge){
            const auto virtualParent=expandedResult.childParentRows[edge];
            if(virtualParent>=virtualToParent.size())
                return fail("expanded source carrier flow references an invalid virtual parent row");
            grouped[virtualToParent[virtualParent]].push_back({virtualParent,
                expandedResult.childParentTransportMassKg[edge],
                expandedResult.childParentTransportMeasureM3[edge],
                expandedResult.childParentTransportDistanceM[edge]});
        }
        const std::size_t firstOutputEdge=out.childParentRows.size();
        ExactDoubleSum childMeasureSum;
        double bestMeasure=-1.0;std::uint32_t dominant=kInvalid;
        for(auto& [parent,pieces]:grouped){
            std::sort(pieces.begin(),pieces.end(),[](const Piece& a,const Piece& b){return a.virtualParent<b.virtualParent;});
            ExactDoubleSum massSum,measureSum;
            long double distanceMass=0.0L;
            for(const auto& piece:pieces){
                if(!std::isfinite(piece.mass)||piece.mass<=0.0||
                   !std::isfinite(piece.measure)||piece.measure<=0.0||
                   !std::isfinite(piece.distance)||piece.distance>originalInput.maximumTransportDistanceM)
                    return fail("expanded source carrier flow contains invalid mass, measure or centroid distance");
                massSum.add(piece.mass);measureSum.add(piece.measure);
                distanceMass+=static_cast<long double>(piece.mass)*piece.distance;
                trace.childParentCarrierRows.push_back(piece.virtualParent);
                trace.childParentCarrierMassKg.push_back(piece.mass);
                trace.childParentCarrierMeasureM3.push_back(piece.measure);
                trace.childParentCarrierDistanceM.push_back(piece.distance);
            }
            const double mass=massSum.rounded(),measure=measureSum.rounded();
            if(!std::isfinite(mass)||mass<=0.0||!std::isfinite(measure)||measure<=0.0)
                return fail("collapsed source carrier parent-child flow is invalid");
            out.childParentRows.push_back(parent);
            out.childParentTransportMassKg.push_back(mass);
            out.childParentTransportMeasureM3.push_back(measure);
            out.childParentTransportDistanceM.push_back(static_cast<double>(distanceMass/mass));
            childMeasureSum.add(measure);
            if(measure>bestMeasure||(measure==bestMeasure&&parent<dominant)){bestMeasure=measure;dominant=parent;}
            for(const auto& piece:pieces){weightedDistance+=static_cast<long double>(piece.mass)*piece.distance;totalMass+=piece.mass;}
            trace.childParentCarrierOffsets.push_back(static_cast<std::uint32_t>(trace.childParentCarrierRows.size()));
        }
        const double childMeasure=childMeasureSum.rounded();
        if(childMeasure<=0.0||!std::isfinite(childMeasure))
            return fail("collapsed source carrier child measure is nonpositive");
        ExactDoubleSum fractionSum;
        for(std::size_t edge=firstOutputEdge;edge<out.childParentRows.size();++edge){
            const double fraction=out.childParentTransportMeasureM3[edge]/childMeasure;
            if(!std::isfinite(fraction)||fraction<=0.0)
                return fail("collapsed source carrier child measure fraction is invalid");
            out.childParentVolumeFractions.push_back(fraction);
            fractionSum.add(fraction);
        }
        out.maximumChildFractionSumError=std::max(out.maximumChildFractionSumError,
            std::abs(fractionSum.rounded()-1.0));
        if(out.maximumChildFractionSumError>2.0e-12)
            return fail("collapsed source carrier child fractions do not sum to one");
        out.childDominantParentRows[child]=dominant;
        out.childParentOffsets[child+1u]=static_cast<std::uint32_t>(out.childParentRows.size());
    }
    out.transportEdgeCount=out.childParentRows.size();
    out.massWeightedMeanTransportDistanceM=totalMass>0.0L?
        static_cast<double>(weightedDistance/totalMass):0.0;
    out.maximumChildFractionSumError=0.0;
    out.semanticSupportSHA256=combinedSupportDigest(originalSupport,
        carrierArrayDigest(originalInput,carrier),expandedResult.semanticSupportSHA256);
    std::string parentHash,childHash,boundsHash;
    hashInputArrays(originalInput,parentHash,childHash,boundsHash);
    out.parentDataSHA256=std::move(parentHash);
    out.childDataSHA256=fractional_semantic_transport_detail::bindSupportToChildHash(childHash,out.semanticSupportSHA256);
    out.materialBoundsSHA256=std::move(boundsHash);
    out.receiptSHA256.clear();
    trace.expandedSolverReceiptSHA256=expandedResult.receiptSHA256;
    spatial_mass_transport_detail::SHA256Builder traceHash;
    traceHash.string("numi.human.source-tetra-centroid-carrier-flow-trace.v1");
    traceHash.string(carrierArrayDigest(originalInput,carrier));
    traceHash.string(expandedResult.receiptSHA256);
    for (const auto value:trace.expandedParentToSourceParentRows) traceHash.u32(value);
    for (const auto value:trace.expandedParentToCarrierRows) traceHash.u32(value);
    for (const auto value:trace.carrierSourceParentRows) traceHash.u32(value);
    for (const auto value:trace.originalCentroidNearestAdmissibleDistanceM) traceHash.real(value);
    for (const auto value:trace.childParentCarrierOffsets) traceHash.u32(value);
    for (const auto value:trace.childParentCarrierRows) traceHash.u32(value);
    for (const auto value:trace.childParentCarrierMassKg) traceHash.real(value);
    for (const auto value:trace.childParentCarrierMeasureM3) traceHash.real(value);
    for (const auto value:trace.childParentCarrierDistanceM) traceHash.real(value);
    trace.receiptSHA256=traceHash.finish();
    const std::string base= fractional_semantic_transport_detail::sharedCapacityReceipt(
        originalInput,out,out.semanticSupportSHA256);
    SHA256Builder receipt;
    receipt.string("numi.human.conservative-spatial-mass-transport.source-carrier-result.v1");
    receipt.string(base);receipt.string(trace.receiptSHA256);receipt.string(expandedResult.receiptSHA256);
    out.receiptSHA256=receipt.finish();
    return true;
}

} // namespace source_carrier_transport_detail

[[nodiscard]] inline bool assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        ConservativeSpatialMassTransportResult& out,
        SourceCellCarrierTransportTrace& trace,
        std::string& error,
        fractional_semantic_transport_detail::SharedCapacityFlowTelemetry* telemetry=nullptr,
        const fractional_semantic_transport_detail::SharedCapacityWarmStart* warmStart=nullptr) {
    using namespace source_carrier_transport_detail;
    std::vector<std::uint32_t> parentGroup;
    std::vector<std::map<std::uint32_t,double>> preferences;
    std::vector<std::vector<std::uint32_t>> admissibleRegions;
    std::vector<Point> carrierPoints;
    std::vector<std::uint32_t> carrierRows;
    std::vector<double> strictNearest;
    if (!validateCarrierInput(input,support,carrier,parentGroup,preferences,admissibleRegions,
                              carrierPoints,carrierRows,strictNearest,error)) return false;
    const std::string carrierSHA=carrierArrayDigest(input,carrier);
    if(warmStart){
        if(warmStart->sourceCarrierPacketSHA256!=carrier.sourceCarrierPacketSHA256 ||
           warmStart->sourceCarrierArraySHA256!=carrierSHA ||
           warmStart->typedSupportSHA256!=support.typedSupportSHA256)
            return (error="source carrier warm-start packet/array/support hashes do not match current input",false);
    }
    std::vector<std::uint32_t> virtualToParent,virtualToCarrier;
    std::vector<std::vector<std::uint32_t>> parentToVirtual(input.parentMassKg.size());
    std::vector<Point> parentCentroids;
    std::vector<double> parentVolumes,parentMasses,donorMasses;
    std::vector<std::uint32_t> parentMaterials,parentDonors,parentRegions,donorOffsets{0u},donorIDs;
    ConservativeSpatialMassTransportInput expandedInput;
    ConservativeSpatialChildSemanticSupportData expandedSupport;
    if (!expandParentRows(input,support,carrier,virtualToParent,virtualToCarrier,parentToVirtual,
                          parentCentroids,parentVolumes,parentMasses,parentMaterials,parentDonors,
                          parentRegions,donorOffsets,donorIDs,donorMasses,expandedInput,
                          expandedSupport,carrierSHA,error)) return false;
    if(warmStart){
        for(const auto& row:warmStart->flows){
            if(row.parentRow>=virtualToParent.size() ||
               row.sourceRoleParentRow!=virtualToParent[row.parentRow] ||
               row.sourceCarrierRow!=virtualToCarrier[row.parentRow] ||
               row.sourceRoleParentRow>=carrier.globalSourceTetRows.size() ||
               row.globalSource330TetRow!=carrier.globalSourceTetRows[row.sourceRoleParentRow])
                return (error="source carrier warm-start expanded/role/global-tet/carrier row maps are stale",false);
        }
    }
    ConservativeSpatialMassTransportResult expandedResult;
    auto expandedWarmStart=warmStart?*warmStart:fractional_semantic_transport_detail::SharedCapacityWarmStart{};
    if(warmStart)expandedWarmStart.typedSupportSHA256=expandedSupport.typedSupportSHA256;
    const auto* effectiveWarmStart=warmStart?&expandedWarmStart:nullptr;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupport(
            expandedInput,expandedSupport.view(),expandedResult,error,telemetry,effectiveWarmStart)) {
        if (telemetry && telemetry->partialFlowCaptured) {
            telemetry->partialExpandedParentRoleRows=virtualToParent;
            telemetry->partialExpandedParentCarrierRows=virtualToCarrier;
        }
        return false;
    }
    SourceCellCarrierTransportTrace nextTrace;
    nextTrace.carrierSourceParentRows=std::move(carrierRows);
    nextTrace.originalCentroidNearestAdmissibleDistanceM=std::move(strictNearest);
    if (!aggregateCarrierResult(input,support,carrier,virtualToParent,virtualToCarrier,
                                expandedResult,out,nextTrace,error)) return false;
    trace=std::move(nextTrace);
    error.clear();return true;
}

[[nodiscard]] inline bool verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
        const ConservativeSpatialMassTransportInput& input,
        const ConservativeSpatialChildSemanticSupport& support,
        const SourceCellCarrierQuadratureInput& carrier,
        const ConservativeSpatialMassTransportResult& supplied,
        const SourceCellCarrierTransportTrace& suppliedTrace,
        std::string& error) {
    ConservativeSpatialMassTransportResult expected;
    SourceCellCarrierTransportTrace expectedTrace;
    if (!assembleConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriers(
            input,support,carrier,expected,expectedTrace,error)) return false;
    if (!fractional_semantic_transport_detail::sameResult(expected,supplied))
        return (error="source carrier aggregate mass-transport result differs from deterministic replay",false);
    if (expectedTrace.sourceCarrierPacketSHA256!=suppliedTrace.sourceCarrierPacketSHA256 ||
        expectedTrace.sourceCarrierArraySHA256!=suppliedTrace.sourceCarrierArraySHA256 ||
        expectedTrace.expandedSolverReceiptSHA256!=suppliedTrace.expandedSolverReceiptSHA256 ||
        expectedTrace.expandedParentToSourceParentRows!=suppliedTrace.expandedParentToSourceParentRows ||
        expectedTrace.expandedParentToCarrierRows!=suppliedTrace.expandedParentToCarrierRows ||
        expectedTrace.carrierSourceParentRows!=suppliedTrace.carrierSourceParentRows ||
        expectedTrace.originalCentroidNearestAdmissibleDistanceM!=suppliedTrace.originalCentroidNearestAdmissibleDistanceM ||
        expectedTrace.childParentCarrierOffsets!=suppliedTrace.childParentCarrierOffsets ||
        expectedTrace.childParentCarrierRows!=suppliedTrace.childParentCarrierRows ||
        expectedTrace.childParentCarrierMassKg!=suppliedTrace.childParentCarrierMassKg ||
        expectedTrace.childParentCarrierMeasureM3!=suppliedTrace.childParentCarrierMeasureM3 ||
        expectedTrace.childParentCarrierDistanceM!=suppliedTrace.childParentCarrierDistanceM ||
        expectedTrace.receiptSHA256!=suppliedTrace.receiptSHA256)
        return (error="source carrier flow trace differs from deterministic replay",false);
    error.clear();return true;
}

} // namespace metalrobo::human::coarse_tissue
