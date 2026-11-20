#pragma once

// Expand the independently checked grouped allocation and pass it through
// Matter's existing source-carrier aggregation and feasibility verifier.
inline bool replayGroupedPairedFlow(
    const std::filesystem::path& path, const std::string& pin, const std::size_t augmentations,
    ConservativeSpatialMassTransportInput input,
    const ConservativeSpatialChildSemanticSupport& support,
    const CurrentSourceCarrierQuadratureData& carrier,
    ConservativeSpatialMassTransportResult& result, SourceCellCarrierTransportTrace& trace,
    std::string& error) {
    using namespace current_fractional_support_npz_detail;
    using source_carrier_transport_detail::Point;
    using source_carrier_transport_detail::kInvalidCarrierRow;
    const auto fail = [&](std::string text) {
        error = std::move(text);
        return false;
    };
    std::string massHash;
    if (!physical_skin_child_mass_detail::sha256File(path, massHash, error) || massHash != pin) {
        return fail("Grouped paired flow differs from its verified artifact");
    }
    std::map<std::string, NpyArray> arrays;
    if (!readRequestedArrays(path, {"carrier_row", "group", "group_for_cell", "source_parent",
                                  "mass_flow_kg", "element_mass_kg"}, arrays, error)) {
        return false;
    }
    std::vector<std::uint32_t> carrierRows, edgeGroup, childGroup, sourceParent;
    std::vector<double> mass, childTarget;
    if (!readU32(arrays.at("carrier_row"), carrierRows, error) ||
        !readU32(arrays.at("group"), edgeGroup, error) ||
        !readU32(arrays.at("group_for_cell"), childGroup, error) ||
        !readU32(arrays.at("source_parent"), sourceParent, error) ||
        !readF64(arrays.at("mass_flow_kg"), mass, error) ||
        !readF64(arrays.at("element_mass_kg"), childTarget, error)) {
        return false;
    }
    arrays.clear();
    const auto np = input.parentMassKg.size();
    const auto nc = input.childReferenceVolumeM3.size();
    if (carrierRows.size()!=mass.size() || edgeGroup.size()!=mass.size() ||
        childGroup.size()!=nc || childTarget.size()!=nc || sourceParent.empty() ||
        *std::max_element(childGroup.begin(), childGroup.end()) >= nc) {
        return fail("Grouped paired flow has misaligned edge or child domains");
    }
    const auto groupCount = *std::max_element(childGroup.begin(), childGroup.end()) + 1u;
    std::vector<double> groupTarget(groupCount);
    for (std::size_t c = 0; c < nc; ++c) {
        groupTarget[childGroup[c]] += childTarget[c];
    }

    input.maximumAllowedEdges = physical_skin_child_capacity_detail::maxCandidateEdges;
    input.maximumAugmentations = physical_skin_child_capacity_detail::maxAugmentations;
    std::vector<std::uint32_t> parentGroup, carrierParents;
    std::vector<std::map<std::uint32_t,double>> preferences;
    std::vector<std::vector<std::uint32_t>> allowed;
    std::vector<Point> witnessed;
    const auto view = carrier.view();
    if (!source_carrier_transport_detail::validateCarrierInput(input, support, view,
            parentGroup, preferences, allowed, witnessed, carrierParents,
            trace.originalCentroidNearestAdmissibleDistanceM, error)) {
        return false;
    }
    decltype(preferences)().swap(preferences);
    decltype(allowed)().swap(allowed);
    trace.carrierSourceParentRows = carrierParents;

    std::vector<std::uint32_t> virtualParent, virtualCarrier, carrierToVirtual;
    std::vector<std::vector<std::uint32_t>> parentVirtual(np);
    std::vector<Point> virtualCentroid;
    std::vector<double> virtualVolume, virtualMass, virtualDonorMass;
    std::vector<std::uint32_t> virtualMaterial, virtualDonor, virtualRegion,
                               virtualDonorOffsets{0}, virtualDonorIDs;
    ConservativeSpatialMassTransportInput expandedInput;
    ConservativeSpatialChildSemanticSupportData expandedSupport;
    const auto carrierHash = source_carrier_transport_detail::carrierArrayDigest(input, view);
    if (!source_carrier_transport_detail::expandParentRows(input, support, view,
            virtualParent, virtualCarrier, parentVirtual, virtualCentroid, virtualVolume,
            virtualMass, virtualMaterial, virtualDonor, virtualRegion, virtualDonorOffsets,
            virtualDonorIDs, virtualDonorMass, expandedInput, expandedSupport, carrierHash, error)) {
        return false;
    }

    // The grouped owner orders Skin carriers, Fat carriers and finally the
    // centroid fallback parents. Matter orders by original parent row.
    const auto skinCount = std::count(carrier.carrierRoleMaterialIndices.begin(),
                                     carrier.carrierRoleMaterialIndices.end(), 15u);
    const auto quadratureCount = carrier.carrierWeights.size();
    carrierToVirtual.assign(sourceParent.size(), kInvalidCarrierRow);
    std::size_t fallback = quadratureCount;
    for (std::uint32_t v = 0; v < virtualParent.size(); ++v) {
        const auto k = virtualCarrier[v];
        const auto row = k==kInvalidCarrierRow ? fallback++ :
            carrier.carrierRoleLocalRows[k] +
                (carrier.carrierRoleMaterialIndices[k]==15u ? 0u : skinCount);
        if (row >= sourceParent.size() || sourceParent[row] != virtualParent[v] ||
            carrierToVirtual[row] != kInvalidCarrierRow) {
            return fail("Grouped carrier and centroid orders do not match the native source packet");
        }
        carrierToVirtual[row] = v;
    }
    if (fallback != sourceParent.size() ||
        std::find(carrierToVirtual.begin(), carrierToVirtual.end(),
                  kInvalidCarrierRow) != carrierToVirtual.end()) {
        return fail("Grouped source flow omits a native carrier");
    }
    std::vector<std::vector<std::pair<std::uint32_t,double>>> incoming(groupCount);
    for (std::size_t i = 0; i < mass.size(); ++i) {
        if (edgeGroup[i] >= groupCount || carrierRows[i] >= carrierToVirtual.size() ||
            !std::isfinite(mass[i]) || mass[i] < 0.0) {
            return fail("Grouped source flow references an invalid carrier or child group");
        }
        if (mass[i] > 0.0) {
            const auto g = edgeGroup[i];
            if (!(groupTarget[g] > 0.0)) return fail("Positive incoming flow has no group demand");
            incoming[g].emplace_back(carrierToVirtual[carrierRows[i]], mass[i]);
        }
    }
    for (auto& rows : incoming) {
        std::sort(rows.begin(), rows.end());
        if (std::adjacent_find(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
                return a.first == b.first;
            }) != rows.end()) {
            return fail("Grouped flow has duplicate positive carrier keys");
        }
    }

    ConservativeSpatialMassTransportResult expanded;
    expanded.method = UnpairedVolumeMassTransferKind::conservativeSpatialMassTransport;
    expanded.parentMassPacketSHA256 = input.parentMassPacketSHA256;
    expanded.parentSourceContributionSHA256 = input.parentSourceContributionSHA256;
    expanded.childMeshSHA256 = input.childMeshSHA256;
    expanded.regionLabelSHA256 = input.regionLabelSHA256;
    expanded.coordinateFrameSHA256 = input.coordinateFrameSHA256;
    expanded.maximumTransportDistanceM = input.maximumTransportDistanceM;
    expanded.relativeClosureTolerance = input.relativeClosureTolerance;
    expanded.absoluteMassToleranceKg = input.absoluteMassToleranceKg;
    expanded.absoluteMeasureToleranceM3 = input.absoluteMeasureToleranceM3;
    expanded.maximumAllowedEdges = input.maximumAllowedEdges;
    expanded.maximumAugmentations = input.maximumAugmentations;
    expanded.augmentationCount = augmentations;
    expanded.usesFractionalSemanticSupport = true;
    expanded.semanticSupportSHA256 = fractional_semantic_transport_detail::supportDigest(expandedSupport.view());
    expanded.receiptSHA256 = massHash;
    expanded.parentMassTransferredKg.resize(virtualParent.size());
    expanded.parentMeasureTransferredM3.resize(virtualParent.size());
    expanded.childMassKg.resize(nc);
    expanded.childAncestryMeasureVolumeM3.resize(nc);
    expanded.childImpliedDensityKgM3.resize(nc);
    expanded.childSourceDonorOffsets.resize(nc+1);
    expanded.childParentOffsets.push_back(0);
    std::size_t outsideCount = 0;
    std::size_t worstCell = 0;
    double worstError = 0.0;
    double worstDensity = 0.0;
    for (std::size_t c = 0; c < nc; ++c) {
        const auto g = childGroup[c];
        const auto fraction = childTarget[c] / groupTarget[g];
        for (const auto& [v, flow] : incoming[g]) {
            const double x = flow * fraction;
            const double measure = x / (virtualMass[v] / virtualVolume[v]);
            const double distance = source_carrier_transport_detail::pointDistance(
                virtualCentroid[v], input.childCentroidsWorld[c]);
            expanded.childParentRows.push_back(v);
            expanded.childParentTransportMassKg.push_back(x);
            expanded.childParentTransportMeasureM3.push_back(measure);
            expanded.childParentTransportDistanceM.push_back(distance);
            expanded.parentMassTransferredKg[v] += x;
            expanded.parentMeasureTransferredM3[v] += measure;
            expanded.childMassKg[c] += x;
            expanded.childAncestryMeasureVolumeM3[c] += measure;
        }
        expanded.childImpliedDensityKgM3[c] = expanded.childMassKg[c] / input.childReferenceVolumeM3[c];

        const auto m = input.childMaterialIndices[c];
        const auto lo = input.materialDensityLowerKgM3[m];
        const auto hi = input.materialDensityUpperKgM3[m];
        const double density = expanded.childImpliedDensityKgM3[c];
        const double violation = std::max(lo - density, density - hi);
        if (violation > spatial_mass_transport_detail::densityBoundRoundoff(lo, hi, density)) {
            ++outsideCount;
            if (violation > worstError) {
                worstCell = c;
                worstError = violation;
                worstDensity = density;
            }
        }
        expanded.childParentOffsets.push_back(static_cast<std::uint32_t>(expanded.childParentRows.size()));
    }
    // The imported candidate consists of these expanded positive edges only.
    expanded.semanticCandidateEdgeCount = expanded.childParentRows.size();
    std::cout << "Native direct density outliers " << outsideCount << " worst cell "
              << worstCell << " density " << worstDensity << " delta kg/m3 "
              << worstError << '\n';
    if (!source_carrier_transport_detail::aggregateCarrierResult(input, support, view,
            virtualParent, virtualCarrier, expanded, result, trace, error)) {
        return false;
    }
    expanded = {};
    expandedSupport = {};
    if (!canonicalizeRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
            input, support, view, result, trace, error)) {
        return false;
    }
    return verifyRecordedConservativeSpatialMassTransportWithSemanticSupportAndSourceCarriersFeasibility(
        input, support, view, result, trace, error);
}
