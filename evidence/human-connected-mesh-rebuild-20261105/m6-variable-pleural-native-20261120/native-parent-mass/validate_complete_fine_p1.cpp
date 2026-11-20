#include "NumiHumanCurrentFractionalSupportNPZ.hpp"
#include "NumiHumanCurrentFractionalRecordedFlowArtifacts.hpp"
#include "NumiHumanPhysicalSkinChildTetGen.hpp"

#include <iomanip>
#include <iostream>

using namespace metalrobo::human::coarse_tissue;

#include "replay_grouped_paired_flow.hpp"

int main(int argc, char** argv) {
    if (argc != 13 && argc != 27 && argc != 31 && argc != 34) {
        std::cerr << "Expected parent mass and labels, TetGen elements, state and parent pins, "
                     "then the skin and fat NPZ, pin and geometry pin, and expected total cells.\n";
        return 2;
    }
    const auto count = static_cast<std::size_t>(std::stoull(argv[12]));
    PhysicalSkinParentMassPacket parent;
    std::string error;
    if (!loadCompiledSkinFatParentMassPacketCSV(argv[1], parent, error) ||
        !loadCompiledSkinFatParentLabelsCSV(argv[2], parent, error)) {
        std::cerr << error << '\n';
        return 3;
    }

    std::ifstream elements(argv[3]);
    std::size_t rowCount = 0;
    unsigned nodesPerCell = 0, attributes = 0;
    if (!(elements >> rowCount >> nodesPerCell >> attributes) ||
        rowCount != count || nodesPerCell != 4 || attributes != 1) {
        std::cerr << "Unexpected TetGen element domain\n";
        return 4;
    }
    std::vector<std::uint32_t> materials, skinRows, fatRows;
    materials.reserve(rowCount);
    const bool withCarriers = argc >= 27;
    std::vector<SpatialMassTransportPoint> positions, centroids;
    std::vector<double> volumes;
    if (withCarriers) {
        std::ifstream nodeFile(argv[13]);
        std::size_t nodeCount = 0;
        unsigned dimension = 0, nodeAttributes = 0, markers = 0;
        if (!(nodeFile >> nodeCount >> dimension >> nodeAttributes >> markers) ||
            dimension != 3 || nodeAttributes != 0 || markers != 0 || nodeCount < 4) {
            std::cerr << "Unexpected TetGen node domain\n";
            return 7;
        }
        positions.resize(nodeCount);
        for (std::size_t i = 0; i < nodeCount; ++i) {
            std::size_t row = 0;
            auto& point = positions[i];
            if (!(nodeFile >> row >> point[0] >> point[1] >> point[2]) || row != i + 1) {
                std::cerr << "Invalid TetGen node row " << i << '\n';
                return 7;
            }
        }
        centroids.resize(rowCount);
        volumes.resize(rowCount);
    }
    for (std::uint32_t row = 0; row < rowCount; ++row) {
        std::uint32_t id = 0, material = 0;
        std::array<std::uint32_t, 4> nodes{};
        if (!(elements >> id >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> material) ||
            id != row + 1 || (material != 15 && material != 16)) {
            std::cerr << "Unexpected TetGen element row " << row << '\n';
            return 4;
        }
        if (withCarriers) {
            std::array<SpatialMassTransportPoint, 4> corners;
            for (std::size_t i = 0; i < corners.size(); ++i) {
                if (nodes[i] == 0 || nodes[i] > positions.size()) {
                    std::cerr << "TetGen node index outside child geometry\n";
                    return 7;
                }
                corners[i] = positions[nodes[i] - 1];
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    centroids[row][axis] += 0.25 * corners[i][axis];
                }
            }
            volumes[row] = parentReferenceTetVolumeM3({0, 1, 2, 3}, corners);
            if (!(volumes[row] > 0.0)) {
                std::cerr << "Unexpected nonpositive child cell " << row << '\n';
                return 7;
            }
        }
        materials.push_back(material);
        (material == 15 ? skinRows : fatRows).push_back(row);
    }

    const CurrentFractionalSupportPins skinPins{argv[7], argv[4], argv[8], argv[8], argv[5]};
    const CurrentFractionalSupportPins fatPins{argv[10], argv[4], argv[11], argv[11], argv[5]};
    ConservativeSpatialChildSemanticSupportData skin, fat, merged;
    if (!loadCurrentP1SemanticSupportNPZ(argv[6], skinPins, parent, skinRows.size(), skin, error) ||
        !loadCurrentP1SemanticSupportNPZ(argv[9], fatPins, parent, fatRows.size(), fat, error) ||
        !mergeCurrentPairedP1SemanticSupport(skin, fat, skinRows, fatRows, materials, parent,
                                             merged, error)) {
        std::cerr << error << '\n';
        return 5;
    }

    auto wrongPins = skinPins;
    wrongPins.volumeSHA256 = std::string(64, '0');
    if (loadCurrentP1SemanticSupportNPZ(argv[6], wrongPins, parent, skinRows.size(), skin, error) ||
        error.find("mass hashes differ") == std::string::npos) {
        std::cerr << "Native reader did not reject the mismatched volume pin\n";
        return 6;
    }
    std::cout << "Native P1 reader and paired merge PASS\n"
              << "parents " << parent.tetrahedra.size()
              << " skin " << skinRows.size() << " fat " << fatRows.size() << '\n'
              << "merged slots " << merged.slotGroupIndices.size()
              << " region entries " << merged.cellAdmissibleRegionIds.size() << '\n'
              << "paired support hash " << merged.typedSupportSHA256 << '\n'
              << "parent ledger mass kg " << std::setprecision(17) << parent.totalMassKg << '\n'
              << "Native reader rejects incorrect volume pin\n";
    if (!withCarriers) return 0;
    CurrentSourceCarrierQuadratureData carrier;
    const CurrentSourceCarrierPins carrierPins{argv[17], argv[19], argv[25], argv[26]};
    if (!loadCurrentPairedSourceCarrierQuadratureNPZ(argv[16], argv[18], argv[14], argv[15],
                                                     carrierPins, parent, skinRows, fatRows,
                                                     carrier, error)) {
        std::cerr << error << '\n';
        return 8;
    }
    ConservativeSpatialMassTransportInput input{};
    input.parentMassKg = parent.tetrahedronMassKg;
    input.parentMaterialIndices = parent.materialIndices;
    input.parentDonorBodyIndices = parent.donorBodyIndices;
    input.parentAnatomicalRegionIndices = parent.anatomicalRegionIndices;
    input.childCentroidsWorld = centroids;
    input.childReferenceVolumeM3 = volumes;
    input.childMaterialIndices = materials;
    input.maximumTransportDistanceM = std::stod(argv[24]);

    spatial_mass_transport_detail::SHA256Builder digest;
    digest.string("paired-primary-carrier-native-reader-check");
    digest.string(argv[21]);
    digest.string(argv[23]);
    const std::string evidenceHash = digest.finish();
    const std::string emptyFileSHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    CurrentCarrierWitnessAllowlistStreamState stream;
    if (!stream.begin(merged, input, carrier, {}, emptyFileSHA, evidenceHash, error, skinRows, fatRows)) {
        std::cerr << error << '\n';
        return 9;
    }
    const auto consume = [&](const CurrentCarrierWitnessEvidenceRow& row, std::string& issue) {
        return stream.consume(row, issue);
    };
    CurrentCarrierWitnessCSVSummary skinWitness, fatWitness;
    if (!loadCurrentCarrierWitnessCSVEach(argv[20], argv[21], 15, skinWitness, consume, error) ||
        !loadCurrentCarrierWitnessCSVEach(argv[22], argv[23], 16, fatWitness, consume, error) ||
        !stream.finish(error)) {
        std::cerr << error << '\n';
        return 10;
    }
    std::cout << "Native carrier witness reader and coordinate replay PASS\n"
              << "skin " << skinWitness.carrierRecordCount << " fat " << fatWitness.carrierRecordCount
              << " total " << stream.primaryCount() << '\n'
              << "additional exact tuple witness rows " << stream.supportCount() << '\n'
              << "typed exact allowlist entries " << merged.childAdmissibleGroupIndices.size() << '\n'
              << "retained compensation parent rows " << carrier.compensationParentRows.size()
              << " retained compensation child rows " << carrier.compensationChildRows.size() << '\n';
    if (argc == 27) return 0;
    PhysicalSkinChildMassMappingBundle child;
    PhysicalSkinChildTetGenImportReport childReport;
    if (!loadPhysicalSkinChildTetGenReference(argv[13], argv[3], argv[27], argv[30],
            std::stoull(argv[29]), input.maximumTransportDistanceM, child, childReport, error) ||
        childReport.labelFileSHA256 != argv[28] || child.childMaterialIndices != materials) {
        std::cerr << "Native TetGen and P1 label admission failed: " << error << '\n';
        return 11;
    }
    std::vector<ConservativeSpatialMassTransportParentNodeSourceMass> source;
    source.reserve(parent.sourceMassContributions.size());
    for (const auto& row : parent.sourceMassContributions) {
        source.push_back({row.parentNodeIndex, row.materialIndex, row.donorBodyIndex, row.massKg});
    }
    std::vector<std::uint32_t> splits, donorIDs;
    std::vector<double> donorMass;
    if (!assembleConservativeSpatialMassTransportParentSourceDonorSplits(
            parent.tetrahedronMassKg, parent.materialIndices, parent.tetrahedra, source,
            1e-10, 5e-4, splits, donorIDs, donorMass, error)) {
        std::cerr << error << '\n';
        return 12;
    }
    std::vector<SpatialMassTransportPoint> parentCentroids(parent.tetrahedra.size());
    std::vector<double> parentVolumes(parent.tetrahedra.size());
    double maximumParentVolumeDifference = 0.0;
    std::size_t changedParentVolumes = 0;
    for (std::size_t row = 0; row < parentCentroids.size(); ++row) {
        parentVolumes[row] = parentReferenceTetVolumeM3(parent.tetrahedra[row], parent.referenceWorldPoints);
        if (parentVolumes[row] != parent.referenceTetVolumeM3[row]) {
            ++changedParentVolumes;
            maximumParentVolumeDifference = std::max(maximumParentVolumeDifference,
                std::abs(parentVolumes[row] - parent.referenceTetVolumeM3[row]));
        }
        for (const auto node : parent.tetrahedra[row]) {
            for (std::size_t axis = 0; axis < 3; ++axis) {
                parentCentroids[row][axis] += 0.25 * parent.referenceWorldPoints[node][axis];
            }
        }
    }
    std::array<double, 17> lower{}, upper{};
    lower[15] = 900.; upper[15] = 1100.;
    lower[16] = 850.; upper[16] = 950.;
    input.parentCentroidsWorld = parentCentroids;
    input.parentReferenceVolumeM3 = parentVolumes;
    for (std::size_t row = 0; row < child.childTetrahedra.size(); ++row) {
        const auto& tet = child.childTetrahedra[row];
        const double liveVolume = parentReferenceTetVolumeM3(tet, child.referenceWorldPoints);
        if (liveVolume != volumes[row]) {
            std::cerr << "Child source volume differs from the live importer at row " << row << '\n';
            return 20;
        }
        SpatialMassTransportPoint liveCentroid{};
        for (const auto node : tet)
            for (unsigned axis = 0; axis < 3; ++axis)
                liveCentroid[axis] += child.referenceWorldPoints[node][axis] * 0.25;
        if (liveCentroid != centroids[row]) {
            std::cerr << "Child source centroid differs from the live importer at row " << row << '\n';
            return 21;
        }
    }
    std::cout << "Live reference-volume replay "
              << "parent determinant rows differing from captured ledger " << changedParentVolumes
              << " maximum absolute change m3 " << maximumParentVolumeDifference << '\n';
    input.parentSourceDonorOffsets = splits;
    input.parentSourceDonorBodyIndices = donorIDs;
    input.parentSourceDonorMassKg = donorMass;
    input.childDonorBodyIndices = child.childDonorBodyIndices;
    input.childAnatomicalRegionIndices = child.childAnatomicalRegionIndices;
    input.materialDensityLowerKgM3 = lower;
    input.materialDensityUpperKgM3 = upper;
    input.parentMassPacketSHA256 = parent.parentMassPacketSHA256;
    input.parentSourceContributionSHA256 = parent.parentSourceContributionSHA256;
    input.childMeshSHA256 = child.childMeshSHA256;
    input.regionLabelSHA256 = child.regionLabelSHA256;
    input.coordinateFrameSHA256 = child.coordinateFrameSHA256;
    if (!bindCurrentRegionalMassCompensation(merged, input, parent, carrier, error)) {
        std::cerr << "Native current mass support or compensation rejected: " << error << '\n';
        return 13;
    }
    std::cout << "Native TetGen and full typed mass support PASS\n"
              << "nodes " << childReport.nodeCount << " cells " << childReport.cellCount
              << " label sha256 " << childReport.labelFileSHA256 << '\n'
              << "minimum emitted reference volume m3 " << childReport.minimumFloat32ReferenceVolumeM3
              << " maximum float32 rounding m " << childReport.maximumFloat32RoundingDistanceM << '\n';
    if (argc == 31) return 0;
    if (const char* path = std::getenv("NUMI_HUMAN_NATIVE_REFERENCE_VOLUME_OUTPUT")) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(volumes.data()),
                   static_cast<std::streamsize>(volumes.size() * sizeof(double)));
        file.close();
        if (!file) {
            std::cerr << "Cannot retain native reference volumes\n";
            return 15;
        }
    }

    ConservativeSpatialMassTransportResult result;
    SourceCellCarrierTransportTrace trace;
    if (!replayGroupedPairedFlow(argv[31], argv[32], std::stoull(argv[33]),
            input, merged.view(), carrier, result, trace, error)) {
        std::cerr << "Native grouped allocation rejected: " << error << '\n';
        return 14;
    }
    spatial_mass_transport_detail::ExactDoubleSum total;
    for (const double mass : result.childMassKg) total.add(mass);
    std::cout << "Native full grouped allocation feasibility replay PASS\n"
              << "total kg " << total.rounded() << " aggregate edges " << result.transportEdgeCount
              << " expanded edges " << trace.childParentCarrierRows.size() << '\n'
              << "max parent mass error kg " << result.maximumParentMassClosureErrorKg
              << " max donor mass error kg " << result.maximumSourceDonorMassClosureErrorKg << '\n'
              << "minimum density kg/m3 " << result.minimumChildImpliedDensityKgM3
              << " maximum density kg/m3 " << result.maximumChildImpliedDensityKgM3 << '\n'
              << "max transport distance m " << result.maximumTransportDistanceUsedM << '\n'
              << "result sha256 " << result.receiptSHA256 << '\n'
              << "carrier trace sha256 " << trace.receiptSHA256 << '\n';
    const char* roleExport = std::getenv("NUMI_HUMAN_NATIVE_ROLE_FLOW_OUTPUT");
    const char* roleReadback = std::getenv("NUMI_HUMAN_NATIVE_ROLE_FLOW_READBACK");
    if (roleExport || roleReadback) {
        const std::filesystem::path directory(roleExport ? roleExport : roleReadback);
        if (!std::filesystem::is_directory(directory) ||
            (roleExport && !std::filesystem::is_empty(directory))) {
            std::cerr << "Role output requires an existing empty directory\n";
            return 16;
        }
        std::string nodesSHA, elementsSHA;
        if (!physical_skin_child_mass_detail::sha256File(argv[13], nodesSHA, error) ||
            !physical_skin_child_mass_detail::sha256File(argv[3], elementsSHA, error)) {
            std::cerr << error << '\n';
            return 17;
        }
        const std::array<std::string,2> supportSHA{argv[7], argv[10]};
        const std::array<std::string,2> packetSHA{argv[17], argv[19]};
        std::array<std::array<std::string,10>,2> expected{};
        for (unsigned role = 0; role < 2; ++role) {
            expected[role] = {parent.parentMassPacketSHA256, parent.parentSourceContributionSHA256,
                child.childMeshSHA256, child.regionLabelSHA256, child.coordinateFrameSHA256,
                nodesSHA, elementsSHA, childReport.labelFileSHA256, supportSHA[role], packetSHA[role]};
        }
        CurrentRecordedRoleFlowArtifacts output;
        if (roleExport) {
            child.parent = parent;
            if (!writeCurrentFractionalRecordedRoleFlowArtifacts(directory, child, result,
                    trace, carrier.view(), supportSHA, packetSHA, expected, output, error)) {
                std::cerr << "Production role export rejected: " << error << '\n';
                return 18;
            }
            std::cout << "Production role-flow writer PASS\n";
        }
        for (unsigned role = 0; role < 2; ++role) {
            using namespace recorded_carrier_allocation_import_detail;
            const std::string prefix = role == 0 ? "skin-" : "fat-";
            PhysicalSkinRecordedCarrierAllocationCSVAssets assets;
            assets.materialIndex = role == 0 ? 15 : 16;
            assets.receiptPath = directory / (prefix + "recorded-flow-receipt.json");
            const char* pinKey = role == 0 ? "NUMI_HUMAN_NATIVE_ROLE_SKIN_SHA256"
                                           : "NUMI_HUMAN_NATIVE_ROLE_FAT_SHA256";
            if (roleExport) {
                assets.receiptSHA256 = output.receiptSHA256[role];
            } else if (const char* pin = std::getenv(pinKey)) {
                assets.receiptSHA256 = pin;
            }
            for (std::size_t k = 0; k < assets.csvPaths.size(); ++k) {
                assets.csvPaths[k] = directory / (prefix + std::string(
                    current_recorded_role_flow_artifact_detail::kNames[k]));
                if (!physical_skin_child_mass_detail::sha256File(assets.csvPaths[k], assets.csvSHA256[k], error)) {
                    std::cerr << error << '\n';
                    return 19;
                }
            }
            assets.expectedInputSHA256.assign(expected[role].begin(), expected[role].end());
            const auto parents = static_cast<std::size_t>(std::count(
                parent.materialIndices.begin(), parent.materialIndices.end(), assets.materialIndex));
            const auto children = role == 0 ? skinRows.size() : fatRows.size();
            PhysicalSkinSourceCarrierRoleRecordedAllocationData restored;
            PhysicalSkinRecordedCarrierAllocationCSVImportReport report;
            if (!loadPhysicalSkinRecordedCarrierAllocationCSVs(assets, assets.materialIndex,
                    parents, children, input.maximumTransportDistanceM, restored, report, error) ||
                !report.verified || report.physicalAdmission ||
                report.sourceCarrierPacketSHA256 != packetSHA[role] ||
                report.transportReceiptSHA256 != result.receiptSHA256) {
                std::cerr << "Production role reader rejected: " << error << '\n';
                return 20;
            }
            std::cout << "Production " << prefix << "role-flow reader PASS "
                      << report.childRows << " cells " << report.carrierFlowRows << " expanded rows "
                      << assets.receiptSHA256 << '\n';
        }
    }
}
