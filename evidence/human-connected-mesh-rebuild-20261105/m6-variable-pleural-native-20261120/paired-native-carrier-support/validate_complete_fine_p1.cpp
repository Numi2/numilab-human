#include "NumiHumanCurrentFractionalSupportNPZ.hpp"

#include <iomanip>
#include <iostream>

using namespace metalrobo::human::coarse_tissue;

int main(int argc, char** argv) {
    if (argc != 13 && argc != 27) {
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
    const bool withCarriers = argc == 27;
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
            volumes[row] = physical_skin_child_mass_detail::signedTetVolume(
                corners[0], corners[1], corners[2], corners[3]);
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
}
