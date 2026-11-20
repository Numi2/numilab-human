#include "NumiHumanCurrentFractionalSupportNPZ.hpp"

#include <iomanip>
#include <iostream>

using namespace metalrobo::human::coarse_tissue;

int main(int argc, char** argv) {
    if (argc != 13) {
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
    for (std::uint32_t row = 0; row < rowCount; ++row) {
        std::uint32_t id = 0, material = 0;
        std::array<std::uint32_t, 4> nodes{};
        if (!(elements >> id >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> material) ||
            id != row + 1 || (material != 15 && material != 16)) {
            std::cerr << "Unexpected TetGen element row " << row << '\n';
            return 4;
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
}
