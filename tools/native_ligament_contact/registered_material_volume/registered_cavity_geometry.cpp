#include "NumiHumanSkinFatFixedCavityReconditioning.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    using namespace metalrobo::human::coarse_tissue;
    using namespace skin_fat_cavity_detail;
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    unsigned nodeCount, oldCount, childCount;
    if (!(input >> nodeCount >> oldCount >> childCount) ||
        !nodeCount || nodeCount > 384 || oldCount < 2 || oldCount > 96 ||
        childCount < 2 || childCount > 96) return 2;
    std::vector<V3> points(nodeCount);
    for (auto& point : points) for (auto& v : point)
        if (!(input >> v) || !std::isfinite(v)) return 2;
    std::vector<Tet> old(oldCount);
    Patch patch;
    patch.cells.resize(childCount);
    for (auto* cells : {&old, &patch.cells}) for (auto& tet : *cells) {
        for (auto& node : tet)
            if (!(input >> node) || node >= nodeCount) return 2;
    }
    patch.oldRows.resize(oldCount);
    std::iota(patch.oldRows.begin(), patch.oldRows.end(), 0u);
    double before = 0, after = 0;
    std::string error;
    if (!certifyMassNeighborhood(points, old, patch, before, after, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "Native reference cavity passed. Minimum shape before " << before
              << " and after " << after << '\n';
    return 0;
}
