#include "NumiHumanParentReferenceMassTransfer.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
    if (argc != 4) return 2;
    std::ifstream nodes(argv[1]), elements(argv[2]);
    std::ofstream output(argv[3], std::ios::binary | std::ios::out);
    std::size_t pointCount, tetCount;
    unsigned dimension, attributes, markers, arity;
    if (!(nodes >> pointCount >> dimension >> attributes >> markers) ||
            dimension != 3 || attributes != 0 || markers != 0)
        throw std::runtime_error("Invalid TetGen node header");
    std::vector<std::array<double, 3>> points(pointCount);
    for (std::size_t i = 0; i < pointCount; ++i) {
        std::size_t row;
        auto& p = points[i];
        if (!(nodes >> row >> p[0] >> p[1] >> p[2]) || row != i + 1)
            throw std::runtime_error("Invalid TetGen node row");
        for (const auto value : p)
            if (!std::isfinite(value) || value != double(float(value)))
                throw std::runtime_error("Node does not match the emitted Float32 point");
    }
    if (!(elements >> tetCount >> arity >> attributes) || arity != 4 || attributes != 1)
        throw std::runtime_error("Invalid TetGen element header");
    std::vector<double> volumes(tetCount);
    for (std::size_t i = 0; i < tetCount; ++i) {
        std::size_t row;
        unsigned material;
        std::array<std::uint32_t, 4> tet;
        if (!(elements >> row >> tet[0] >> tet[1] >> tet[2] >> tet[3] >> material)
                || row != i + 1 || (material != 15 && material != 16))
            throw std::runtime_error("Invalid TetGen element row");
        for (auto& v : tet) {
            if (!v || v > pointCount) throw std::runtime_error("Invalid TetGen node index");
            --v;
        }
        using metalrobo::human::coarse_tissue::parentReferenceTetVolumeM3;
        volumes[i] = parentReferenceTetVolumeM3(tet, points);
        if (!(volumes[i] > 1e-18)) throw std::runtime_error("Reference volume below native gate");
    }
    if (!output.write(reinterpret_cast<const char*>(volumes.data()),
                      volumes.size() * sizeof(double)))
        throw std::runtime_error("Failed to write reference volumes");
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
