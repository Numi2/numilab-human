#include "NumiHumanPhysicalSkinChildTetGen.hpp"
#include "NumiHumanMaterialCellShapeAudit.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>

namespace tissue = metalrobo::human::coarse_tissue;

static std::vector<double> read_masses(const std::filesystem::path& path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    std::vector<double> values(count);
    if (!file.read(reinterpret_cast<char*>(values.data()), count * sizeof(double)) || file.peek() != EOF)
        throw std::runtime_error("Mass array length differs from the imported mesh");
    return values;
}

int main(int argc, char** argv) try {
    if (argc != 5) return 2;
    const std::filesystem::path path = argv[1];
    const size_t bodies = std::stoull(argv[3]);
    tissue::PhysicalSkinChildMassMappingBundle bundle;
    tissue::PhysicalSkinChildTetGenImportReport imported;
    std::string error;
    if (!tissue::loadPhysicalSkinChildTetGenReference(
            path / "mesh.node", path / "mesh.ele", path / "current-child.labels.csv",
            argv[2], bodies, std::stod(argv[4]), bundle, imported, error))
        throw std::runtime_error(error);
    const auto& points = bundle.referenceWorldPoints;
    const auto& tets = bundle.childTetrahedra;
    const auto mass = read_masses(path / "cell-mass.f64", tets.size());
    const auto expected_nodes = read_masses(path / "node-mass.f64", points.size());
    std::vector<double> nodal(points.size(), 0.0);
    numiHumanCellShape::Summary shape;
    size_t bad_density = 0;
    double skin_mass = 0.0, fat_mass = 0.0, max_node_error = 0.0;
    for (size_t i = 0; i < tets.size(); ++i) {
        const auto& cell = tets[i];
        std::array<double,3> e[3]{};
        for (unsigned j = 0; j < 3; ++j)
            for (unsigned k = 0; k < 3; ++k)
                e[j][k] = points[cell[j+1]][k] - points[cell[0]][k];
        const auto& a = e[0]; const auto& b = e[1]; const auto& c = e[2];
        const double det = a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]);
        shape.observe(det,numiHumanCellShape::normalizedDeterminant(det,a,b,c),static_cast<uint32_t>(i));
        const auto material = bundle.childMaterialIndices[i];
        if (material != 15 && material != 16) throw std::runtime_error("Unknown tissue");
        const double lo = material == 15 ? 900.0 : 850.0, hi = material == 15 ? 1100.0 : 950.0;
        const double rho = mass[i] / (det / 6);
        const auto roundoff = tissue::spatial_mass_transport_detail::densityBoundRoundoff(lo,hi,rho);
        if (!std::isfinite(rho) || rho < lo-roundoff || rho > hi+roundoff) ++bad_density;
        (material == 15 ? skin_mass : fat_mass) += mass[i];
        for (auto v : cell) nodal[v] += .25 * mass[i];
    }
    size_t empty = 0;
    for (size_t i = 0; i < nodal.size(); ++i) {
        if (!(nodal[i] > 0) || !std::isfinite(nodal[i])) ++empty;
        max_node_error = std::max(max_node_error, std::abs(nodal[i] - expected_nodes[i]));
    }
    std::cout << std::setprecision(17) << "{\"native_imported\":true,\"nodes\":" << imported.nodeCount
              << ",\"cells\":" << imported.cellCount << ",\"inverted\":" << shape.inverted
              << ",\"below_shape\":" << shape.belowShape << ",\"below_volume\":" << shape.belowVolume
              << ",\"nonfinite\":" << shape.nonfinite << ",\"density_outliers\":" << bad_density
              << ",\"zero_mass_nodes\":" << empty << ",\"maximum_nodal_error_kg\":" << max_node_error
              << ",\"skin_kg\":" << skin_mass << ",\"fat_kg\":" << fat_mass
              << ",\"node_sha256\":\"" << imported.nodeFileSHA256 << "\",\"element_sha256\":\""
              << imported.elementFileSHA256 << "\",\"label_sha256\":\"" << imported.labelFileSHA256 << "\"}\n";
    return shape.inverted || shape.nonfinite || shape.belowShape || shape.belowVolume ||
           bad_density || empty || max_node_error > 1e-14 ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
