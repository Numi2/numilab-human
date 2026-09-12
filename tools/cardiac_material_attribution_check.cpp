// Independent exact geometry/material-attribution check. No world cooking or
// physical stepping. GMP supplies independent exact integer/rational arithmetic.
#include <gmpxx.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Integer = mpz_class;
using Rational = mpq_class;
using Point = std::array<Integer, 3>;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> std::vector<T> read(const std::filesystem::path& path,
                                    std::size_t maximum) {
    require(std::endian::native == std::endian::little, "little-endian host required");
    require(!std::filesystem::is_symlink(path) && std::filesystem::is_regular_file(path), "not a regular buffer");
    const auto size = std::filesystem::file_size(path);
    require(size % sizeof(T) == 0 && size / sizeof(T) <= maximum, "invalid buffer size");
    std::vector<T> result(size / sizeof(T));
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "buffer open failed");
    stream.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(size));
    require(stream.gcount() == static_cast<std::streamsize>(size) && stream.peek() == std::char_traits<char>::eof(), "buffer changed or truncated");
    return result;
}
struct Dyadic { std::int64_t mantissa; int exponent; };
Dyadic decompose(double value) {
    require(std::isfinite(value), "nonfinite coordinate");
    const auto bits = std::bit_cast<std::uint64_t>(value);
    const auto encoded = int((bits >> 52u) & 2047u);
    const auto fraction = bits & ((std::uint64_t(1) << 52u) - 1u);
    std::int64_t mantissa = std::int64_t(fraction | (encoded ? std::uint64_t(1) << 52u : 0u));
    if (bits >> 63u) mantissa = -mantissa;
    return {mantissa, encoded ? encoded - 1023 - 52 : -1074};
}
Point difference(const Point& a, const Point& b) {
    return {a[0]-b[0], a[1]-b[1], a[2]-b[2]};
}
Integer determinant(const Point& a, const Point& b, const Point& c) {
    return a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0])
        + a[2]*(b[0]*c[1]-b[1]*c[0]);
}
std::uint32_t material(std::uint32_t label) {
    require(label >= 1 && label <= 24, "unknown source label");
    if (label <= 2) return 0;
    if (label <= 4 || label >= 18) return 1;
    if (label == 5) return 2;
    if (label == 6) return 3;
    if (label <= 10) return 4;
    return std::numeric_limits<std::uint32_t>::max();
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 6, "usage: cardiac-material-attribution-check NODES TETS LABELS CLASSES CELLS");
        const std::string argument(argv[5]); std::size_t parsed = 0;
        const auto count = std::stoull(argument, &parsed);
        require(parsed == argument.size() && count > 0 && count <= 5000000, "invalid cell count");
        const auto nodes = read<double>(argv[1], 3000000);
        const auto tets = read<std::uint32_t>(argv[2], count * 4);
        const auto labels = read<std::uint32_t>(argv[3], count);
        const auto classes = read<std::uint32_t>(argv[4], count);
        require(nodes.size() >= 12 && nodes.size() % 3 == 0 &&
            tets.size() == count * 4 && labels.size() == count && classes.size() == count, "field shape mismatch");
        int exponent = 0;
        for (const auto value : nodes) {
            const auto d = decompose(value);
            if (d.mantissa) exponent = std::min(exponent, d.exponent);
        }
        // Bound integer work for this offline geometry check. This is far wider
        // than anatomical coordinate ranges; unbounded subnormal inputs reject.
        require(exponent >= -256, "coordinate exponent outside exact-check range");
        std::vector<Point> points(nodes.size() / 3);
        for (std::size_t node = 0; node < points.size(); ++node)
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const auto d = decompose(nodes[3*node + axis]);
                if (d.mantissa) {
                    require(d.exponent - exponent <= 512, "coordinate magnitude outside exact-check range");
                    points[node][axis] = Integer(static_cast<long>(d.mantissa)) << (d.exponent - exponent);
                }
            }
        std::array<std::uint64_t, 25> labelCounts{};
        std::array<std::uint64_t, 5> classCounts{};
        std::array<Integer, 25> determinants{};
        std::uint64_t unresolved = 0;
        for (std::size_t cell = 0; cell < count; ++cell) {
            const auto label = labels[cell];
            const auto expected = material(label);
            require(classes[cell] == expected, "source material class mismatch");
            if (expected == std::numeric_limits<std::uint32_t>::max()) ++unresolved;
            else ++classCounts[expected];
            ++labelCounts[label];
            const auto* ids = tets.data() + 4*cell;
            for (std::size_t i = 0; i < 4; ++i) {
                require(ids[i] < points.size(), "tet node outside source nodes");
                for (std::size_t j = 0; j < i; ++j) require(ids[i] != ids[j], "repeated tetrahedron node");
            }
            const auto& origin = points[ids[0]];
            const Integer det = determinant(difference(points[ids[1]], origin),
                difference(points[ids[2]], origin), difference(points[ids[3]], origin));
            require(det > 0, "nonpositive source tetrahedron");
            determinants[label] += det;
        }
        const Integer denominator = Integer(6) << (-3*exponent);
        Rational volume(determinants[1], denominator); volume.canonicalize();
        Rational mass = volume * 1050; mass.canonicalize();
        std::cout << std::setprecision(17) << "{\"status\":\"pass\",\"cells\":" << count
            << ",\"unresolved_cells\":" << unresolved << ",\"counts_by_source_label\":{";
        for (unsigned label = 1; label <= 24; ++label) {
            if (label != 1) std::cout << ',';
            std::cout << '"' << label << "\":" << labelCounts[label];
        }
        std::cout << "},\"counts_by_class\":{";
        for (unsigned id = 0; id < 5; ++id) {
            if (id) std::cout << ',';
            std::cout << '"' << id << "\":" << classCounts[id];
        }
        std::cout << "},\"lv_geometric_mass\":{\"cell_count\":" << labelCounts[1]
            << ",\"volume_m3\":" << volume.get_d()
            << ",\"density_kg_per_m3\":1050,\"mass_kg\":" << mass.get_d()
            << ",\"volume_exact\":{\"numerator\":\"" << volume.get_num()
            << "\",\"denominator\":\"" << volume.get_den()
            << "\"},\"mass_exact\":{\"numerator\":\"" << mass.get_num()
            << "\",\"denominator\":\"" << mass.get_den()
            << "\"}},\"native_inertial_density_supplied\":false,\"blood_mass_partitioned\":false,"
            << "\"native_matter_package\":false,\"physical_steps\":0}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cardiac material attribution check failed: " << error.what() << '\n';
        return 1;
    }
}
