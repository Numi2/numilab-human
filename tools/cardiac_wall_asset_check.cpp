// Independent offline check of HumanPack.cardiac-wall-source-asset.v1 buffers.
// C++20, standard library only. No mechanics solver, source repair, or stepping.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fs = std::filesystem;
constexpr std::size_t maximumNodes = 1000000;
constexpr std::size_t maximumCells = 5000000;
constexpr double volumeTolerance = 1e-10;
using Point = std::array<double, 3>;
using Face = std::array<std::uint32_t, 3>;
void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
struct Sum {
    double main = 0, correction = 0;
    void add(double value) {
        require(std::isfinite(value), "nonfinite geometric summand");
        const double next = main + value;
        correction += std::abs(main) >= std::abs(value)
            ? (main - next) + value : (value - next) + main;
        main = next;
    }
    double value() const {
        const double result = main + correction;
        require(std::isfinite(result), "nonfinite compensated geometric sum");
        return result;
    }
};
std::string quoted(const std::string& value) {
    std::ostringstream output;
    output << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') output << '\\' << c;
        else if (c < 32) output << "\\u" << std::hex << std::setfill('0') << std::setw(4) << unsigned(c) << std::dec;
        else output << c;
    }
    output << '"';
    return output.str();
}
template<class T> std::vector<T> read(const fs::path& root, const char* name,
                                    std::size_t maximumCount) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    const fs::path path = root / name;
    const auto status = fs::symlink_status(path);
    require(fs::is_regular_file(status) && !fs::is_symlink(status), std::string("missing or nonregular buffer: ") + name);
    const auto bytes = fs::file_size(path);
    require(bytes % sizeof(T) == 0, std::string("truncated scalar buffer: ") + name);
    require(bytes / sizeof(T) <= maximumCount, std::string("buffer capacity exceeded: ") + name);
    std::vector<T> values(static_cast<std::size_t>(bytes / sizeof(T)));
    std::ifstream input(path, std::ios::binary);
    require(bool(input), std::string("cannot open buffer: ") + name);
    if (bytes) {
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
        require(input.gcount() == static_cast<std::streamsize>(bytes), std::string("truncated buffer read: ") + name);
    }
    require(input.peek() == std::char_traits<char>::eof(), std::string("buffer changed while reading: ") + name);
    if constexpr (std::endian::native == std::endian::big) {
        for (auto& value : values) {
            auto* first = reinterpret_cast<unsigned char*>(&value);
            std::reverse(first, first + sizeof(T));
        }
    }
    return values;
}
void finiteField(const fs::path& root, const char* name, std::size_t count) {
    const auto values = read<double>(root, name, count);
    require(values.size() == count, std::string("wrong field length: ") + name);
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) require(false, std::string("nonfinite field: ") + name + " scalar=" + std::to_string(i));
}
Point subtract(const Point& a, const Point& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
double determinant(const Point& a, const Point& b, const Point& c) {
    return a[0]*(b[1]*c[2]-b[2]*c[1]) + a[1]*(b[2]*c[0]-b[0]*c[2]) + a[2]*(b[0]*c[1]-b[1]*c[0]);
}
struct FaceRecord {
    Face key;
    std::uint32_t encoded;
};
static_assert(sizeof(FaceRecord) == 16);
Face outwardFace(const std::vector<std::uint32_t>& cells, std::uint32_t encoded) {
    const auto cell = encoded / 4, omitted = encoded % 4;
    Face face{};
    unsigned index = 0;
    for (unsigned local = 0; local < 4; ++local)
        if (local != omitted) face[index++] = cells[std::size_t(cell)*4+local];
    // Alternating simplex boundary orientation for a positive determinant.
    if (omitted % 2) std::swap(face[1], face[2]);
    return face;
}
unsigned parity(const Face& face) {
    return (unsigned(face[0] > face[1]) + unsigned(face[0] > face[2]) + unsigned(face[1] > face[2])) % 2;
}
Face sorted(Face face) { std::sort(face.begin(), face.end()); return face; }
struct Region {
    std::uint64_t cells = 0, boundaryFaces = 0;
    Sum tetrahedra, boundary, absoluteBoundary;
};
struct EdgeRecord {
    std::uint32_t a, b, face;
    bool increasing;
};
struct Components {
    std::vector<std::uint32_t> parent;
    explicit Components(std::size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    std::uint32_t find(std::uint32_t i) {
        while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
        return i;
    }
    void join(std::uint32_t a, std::uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) parent[std::max(a,b)] = std::min(a,b);
    }
};
double conditionedError(double volume, double boundary, double absoluteBoundary) {
    const double scale = std::abs(volume) + absoluteBoundary;
    require(std::isfinite(scale) && scale > 0, "invalid volume comparison conditioning scale");
    return std::abs(volume-boundary) / scale;
}
void check(const fs::path& root) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
    require(fs::is_directory(root) && !fs::is_symlink(fs::symlink_status(root)), "asset must be a real directory");
    const auto coordinates = read<double>(root, "nodes.f64le", maximumNodes*3);
    require(coordinates.size() % 3 == 0 && coordinates.size() >= 12, "wrong node coordinate shape");
    const std::size_t nodeCount = coordinates.size()/3;
    Point minimum{INFINITY, INFINITY, INFINITY}, maximum{-INFINITY, -INFINITY, -INFINITY};
    for (std::size_t i = 0; i < coordinates.size(); ++i) {
        if (!std::isfinite(coordinates[i])) require(false, "nonfinite node coordinate scalar="+std::to_string(i));
        minimum[i%3] = std::min(minimum[i%3], coordinates[i]);
        maximum[i%3] = std::max(maximum[i%3], coordinates[i]);
    }
    const Point origin{std::midpoint(minimum[0],maximum[0]), std::midpoint(minimum[1],maximum[1]), std::midpoint(minimum[2],maximum[2])};
    auto point = [&](std::uint32_t node) -> Point {
        require(node < nodeCount, "node index outside coordinate buffer");
        return {coordinates[3*std::size_t(node)], coordinates[3*std::size_t(node)+1], coordinates[3*std::size_t(node)+2]};
    };
    auto surfaceTerm = [&](const Face& face) {
        return determinant(subtract(point(face[0]),origin), subtract(point(face[1]),origin), subtract(point(face[2]),origin))/6.;
    };
    const auto cells = read<std::uint32_t>(root, "tetrahedra.u32le", maximumCells*4);
    require(!cells.empty() && cells.size()%4 == 0, "wrong tetrahedron buffer shape");
    const std::size_t cellCount = cells.size()/4;
    const auto labels = read<std::uint32_t>(root, "labels.u32le", cellCount);
    require(labels.size() == cellCount, "label count differs from tetrahedron count");
    const auto reversed = read<std::uint32_t>(root, "source_reversed_cells.u32le", cellCount);
    for (std::size_t i = 0; i < reversed.size(); ++i)
        require(reversed[i] < cellCount && (i == 0 || reversed[i] > reversed[i-1]), "invalid or repeated source reversal index");
    std::array<Region,25> regions{};
    Sum total;
    double minimumVolume = INFINITY, maximumVolume = 0;
    std::vector<FaceRecord> allFaces;
    allFaces.reserve(cellCount*4);
    std::vector<unsigned char> used(nodeCount);
    for (std::uint32_t cell = 0; cell < cellCount; ++cell) {
        const std::array<std::uint32_t,4> indices{cells[4*std::size_t(cell)],cells[4*std::size_t(cell)+1],cells[4*std::size_t(cell)+2],cells[4*std::size_t(cell)+3]};
        for (unsigned i = 0; i < 4; ++i) {
            if (indices[i] >= nodeCount) require(false, "out-of-range tetrahedron index cell="+std::to_string(cell));
            used[indices[i]] = 1;
            for (unsigned j = 0; j < i; ++j) if (indices[i] == indices[j]) require(false, "repeated vertex in tetrahedron cell="+std::to_string(cell));
        }
        const Point a = point(indices[0]);
        const double volume = determinant(subtract(point(indices[1]),a),subtract(point(indices[2]),a),subtract(point(indices[3]),a))/6.;
        if (!std::isfinite(volume) || volume <= 0) require(false, "nonpositive or nonfinite tetrahedron volume cell="+std::to_string(cell));
        if (labels[cell] < 1 || labels[cell] > 24) require(false, "unknown anatomical region cell="+std::to_string(cell));
        auto& region = regions[labels[cell]];
        ++region.cells; region.tetrahedra.add(volume); total.add(volume);
        minimumVolume = std::min(minimumVolume,volume); maximumVolume = std::max(maximumVolume,volume);
        for (unsigned face = 0; face < 4; ++face) {
            const std::uint32_t encoded = cell*4+face;
            allFaces.push_back({sorted(outwardFace(cells,encoded)),encoded});
        }
    }
    require(std::all_of(used.begin(),used.end(),[](auto value){return value != 0;}), "unused source node");
    std::sort(allFaces.begin(),allFaces.end(),[](const auto& a,const auto& b){ return a.key != b.key ? a.key < b.key : a.encoded < b.encoded; });
    std::vector<unsigned char> exterior(cellCount*4);
    std::uint64_t exteriorCount = 0, interiorCount = 0, materialInterfaceCount = 0;
    auto regionFace = [&](std::uint32_t encoded) {
        auto& region = regions[labels[encoded/4]];
        const double term = surfaceTerm(outwardFace(cells,encoded));
        region.boundary.add(term); region.absoluteBoundary.add(std::abs(term)); ++region.boundaryFaces;
    };
    for (std::size_t i = 0; i < allFaces.size();) {
        std::size_t end = i+1;
        while (end < allFaces.size() && allFaces[end].key == allFaces[i].key) ++end;
        require(end-i <= 2, "nonmanifold tetrahedral face");
        const auto first = allFaces[i].encoded;
        if (end-i == 1) {
            exterior[first] = 1; ++exteriorCount; regionFace(first);
        } else {
            const auto second = allFaces[i+1].encoded;
            require(parity(outwardFace(cells,first)) != parity(outwardFace(cells,second)), "same-side or duplicated tetrahedra at shared face");
            ++interiorCount;
            if (labels[first/4] != labels[second/4]) { ++materialInterfaceCount; regionFace(first); regionFace(second); }
        }
        i = end;
    }
    // Release the large face-sort arena before reading all surface buffers.
    std::vector<FaceRecord>().swap(allFaces);
    const auto boundary = read<std::uint32_t>(root,"boundary.u32le",cellCount*12);
    require(!boundary.empty() && boundary.size()%3 == 0, "wrong boundary shape");
    const std::size_t boundaryCount = boundary.size()/3;
    require(boundaryCount == exteriorCount, "boundary is not the complete tetrahedral material exterior");
    const auto owners = read<std::uint32_t>(root,"boundary_owners.u32le",boundaryCount);
    const auto componentIds = read<std::uint32_t>(root,"boundary_components.u32le",boundaryCount);
    require(owners.size() == boundaryCount && componentIds.size() == boundaryCount, "boundary owner/component length mismatch");
    std::vector<unsigned char> seen(cellCount*4);
    Sum boundaryVolume, absoluteBoundary;
    std::vector<EdgeRecord> edges;
    edges.reserve(boundaryCount*3);
    for (std::uint32_t i = 0; i < boundaryCount; ++i) {
        const Face face{boundary[3*std::size_t(i)],boundary[3*std::size_t(i)+1],boundary[3*std::size_t(i)+2]};
        require(owners[i] < cellCount && componentIds[i] < boundaryCount, "out-of-range boundary owner/component");
        require(face[0] != face[1] && face[1] != face[2] && face[0] != face[2], "repeated boundary face vertex");
        for (const auto node : face) require(node < nodeCount, "out-of-range boundary vertex");
        const Face key = sorted(face);
        std::uint32_t encoded = std::numeric_limits<std::uint32_t>::max();
        for (unsigned local = 0; local < 4; ++local) {
            const auto candidate = owners[i]*4+local;
            if (sorted(outwardFace(cells,candidate)) == key) { encoded = candidate; break; }
        }
        require(encoded != std::numeric_limits<std::uint32_t>::max(), "boundary face does not belong to named owner");
        require(exterior[encoded] && !seen[encoded], "boundary face is interior or duplicated");
        require(parity(face) == parity(outwardFace(cells,encoded)), "boundary face is not material-outward");
        seen[encoded] = 1;
        const double term = surfaceTerm(face);
        boundaryVolume.add(term); absoluteBoundary.add(std::abs(term));
        for (unsigned k = 0; k < 3; ++k) {
            const auto a = face[k], b = face[(k+1)%3];
            edges.push_back({std::min(a,b),std::max(a,b),i,a < b});
        }
    }
    require(seen == exterior, "missing material exterior face");
    std::sort(edges.begin(),edges.end(),[](const auto& a,const auto& b){return std::array{a.a,a.b,a.face} < std::array{b.a,b.b,b.face};});
    Components connected(boundaryCount);
    std::uint64_t edgeDefects = 0;
    for (std::size_t i = 0; i < edges.size();) {
        std::size_t end = i+1;
        while (end < edges.size() && edges[end].a == edges[i].a && edges[end].b == edges[i].b) ++end;
        if (end-i != 2 || edges[i].increasing == edges[i+1].increasing) ++edgeDefects;
        for (std::size_t j = i+1; j < end; ++j) connected.join(edges[i].face,edges[j].face);
        i = end;
    }
    std::vector<std::uint32_t> canonicalIds(boundaryCount,std::numeric_limits<std::uint32_t>::max());
    std::uint32_t componentCount = 0;
    for (std::uint32_t i = 0; i < boundaryCount; ++i) {
        const auto representative = connected.find(i);
        if (canonicalIds[representative] == std::numeric_limits<std::uint32_t>::max()) canonicalIds[representative] = componentCount++;
        require(componentIds[i] == canonicalIds[representative], "boundary component IDs disagree with independent connectivity");
    }
    for (const auto* name : {"fibres.f64le","sheets.f64le"}) finiteField(root,name,cellCount*3);
    for (const auto* name : {"uvc_rho.f64le","uvc_phi.f64le","uvc_z.f64le","uvc_v.f64le"}) finiteField(root,name,nodeCount);
    const double globalError = conditionedError(total.value(),boundaryVolume.value(),absoluteBoundary.value());
    require(globalError <= volumeTolerance, "conditioned material boundary/tetrahedron volume mismatch");
    double maximumRegionalError = 0;
    for (unsigned label = 1; label <= 24; ++label) if (regions[label].cells) {
        const auto& region = regions[label];
        const double error = conditionedError(region.tetrahedra.value(),region.boundary.value(),region.absoluteBoundary.value());
        maximumRegionalError = std::max(maximumRegionalError,error);
        require(error <= volumeTolerance, "conditioned regional boundary/tetrahedron volume mismatch label="+std::to_string(label));
    }
    std::cout << std::setprecision(17)
        << "{\"schema\":\"HumanPack.cardiac-wall-source-check.v1\",\"status\":\"pass\",\"arithmetic\":\"IEEE754 binary64 with Neumaier sums\","
        << "\"nodes\":" << nodeCount << ",\"tetrahedra\":" << cellCount << ",\"boundary_faces\":" << boundaryCount
        << ",\"interior_faces\":" << interiorCount << ",\"material_interface_faces\":" << materialInterfaceCount
        << ",\"boundary_components\":" << componentCount << ",\"boundary_edge_manifold_orientation_defects\":" << edgeDefects
        << ",\"reversed_source_cells\":" << reversed.size() << ",\"minimum_tetrahedron_volume_m3\":" << minimumVolume
        << ",\"maximum_tetrahedron_volume_m3\":" << maximumVolume << ",\"total_tetrahedron_volume_m3\":" << total.value()
        << ",\"material_boundary_signed_volume_m3\":" << boundaryVolume.value()
        << ",\"material_boundary_absolute_terms_m3\":" << absoluteBoundary.value()
        << ",\"conditioned_volume_error\":" << globalError << ",\"maximum_regional_conditioned_volume_error\":" << maximumRegionalError
        << ",\"conditioned_relative_tolerance\":" << volumeTolerance
        << ",\"conditioning_definition\":\"abs(tetra_volume-boundary_volume)/(abs(tetra_volume)+sum(abs(boundary_terms)))\",\"regions\":[";
    bool first = true;
    for (unsigned label = 1; label <= 24; ++label) if (regions[label].cells) {
        if (!first) std::cout << ',';
        first = false;
        const auto& region = regions[label];
        std::cout << "{\"label\":" << label << ",\"cells\":" << region.cells << ",\"volume_m3\":" << region.tetrahedra.value()
            << ",\"region_boundary_faces_including_interfaces\":" << region.boundaryFaces
            << ",\"region_boundary_signed_volume_m3\":" << region.boundary.value()
            << ",\"region_boundary_absolute_terms_m3\":" << region.absoluteBoundary.value() << '}';
    }
    std::cout << "],\"all_field_buffers_finite\":true,\"physical_steps\":0,\"native_matter_runtime_qualified\":false,"
        << "\"global_embedding_checked\":false,\"boundary_vertex_manifold_checked\":false,\"source_hashes_checked\":false,"
        << "\"frame_orthonormality_required\":false,\"blood_volume_or_mass_assigned\":false}\n";
}
} // namespace
int main(int argc,char** argv) {
    try {
        require(argc == 2, "usage: cardiac-wall-asset-check ASSET_DIRECTORY");
        check(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cout << "{\"schema\":\"HumanPack.cardiac-wall-source-check.v1\",\"status\":\"failed\",\"error\":"
            << quoted(error.what()) << ",\"physical_steps\":0,\"native_matter_runtime_qualified\":false}\n";
        return 1;
    }
}
