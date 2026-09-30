// Source-bound Apple CPU diagnostic for one LV/RV unit-diffusion update.
// The verified full-face source-node quotient is supplied by the independent
// activation gate. This is neither a voltage/ionic model nor a HumanPack step.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vec3 { double x, y, z; };
static_assert(sizeof(Vec3) == 24);

void need(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <typename T> std::vector<T> read(const std::filesystem::path& path,
                                           std::size_t expected) {
    need(std::filesystem::is_regular_file(path) &&
         std::filesystem::file_size(path) == expected * sizeof(T),
         "source input size or type changed");
    std::vector<T> result(expected);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(expected * sizeof(T)));
    need(bool(stream), "source input read failed");
    return result;
}

template <typename T> void write(const std::filesystem::path& path,
                                  const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    need(bool(stream), "cannot open diagnostic output");
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    need(bool(stream), "diagnostic output write failed");
}

Vec3 subtract(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x};
}
double dot(Vec3 a, Vec3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

constexpr std::array<std::array<unsigned, 2>, 6> edges{{
    {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}, {{1, 3}}, {{2, 3}}
}};

} // namespace

int main(int argc, char** argv) {
    try {
        need((argc == 6 || argc == 7) &&
             (argc == 6 || std::string(argv[6]) == "--block-interface"),
             "usage: probe asset-dir source-nodes.u32le dof-regions.u32le "
             "candidate.f64le candidate-source-nodes.u32le [--block-interface]");
        const bool blocked = argc == 7;
        const std::filesystem::path asset = argv[1];
        const auto nodes = read<Vec3>(asset / "nodes.f64le", 300965);
        const auto tetrahedra = read<std::uint32_t>(
            asset / "tetrahedra.u32le", 1470083u * 4u);
        const auto labels = read<std::uint32_t>(asset / "labels.u32le", 1470083);
        auto sourceNodes = read<std::uint32_t>(argv[2], 218080);
        const auto dofRegions = read<std::uint32_t>(argv[3], 218080);
        std::vector<std::int32_t> base(nodes.size(), -1);
        for (std::uint32_t index = 0; index < 218077u; ++index) {
            const auto source = sourceNodes[index];
            need(source < nodes.size() && dofRegions[index] == 0u &&
                 (index == 0u || source > sourceNodes[index-1]),
                 "unique source ventricular DOF order changed");
            base[source] = static_cast<std::int32_t>(index);
        }
        std::vector<std::int32_t> rv = base;
        for (std::uint32_t index = 218077u; index < 218080u; ++index) {
            const auto source = sourceNodes[index];
            need(source < nodes.size() && base[source] >= 0 &&
                 dofRegions[index] == 2u &&
                 (index == 218077u || source > sourceNodes[index-1]),
                 "point-only DOF record changed");
            rv[source] = static_cast<std::int32_t>(index);
        }
        std::vector<std::uint8_t> lvMask(nodes.size()), rvMask(nodes.size());
        std::size_t lvCells = 0u, rvCells = 0u;
        for (std::size_t cell = 0; cell < labels.size(); ++cell) {
            if (labels[cell] != 1u && labels[cell] != 2u) continue;
            if (labels[cell] == 1u) ++lvCells;
            else ++rvCells;
            for (unsigned corner = 0; corner < 4u; ++corner) {
                const auto node = tetrahedra[4u*cell + corner];
                need(node < nodes.size() && base[node] >= 0,
                     "ventricular cell has unmapped source node");
                (labels[cell] == 1u ? lvMask[node] : rvMask[node]) = 1u;
            }
        }
        need(lvCells == 722773u && rvCells == 374761u,
             "source ventricular cell partition changed");
        std::size_t shared = 0u, connected = 0u, pointOnly = 0u;
        for (std::uint32_t node = 0; node < nodes.size(); ++node) {
            if (!lvMask[node] || !rvMask[node]) continue;
            ++shared;
            if (rv[node] == base[node]) {
                ++connected;
                if (blocked) {
                    rv[node] = static_cast<std::int32_t>(sourceNodes.size());
                    sourceNodes.push_back(node);
                }
            } else ++pointOnly;
        }
        need(shared == 2631u && connected == 2628u && pointOnly == 3u &&
             sourceNodes.size() == (blocked ? 220708u : 218080u),
             "verified LV/RV face and point interface changed");

        std::vector<double> initial(sourceNodes.size(), 0.0);
        for (std::uint32_t index = 0; index < 218077u; ++index)
            if (lvMask[sourceNodes[index]]) initial[index] = 1.0;
        std::vector<double> capacity(sourceNodes.size(), 0.0);
        std::vector<double> rowBound(sourceNodes.size(), 0.0);
        std::vector<double> residual(sourceNodes.size(), 0.0);
        double energyBefore = 0.0;
        auto visit = [&](bool assemble) {
            double energy = 0.0;
            for (std::size_t cell = 0; cell < labels.size(); ++cell) {
                const auto label = labels[cell];
                if (label != 1u && label != 2u) continue;
                std::array<std::uint32_t, 4> mapped{};
                std::array<Vec3, 4> p{};
                for (unsigned corner = 0; corner < 4u; ++corner) {
                    const auto source = tetrahedra[4u*cell + corner];
                    mapped[corner] = static_cast<std::uint32_t>(
                        label == 1u ? base[source] : rv[source]);
                    p[corner] = nodes[source];
                }
                const double six = dot(subtract(p[1], p[0]),
                                       cross(subtract(p[2], p[0]),
                                             subtract(p[3], p[0])));
                need(std::isfinite(six) && six > 0.0,
                     "inverted source ventricular tetrahedron");
                const double volume = six / 6.0;
                if (assemble)
                    for (auto node : mapped) capacity[node] += volume / 4.0;
                for (auto pair : edges) {
                    const auto a = pair[0], b = pair[1];
                    const auto delta = subtract(p[a], p[b]);
                    const double lengthSquared = dot(delta, delta);
                    need(std::isfinite(lengthSquared) && lengthSquared > 0.0,
                         "collapsed ventricular edge");
                    const double weight = volume / (6.0 * lengthSquared);
                    const auto u = mapped[a], v = mapped[b];
                    need(u != v && std::isfinite(weight) && weight > 0.0,
                         "invalid source-connected edge");
                    if (assemble) {
                        rowBound[u] += weight;
                        rowBound[v] += weight;
                        const double flux = weight * (initial[u] - initial[v]);
                        residual[u] += flux;
                        residual[v] -= flux;
                        energy += 0.5 * weight *
                                  (initial[u]-initial[v]) *
                                  (initial[u]-initial[v]);
                    }
                }
            }
            return energy;
        };
        energyBefore = visit(true);
        double safeStep = std::numeric_limits<double>::infinity();
        for (std::size_t index = 0; index < capacity.size(); ++index) {
            need(capacity[index] > 0.0 && rowBound[index] > 0.0,
                 "unowned or isolated ventricular DOF");
            safeStep = std::min(safeStep, capacity[index] / rowBound[index]);
        }
        safeStep *= 0.25;
        need(std::isfinite(safeStep) && safeStep > 0.0,
             "no stable diagnostic step");
        std::vector<double> candidate(initial.size());
        double initialIntegral = 0.0, acceptedIntegral = 0.0;
        std::size_t changed = 0u, rvExclusiveChanged = 0u;
        double minimum = 1.0, maximum = 0.0, maximumChange = 0.0;
        for (std::size_t index = 0; index < initial.size(); ++index) {
            candidate[index] = initial[index] -
                               safeStep * residual[index] / capacity[index];
            need(std::isfinite(candidate[index]) &&
                 candidate[index] >= -1e-14 && candidate[index] <= 1.0+1e-14,
                 "diagnostic field left positive unit interval");
            initialIntegral += capacity[index] * initial[index];
            acceptedIntegral += capacity[index] * candidate[index];
            minimum = std::min(minimum, candidate[index]);
            maximum = std::max(maximum, candidate[index]);
            maximumChange = std::max(maximumChange,
                std::abs(candidate[index] - initial[index]));
            if (candidate[index] != initial[index]) ++changed;
        }
        for (std::uint32_t source = 0; source < nodes.size(); ++source)
            if (rvMask[source] && !lvMask[source] &&
                candidate[static_cast<std::uint32_t>(base[source])] > 0.0)
                ++rvExclusiveChanged;
        double energyAfter = 0.0;
        for (std::size_t cell = 0; cell < labels.size(); ++cell) {
            const auto label = labels[cell];
            if (label != 1u && label != 2u) continue;
            Vec3 p[4]; std::uint32_t mapped[4];
            for (unsigned corner = 0; corner < 4u; ++corner) {
                const auto source = tetrahedra[4u*cell + corner];
                p[corner] = nodes[source];
                mapped[corner] = static_cast<std::uint32_t>(
                    label == 1u ? base[source] : rv[source]);
            }
            const double volume = dot(subtract(p[1],p[0]),
                                      cross(subtract(p[2],p[0]),
                                            subtract(p[3],p[0]))) / 6.0;
            for (auto pair : edges) {
                const auto a = pair[0], b = pair[1];
                const auto delta = subtract(p[a],p[b]);
                const double difference = candidate[mapped[a]] -
                                          candidate[mapped[b]];
                energyAfter += 0.5 * volume * difference * difference /
                               (6.0 * dot(delta,delta));
            }
        }
        const double relativeConservationError =
            std::abs(acceptedIntegral-initialIntegral) / initialIntegral;
        need(relativeConservationError < 1e-12 &&
             (blocked ? (energyBefore == 0.0 && energyAfter == 0.0 &&
                         changed == 0u && rvExclusiveChanged == 0u) :
                        (energyBefore > 0.0 && energyAfter < energyBefore &&
                         rvExclusiveChanged > 0u)),
             "field integral, graph energy, or interface-transfer control failed");
        write(argv[4], candidate);
        write(argv[5], sourceNodes);
        std::printf("{\"schema\":\"numi.human.ventricular-unit-conduction-native.v1\","
                    "\"blocked_interface\":%s,\"apple_native_cpu_step\":true,"
                    "\"source_ventricular_tetrahedra\":%zu,\"source_lv_cells\":%zu,"
                    "\"source_rv_cells\":%zu,\"ventricular_dofs\":%zu,"
                    "\"lv_rv_shared_source_nodes\":%zu,"
                    "\"full_face_connected_interface_nodes\":%zu,"
                    "\"point_only_split_nodes\":%zu,"
                    "\"unit_diffusivity_fixture_m2_per_s\":1,"
                    "\"diagnostic_step_seconds\":%.17g,"
                    "\"initial_volume_weighted_field_m3\":%.17g,"
                    "\"accepted_volume_weighted_field_m3\":%.17g,"
                    "\"relative_conservation_error\":%.17g,"
                    "\"graph_energy_before\":%.17g,"
                    "\"graph_energy_after\":%.17g,"
                    "\"changed_dofs\":%zu,\"rv_exclusive_dofs_activated\":%zu,"
                    "\"minimum_field\":%.17g,\"maximum_field\":%.17g,"
                    "\"maximum_field_change\":%.17g,"
                    "\"production_native_electrical_steps\":0,"
                    "\"source_voltage_or_ionic_model\":false,"
                    "\"heartbeat_qualified\":false}\n",
                    blocked ? "true" : "false", lvCells + rvCells,
                    lvCells, rvCells, candidate.size(), shared, connected,
                    pointOnly, safeStep, initialIntegral, acceptedIntegral,
                    relativeConservationError, energyBefore, energyAfter, changed,
                    rvExclusiveChanged, minimum, maximum, maximumChange);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ventricular unit conduction: %s\n", error.what());
        return 1;
    }
}
