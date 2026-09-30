// Transfer the pinned full-source active-force reference to the corrected
// 218080-node ventricular mechanical quotient. This is an offline reference,
// not a native accepted force or a physiological heartbeat.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Vec3 { double x, y, z; };
static_assert(sizeof(Vec3) == 3*sizeof(double));
Vec3 operator+(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 operator*(Vec3 a, double b) { return {a.x*b, a.y*b, a.z*b}; }
Vec3& operator+=(Vec3& a, Vec3 b) { a=a+b; return a; }
Vec3& operator-=(Vec3& a, Vec3 b) { a=a-b; return a; }
double dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
double length(Vec3 a) { return std::sqrt(dot(a,a)); }

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <typename T>
std::vector<T> read(const std::filesystem::path& path, std::size_t count) {
    require(std::filesystem::is_regular_file(path) &&
            std::filesystem::file_size(path) == count*sizeof(T),
            "source or reference input size changed");
    std::vector<T> result(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(result.data()),
                static_cast<std::streamsize>(count*sizeof(T)));
    require(bool(stream), "source or reference input read failed");
    return result;
}

double tanhStress(double now, double arrival, double peak, double delay,
                  double contraction, double relaxation, double duration) {
    const double elapsed = now-arrival-delay;
    if (!(elapsed > 0.0 && elapsed < duration)) return 0.0;
    const double rise = std::tanh(elapsed/contraction);
    const double fall = std::tanh((duration-elapsed)/relaxation);
    return peak*rise*rise*fall*fall;
}

} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 12,
            "usage: split-active-force asset-dir activation-dir old-residual.f64le "
            "cooked-source-nodes.u32le output.f64le time-ms peak-pa delay-ms "
            "contraction-ms relaxation-ms duration-ms");
        const std::filesystem::path asset = argv[1], activation = argv[2];
        const auto old = read<Vec3>(argv[3], 218077u);
        const auto cookedSources = read<std::uint32_t>(argv[4], 218080u);
        const std::filesystem::path output = argv[5];
        const double now = std::strtod(argv[6], nullptr);
        const double peak = std::strtod(argv[7], nullptr);
        const double delay = std::strtod(argv[8], nullptr);
        const double contraction = std::strtod(argv[9], nullptr);
        const double relaxation = std::strtod(argv[10], nullptr);
        const double duration = std::strtod(argv[11], nullptr);
        require(std::isfinite(now) && now >= 0.0 && std::isfinite(peak) &&
                peak > 0.0 && std::isfinite(delay) && delay >= 0.0 &&
                std::isfinite(contraction) && contraction > 0.0 &&
                std::isfinite(relaxation) && relaxation > 0.0 &&
                std::isfinite(duration) && duration > 0.0,
                "invalid source active-force parameters");
        const auto positions = read<Vec3>(asset/"nodes.f64le", 300965u);
        const auto cells = read<std::uint32_t>(asset/"tetrahedra.u32le", 1470083u*4u);
        const auto labels = read<std::uint32_t>(asset/"labels.u32le", 1470083u);
        const auto fibres = read<Vec3>(asset/"fibres.f64le", 1470083u);
        const auto sourceNodes = read<std::uint32_t>(
            activation/"ventricular-source-nodes.u32le", 218080u);
        const auto regions = read<std::uint32_t>(
            activation/"ventricular-dof-regions.u32le", 218080u);
        const auto arrival = read<double>(activation/"refined-arrival.f64le", 218080u);
        constexpr std::array<std::uint32_t, 3> pointOnly{17565u, 170947u, 235754u};
        std::vector<std::int32_t> sourceLut(positions.size(), -1);
        std::vector<std::int32_t> cookedLut(positions.size(), -1);
        for (std::uint32_t i=0; i<218077u; ++i) {
            const auto source = sourceNodes[i];
            require(source < positions.size() && regions[i] == 0u &&
                    (i == 0u || source > sourceNodes[i-1u]) &&
                    std::isfinite(arrival[i]) && arrival[i] >= 0.0,
                    "source electrical quotient changed");
            sourceLut[source] = static_cast<std::int32_t>(i);
        }
        for (std::uint32_t i=0; i<218080u; ++i) {
            const auto source = cookedSources[i];
            require(source < positions.size(), "cooked source node out of range");
            if (i < 218077u) {
                require(sourceLut[source] >= 0 && cookedLut[source] == -1,
                        "cooked base node permutation changed");
                cookedLut[source] = static_cast<std::int32_t>(i);
            } else {
                require(source == pointOnly[i-218077u] &&
                        sourceNodes[i] == source && regions[i] == 2u &&
                        std::isfinite(arrival[i]) && arrival[i] >= 0.0,
                        "point-only RV node identity changed");
            }
        }
        std::vector<Vec3> split(218080u, Vec3{0.0,0.0,0.0});
        for (std::uint32_t i=0; i<218077u; ++i) {
            const auto source = cookedSources[i];
            split[i] = old[static_cast<std::uint32_t>(sourceLut[source])];
        }
        const double qa=(1.0+3.0/std::sqrt(5.0))/4.0;
        const double qb=(1.0-qa)/3.0;
        std::array<std::uint32_t, 3> incident{};
        std::array<Vec3, 3> transferred{};
        for (std::size_t cell=0; cell<labels.size(); ++cell) {
            if (labels[cell] != 2u) continue;
            int target = -1, cornerTarget = -1;
            for (std::uint32_t corner=0; corner<4u; ++corner)
                for (std::uint32_t point=0; point<pointOnly.size(); ++point)
                    if (cells[4u*cell+corner] == pointOnly[point]) {
                        require(target == -1, "RV cell contains multiple point-only nodes");
                        target = static_cast<int>(point);
                        cornerTarget = static_cast<int>(corner);
                    }
            if (target < 0) continue;
            Vec3 p[4];
            double times[4];
            for (std::uint32_t corner=0; corner<4u; ++corner) {
                const auto source = cells[4u*cell+corner];
                require(source < positions.size() && sourceLut[source] >= 0,
                        "RV active cell source node missing");
                p[corner] = positions[source];
                std::uint32_t dof = static_cast<std::uint32_t>(sourceLut[source]);
                if (source == pointOnly[target]) dof = 218077u+target;
                times[corner] = 1000.0*arrival[dof];
            }
            const Vec3 e1=p[1]-p[0], e2=p[2]-p[0], e3=p[3]-p[0];
            const double determinant = dot(e1, cross(e2,e3));
            require(std::isfinite(determinant) && determinant > 0.0,
                    "RV point-only cell has inverted geometry");
            Vec3 fibre = fibres[cell];
            const double norm = length(fibre);
            require(std::isfinite(norm) && norm > 0.0,
                    "RV point-only cell has invalid fibre");
            fibre = fibre*(1.0/norm);
            const double timeSum = times[0]+times[1]+times[2]+times[3];
            double stress=0.0;
            for (std::uint32_t corner=0; corner<4u; ++corner)
                stress += tanhStress(now, qb*timeSum+(qa-qb)*times[corner],
                                     peak, delay, contraction, relaxation,
                                     duration)/4.0;
            const std::array<Vec3, 4> gradients{
                (cross(e2,e3)+cross(e3,e1)+cross(e1,e2))*(-1.0/determinant),
                cross(e2,e3)*(1.0/determinant),
                cross(e3,e1)*(1.0/determinant),
                cross(e1,e2)*(1.0/determinant),
            };
            const Vec3 contribution = fibre *
                ((determinant/6.0)*stress*dot(fibre,gradients[cornerTarget]));
            const auto left = static_cast<std::uint32_t>(
                cookedLut[pointOnly[target]]);
            const auto right = 218077u+static_cast<std::uint32_t>(target);
            split[left] -= contribution;
            split[right] += contribution;
            transferred[target] += contribution;
            ++incident[target];
        }
        require(incident == std::array<std::uint32_t,3>{1u,1u,1u},
                "RV point-only incidence changed");
        std::ofstream stream(output, std::ios::binary | std::ios::trunc);
        require(bool(stream), "cannot open split force output");
        stream.write(reinterpret_cast<const char*>(split.data()),
                     static_cast<std::streamsize>(split.size()*sizeof(Vec3)));
        require(bool(stream), "cannot write split force output");
        std::printf("{\"status\":\"source_active_force_point_contact_split\","
                    "\"frame_ms\":%.17g,\"ventricular_nodes\":%zu,"
                    "\"rv_point_only_cells\":3,\"rv_transfer_force_norm_n\":["
                    "%.17g,%.17g,%.17g],"
                    "\"native_accepted_steps\":0,\"heartbeat_qualified\":false}\n",
                    now, split.size(), length(transferred[0]),
                    length(transferred[1]), length(transferred[2]));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "split active force reference: %s\n", error.what());
        return 1;
    }
}
