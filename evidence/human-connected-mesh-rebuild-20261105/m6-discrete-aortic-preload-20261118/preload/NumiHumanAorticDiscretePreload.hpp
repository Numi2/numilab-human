#pragma once

#include <Accelerate/Accelerate.h>
#include <simd/simd.h>
#include <iostream>

namespace numi::human::aortic_wall {
namespace discrete_preload_detail {
using Vector = simd_double3;
using Matrix = simd_double3x3;
inline Vector vector(const Vec& point) { return {point[0],point[1],point[2]}; }
inline Matrix edges(const std::vector<Vector>& points,
                    const std::array<std::uint32_t,4>& n) {
    const auto a = points[n[0]];
    return {points[n[1]] - a, points[n[2]] - a, points[n[3]] - a};
}
inline double euclideanNorm(const std::vector<double>& r) {
    double total = 0.0;
    for (const double v : r) total += v*v;
    return std::sqrt(total);
}
struct Cell {
    std::array<std::uint32_t,4> nodes{};
    Matrix currentEdges{};
    std::array<Vector,4> currentGradients{};
    double currentVolume = 0.0;
};
} // namespace discrete_preload_detail

struct DiscretePreloadResult {
    double initialResidualN = 0.0;
    double finalResidualN = 0.0;
    double maximumReferenceChangeM = 0.0;
    double minimumDeterminant = 0.0;
    unsigned acceptedIterations = 0u;
};

// Inverse elastostatics at a fixed, registered current geometry. Update the
// stress-free reference coordinates so the displacement FEM residual balances
// the initial physical pressure and gravity. This runs only in asset prep.
// The pressure, geometry, material law, element mass, and anchors are inputs.
inline DiscretePreloadResult fitDiscreteReference(
    ObjectSource& object, const VascularCavitySource& cavity,
    std::span<const NMTetrahedronGPU> compiledTets,
    std::span<const NMFEMNodeStateGPU> compiledNodes,
    const Vec& gravity, const double externalPressurePa,
    const Config& config = {}) {
    using namespace discrete_preload_detail;
    const auto count = object.femNodes.size();
    if (count != compiledNodes.size() || count != object.femReferenceNodes.size() ||
        object.tetrahedra.size() != compiledTets.size() || count > 1024u ||
        object.femFixedNodes.empty() || cavity.faces.empty())
        throw std::runtime_error("discrete wall fit requires the exact small compiled wall");
    std::vector<Vector> x, reference, load(count);
    std::vector<double> mass;
    std::vector<int> base(count,-1);
    x.reserve(count);
    reference.reserve(count);
    int n = 0;
    for (std::size_t i = 0; i < count; ++i) {
        const auto& p = compiledNodes[i].positionAndMass;
        x.push_back(Vector{p.x,p.y,p.z});
        reference.push_back(vector(object.femReferenceNodes[i]));
        load[i] = double(p.w) * vector(gravity);
        if (compiledNodes[i].restAndFixed.w == 0.0f) {
            base[i] = n;
            n += 3;
        }
    }
    if (n == 0) throw std::runtime_error("discrete wall fit has no free nodes");

    const auto origin = x[cavity.faces.front().nodes[0]];
    const double pressure = cavity.initialPressure - externalPressurePa;
    for (const auto& face : cavity.faces) {
        for (unsigned corner = 0; corner < 3; ++corner) {
            load[face.nodes[corner]] += (pressure / 6.0) * simd_cross(
                x[face.nodes[(corner+1u)%3u]] - origin,
                x[face.nodes[(corner+2u)%3u]] - origin);
        }
    }

    std::vector<Cell> cells;
    cells.reserve(compiledTets.size());
    mass.reserve(compiledTets.size());
    for (std::size_t i = 0; i < compiledTets.size(); ++i) {
        const auto& nodes = object.tetrahedra[i].nodes;
        const auto h = edges(x,nodes);
        const double v = simd_determinant(h) / 6.0;
        if (!(v > 0.0))
            throw std::runtime_error("discrete wall fit encountered inverted current geometry");
        const auto gradients = simd_transpose(simd_inverse(h));
        const auto a = gradients.columns[0];
        const auto b = gradients.columns[1];
        const auto c = gradients.columns[2];
        cells.push_back({nodes,h,{-a-b-c,a,b,c},v});
        mass.push_back(config.densityKgM3 * double(compiledTets[i].inverseRestRow0.w));
    }

    const double mu = config.shearModulusPa, lambda = config.bulkPenaltyPa;
    std::vector<double> residual(std::size_t(n),0.0), candidateResidual(std::size_t(n),0.0);
    std::vector<double> jacobian(std::size_t(n)*n,0.0);
    const Matrix identity = matrix_identity_double3x3;
    auto evaluate = [&](const std::vector<Vector>& y, std::vector<double>& r,
                        double* derivative, double& minJ) {
        for (std::size_t i = 0; i < count; ++i) {
            if (base[i] == -1) continue;
            for (int a = 0; a < 3; ++a) r[base[i]+a] = load[i][a];
        }
        if (derivative != nullptr)
            std::fill(derivative,derivative + std::size_t(n)*n,0.0);
        minJ = std::numeric_limits<double>::infinity();
        for (const auto& cell : cells) {
            const auto ref = edges(y,cell.nodes);
            const auto inverse = simd_inverse(ref);
            const auto F = simd_mul(cell.currentEdges,inverse);
            const double J = simd_determinant(F);
            if (!(J > 0.5 && J < 1.8)) return false;
            minJ = std::min(minJ,J);
            const auto Ft = simd_transpose(F);
            const auto tau = simd_add(simd_mul(mu,simd_sub(simd_mul(F,Ft),identity)),
                simd_mul(lambda * std::log(J),identity));
            const double volume = cell.currentVolume / J;
            for (unsigned i = 0; i < 4; ++i) {
                const int row = base[cell.nodes[i]];
                if (row == -1) continue;
                const auto f = -volume * simd_mul(tau,cell.currentGradients[i]);
                for (int a = 0; a < 3; ++a) r[row+a] += f[a];
            }
            if (derivative == nullptr) continue;
            for (unsigned j = 0; j < 4; ++j) {
                const int col = base[cell.nodes[j]];
                if (col == -1) continue;
                for (int b = 0; b < 3; ++b) {
                    Vector unit{0.0,0.0,0.0};
                    unit[b] = 1.0;
                    const Vector v0 = j == 0u ? -unit : (j == 1u ? unit : Vector{});
                    const Vector v1 = j == 0u ? -unit : (j == 2u ? unit : Vector{});
                    const Vector v2 = j == 0u ? -unit : (j == 3u ? unit : Vector{});
                    const Matrix dref{v0,v1,v2};
                    const auto transform = simd_mul(dref,inverse);
                    const double dlogJ = -(transform.columns[0].x +
                        transform.columns[1].y + transform.columns[2].z);
                    const auto dF = simd_mul(-1.0,simd_mul(F,transform));
                    const auto dB = simd_mul(dF,Ft);
                    const auto integratedTangent =
                        simd_add(simd_mul(mu,simd_add(dB,simd_transpose(dB))),
                        simd_mul(dlogJ,simd_sub(simd_mul(lambda,identity),tau)));
                    for (unsigned i = 0; i < 4; ++i) {
                        const int row = base[cell.nodes[i]];
                        if (row == -1) continue;
                        const auto df = -volume *
                            simd_mul(integratedTangent,cell.currentGradients[i]);
                        for (int a = 0; a < 3; ++a)
                            derivative[std::size_t(col+b)*n+row+a] += df[a];
                    }
                }
            }
        }
        return true;
    };

    const auto initial = reference;
    DiscretePreloadResult result;
    if (!evaluate(reference,residual,nullptr,result.minimumDeterminant))
        throw std::runtime_error("analytic wall reference leaves discrete determinant trust range");
    result.initialResidualN = euclideanNorm(residual);
    result.finalResidualN = result.initialResidualN;
    std::vector<int> pivots(std::size_t(n),0);
    auto trial = reference;
    for (unsigned iteration = 0; iteration < 20 && result.finalResidualN > 1e-8; ++iteration) {
        if (!evaluate(reference,residual,jacobian.data(),result.minimumDeterminant))
            throw std::runtime_error("admitted discrete wall reference became invalid");
        auto correction = residual;
        for (double& value : correction) value = -value;
        int nrhs = 1, info = 0;
        dgesv_(&n,&nrhs,jacobian.data(),&n,pivots.data(),correction.data(),&n,&info);
        if (info != 0) throw std::runtime_error("discrete wall reference Newton matrix is singular");
        bool advanced = false;
        for (double step = 1.0; step >= 1.0/128.0; step *= 0.5) {
            double maximum = 0.0;
            for (std::size_t i = 0; i < count; ++i) {
                if (base[i] == -1) continue;
                const auto* d = correction.data()+base[i];
                trial[i] = reference[i] + step * Vector{d[0],d[1],d[2]};
                maximum = std::max(maximum,simd_length(trial[i] - initial[i]));
            }
            double minJ = 0.0;
            if (maximum > 0.015 || !evaluate(trial,candidateResidual,nullptr,minJ))
                continue;
            const double norm = euclideanNorm(candidateResidual);
            if (!(norm < result.finalResidualN)) continue;
            result.finalResidualN = norm;
            result.minimumDeterminant = minJ;
            result.maximumReferenceChangeM = maximum;
            ++result.acceptedIterations;
            reference.swap(trial);
            advanced = true;
            break;
        }
        if (!advanced) break;
        std::cerr << "wall_inverse_iter count=" << result.acceptedIterations
            << " residual_N=" << result.finalResidualN
            << " minimum_J=" << result.minimumDeterminant
            << " max_reference_delta_m=" << result.maximumReferenceChangeM << std::endl;
    }
    if (!(result.finalResidualN < 1e-5))
        throw std::runtime_error("discrete wall reference failed nodal load balance: " +
            std::to_string(result.initialResidualN) + " -> " +
            std::to_string(result.finalResidualN));

    for (std::size_t i = 0; i < count; ++i)
        object.femReferenceNodes[i] = {reference[i].x,reference[i].y,reference[i].z};
    object.femReferenceSourceIdentity[3] = 0x4449534352455445ULL; // DISCRETE
    object.femTetrahedronMassesKg = std::move(mass);
    object.femTetrahedronMassSourceIdentity = {
        0x414f5254414d4153ULL,0x5352455441494e45ULL,
        0x44414e414c595449ULL,0x4357414c4c303031ULL};
    return result;
}
} // namespace numi::human::aortic_wall
