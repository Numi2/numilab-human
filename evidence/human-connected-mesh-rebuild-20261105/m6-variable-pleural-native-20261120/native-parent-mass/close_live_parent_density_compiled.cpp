#include "NumiHumanConservativeSpatialMassTransport.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {
template<class T>
std::vector<T> read(std::istream& stream, std::size_t count) {
    std::vector<T> values(count);
    if (!stream.read(reinterpret_cast<char*>(values.data()), count * sizeof(T)))
        throw std::runtime_error("truncated flow problem");
    return values;
}

template<class T>
void write(std::ostream& stream, const std::vector<T>& values) {
    if (!stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T)))
        throw std::runtime_error("cannot write flow result");
}
}

int main(int argc, char** argv) try {
    using namespace metalrobo::human::coarse_tissue::spatial_mass_transport_detail;
    if (argc != 4) return 2;
    std::ifstream input(argv[1], std::ios::binary);
    const auto counts = read<std::uint32_t>(input, 3);
    const auto carriers = counts[0], groups = counts[1], edges = counts[2];
    const auto donor = read<std::uint32_t>(input, edges);
    const auto receiver = read<std::uint32_t>(input, edges);
    auto mass = read<double>(input, edges);
    const auto quota = read<double>(input, carriers);
    const auto lower = read<double>(input, groups);
    const auto upper = read<double>(input, groups);
    const auto compensation = read<std::uint8_t>(input, edges);
    const auto source = carriers + groups, sink = source + 1u;
    std::vector<double> assigned(groups), unmet(groups);
    for (std::size_t i = 0; i < edges; ++i) {
        if (donor[i] >= carriers || receiver[i] >= groups || !std::isfinite(mass[i]) || mass[i] < 0)
            throw std::runtime_error("invalid route");
        assigned[receiver[i]] += mass[i];
    }
    std::vector<std::uint8_t> cut(sink + 1u);
    for (const bool fillMinimum : {false, true}) {
        MinCostFlow solver(sink + 1u);
        std::vector<EdgeRef> add(edges), remove(edges), obligation(groups);
        for (std::size_t i = 0; i < edges; ++i) {
            const auto c = donor[i], g = carriers + receiver[i];
            add[i] = {.from = c, .edge = solver.addEdge(c, g, compensation[i] ? 0.0 : quota[c], 0.0)};
            remove[i] = {.from = g, .edge = solver.addEdge(g, c, mass[i], 0.0)};
        }
        std::vector<double> needs(groups);
        for (std::uint32_t i = 0; i < groups; ++i) {
            const auto g = carriers + i;
            const double bound = fillMinimum ? lower[i] : upper[i];
            const double excess = assigned[i] - bound;
            if (excess > 0)
                obligation[i] = {.from = source, .edge = solver.addEdge(source, g, excess, 0.0)};
            if (excess < 0)
                obligation[i] = {.from = g, .edge = solver.addEdge(g, sink, -excess, 0.0)};
            needs[i] = std::max(0.0, fillMinimum ? -excess : excess);
        }
        std::size_t augments = 0;
        double amount = 0;
        std::string error;
        if (!solver.runFeasible(source, sink, 10000000u, augments, amount, error))
            throw std::runtime_error(error);
        double maximumUnmet = 0, totalUnmet = 0;
        std::size_t unmetCount = 0;
        for (std::size_t i = 0; i < groups; ++i) {
            if (needs[i] == 0) continue;
            unmet[i] = std::max(unmet[i], needs[i] - solver.reverseFlow(obligation[i]));
            maximumUnmet = std::max(maximumUnmet, unmet[i]);
            totalUnmet += unmet[i];
            unmetCount += unmet[i] > 1.e-15;
        }
        std::cout << std::setprecision(17) << (fillMinimum ? "minimum" : "maximum")
                  << " augmentations=" << augments << " moved_kg=" << amount
                  << " unassigned_kg=" << totalUnmet << " largest_kg=" << maximumUnmet
                  << " groups_above_1e-15=" << unmetCount << std::endl;
        for (std::size_t i = 0; i < edges; ++i) {
            const double delta = solver.reverseFlow(add[i]) - solver.reverseFlow(remove[i]);
            mass[i] += delta;
        }
        if (maximumUnmet > 1.e-15) {
            const auto reached = solver.residualReachable(source, false);
            std::copy(reached.begin(), reached.end(), cut.begin());
            break;
        }
        std::fill(assigned.begin(), assigned.end(), 0.0);
        for (std::size_t i = 0; i < edges; ++i) assigned[receiver[i]] += mass[i];
    }
    std::ofstream output(argv[2], std::ios::binary);
    write(output, mass);
    std::ofstream diagnostic(argv[3], std::ios::binary);
    write(diagnostic, unmet);
    write(diagnostic, cut);
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
