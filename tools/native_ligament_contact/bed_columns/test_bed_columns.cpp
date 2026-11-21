#include "NumiHumanBedColumnQuadrature.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>

static void close(double actual, double expected) {
    assert(std::abs(actual - expected) <= 2e-14 * std::abs(expected));
}
int main() {
    NumiHumanBedColumnQuadrature homogenous;
    homogenous.add(0.006, 0.002, 0.01);
    homogenous.add(0.004, 0.002, 0.01);
    close(homogenous.skinComplianceMPerN(), 0.002 / (200000.0 * 0.01));
    close(homogenous.fatComplianceMPerN(), 0.01 / (55000.0 * 0.01));

    NumiHumanBedColumnQuadrature field;
    double force = 0.0, skinEnergy = 0.0, fatEnergy = 0.0;
    constexpr double approach = 0.00025;
    for (auto column : {std::array{0.004, 0.001, 0.003}, std::array{0.006, 0.003, 0.014}}) {
        const auto [area, skin, fat] = column;
        field.add(area, skin, fat);
        const double pressure = approach / (skin / 200000.0 + fat / 55000.0);
        force += area * pressure;
        skinEnergy += 0.5 * area * skin * pressure * pressure / 200000.0;
        fatEnergy += 0.5 * area * fat * pressure * pressure / 55000.0;
    }
    close(force, approach / (field.skinComplianceMPerN() + field.fatComplianceMPerN()));
    close(skinEnergy, 0.5 * force * force * field.skinComplianceMPerN());
    close(fatEnergy, 0.5 * force * force * field.fatComplianceMPerN());
    for (auto bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
        bool rejected = false;
        try { field.add(0.01, bad, 0.01); } catch (const std::domain_error&) { rejected = true; }
        assert(rejected);
    }
    bool rejected = false;
    try { NumiHumanBedColumnQuadrature{}.fatComplianceMPerN(); }
    catch (const std::domain_error&) { rejected = true; }
    assert(rejected);
}
