#pragma once

#include <cmath>
#include <stdexcept>

// Skin and fat act in series within each bed-normal material column.
// Columns in one contact row share a normal displacement and act in parallel.
struct NumiHumanBedColumnQuadrature {
    static constexpr double skinModulusPa = 200000.0;
    static constexpr double fatModulusPa = 55000.0;

    double stiffnessNPerM = 0.0;
    double skinEnergyWeightNPerM = 0.0;
    double fatEnergyWeightNPerM = 0.0;

    void add(double areaM2, double skinM, double fatM) {
        if (!(std::isfinite(areaM2) && areaM2 > 0.0 &&
              std::isfinite(skinM) && skinM > 0.0 &&
              std::isfinite(fatM) && fatM > 0.0))
            throw std::domain_error("invalid bed material column");
        const double skin = skinM / skinModulusPa;
        const double fat = fatM / fatModulusPa;
        const double complianceArea = skin + fat;
        const double stiffness = areaM2 / complianceArea;
        stiffnessNPerM += stiffness;
        skinEnergyWeightNPerM += stiffness * (skin / complianceArea);
        fatEnergyWeightNPerM += stiffness * (fat / complianceArea);
    }

    double energyCompliance(double weightNPerM) const {
        if (!(std::isfinite(stiffnessNPerM) && stiffnessNPerM > 0.0 &&
              std::isfinite(weightNPerM) && weightNPerM > 0.0))
            throw std::domain_error("empty or invalid bed material quadrature");
        return (weightNPerM / stiffnessNPerM) / stiffnessNPerM;
    }

    double skinComplianceMPerN() const { return energyCompliance(skinEnergyWeightNPerM); }
    double fatComplianceMPerN() const { return energyCompliance(fatEnergyWeightNPerM); }
};
