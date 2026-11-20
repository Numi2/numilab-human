#include "numi/matter/matter.hpp"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    using namespace numi::matter;
    WorldSource source;
    source.environmentCount = 1;
    source.frameTimestep = 0.001;
    source.gravity = {0, 0, 0};
    auto& vascular = source.vascular;
    vascular.contentIdentity = {1u, 2u, 3u, 4u};
    vascular.sourceIdentity = {5u, 6u, 7u, 8u};
    vascular.authoredIdentity = {9u, 10u, 11u, 12u};
    vascular.compartments = {
        {1u, "fixture/a", 0.001, 0.001, 0, 1e-7, 0, 0.001, 1e-5, {}},
        {2u, "fixture/b", 0.001, 0.001, 0, 1e-7, 0, 0.001, 1e-5, {}},
    };
    vascular.connections = {{1u, 1u, 2u, 1e8, 1e5, 0, 1e-5, 1000, 1e-5}};
    const auto compiled = compileWorld(source, {.maximumRateExponent = 0});
    if (!compiled.succeeded()) {
        for (const auto& d : compiled.diagnostics) std::cerr << d.message << '\n';
        return 2;
    }
    RuntimeConfiguration config;
    config.metallib = argv[1];
    config.environmentCount = 1;
    config.captureEvents = false;
    config.adaptiveTransfer = false;
    Runtime runtime;
    setenv("NM_FGMRES_FLAT_COLUMN_ICB", "1", 1);
    const auto rejected = runtime.initialize(compiled.world, config);
    const auto rejectedTimestep = runtime.timestepSeconds();
    unsetenv("NM_FGMRES_FLAT_COLUMN_ICB");
    const auto accepted = runtime.initialize(compiled.world, config);
    const auto acceptedTimestep = runtime.timestepSeconds();
    std::cout << "invalid_configuration_encoded=" << rejected.encoded
              << " invalid_configuration_dt_s=" << rejectedTimestep
              << " rejection=\"" << rejected.message << "\"\n"
              << "valid_configuration_encoded=" << accepted.encoded
              << " valid_configuration_dt_s=" << acceptedTimestep
              << " device=\"" << accepted.device << "\"\n";
    return !rejected.encoded && rejectedTimestep == 0.0f &&
        std::string_view(rejected.message).starts_with("flat-column ICB requires") &&
        accepted.encoded && std::abs(acceptedTimestep - 0.001f) < 1e-8f ? 0 : 1;
}
