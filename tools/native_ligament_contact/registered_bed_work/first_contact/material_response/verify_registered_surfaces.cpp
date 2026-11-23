#include "NumiHumanPhysicalSkinChildRegisteredSurfaceSidecars.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

namespace tissue = metalrobo::human::coarse_tissue;

static std::string setting(const std::string& key) {
    const auto name = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_" + key;
    const char* value = std::getenv(name.c_str());
    if (!value) throw std::runtime_error("Missing " + name);
    return value;
}

#include "verify_volume_bindings.hpp"

int main(int argc, char** argv) try {
    if (argc != 3 && argc != 4) return 2;
    tissue::PhysicalSkinRegisteredSurfaceCSVFiles files;
    files.donorCount = std::stoul(argv[2]);
    std::vector<tissue::PhysicalSkinDonorFrameRow> frames(files.donorCount);
    std::ifstream pose(argv[1]);
    unsigned binding, donor;
    while (pose >> binding >> donor) {
        auto& row = frames.at(donor);
        if (row.available) throw std::runtime_error("Duplicate pose");
        row.available = true;
        row.binding = binding;
        row.arithmetic = tissue::PhysicalSkinPoseArithmetic::nativeFloat32;
        for (auto& v : row.currentOrigin) pose >> v;
        for (auto& v : row.currentR) pose >> v;
        for (auto& v : row.restOrigin) pose >> v;
        for (auto& v : row.restR) pose >> v;
        if (!pose) throw std::runtime_error("Incomplete pose");
    }
    if (!pose.eof()) throw std::runtime_error("Invalid pose");
    files.nativeFloat32FramesByDonor = frames;
    unsigned index = 0;
    for (std::string name : {"O", "I", "D"}) {
        const auto key = "SURFACE_" + name + "_";
        auto& layer = files.layers[index];
        layer.nodes = setting(key + "NODES_CSV");
        layer.faces = setting(key + "FACES_CSV");
        layer.nodesSHA256 = setting(key + "NODES_SHA256");
        layer.facesSHA256 = setting(key + "FACES_SHA256");
        layer.stableIdentifier = std::stoul(setting(key + "STABLE_IDENTIFIER"));
        layer.name = "registered " + name + " surface";
        layer.sourceMember = layer.nodes.string();
        const auto role = "ROLE_" + name + "_SOURCE_";
        files.currentSourceDomains[index++] = {setting(role + "DOMAIN"), setting(role + "ASSET_SHA256"),
            std::stoul(setting(role + "VERTEX_COUNT")), std::stoul(setting(role + "FACE_COUNT"))};
    }
    files.donorFrames = setting("DONOR_FRAMES_CSV");
    files.donorFramesSHA256 = setting("DONOR_FRAMES_SHA256");
    files.nodeRegistration = setting("NODE_REGISTRATION_CSV");
    files.nodeRegistrationSHA256 = setting("NODE_REGISTRATION_SHA256");
    files.outerFaceAncestry = setting("OUTER_FACE_ANCESTRY_CSV");
    files.outerFaceAncestrySHA256 = setting("OUTER_FACE_ANCESTRY_SHA256");
    files.d37InnerFaceCount = std::stoul(setting("D37_INNER_FACE_COUNT"));
    files.d37DeepFaceCount = std::stoul(setting("D37_DEEP_FACE_COUNT"));
    files.maxParentRegistrationDistanceM = std::stod(setting("MAX_PARENT_REGISTRATION_DISTANCE_M"));
    if (argc == 4) {
        if (std::string(argv[3]) != "--volume") return 2;
        verifyVolume(files,frames);
        return 0;
    }
    tissue::PhysicalSkinRegisteredSurfaceCSVSet imported;
    tissue::PhysicalSkinRegisteredSurfaceCSVReport audit;
    std::string error;
    if (!tissue::loadPhysicalSkinRegisteredSurfaceCSVSet(files, imported, audit, error))
        throw std::runtime_error(error);
    std::cout << "{\"registered_nodes\":[" << audit.nodes[0] << ',' << audit.nodes[1] << ',' << audit.nodes[2]
              << "],\"registered_faces\":[" << audit.faces[0] << ',' << audit.faces[1] << ',' << audit.faces[2]
              << "],\"donor_frames\":" << audit.donorFrames << ",\"payload_sha256\":\"" << audit.payloadSHA256 << "\"}\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
