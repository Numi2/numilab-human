#include "NumiHumanPhysicalSkinChildRegisteredVolumeIntegration.hpp"
#include <iomanip>
#include <iostream>

using namespace metalrobo::human::coarse_tissue;
using physical_skin_child_mass_detail::splitCSV;

int main(int argc, char** argv) {
    if (argc != 12) return 2;
    const std::filesystem::path registration(argv[1]);
    const std::filesystem::path archived(argv[2]);
    const std::filesystem::path parent(argv[3]);
    const std::filesystem::path mesh(argv[4]);
    const std::filesystem::path labels(argv[5]);
    std::string error;
    PhysicalSkinChildMassMappingBundle bundle;
    if (const char* output = std::getenv("NUMI_HUMAN_VERIFY_VOLUME_FLOOR_ONLY")) {
        PhysicalSkinChildTetGenImportReport report;
        physical_skin_registered_volume_integration_detail::PhysicalSkinRuntimeMeshQualityPreflightReport quality;
        if (!loadPhysicalSkinChildTetGenReference(mesh/"mesh.node", mesh/"mesh.ele", labels,
                argv[7], std::stoul(argv[8]), 0.187444, bundle, report, error) ||
            !physical_skin_registered_volume_integration_detail::preflightPhysicalSkinRuntimeMeshQuality(bundle, quality, error)) {
            std::cerr << error << '\n';
            return 10;
        }
        std::ofstream file(output, std::ios::binary);
        for (const auto& tet : bundle.childTetrahedra) {
            const auto value = parentReferenceTetVolumeM3(tet, bundle.referenceWorldPoints);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
        file.close();
        if (!file) return 11;
        std::cout << "Production normalized shape and absolute volume gates PASS "
                  << report.nodeCount << " nodes " << report.cellCount << " tets\n";
        return 0;
    }

    if (!loadCompiledSkinFatParentMassPacketCSV(parent, bundle.parent, error) ||
        !loadCompiledSkinFatParentLabelsCSV(parent.string()+".labels.csv", bundle.parent, error)) {
        std::cerr << error << '\n';
        return 3;
    }
    const auto hash = [&](const std::filesystem::path& file) {
        std::string digest;
        if (!physical_skin_child_mass_detail::sha256File(file, digest, error))
            throw std::runtime_error(error);
        return digest;
    };
    PhysicalSkinRegisteredSurfaceCSVFiles files;
    files.donorFrames = archived / "volume-export/registered/donor-frames.csv";
    if (hash(files.donorFrames) != argv[11]) return 4;
    files.donorFramesSHA256 = argv[11];
    files.outerFaceAncestry = archived / "volume-export/registered/outer-face-ancestry.csv";
    files.outerFaceAncestrySHA256 = hash(files.outerFaceAncestry);
    files.nodeRegistration = registration / "node-parent-registration.csv";
    files.nodeRegistrationSHA256 = hash(files.nodeRegistration);
    files.donorCount = std::stoul(argv[8]);
    files.maxParentRegistrationDistanceM = 0.02;
    files.parentRegistrationIsHistoricalProvenanceOnly = true;

    // The poses are an offline replay of the pinned capture used by the
    // registration authoring tools. The installed scene checks live poses.
    std::vector<PhysicalSkinDonorFrameRow> trusted(files.donorCount);
    std::vector<PhysicalSkinRegisteredBodyPose> poses(files.donorCount);
    std::ifstream frames(files.donorFrames);
    std::string line;
    std::getline(frames, line);
    while (std::getline(frames, line)) {
        const auto row = splitCSV(line);
        if (row.size() != 26) return 5;
        const auto index = std::stoul(row[1]);
        auto& frame = trusted.at(index);
        frame.available = true;
        frame.arithmetic = PhysicalSkinPoseArithmetic::nativeFloat32;
        frame.binding = static_cast<std::uint32_t>(std::stoul(row[0]));
        for (unsigned k = 0; k < 3; ++k) {
            frame.currentOrigin[k] = std::stod(row[2+k]);
            frame.restOrigin[k] = std::stod(row[14+k]);
        }
        for (unsigned k = 0; k < 9; ++k) {
            frame.currentR[k] = std::stod(row[5+k]);
            frame.restR[k] = std::stod(row[17+k]);
        }
        poses[index] = {static_cast<std::uint32_t>(index), frame.binding, frame.available,
            frame.arithmetic, frame.currentOrigin, frame.restOrigin, frame.currentR, frame.restR};
    }
    std::ifstream allBodyFrames(argv[10]);
    for (std::size_t body = 0; body < poses.size(); ++body) {
        if (!std::getline(allBodyFrames, line)) return 5;
        const auto row = splitCSV(line);
        if (row.size() != 25 || std::stoul(row[0]) != body) return 5;
        if (trusted[body].available) continue;
        auto& pose = poses[body];
        pose.bodyIndex = static_cast<std::uint32_t>(body);
        pose.binding = NM_INVALID_INDEX;
        pose.available = true;
        pose.arithmetic = PhysicalSkinPoseArithmetic::nativeFloat32;
        for (unsigned axis = 0; axis < 3; ++axis) {
            pose.currentOrigin[axis] = std::stod(row[1+axis]);
            pose.restOrigin[axis] = std::stod(row[13+axis]);
        }
        for (unsigned j = 0; j < 9; ++j) {
            pose.currentLocalToWorld[j] = std::stod(row[4+j]);
            pose.restLocalToWorld[j] = std::stod(row[16+j]);
        }
    }
    if (std::getline(allBodyFrames, line)) return 5;
    files.nativeFloat32FramesByDonor = trusted;
    constexpr std::array roles{PhysicalSkinSourceDomainRole::outerSurface,
        PhysicalSkinSourceDomainRole::innerSurface, PhysicalSkinSourceDomainRole::deepSurface};
    std::array<PhysicalSkinSourceDomainRegistration,3> domains;
    std::ifstream registry(argv[9]);
    for (unsigned i = 0; i < 3; ++i) {
        const std::string role(1, "OID"[i]);
        auto& layer = files.layers[i];
        layer.nodes = registration / ("registered-"+role+".csv");
        layer.faces = archived / ("volume-export/registered/registered-"+role+"-faces.csv");
        layer.nodesSHA256 = hash(layer.nodes);
        layer.facesSHA256 = hash(layer.faces);
        layer.name = "registered " + role;
        layer.sourceMember = layer.nodes.string();
        layer.stableIdentifier = 291 + i;
        auto& source = files.currentSourceDomains[i];
        if (!(registry >> source.name >> source.assetSHA256 >> source.vertexCount >> source.faceCount)) return 6;
        domains[i] = {source.name, roles[i], source.assetSHA256,
            static_cast<std::uint32_t>(source.vertexCount), static_cast<std::uint32_t>(source.faceCount)};
    }
    files.d37InnerFaceCount = files.currentSourceDomains[1].faceCount;
    files.d37DeepFaceCount = files.currentSourceDomains[2].faceCount;
    PhysicalSkinRegisteredSurfaceCSVSet surfaces;
    PhysicalSkinRegisteredSurfaceCSVReport surfaceReport;
    PhysicalSkinSurfaceParentRegistrationReport parentReport;
    if (!loadPhysicalSkinRegisteredSurfaceCSVSetForParent(files, bundle.parent, surfaces,
            surfaceReport, parentReport, error)) {
        std::cerr << "Production O/I/D reader rejected: " << error << '\n';
        return 7;
    }
    std::cout << "Production O/I/D reader PASS "
              << surfaceReport.nodes[0] << ' ' << surfaceReport.nodes[1] << ' '
              << surfaceReport.nodes[2] << '\n' << std::flush;
    std::ifstream requestsFile(archived/"bindings/deep-attachment-requests.csv");
    std::getline(requestsFile, line);
    std::vector<PhysicalSkinStableDeepAttachmentRequest> requests;
    while (std::getline(requestsFile, line)) {
        const auto row = splitCSV(line);
        if (row.size() != 4) return 8;
        requests.push_back({std::stoull(row[0]), static_cast<std::uint32_t>(std::stoul(row[1])),
            static_cast<std::uint32_t>(std::stoul(row[2])), row[3]});
    }
    if (requests.size() != 14) return 8;
    PhysicalSkinRegisteredVolumeAssetFiles volume;
    volume.registrationDomain = PhysicalSkinRegisteredVolumeRegistrationDomain::currentFractionalReference;
    volume.tetgenNode = mesh/"mesh.node";
    volume.tetgenElements = mesh/"mesh.ele";
    volume.childLabels = labels;
    volume.combinedNodeMapCSV = registration/"combined-node-map.csv";
    volume.facetAncestryCSV = argv[6];
    volume.expectedSHA256 = {hash(volume.tetgenNode), hash(volume.tetgenElements), hash(labels),
        hash(volume.combinedNodeMapCSV), hash(volume.facetAncestryCSV)};
    PhysicalSkinRegisteredVolumeImportSidecars sidecars;
    PhysicalSkinRegisteredVolumeFileImportReport report;
    if (!importPhysicalSkinRegisteredVolumeFiles(volume, argv[7], files.donorCount, 0.187444,
            files.maxParentRegistrationDistanceM, "", 0, 0, surfaces, requests, poses,
            bundle, sidecars, report, error, domains)) {
        std::cerr << "Production registered volume rejected: " << error << '\n';
        return 9;
    }
    std::cout << std::setprecision(17)
              << "Production registered volume PASS " << report.tetgen.nodeCount << " nodes "
              << report.tetgen.cellCount << " tets "
              << bundle.stableDeepAttachments.size() << " stable deep attachments\n"
              << "Operational domain " << report.operationalReferenceDomainSHA256 << '\n'
              << "Source parent reconstruction m "
              << report.registeredVolume.maximumCurrentParentReconstructionErrorM << '\n';
}
