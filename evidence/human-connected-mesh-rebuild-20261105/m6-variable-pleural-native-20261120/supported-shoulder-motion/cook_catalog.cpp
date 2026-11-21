#include "NumiHumanRegisteredLigaments.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

using namespace metalrobo;
namespace lig = registered_ligaments;
struct BoneRecord {
    uint32_t bodyIndex, firstVertex, vertexCount, firstIndex, indexCount, stableId;
    float translationX, translationY, translationZ, quaternionX, quaternionY, quaternionZ, quaternionW, uniformScale;
    uint32_t sourceRecord;
};
struct BoneVertex { float positionX, positionY, positionZ, normalX, normalY, normalZ; };
static_assert(sizeof(BoneRecord) == 60 && sizeof(BoneVertex) == 24);
static_assert(sizeof(MRJointDescriptorGPU) == 144);

template<class T> std::vector<T> readRecords(std::istream& in, size_t count) {
    std::vector<T> data(count);
    if (!in.read(reinterpret_cast<char*>(data.data()), sizeof(T) * count)) throw std::runtime_error("truncated native asset");
    return data;
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    std::ifstream bones(argv[1], std::ios::binary), rigid(argv[2], std::ios::binary);
    const auto magic = readRecords<char>(bones, 8);
    const auto header = readRecords<uint32_t>(bones, 5);
    if (std::string_view(magic.data(),8) != "NHBONES1" || header[0] != 3 || header[1] != 185) return 3;
    bones.seekg(60);
    const auto records = readRecords<BoneRecord>(bones, header[1]);
    const auto vertices = readRecords<BoneVertex>(bones, header[2]);
    const auto indices = readRecords<uint32_t>(bones, header[3]);
    if (bones.peek() != EOF) return 4;
    const auto rigidMagic = readRecords<char>(rigid, 8);
    const auto layout = readRecords<uint32_t>(rigid, 10);
    if (std::string_view(rigidMagic.data(),8) != "NHRIGID2" || layout[0] != 1) return 5;
    EngineModel model;
    rigid.seekg(224 + static_cast<std::streamoff>(layout[3]) * 160);
    model.joints = readRecords<MRJointDescriptorGPU>(rigid, layout[4]);
    std::vector<MRBodyStateGPU> poses(layout[3]);
    for (auto& pose : poses) {
        if (!(std::cin >> pose.position.x >> pose.position.y >> pose.position.z
                       >> pose.orientation.x >> pose.orientation.y >> pose.orientation.z >> pose.orientation.w)) return 6;
    }
    // Input SHA-256 is independently verified by the invoking driver.
    std::vector<inferred_acl::BoneSurfaceStorage> storage;
    for (const auto& bank : records) {
        storage.push_back(inferred_acl::bakeRegisteredBoneSurface(bank, bank.sourceRecord,
            std::span<const BoneVertex>(vertices), std::span<const uint32_t>(indices),
            poses.at(bank.bodyIndex), lig::kReferenceBonesPayloadSha256));
    }
    std::vector<inferred_acl::BoneSurface> surfaces;
    for (const auto& bank : storage) surfaces.push_back(bank.view());
    std::cout << std::setprecision(17)
              << "name,side,stable0,stable1,face0,face1,x0,y0,z0,x1,y1,z1,length_m,area_m2,c3,c4,c5,transition\n";
    for (const auto& spec : lig::referenceCatalog()) {
        const auto built = lig::cookReference(spec, model, poses, surfaces, {}, {false,true,false,false});
        const auto& p = built.provenance;
        const auto& r = built.row;
        std::cout << p.name << ',' << p.side << ',' << p.stable0 << ',' << p.stable1 << ','
                  << p.triangle0 << ',' << p.triangle1 << ',' << r.firstLocalPoint.x << ','
                  << r.firstLocalPoint.y << ',' << r.firstLocalPoint.z << ',' << r.secondLocalPoint.x
                  << ',' << r.secondLocalPoint.y << ',' << r.secondLocalPoint.z << ',' << r.reference.x
                  << ',' << r.reference.y << ',' << r.material.x << ',' << r.material.y << ','
                  << r.material.z << ',' << r.material.w << '\n';
    }
}
