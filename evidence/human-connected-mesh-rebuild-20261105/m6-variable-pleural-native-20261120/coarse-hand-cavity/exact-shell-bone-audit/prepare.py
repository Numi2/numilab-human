"""Prepare an opt-in export of the exact physical bone inventory for the shell audit."""
from difflib import unified_diff
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "last-clearance-029/build"
SOURCE = Path.home() / "Work/skin-fat-native-integration-209"
APP = "numilab_human_myosim_visual_probe.mm"
MARK = "        const char* fullBankComparisonPath=std::getenv("
EXPORT = r'''        if (const char* path = std::getenv("NUMI_HUMAN_INITIAL_BONE_GEOMETRY_AUDIT")) {
            require(staticBVHEnabled, "bone geometry audit requires the complete registered banks");
            namespace witness = numi::human::initial_skin_bone_contact_witness;
            std::ofstream audit(path, std::ios::binary | std::ios::trunc);
            require(audit.good(), "bone geometry audit output unavailable");
            for (std::uint32_t bodyIndex = 0; bodyIndex < bodyBanks.size(); ++bodyIndex) {
                const auto& bank = bodyBanks[bodyIndex];
                if (!bodyBankEligible[bodyIndex]) continue;
                const auto& pose = bodies[bodyIndex];
                const witness::Vec3 origin{pose.position.x, pose.position.y, pose.position.z};
                const std::array<double, 4> orientation{pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w};
                for (const auto& triangle : bank.triangles) {
                    const std::array<std::uint32_t, 3> identity{
                        bodyIndex, triangle.boneStableId, triangle.triangleStableId};
                    std::array<float, 9> vertices;
                    unsigned component = 0;
                    for (const auto& local : {triangle.bodyLocalVertex0, triangle.bodyLocalVertex1,
                                               triangle.bodyLocalVertex2}) {
                        const auto world = witness::add(origin,
                            witness::rotate(orientation, witness::read3(local)));
                        vertices[component++] = static_cast<float>(world.x);
                        vertices[component++] = static_cast<float>(world.y);
                        vertices[component++] = static_cast<float>(world.z);
                    }
                    audit.write(reinterpret_cast<const char*>(identity.data()), sizeof(identity));
                    audit.write(reinterpret_cast<const char*>(vertices.data()), sizeof(vertices));
                }
            }
            audit.flush();
            require(audit.good(), "bone geometry audit output failed");
        }
'''

def main():
    original = (BASE / APP).read_text()
    assert original.count(MARK) == 1
    patched = original.replace(MARK, EXPORT + MARK)
    (HERE / APP).write_text(patched)
    with (HERE / "compiled-bone-export.patch").open("w") as stream:
        path = "apps/" + APP
        stream.writelines(unified_diff(original.splitlines(True), patched.splitlines(True),
                                      "a/" + path, "b/" + path, n=0))
    commands = json.loads((BASE / "build-commands.json").read_text())
    commands = [[str(HERE / Path(arg).name) if Path(arg).parent == BASE else arg
                 for arg in argv] for argv in commands]
    (HERE / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    if "--build" in sys.argv:
        import subprocess
        for argv in commands:
            subprocess.run(argv, cwd=SOURCE / "build-native", check=True)

if __name__ == "__main__":
    main()
