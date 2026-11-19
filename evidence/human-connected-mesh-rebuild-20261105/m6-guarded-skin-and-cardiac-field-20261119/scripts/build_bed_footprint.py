from pathlib import Path
import os
import shutil
import sys
import subprocess

here=Path.cwd()
prior=here.parent/"resting-clearance-and-atrium-003"
control=here/"guarded-skin-build"
work=here/"bed-footprint-build"
work.mkdir()
for f in control.iterdir():
    if f.is_symlink():
        (work/f.name).symlink_to(f.resolve(),target_is_directory=True)
    elif f.suffix in (".hpp",".o",".py",".mm") and not f.name.startswith("numi-human-native") and f.name!="NumiHumanRestingVisual.hpp":
        os.link(f,work/f.name)
header=control/"NumiHumanRestingVisual.hpp"
data=header.read_text()
old='retainedSourceSkinSHA256 == "6442daaf259a83e186183c9a32ee8d239945c11228bf5b0fe998ff8595c78965"'
baseline='retainedSourceSkinSHA256 == "00709724af1ae4684f057255e0f98297add7e83671d3e229f040a3d91daeb1c8"'
assert data.count(old)==1
data=data.replace(old,f"({old} || {baseline})")
data=data.replace("for (unsigned v = 0; v < posedSkin.size(); ++v) {\n                    if (tributaryAreas[v] == 0) continue;",
"""std::vector<unsigned> chosenThicknessVertex(posedSkin.size(), MR_INVALID_INDEX);
                for (unsigned v = 0; v < posedSkin.size(); ++v) {
                    if (tributaryAreas[v] == 0) continue;""")
data=data.replace("skinThicknessByRow[row] += tributaryAreas[v]",
"""chosenThicknessVertex[v] = kWholeBodySkinSourceVertices[nearest];
                    skinThicknessByRow[row] += tributaryAreas[v]""")
key="                auto* uploadedMaps = static_cast<MRHumanRestingVertexMap*>(mapping.contents);"
assert data.count(key)==1
data=data.replace(key,"""                if (std::getenv("NUMI_HUMAN_TEST_BED_SECTION_FOOTPRINT")) {
                    std::ofstream footprint(outputDirectory / "resting-bed-section-footprint.csv");
                    footprint << "vertex,region,x,y,z,region_min,tributary_area,nearest_thickness_vertex\\n"
                              << std::setprecision(17);
                    for (unsigned v=0; v<posedSkin.size(); ++v) {
                        const auto row=regionForVertex[v];
                        footprint << v << ',' << row << ',' << posedSkin[v].x << ',' << posedSkin[v].y
                                  << ',' << posedSkin[v].z << ',' << patchMinimum[row] << ','
                                  << tributaryAreas[v] << ',' << chosenThicknessVertex[v] << '\\n';
                    }
                    require(bool(footprint), "bed section footprint receipt write failed");
                }
"""+key)
(work/header.name).write_text(data)
subprocess.run([sys.executable,str(work/"build.py")],check=True)
