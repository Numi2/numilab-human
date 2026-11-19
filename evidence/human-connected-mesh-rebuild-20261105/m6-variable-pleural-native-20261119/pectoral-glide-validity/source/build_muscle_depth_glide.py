"""Use donor muscle depth in the existing pectoral elastic foundation."""
import difflib
import json
import os
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
base = here / "deep-pectoral-build"
out = here / "deep-pectoral-depth-build"
out.mkdir()
source = "numilab_human_myosim_visual_probe.mm"
app = "numi-human-native"
for path in (*base.glob("*.hpp"), base / "runtime.mm.o"):
    os.link(path, out / path.name)
os.symlink(base / "metalrobo", out / "metalrobo", target_is_directory=True)
original = (base / source).read_text()
updated = original

def replace(old, new):
    global updated
    assert updated.count(old) == 1, old
    updated = updated.replace(old, new)

replace("""    std::vector<mr_float4> regionFibreDirections(fascia.regions.size());""",
"""    std::vector<mr_float4> pectoralProjectedMusclePoints(fascia.nodes.size());
    std::vector<mr_float4> regionFibreDirections(fascia.regions.size());""")
replace("""                point.x+=kPreservedSurfaceRemapResidualFraction*
                    remapResidualWorld.x;""",
"""                pectoralProjectedMusclePoints[global] = point;
                point.x+=kPreservedSurfaceRemapResidualFraction*
                    remapResidualWorld.x;""")
replace("""    constexpr double kGlidingFoundationModulusPascals=20'000.0;
    if(fasciaBodyGlideContactEnabled) {""",
"""    constexpr double kGlidingFoundationModulusPascals=20'000.0;
    // The sheet is resolved by FEM. The gliding foundation is the donor
    // muscle underneath it. Measure the latter's load path in the registered
    // muscle mesh along the same normal used in the contact law.
    const auto inwardMuscleDepth = [&](const SoftTissueRecord& tissue,
                                       std::span<const mr_float4> surface,
                                       mr_float4 surfacePoint, mr_float4 outward) {
        constexpr float entranceGuard = 5.0e-5f;
        const auto ray = femScale(outward, -1.0f);
        double firstExit=std::numeric_limits<double>::infinity();
        for (std::uint32_t k=0u; k<tissue.indexCount; k+=3u) {
            const auto index = [&](std::uint32_t lane) {
                return tissues.indices[tissue.firstIndex+k+lane]-tissue.firstVertex;
            };
            const auto a=surface[index(0u)];
            const auto e1=femSubtract(surface[index(1u)],a);
            const auto e2=femSubtract(surface[index(2u)],a);
            const auto p=femCross(ray,e2);
            const double det=femDot(e1,p);
            // For outward windings an inward ray exits at a negative
            // determinant. The projection's entrance triangle is excluded.
            if (!(det < -1.0e-11)) continue;
            const auto t=femSubtract(surfacePoint,a);
            const double u=femDot(t,p)/det;
            if(u<0.0 || u>1.0) continue;
            const auto q=femCross(t,e1);
            const double v=femDot(ray,q)/det;
            if(v<0.0 || u+v>1.0) continue;
            const double distance=femDot(e2,q)/det;
            if(distance>entranceGuard) firstExit=std::min(firstExit,distance);
        }
        return firstExit;
    };
    if(fasciaBodyGlideContactEnabled) {""")
replace("""        for(std::uint32_t local=0u;local<layerWidth;++local) {
            const std::uint32_t outerNode=region.firstNode+local;""",
"""        const bool pectoral=regionIndex<kPectoralisFasciaRegionCount;
        const SoftTissueRecord* donor=nullptr;
        std::vector<mr_float4> muscleVertices;
        if (pectoral) {
            const auto match=std::find_if(tissues.records.begin(),tissues.records.end(),
                [&region](const SoftTissueRecord& tissue){
                    return tissue.stableId==region.softTissueStableId;
                });
            require(match!=tissues.records.end(), "pectoral gliding donor is missing");
            donor=&*match;
            muscleVertices.reserve(donor->vertexCount);
            for(std::uint32_t v=0u; v<donor->vertexCount; ++v)
                muscleVertices.push_back(softTissueVertexBlendedWorld(
                    *donor,tissues.vertices[donor->firstVertex+v],bodies));
        }
        double minDepth=std::numeric_limits<double>::infinity(),maxDepth=0.0;
        std::uint32_t depthRows=0u,unsupportedRows=0u;
        for(std::uint32_t local=0u;local<layerWidth;++local) {
            const std::uint32_t outerNode=region.firstNode+local;""")
replace("""            sample.stiffnessAndNormalStrainPerPressure={
                static_cast<float>(kGlidingFoundationModulusPascals/layerThickness),""",
"""            double foundationDepth=layerThickness;
            if (donor) {
                const double measured=inwardMuscleDepth(*donor,muscleVertices,
                    pectoralProjectedMusclePoints[sourceNode],worldNormal);
                if (std::isfinite(measured) && measured>=layerThickness) {
                    foundationDepth=measured;
                    minDepth=std::min(minDepth,measured);
                    maxDepth=std::max(maxDepth,measured);
                    ++depthRows;
                } else {
                    ++unsupportedRows;
                }
            }
            sample.stiffnessAndNormalStrainPerPressure={
                static_cast<float>(kGlidingFoundationModulusPascals/foundationDepth),""")
replace("""            ++fasciaBodyContactSamplesPerRegion[regionIndex];
        }
    }
    }""",
"""            ++fasciaBodyContactSamplesPerRegion[regionIndex];
        }
        if (donor) {
            std::cout << "pectoral_gliding_donor_depth region=" << regionIndex
                << " rows=" << depthRows
                << " unsupported_rows_using_input_foundation=" << unsupportedRows
                << " min_depth_m=" << (depthRows ? minDepth : 0.0)
                << " max_depth_m=" << maxDepth << '\\n';
        }
    }
    }""")
(out / source).write_text(updated)
(here / "muscle-depth-glide.patch").write_text("".join(difflib.unified_diff(
    original.splitlines(keepends=True),updated.splitlines(keepends=True),
    "a/apps/" + source,"b/apps/" + source)))
paths = {str(base / f): str(out / f) for f in (app, app + ".o", source, "runtime.mm.o")}
commands = [[paths.get(a,a) for a in cmd] for cmd in json.loads(
    (base / "build-commands.json").read_text())]
(out / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for command in commands:
    subprocess.run(command, cwd="/Users/numi/Work/skin-fat-native-integration-209/build-native",
                   check=True)
