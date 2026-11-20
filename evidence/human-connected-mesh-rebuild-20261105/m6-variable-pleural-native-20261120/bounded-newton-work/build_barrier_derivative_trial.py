"""Build the retained Metal graph with a geometry-matched edge barrier slope."""
from pathlib import Path
import difflib
import json
import os
import subprocess
import sys

here = Path.cwd()
base = here.parent/"resting-clearance-and-atrium-013/glide-refresh-build/shader"
recipe = here.parent/"resting-clearance-and-atrium-012/residual-bank-shader"
rounded_chord = sys.argv[1:] == ["rounded-chord"]
build_name = "edge-barrier-rounded-chord" if rounded_chord else "edge-barrier-derivative"
destination = here/build_name
(destination/"src").mkdir(exist_ok=True,parents=True)
for p in (base/"src").rglob("*"):
    dest = destination/p.relative_to(base)
    if p.is_dir():
        dest.mkdir(exist_ok=True,parents=True)
    elif p.is_file() and not dest.exists():
        os.link(p.resolve(),dest)

def replace(source,before,after):
    if source.count(before) != 1:
        raise ValueError(before)
    return source.replace(before,after)

def emit(name,old,new):
    src = "src/metal/"+name
    dest = destination/src
    dest.unlink()
    dest.write_text(new)
    return "".join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                        "a/matter/"+src,"b/matter/"+src))

common = (base/"src/metal/common.metalinc").read_text()
start = common.index("inline MRCompensatedScalar ipcFEMBarrierEnergyCompensated(")
end = common.index("\ninline bool finite3",start)
src = common[start:end]
src = replace(src,"    const float clearanceOffset = 0.0f\n) {",
    "    const float clearanceOffset = 0.0f,\n"
    "    thread MRCompensatedScalar* radialDerivative = nullptr\n) {\n"
    "    if (radialDerivative != nullptr) *radialDerivative = {0.0f, 0.0f};")
lo = src.index("    if (!belowFloor)\n")
hi = src.index("    const auto firstSquared",lo)
before = src[lo:hi]
src = src[:lo]+src[hi:]
lo = src.index("    const auto firstSquared")
hi = src.index("    const auto continuation",lo)
derivative = src[lo:hi]
src = src[:lo]+(
    "    if (radialDerivative != nullptr) {\n"
    +"".join("    "+line for line in derivative.splitlines(True))+
    "        *radialDerivative = energyDerivativeAtFloor;\n"
    "    }\n"+before)+src[lo:]
# Use one shared derivative both for the existing floor continuation and
# the analytic directional derivative requested by the line search.
lo = src.index("    if (radialDerivative != nullptr) {")
hi = src.index("    const auto continuation",lo)
src = src[:lo]+(
    "    MRCompensatedScalar energyDerivative = {0.0f, 0.0f};\n"
    "    if (belowFloor || radialDerivative != nullptr) {\n"
    +"".join("    "+line for line in
           derivative.replace("const auto energyDerivativeAtFloor =","energyDerivative =").splitlines(True))+
    "        if (radialDerivative != nullptr) *radialDerivative = energyDerivative;\n"
    "    }\n"+before)+src[hi:]
src = src.replace("mrCompensatedMultiply(energyDerivativeAtFloor, floorDelta)",
                  "mrCompensatedMultiply(energyDerivative, floorDelta)")
patch = emit("common.metalinc",common,common[:start]+src+common[end:])

contact = (base/"src/metal/contact.metalinc").read_text()
start = contact.index("inline bool deformableBarrierEnergyAt(")
end = contact.index("inline bool deformableBarrierEnergyAt(",start+1)
src = contact[start:end]
src = replace(src,"    thread MRCompensatedScalar* energyWords = nullptr\n) {",
    "    thread MRCompensatedScalar* energyWords = nullptr,\n"
    "    thread const float3* edgeDirection = nullptr,\n"
    "    thread float* edgeDerivative = nullptr\n) {")
src = replace(src,"    MRCompensatedPositionGPU relative;",
    "    MRCompensatedPositionGPU relative;\n"
    "    float3 relativeDirection = float3(0.0f);\n"
    "    float mollifierDerivative = 0.0f;\n"
    "    if (edgeDerivative != nullptr) *edgeDerivative = NAN;")
needle = """            firstEdge, secondEdge, firstGradient, secondGradient,
            energyWords != nullptr ? &featureMollifierWords : nullptr);
"""
src = replace(src,needle,needle+"""
        if (edgeDirection != nullptr) {
            const float3 firstVelocity = edgeDirection[1] - edgeDirection[0];
            const float3 secondVelocity = edgeDirection[3] - edgeDirection[2];
            relativeDirection = (edgeDirection[0] - edgeDirection[2]) +
                firstParameter * firstVelocity - secondParameter * secondVelocity;
            mollifierDerivative = dot(firstGradient, firstVelocity) +
                dot(secondGradient, secondVelocity);
        }
""")
src = replace(src,"    const MRCompensatedScalar barrierEnergy =",
    "    MRCompensatedScalar radialDerivative = {0.0f, 0.0f};\n"
    "    const MRCompensatedScalar barrierEnergy =")
src = replace(src,"            nmFEMClearanceAlignedBarrierActive ? contact.barrierHessianRow2.w : 0.0f);",
    """            nmFEMClearanceAlignedBarrierActive ? contact.barrierHessianRow2.w : 0.0f,
            edgeDerivative != nullptr ? &radialDerivative : nullptr);
    if (edgeDerivative != nullptr && separation > 0.0f) {
        const auto speed = mrCompensatedVectorDot(relative,
            mrCompensatedVector(float4(relativeDirection, 0.0f)));
        const auto distanceDerivative = mrCompensatedDivide(speed, distance);
        const auto radialTerm = mrCompensatedMultiply(featureMollifierWords,
            mrCompensatedMultiply(radialDerivative, distanceDerivative));
        const auto angularTerm = mrCompensatedMultiply(
            barrierEnergy, {mollifierDerivative, 0.0f});
        const auto derivative = mrCompensatedAdd(radialTerm, angularTerm);
        *edgeDerivative = derivative.high + derivative.low;
    }""")
updated = contact[:start]+src+contact[end:]
needle = """        float currentEnergy;
        float currentSeparation;
        MRCompensatedScalar currentEnergyWords = {NAN, 0.0f};
"""
updated = replace(updated,needle,"""        // Match the edge-edge Armijo slope to the same compensated geometry
        // and potential as the trial energy. Stored force rows are assembled
        // separately and may use a differently rounded point or mollifier.
        const bool matchedEdgeSlope = referenceModeContact &&
            contact.identity.z == NM_DEFORMABLE_CONTACT_EDGE_EDGE &&
            (!nmFEMClearanceAlignedBarrierActive || contact.barrierHessianRow2.w == 0.0f);
        float3 edgeDirection[4];
        if (matchedEdgeSlope) {
            const bool mapped = (contact.identity.w & NM_CONTACT_MAPPED_FEATURE) != 0u;
            for (uint feature = 0u; feature < 4u; ++feature) {
                if (mapped) {
                    const uint base = contact.featureSideByVertex[feature] == 0u ? 0u : 3u;
                    edgeDirection[feature] =
                        contact.featureBarycentricX[feature] * analyticDisplacement[base] +
                        contact.featureBarycentricY[feature] * analyticDisplacement[base + 1u] +
                        contact.featureBarycentricZ[feature] * analyticDisplacement[base + 2u];
                } else {
                    edgeDirection[feature] = analyticDisplacement[feature];
                }
            }
        }
        float matchedDerivative = NAN;
"""+needle)
updated = replace(updated,
    "                currentEnergy, currentSeparation, &currentEnergyWords)",
    """                currentEnergy, currentSeparation, &currentEnergyWords,
                matchedEdgeSlope ? edgeDirection : nullptr,
                matchedEdgeSlope ? &matchedDerivative : nullptr)""")
updated = replace(updated,
    "            const float directionalDerivative =\n                contactBarrierDirectionalDerivative(",
    "            const float directionalDerivative = matchedEdgeSlope\n"
    "                ? matchedDerivative : contactBarrierDirectionalDerivative(")
if rounded_chord:
    start = updated.index("        if (matchedEdgeSlope) {")
    end = updated.index("        float matchedDerivative",start)
    updated = updated[:start]+"""        if (matchedEdgeSlope) {
            for (uint feature = 0u; feature < 4u; ++feature) {
                const auto chord = mrCompensatedVectorAdd(
                    featurePotentialEndpoint[feature],
                    mrCompensatedVectorNegate(featurePotentialPosition[feature]));
                edgeDirection[feature] = contactCompensatedPositionProjection(chord);
            }
        }
"""+updated[end:]
patch += emit("contact.metalinc",contact,updated)
(here/(build_name+".patch")).write_text(patch)
commands = json.loads((recipe/"build-commands.json").read_text())
commands = [[item.replace(str(recipe)+"/",str(destination)+"/") for item in cmd]
            for cmd in commands]
(destination/"build-commands.json").write_text(json.dumps(commands,indent=2)+"\n")
with (here/(build_name+"-build.log")).open("wb") as log:
    for cmd in commands:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
Path(commands[0][-1]).unlink()
print(destination/"NumiMatter.metallib")
