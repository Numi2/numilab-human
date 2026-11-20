"""Rebuild only the Human launch executable with a fixed-bed seed option."""
import difflib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / "Work/skin-fat-native-integration-209"
BASE = ROOT.parent / "last-clearance-029/build"

def replace(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)

def write(source, text, relative):
    before = source.read_text()
    target = ROOT / source.name
    target.write_text(text)
    patch = difflib.unified_diff(before.splitlines(True), text.splitlines(True),
                                f"a/{relative}", f"b/{relative}", n=0)
    (ROOT / (source.name + ".patch")).write_text("".join(patch))
    return target

header = SOURCE / "apps/NumiHumanFEMOuterGroundAlignment.hpp"
text = header.read_text()
text = replace(text, "        FEMOuterGroundAlignment& result,\n        std::string& error) {",
               """        FEMOuterGroundAlignment& result,
        std::string& error, double firstTouchMarginM = 1.0e-8) {""")
text = replace(text, "    constexpr double firstTouchMarginM=1.0e-8;",
               """    if (!std::isfinite(firstTouchMarginM) || firstTouchMarginM < 1.0e-8 ||
        firstTouchMarginM >= 1.0e-6)
        return fail("FEM outer first-touch clearance is outside the supported positive range");""")
write(header, text, "apps/" + header.name)
app = BASE / "numilab_human_myosim_visual_probe.mm"
text = app.read_text()
mark = "                    FEMOuterGroundAlignment physicalAlignment;std::string alignmentError;"
text = replace(text, mark, """                    double firstTouchGapM = 1.0e-8;
                    if (const char* requestedGap =
                            std::getenv("NUMI_HUMAN_FEM_TRIAL_INITIAL_CLEARANCE_M")) {
                        int parsed = 0;
                        require(!std::getenv("NUMI_HUMAN_FEM_TRIAL_BED_PLANE") &&
                            !std::getenv("NUMI_HUMAN_FEM_SUPPORT_BARRIER_RELEASE") &&
                            std::sscanf(requestedGap, "%lf%n", &firstTouchGapM, &parsed) == 1 &&
                            requestedGap[parsed] == '\\0' && std::isfinite(firstTouchGapM),
                            "seeded FEM bed clearance requires the unchanged fixed plane");
                    }
""" + mark)
text = replace(text, "                                evaluateOuter,physicalAlignment,alignmentError),",
               "                                evaluateOuter,physicalAlignment,alignmentError,firstTouchGapM),")
text = replace(text, '                        <<" aligned_min_gap_m="<<physicalAlignment.alignedMinimumGapM',
               '                        <<" requested_gap_m="<<firstTouchGapM\n'
               '                        <<" aligned_min_gap_m="<<physicalAlignment.alignedMinimumGapM')
write(app, text, "apps/" + app.name)
commands = json.loads((BASE / "build-commands.json").read_text())
commands = [[str(ROOT / Path(arg).name) if Path(arg).parent == BASE else arg
             for arg in argv] for argv in commands]
(ROOT / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
for argv in commands:
    subprocess.run(argv, cwd=SOURCE / "build-native", check=True)
