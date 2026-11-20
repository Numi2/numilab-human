"""Build the registered-contact first-touch trial with the existing contact owner."""
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess

here = Path.cwd()
base = here / 'settled-ligament-preflight-002'
build = here / 'registered-first-touch-build'
build.mkdir(exist_ok=False)
source_name = 'numilab_human_myosim_visual_probe.mm'
original = (base / source_name).read_text()
initialization = '''            // One-time support-owned placement for a fresh authored release.'''
declaration = '''            // A premeasured flat-bed pose may already be aligned to the
            // registered skin contact witnesses. Check it before recruitment
            // instead of moving it with the legacy NHCNT proxies.
            const bool useRegisteredFirstTouch =
                std::getenv("NUMI_HUMAN_RESTING_REGISTERED_FIRST_TOUCH") != nullptr;
            require(!useRegisteredFirstTouch ||
                        (restingScene.has_value() && persistentMetalStand &&
                         restingReleaseInitialization && requestedRootPose.has_value() &&
                         restingInitialPosture.empty() && supportContactPayload.has_value() &&
                         skinPayload.has_value() && !femSkinSupportRequested() &&
                         restingBedSurface.empty() && muscleStepSeconds.has_value() &&
                         muscleStepCount.has_value()),
                    "registered first touch requires an explicit fresh flat-bed skin pose");
'''
assert original.count(initialization) == 1
updated = original.replace(initialization, declaration + initialization)
legacy = '''                restingInitialPosture.empty() && supportContactPayload.has_value() &&
                muscleStepSeconds.has_value() && muscleStepCount.has_value()) {'''
assert updated.count(legacy) == 1
updated = updated.replace(legacy, legacy.replace('restingInitialPosture.empty()','!useRegisteredFirstTouch && restingInitialPosture.empty()'))
capture = '''        const auto initialWeightedGaps = summarizeWeightedSupportGaps(
            weightedCapture->staticSupports);'''
admit = '''
        if (std::getenv("NUMI_HUMAN_RESTING_REGISTERED_FIRST_TOUCH")) {
            require(!localBedSections.empty() && !useFEMSkinSupport &&
                        !supportGeometryProgram->usePerContactSupportPlanes &&
                        initialWeightedGaps.minimumGapMeters >= -1.0e-6 &&
                        initialWeightedGaps.minimumGapMeters <= 5.0e-5,
                    "registered first-touch pose failed the weighted bed gap check");
            std::cout << std::setprecision(12)
                      << "resting_registered_first_touch=admitted minimum_gap_m="
                      << initialWeightedGaps.minimumGapMeters
                      << " force_rows_added=0 pose_updates=0" << std::endl;
        }'''
assert updated.count(capture) == 1
updated = updated.replace(capture, capture + admit)
scope = '''    const bool useWeightedSkinSupport = restingSupportedInitialization ||'''
if scope not in updated:
    # Validate the owner before any possible bypass of weighted capture.
    scope = '''    require(!restingSupportedInitialization || weightedSupportSetting == nullptr ||'''
check = '''    require(!std::getenv("NUMI_HUMAN_RESTING_REGISTERED_FIRST_TOUCH") ||
                (weightedSupportSetting != nullptr &&
                 std::strcmp(weightedSupportSetting, "1") == 0 &&
                 !localBedSections.empty()),
            "registered first touch requires the sole weighted skin and fat bed owner");
'''
assert updated.count(scope) == 1
updated = updated.replace(scope, check + scope)
(build / source_name).write_text(updated)
(here / 'registered-first-touch.patch').write_text(''.join(difflib.unified_diff(
    original.splitlines(keepends=True), updated.splitlines(keepends=True),
    'a/apps/' + source_name, 'b/apps/' + source_name)))
for path in (*base.glob('*.hpp'), base / 'runtime.mm.o'):
    os.link(path, build / path.name)
os.symlink(base / 'metalrobo', build / 'metalrobo', target_is_directory=True)
commands = json.loads((base / 'build-commands.json').read_text())
commands = [[a.replace(str(base) + '/', str(build) + '/') for a in cmd] for cmd in commands]
(build / 'build-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
with (here / 'registered-first-touch-build.log').open('wb') as out:
    for cmd in commands:
        subprocess.run(cmd, cwd=Path.home() / 'Work/skin-fat-native-integration-209/build-native',
                       stdout=out, stderr=subprocess.STDOUT, check=True)
print(hashlib.sha256((build / 'numi-human-native').read_bytes()).hexdigest(), flush=True)
