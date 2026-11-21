#!/bin/bash
set -euo pipefail
work="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/pelvic-contact-response-048"
cd "$work"
e=$(dirname "$work")
part="$e/pelvic-material-interface-047"
prior="$e/anatomical-extensor-and-bed-045"
prev="$e/material-bed-equilibrium-046"
app="$HOME/Work/skin-fat-native-integration-209"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
ref="$prev/native-release-motion/step-8.receipt.json"
export PYTHONPATH="$work:$part:$(/usr/bin/python3 -c 'import json,sys;print(":".join(json.load(open(sys.argv[1]))["authoring_pythonpath"]))' "$part/tool-versions.json"):${PYTHONPATH:-}"
"$py" prepare_finite_facets.py repartition repartition/relative-assets "$ref" repartition/facet-assets
"$py" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$work/repartition/facet-assets" "$app" "$work/repartition/replay-facet-build"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -O2 -fobjc-arc -framework Metal -framework Foundation verify_finite_facet_kernel.mm -o verify-finite-facet
"$py" test_finite_facet_kernel.py "$work/verify-finite-facet" "$work/repartition/replay-facet-build/metalrobo/HumanRestingSkin.metallib" repartition/facet-assets repartition/replay-facet-build/facet-kernel-parity.json
"$py" - "$prev/supported-release/config.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);d=json.loads(p.read_text());d['environment']['NUMI_HUMAN_PELVIC_CAPTURE_SURFACES']='1'
out=Path('repartition/replay-facet-config');out.mkdir(exist_ok=True)
(out/'config.json').write_text(json.dumps(d,indent=2)+'\n')
if not (out/'run_supported.py').exists():(out/'run_supported.py').hardlink_to(p.parent/'run_supported.py')
PY
for steps in 8 8000;do
    run="$work/repartition/replay-facets-$steps"
    test ! -e "$run"
    bash run_cortical_trial.sh "$work/repartition/replay-facet-config/config.json" "$work/repartition/facet-assets" "$work/repartition/replay-facet-build/numi-human-native" "$run" "$py" "$steps" 1 || test -f "$run/failure.json"
    "$py" audit_complete_pelvic_forces.py repartition repartition/response repartition/facet-assets "$run/accepted-pelvic-state.csv" "$run/bulk-force.json"
    "$py" audit_emitted_cortical_crossings.py repartition/facet-assets "$run/accepted-pelvic-surfaces.csv" "$prior/changed_surface_pairs" "$run/native-crossings.json"
    "$py" audit_accepted_cortical_volume.py repartition repartition/facet-assets "$run/accepted-pelvic-state.csv" "$run/solid-overlap.json"
    "$py" evaluate_pelvic_trial.py "$run"
done
