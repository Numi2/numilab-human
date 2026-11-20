#!/bin/bash
set -euo pipefail
T="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024"
J="$T/posterior-load-path-042"
N="$T/newton-terminal-schedule-043"
V="$T/posterior-gluteal-continuum-041"
Q="$T/effective-newton-terminal-044"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
steps="${1:-4}"
[[ "$steps" = 4 || "$steps" = 90 ]]
cd "$Q"
shasum -a 256 -c preregistration.sha256
test -s app/identity.json && test -s library/identity.json
vol="NumiEffectiveNewton044"
ram="/Volumes/$vol"
test ! -e "$ram"
disk=$(hdiutil attach -nomount ram://900000 | awk 'NR==1 {print $1}')
[[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
trap 'diskutil eject "$disk"' EXIT
diskutil eraseDisk APFS "$vol" GPT "$disk" > "ram-volume-$steps.log"
diskutil info "$ram" | rg 'Protocol:.*Disk Image'
/opt/homebrew/bin/gh release download numi-human-bed-column-20261120 \
    -R Numi2/workstation-checkpoint-archive \
    --pattern 20261120-numi-posterior-ilium-native.mrvpack -O "$ram/initial.mrvpack"
reference="$V/native-candidate-128/launch.json"
"$py" - "$reference" "$ram/staged-source.json" "$ram/fixed-source.json" <<'PY'
import json
import sys
from pathlib import Path
reference = json.loads(Path(sys.argv[1]).read_text())
environment = reference["environment"]
cap = int(environment["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"])
assert cap == 24
environment["NUMI_MATTER_KEEP_ORIGINAL_NEWTON_TERMINAL"] = str(cap)
Path(sys.argv[2]).write_text(json.dumps(reference, indent=2) + "\n")
environment["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"] = "8"
Path(sys.argv[3]).write_text(json.dumps(reference, indent=2) + "\n")
PY
modes=(full staged)
if [[ "$steps" = 4 ]]; then modes+=(fixed); fi
for mode in "${modes[@]}"; do
    input="$reference"
    flags=()
    if [[ "$mode" = staged ]]; then
        input="$ram/staged-source.json"
        flags=(--staged)
    elif [[ "$mode" = fixed ]]; then
        input="$ram/fixed-source.json"
    fi
    folder="$mode-$steps"
    result=0
    "$py" "$J/replay_outlier.py" "$input" "$ram/initial.mrvpack" "$ram/$folder" \
        --steps "$steps" --root "$((steps > 84 ? 84 : 0))" --shader "$N/shader/guarded-short.metallib" \
        --binary "$Q/app/numi-human-native" --library "$Q/library/libmetalrobo.dylib" \
        --fields ${flags[@]+"${flags[@]}"} || result=$?
    mkdir "$folder" "$folder/native"
    for file in stdout.log.gz stderr.log launch.json execution.json native/resting-coupled.csv; do
        if [[ -f "$ram/$folder/$file" ]]; then
            cp "$ram/$folder/$file" "$folder/$file"
            cmp "$ram/$folder/$file" "$folder/$file"
        fi
    done
    gzip -t "$folder/stdout.log.gz"
    echo "$folder status=$result"
done
"$py" "$J/compare_solver_prefixes.py" "$N/full" "full-$steps" "prior-parity-$steps.json"
"$py" "$J/compare_solver_prefixes.py" "full-$steps" "staged-$steps" "staged-prefix-$steps.json"
"$py" "$J/compare_latch.py" "full-$steps" "staged-$steps" "staged-work-$steps.json"
"$py" "$T/next-stage-profile/compare_solver_fields.py" "full-$steps/stdout.log.gz" "staged-$steps/stdout.log.gz" "staged-fields-$steps.json" > "staged-fields-$steps.log"
if [[ "$steps" = 4 ]]; then
    "$py" "$J/compare_solver_prefixes.py" full-4 fixed-4 fixed-prefix-4.json
    "$py" "$T/next-stage-profile/compare_solver_fields.py" full-4/stdout.log.gz fixed-4/stdout.log.gz fixed-fields-4.json > fixed-fields-4.log
fi
"$py" - "$steps" <<'PY'
import json
from pathlib import Path
import sys
steps = int(sys.argv[1])

def read(name):
    return json.loads(Path(f"{name}-{steps}.json").read_text())

prefix = read("staged-prefix")
work = read("staged-work")
fields = read("staged-fields")
terminals = {}
with Path(f"staged-{steps}/stderr.log").open() as stream:
    for line in stream:
        if line.startswith("matter_fgmres_active_work="):
            row = json.loads(line.partition("=")[2])
            if row["status_code"] == 0:
                terminals[row["root"]] = row["last_newton_iteration"]
gates = {
    "reference_prefix_parity": read("prior-parity")["first_accepted_difference"] is None,
    "coupled_trace_parity": prefix["first_accepted_difference"] is None,
    "accepted_roots": prefix["shared_accepted_steps"] == steps,
    "zero_root_assistance": prefix["candidate"]["zero_root_assistance"],
    "terminal_logical_iteration": len(terminals) == steps and set(terminals.values()) == {23},
    "conservative_rollback": bool(work["candidate"].get("verified_rollbacks")),
    "force_parity": not fields["force_regions_over_1pct"],
    "strain_parity": not fields["strain_metrics_over_1pct"],
    "unchanged_connectivity_and_mass": all(fields[name] for name in
        ("exact_element_connectivity", "reference_positions_match", "nodal_masses_match")),
}
print(json.dumps(gates, indent=2))
Path(f"gates-{steps}.json").write_text(json.dumps(gates, indent=2) + "\n")
raise SystemExit(0 if all(gates.values()) else 1)
PY
