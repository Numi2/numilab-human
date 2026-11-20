#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
P="$W/coarse-hand-cavity-024/posterior-muscle-interface-045"
repo="$W/resting-clearance-and-atrium-010/publication-git"
scope="evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/posterior-muscle-interface"
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
cd "$P"
stage=$(mktemp -d "$P/publication.XXXXXX")
for dir in trial-02 bilateral-03 protected-04 bilateral-05 paired-06 paired-native-128; do
    mkdir "$stage/$dir"
done
cp README.md *.sha256 *experiment.md containment-05.json donor-screen.json donor-bindings.json \
    fit_muscle_interface.py verify_bilateral.py verify_containment.py screen_donors.py \
    bind_donor_geometry.py fit_paired_boundary.py check_native_endpoint.py validate_native.sh \
    retire_duplicate_native_packs.sh native-pack-recovery.txt publish.sh "$stage"
for trial in trial-02 bilateral-03 protected-04 bilateral-05 paired-06; do
    cp "$trial/assessment.json" "$stage/$trial"
done
cp bilateral-05/gluteal-{29,30}.npz "$stage/bilateral-05"
cp paired-06/{posterior-interface-fit.npz,balanced-mass.npz,mass.json,containment.json,balanced-mass.log} "$stage/paired-06"
run=paired-native-128
cp "$run/"{launch.json,execution.json,verified-mass.json,verified-cells.json,verified-shoulders.json} "$stage/$run"
cp "$run/native/resting-coupled.csv" "$stage/$run"
cp paired-native-build/build-commands.json "$stage/$run"
{
    sw_vers
    sysctl -n machdep.cpu.brand_string hw.memsize
    "$HOME/numi-human-prep-venv-20261005/bin/python3" --version
    git -C "$HOME/Work/skin-fat-native-integration-209" rev-parse HEAD
    shasum -a 256 paired-native-build/{PosteriorBedReference.inc,numi-human-native,numilab_human_myosim_visual_probe.mm} \
        "$run/native/resting-human.mrvpack"
} > "$stage/$run/device-and-build.txt"
rg '^(posterior_reference_mass_install=|resting_integrated_(body|throughput)|resting_fem_skin_bed_terminal_history |resting_fascia_fem_terminal_audit=|resting_FEM_terminal_audit=)' \
    "$run/stdout.log" > "$stage/$run/terminal.log"
"$HOME/numi-human-prep-venv-20261005/bin/python3" - "$run/bone-endpoints.json" "$stage/$run/bone-summary.json" <<'PY'
import json
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
report = json.loads(source.read_text())
for pose in ("initial", "terminal"):
    report[pose].pop("witnesses")
target.write_text(json.dumps(report, indent=2) + "\n")
PY
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)

export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
[[ "$parent" == "$(cat "$W/coarse-hand-cavity-024/effective-newton-terminal-044/published-commit.txt")" ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Repair posterior fat and gluteal interface with conservative native mass mapping')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$P/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
