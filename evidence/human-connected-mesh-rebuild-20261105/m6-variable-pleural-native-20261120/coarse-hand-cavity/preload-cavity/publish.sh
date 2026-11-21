#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
N="$T/preload-cavity-topology-051"
repo="$W/resting-clearance-and-atrium-010/publication-git"
scope='evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/preload-cavity'
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
cd "$N"
test -s native-result.json
for plan in experiment regional-plan mapping-plan native-plan; do shasum -a 256 -c "$plan.sha256"; done
stage=$(mktemp -d "$N/publication.XXXXXX")
for name in README.md cavity.py regional_cavity.py map_cavity.py build_native.py verify_native_cavity.py test_cavity.py \
            unit-tests.log run_native.sh publish.sh load-predictor.patch geometry-audit.patch \
            experiment.md regional-plan.md mapping-plan.md native-plan.md native-pack-recovery.txt *.sha256 \
            report-01.json regional-01.json native-result.json; do
    cp "$name" "$stage/"
done
mkdir "$stage/remap-01" "$stage/fit-01" "$stage/native"
cp remap-01/{receipt.json,support-hull-01.json} "$stage/remap-01/"
cp fit-01/{result.json,trace.json,warm-trace.json,geometry.json,last-equilibrated.npz,seed.npz} "$stage/fit-01/"
cp native-build-03/{build-commands.json,build-identity.json} "$stage/native/"
for run in native-setup-rejected-01 native-setup-rejected-02; do
    mkdir "$stage/$run"
    cp "$run"/{execution.json,launch.json} "$stage/$run/"
    tail -6 "$run/stderr.log" > "$stage/$run/stderr-tail.log"
done
if [[ -f native-root-128/execution.json ]]; then
    cp native-root-128/{execution.json,launch.json} "$stage/native/"
    for name in terminal.log verified-cells.json verified-mass.json verified-shoulders.json; do
        if [[ -f "native-root-128/$name" ]]; then cp "native-root-128/$name" "$stage/native/"; fi
    done
    if [[ -f native-root-128/native/resting-coupled.csv ]]; then
        cp native-root-128/native/resting-coupled.csv "$stage/native/"
    fi
fi
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; git -C "$HOME/Work/skin-fat-native-integration-209" rev-parse HEAD; } > "$stage/device.txt"
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)
export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
[[ "$parent" == ce4538825614aa689a7f33578f402e9424212ff4 ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope" ":(exclude)$scope/*.patch"
git -c core.whitespace=-blank-at-eol diff --cached --check "$parent" -- "$scope/*.patch"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Remesh the posterior fat load cavity and conserve registered source mass')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$N/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
