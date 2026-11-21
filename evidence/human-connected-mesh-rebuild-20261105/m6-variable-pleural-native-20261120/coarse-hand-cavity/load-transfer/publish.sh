#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
D="$T/load-transfer-capacity-048"
C="$T/gluteal-conforming-remesh-047"
repo="$W/resting-clearance-and-atrium-010/publication-git"
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
scope='evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/load-transfer'
cd "$D"
stage=$(mktemp -d "$D/publication.XXXXXX")
cp ./*.py ./*.sh README.md trial-contract.md trial-contract.sha256 \
    existing-load-routes.json separated-posterior*.json verified-map-02.json \
    archive-members.txt archive-recovery.txt "$stage/"
mkdir "$stage/gluteal-conforming-remesh"
cp "$C"/{screen_conforming_remesh.py,surface-split-screen.json} "$stage/gluteal-conforming-remesh/"
for a in engagement-fit-01 engagement-fit-02 engagement-fit-03 engagement-dose-050 engagement-dose-075; do
    mkdir "$stage/$a"
    for file in assessment.json mass.json balanced-mass.log; do
        if [[ -f "$a/$file" ]]; then
            cp "$a/$file" "$stage/$a/"
        fi
    done
    if [[ -f "$a.log" ]]; then
        cp "$a.log" "$stage/"
    fi
done
cp engagement-fit-02/{posterior-interface-fit.npz,balanced-mass.npz} "$stage/engagement-fit-02/"
mkdir "$stage/overlay-source-02" "$stage/native-engagement-build-02"
cp overlay-source-02/source-identity.json "$stage/overlay-source-02/"
cp native-engagement-build-02/{build-identity.json,build-commands.json} "$stage/native-engagement-build-02/"
for a in native-engagement-128 native-engagement-128-02; do
    mkdir "$stage/$a"
    find "$a" -type f -maxdepth 1 -exec cp '{}' "$stage/$a/" \;
done
{
    sw_vers
    sysctl -n machdep.cpu.brand_string hw.memsize
    "$HOME/numi-human-prep-venv-20261005/bin/python3" --version
    git -C "$HOME/Work/skin-fat-native-integration-209" rev-parse HEAD
} > "$stage/device.txt"
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)

export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
[[ "$parent" == 2dcf93fddf35f6ee5ceeba9f36cbb4d8996ff141 ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Measure and verify posterior tissue load transfer with mapped native mass')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$D/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
