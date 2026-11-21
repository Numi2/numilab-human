#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
N="$T/scalp-diagonal-clearance-052"
repo="$W/resting-clearance-and-atrium-010/publication-git"
scope='evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/scalp-diagonal-clearance'
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
cd "$N"
test -s exact-02/verification.json
for plan in experiment volume-plan transport-plan; do shasum -a 256 -c "$plan.sha256"; done
stage=$(mktemp -d "$N/publication.XXXXXX")
for name in README.md screen.py volume.py verify.py check_registration.py publish.sh \
            experiment.md volume-plan.md transport-plan.md *.sha256 surface-screen-01.json; do
    cp "$name" "$stage/"
done
for trial in volume-01 volume-02 exact-02; do mkdir "$stage/$trial"; done
cp volume-01/geometry.json "$stage/volume-01/"
cp volume-02/{geometry.json,candidate.npz} "$stage/volume-02/"
cp exact-02/{verification.json,registration.json} "$stage/exact-02/"
gzip -nc exact-02/bone-witnesses.json > "$stage/exact-02/bone-witnesses.json.gz"
cmp exact-02/bone-witnesses.json <(gzip -dc "$stage/exact-02/bone-witnesses.json.gz")
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; git -C "$HOME/Work/skin-fat-native-integration-209" rev-parse HEAD; } > "$stage/device.txt"
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)
export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
[[ "$parent" == 4e8213ba86669db2d8291128697bf85eedd0e508 ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Verify a connected scalp skin and fat remesh at retained native poses')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$N/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
