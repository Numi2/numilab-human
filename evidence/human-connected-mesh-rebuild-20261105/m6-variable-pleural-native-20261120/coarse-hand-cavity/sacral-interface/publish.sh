#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
S="$T/sacral-interface-046"
repo="$W/resting-clearance-and-atrium-010/publication-git"
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
scope='evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/sacral-interface'
cd "$S"
stage=$(mktemp -d "$S/publication.XXXXXX")
cp README.md ./*.py ./*.md ./*.sha256 classified.json containment-11.json emitted-shell-10.json publish.sh     retire_duplicate_witness_pack.sh "$stage/"
if [[ -f witness-pack-recovery.txt ]]; then
    cp witness-pack-recovery.txt "$stage/"
fi
for i in 01 02 05 06 07 08 09 10 11; do
    trial=partition-$i
    mkdir "$stage/$trial"
    cp "$trial/assessment.json" "$stage/$trial/"
    if [[ -f "$trial.log" ]]; then
        cp "$trial.log" "$stage/$trial/"
    fi
done
for trial in partition-06 partition-10 partition-11; do
    cp "$trial"/gluteal-{29,30}.npz "$stage/$trial/"
done
for trial in contract-10 contract-11; do
    mkdir "$stage/$trial"
    cp "$trial/assessment.json" "$stage/$trial/" 2>/dev/null || cp "$trial"/*.json "$stage/$trial/"
done
for trial in volume-06-{29,30} volume-06-strict-{29,30} volume-{08,09,10,11}-{29,30}     volume-polished-{10-29,10-30,11-30} volume-cavity-{10-29,10-30,11-30} volume-monotonic-{10-29,10-30,11-30}; do
    mkdir "$stage/$trial"
    if [[ -f "$trial/assessment.json" ]]; then
        cp "$trial/assessment.json" "$stage/$trial/"
    fi
    if [[ -f "$trial.log" ]]; then
        cp "$trial.log" "$stage/$trial/"
    fi
    if [[ -f "$trial/unqualified-witnesses.npz" && "$trial" == volume-monotonic-* ]]; then
        cp "$trial/unqualified-witnesses.npz" "$stage/$trial/"
    fi
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
[[ "$parent" == 77699beefad88b9c0a57d96c1c7507ab956586d6 ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Resolve bilateral gluteal reference intersections and retain FEM volume witnesses')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$S/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
