#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
N="$W/coarse-hand-cavity-024/sacral-retinacula-049"
repo="$W/resting-clearance-and-atrium-010/publication-git"
scope='evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/sacral-preload'
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
cd "$N"
stage=$(mktemp -d "$N/publication.XXXXXX")
cp ./*.py ./*.md ./*.sha256 ./publish.sh support-hull-01.json pressure-sensitivity-01.json screen-02.json "$stage/"
mkdir "$stage/descent-fallback"
cp descent-fallback/*.patch "$stage/descent-fallback/"
for trial in preload-normal-01 preload-lateral-01 preload-warm-normal-01 preload-warm-strip-01 preload-shift-normal-01 preload-shift-strip-01 preload-deep-slip-01; do
    mkdir "$stage/$trial"
    cp "$trial"/*.json "$trial/seed.npz" "$stage/$trial/"
    if [[ -s "$trial/last-equilibrated.npz" ]]; then
        cp "$trial/last-equilibrated.npz" "$stage/$trial/"
    fi
done
{
    sw_vers
    sysctl -n machdep.cpu.brand_string hw.memsize
    git -C "$HOME/Work/skin-fat-native-integration-209" rev-parse HEAD
} > "$stage/device.txt"
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)
export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
[[ "$parent" == 57c6445c2e2465a4979a6e7e3e80e9065e72085c ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope" ":(exclude)$scope/descent-fallback/*.patch"
# A unified diff uses a single space as the prefix for blank context lines.
git -c core.whitespace=-blank-at-eol diff --cached --check "$parent" -- "$scope/descent-fallback"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Test distributed flat-bed preload and quantify shape and contact limits')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$N/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
