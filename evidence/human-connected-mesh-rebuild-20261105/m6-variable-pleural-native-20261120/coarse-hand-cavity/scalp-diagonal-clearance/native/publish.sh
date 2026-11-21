#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
N="$T/scalp-diagonal-clearance-052/native"
repo="$W/resting-clearance-and-atrium-010/publication-git"
scope='evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/scalp-diagonal-clearance/native'
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
(cd "$N/.." && shasum -a 256 -c native/trial.sha256)
cd "$N"
stage=$(mktemp -d "$N/publication.XXXXXX")
cp README.md trial.md trial.sha256 build.py patch.py verify_mass.py qualify.py run.sh archive.sh publish.sh archive-recovery.txt "$stage/"
cp build-01/{build-commands.json,build-identity.json,registration-provenance.json} "$stage/"
cp root-128-01/{launch.json,execution.json,comparison.json,qualification.json,verified-cells.json,verified-mass.json,verified-shoulders.json} "$stage/"
cp root-128-01/native/resting-coupled.csv "$stage/"
rg '^(posterior_reference_mass_install=|resting_fem_skin_bed_accepted_root_reaction |resting_fascia_fem_terminal_audit=|resting_integrated_body=|resting_initialization=)' \
    root-128-01/stdout.log > "$stage/terminal.log"
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; git -C "$HOME/Work/skin-fat-native-integration-209" rev-parse HEAD; } > "$stage/device.txt"
(cd "$stage" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)
export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
[[ "$parent" == 84ddfaa1898cd317284e6c56a561c21b41b63ded ]]
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope"
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Run the derived scalp skin and fat reference through native Metal motion')
git -c "$cred" push origin "$commit:refs/heads/main"
[[ "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" == "$commit" ]]
printf '%s\n' "$commit" > "$N/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
