#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
Q="$W/coarse-hand-cavity-024/effective-newton-terminal-044"
N="$W/coarse-hand-cavity-024/newton-terminal-schedule-043"
repo="$W/resting-clearance-and-atrium-010/publication-git"
scope="evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/coarse-hand-cavity/effective-newton-terminal"
cred='credential.helper=!/opt/homebrew/bin/gh auth git-credential'
cd "$Q"
stage=$(mktemp -d "$Q/publication.XXXXXX")
cp README.md experiment.md preregistration.sha256 diagnostic-90.md diagnostic-90.sha256 \
    build_trial.py run_trial.sh publish.sh archive_superseded_roles.sh \
    staged-prefix-*.json staged-fields-*.json staged-work-*.json prior-parity-*.json \
    fixed-prefix-4.json fixed-fields-4.json gates-*.json "$stage"
if [[ -f role-archive-recovery.txt ]]; then cp role-archive-recovery.txt "$stage"; fi
cp "$N/recovery-043-builds.txt" "$stage"
for owner in app library; do
    cp "$owner/identity.json" "$stage/$owner-identity.json"
    cp "$owner/commands.json" "$stage/$owner-commands.json"
    cp "$owner/"*.patch "$stage"
done
for run in full-4 staged-4 fixed-4 full-90 staged-90; do
    [[ -s "$run/execution.json" ]] || continue
    for name in launch.json execution.json native/resting-coupled.csv; do
        cp "$run/$name" "$stage/$run-$(basename "$name")"
    done
    rg '^(matter_solver_domain_norms=|matter_fgmres_active_work=|resting_common_failure )' \
        "$run/stderr.log" > "$stage/$run-solver.log"
    gzip -dc "$run/stdout.log.gz" | \
        rg '^(human_staged_solver_attempt|human_staged_solver_rollback|resting_integrated_body=|resting_fem_skin_bed_terminal_history |resting_fascia_fem_terminal_audit=|resting_FEM_terminal_audit=)' > "$stage/$run-terminal.log"
done
(cd "$stage" && shasum -a 256 ./* > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null)
export GIT_INDEX_FILE="$stage.index"
export GIT_AUTHOR_NAME=numi2 GIT_AUTHOR_EMAIL=33553832+Numi2@users.noreply.github.com
export GIT_COMMITTER_NAME="$GIT_AUTHOR_NAME" GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"
cd "$repo"
parent=$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)
test "$parent" = "$(cat "$N/published-commit.txt")"
git read-tree "$parent"
while IFS= read -r file; do
    blob=$(git hash-object -w "$file")
    git update-index --add --cacheinfo "100644,$blob,$scope/${file#"$stage"/}"
done < <(find "$stage" -type f | LC_ALL=C sort)
git diff --cached --check "$parent" -- "$scope" ':(exclude,glob)**/*.patch'
tree=$(git write-tree --missing-ok)
commit=$(git commit-tree "$tree" -p "$parent" -m 'Measure Newton staging against the effective native iteration cap')
git -c "$cred" push origin "$commit:refs/heads/main"
test "$(git -c "$cred" ls-remote origin refs/heads/main | cut -f1)" = "$commit"
printf '%s\n' "$commit" > "$Q/published-commit.txt"
rm "$GIT_INDEX_FILE"
git show -s --format='%h %an <%ae> %s' "$commit"
