# Human mechanics, joint research, and evidence

Numi Human owns the biomechanics material in this handoff. The snapshots cover joint mechanics, muscle and tendon models, contact and reference probes, and selected experimental evidence. The branch ledger also covers the archived conditioned-limit comparison branch. Its tip is already an ancestor of Numi Lab `main` and records workflow and script changes recoverable from the archive.

This handoff indexes 36 former Numi Lab branch heads and imports 146 byte-preserved source and evidence files. The snapshots are historical research inputs stored under `docs` or `Docs`. Reproducing a result requires the original source revision, hardware, inputs, and acceptance criteria recorded in its history. The owning repository’s live build keeps its existing source and targets.

## Branch and file records

- [Branch ledger](branches.tsv) records the original branch name, tip, root tree, archive tag, comparison base, and selected snapshot paths.
- [Changed paths](changed-paths.tsv.gz) lists every path changed against the ledger’s comparison base, including deletions and original Git object IDs. Branches that share their tip with their comparison base have an empty delta, and branches without a shared base are compared with an empty tree.
- [Source manifest](source-files.tsv) and [SHA-256 checksums](source-files.sha256) record the original revision, source path, Git mode, blob ID, and content of each imported file. From this directory, run `shasum -a 256 -c source-files.sha256` to verify the imports.

## Original Numi Lab draft reviews

- [Open Knee native reference (Numi Lab #6)](https://github.com/Numi2/numi-lab/pull/6). The native source-time continuation study reported no accepted microsteps at its convergence gate. The branch and full review diff preserve that result for follow-up. [Original diff](source-reviews/numi-lab-pr-6.diff).
- [Source-bound NHBONES ABI 3 (Numi Lab #7)](https://github.com/Numi2/numi-lab/pull/7). The draft changes the native visual probe. Numi Lab `main` carries later source-record-index validation in that probe. This package preserves the original single-file review diff and its exact branch tip. [Original diff](source-reviews/numi-lab-pr-7.diff).

[Review diff checksums](source-reviews/SHA256SUMS).

## Restore a branch

The complete Git history and bulk artifacts remain available from the annotated archive tags in [Numi Lab](https://github.com/Numi2/numi-lab/tags). Copy the original branch name and archive tag from the ledger. This example fetches the complete object history for that tag into a fresh local repository and recreates the original branch.

```sh
git init --bare restored-numi-lab.git
branch='<source_branch from branches.tsv>'
tag='<archive_tag from branches.tsv>'
git -C restored-numi-lab.git fetch --no-tags https://github.com/Numi2/numi-lab.git \
  "refs/tags/$tag:refs/tags/$tag"
git -C restored-numi-lab.git branch "$branch" "refs/tags/$tag^{commit}"
git -C restored-numi-lab.git rev-parse "refs/heads/$branch" "refs/heads/$branch^{tree}"
```

The final command prints the source tip and root tree from the ledger.
