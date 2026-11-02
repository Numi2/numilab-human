# Current native checkpoint: six serialization replacements

All work ran on SSH Numi. Source, three matched GPU parity runs, owner syntax
and native attachment/deformation/rollback/replay controls are published in
Numi2/numi-lab main. Author and committer are numi2.

- Source: https://github.com/Numi2/numi-lab/commit/3746de2401f7c3a52855a2f0eb7ff89bf29a9294
- Evidence: https://github.com/Numi2/numi-lab/tree/7d22e8a971d03be06c009dd27a51d861699ffa9c/tools/evidence/human-whole-body-serialization121-20261102

Node validation, force/reaction audits, sparse anchor projection, GPU-gated
parallel working-set reset, deterministic tet-corner incidence, and local
blood-particle compaction are implemented and tested. Conditional paths were
exercised in GPU fixtures; blood MPM is disabled in the current full scene.

Native122 stopped during preparation after50.254s under the retained12GiB
disk guard. No physical step or motion was accepted; integrated throughput
and0.1x remain unmeasured. Source geometry, mass mapping, material laws and
physical acceptance gates are unchanged. No visual inspection was performed.

The attempted lossless shared-dump compression yielded no allocation gain
and was not promoted. All12 original paths, required mass proofs and OS swap
remain intact. See integration-checkpoint.json for the precise next blocking
work: full-scene preparation/working-set memory admission, then accepted
coupled timing and motion. Do not repeat the same paging-bound trial.

Unrelated untracked Human evidence was preserved and excluded from this commit.
