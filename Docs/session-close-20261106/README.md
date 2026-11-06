# Numi Human — session closed, 6 November 2026

Stopped at the user's request. The goal is **unfinished**. Simulation processes and queued trials were stopped; nothing is scheduled to resume. No further repairs or benchmarks were performed during closure.

## Recovery

All closure material belongs to [Numi2/numilab-human](https://github.com/Numi2/numilab-human). Bulk assets, native captures and source snapshots are in the [verified recovery release](https://github.com/Numi2/numilab-human/releases/tag/numi-human-20261105-intermediate-recovery). See the adjacent `archive-index.json` for coverage and exact asset links. Each new tar.zst was downloaded independently, every member restored and SHA-256 checked. Local originals were retained.

The full release inventory maps original paths and identical-content aliases to archive members. It preserves failed trials as failures. This is a project checkpoint, not a machine backup: OS/toolchains, credentials, personal data and Git object databases are excluded. Historical receipts also contain unresolved old hash references; these are listed rather than represented as recoverable experiments. Required inputs of the selected 565, 645 and 655 launch plans are accounted for.

## Last actual result

| Item | Status |
| --- | --- |
| Human preparation source | `2077bf1336a10e539137f723b36b25085af81683`, pushed; 119 Human tests passed. Handles growing cavities consistently in construction, response screening and witness replay. |
| Unchanged comparison | Trial 495 geometry and matched 565 control. A comparison reference, **not** a completed resting-human qualification. |
| Last completed native candidate | Trial 645: four accepted roots, **62.5 microseconds simulated**, 65.287 seconds body time, 241.864 seconds launch time. Body-only factor **0.000000957×**; this short release transient does not establish sustained throughput. |
| Candidate fidelity | Versus unchanged reference: **102 force and 139 strain gate failures**. Not promoted. Earlier 602/624/631 variants also failed; they remain rejected evidence. |
| Last prepared candidate | Trial 652: 855 screened fixed-boundary cavities, exact surface coordinates and provenance preserved. Source/geometry and analytical response checks only. |
| Stopped trial | 655 was terminated during preparation after about 15 seconds. No physical steps or completed source/mass admission. This was a user stop, not a physical rejection. Comparison 656 did not run. |

The 855-cavity packet is `reference-cavity-response-652/response/fixed-boundary-volume-proposal.npz`, SHA-256 `79d22fdd79cc2fc60ca12d212d35ad340d0cb52b76df5e06a27043d2662c2313`. It has **no demonstrated native fidelity or speed improvement**.

## Unresolved

The requested continuous, mechanically coupled human and ≥0.1× performance have not been delivered. The large skin/fat discretization, active-solve convergence, assembly/contact cost and source-to-native force/strain fidelity remain blockers. Supported equilibrium, anatomy through motion, sustained breathing/cardiac/gas coupling, five-minute baseline, intervention and continuous native recording remain unqualified. Positive source cells and short accepted steps do not close those gaps.

Do not resume the growing chain of conditioning patches automatically. If work resumes, first decide whether to retain this discretization and solver approach; the latest local conditioning batch has not demonstrated a useful full-scene result. Keep the unchanged reference and rejected evidence distinct.

## Exact checkpoint locations

All paths below are on **SSH Numi**.

- Work/evidence: `/Users/numi/Work/human-integrated-gaps-20261104`; closure: `session-close-657`.
- Human publication checkout: `/Users/numi/Work/human-checkpoint-owner-20261105`.
- Native dirty checkout: `/Users/numi/Work/skin-fat-native-integration-209`, base `f75955f08d6edf0f85027eac7614017ac9fbc996`. Its unpublished base commit is bundled with prerequisite `868486402da2ff0a208bf60c267df0928e2f91e0`; tracked patch and untracked files are archived. Restore the base before applying the patch.
- Runtime owner: `/Users/numi/Work/current-fractional-020-publication-20261101`, revision `f8efdb0dc802284137372ef10bb9c48f62850850`.
- Native binary SHA-256: `3900d587e78526d70157f864ba438a48acf48b652fd873e6542319ebf3e2bc8a`; library/shader hashes, exact commands, configuration and inputs remain in `reference-batch-native-655/install-motion.plan.json`.

For recovery, download the index and named manifests/assets with `gh release download`, verify the compressed SHA-256, decompress with `zstd -d --long=29` into a staging directory, verify member hashes, then restore the catalogued paths/aliases. Do not launch a retained queue script merely to restore files. The archive preserves the state; it does not qualify it.
