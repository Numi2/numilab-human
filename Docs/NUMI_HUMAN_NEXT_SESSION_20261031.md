# Numi Human — next development session

Session concluded at the user's request on 2026-10-31. All task workers stopped; no new simulation, geometry search, or mass allocation should be running. Continue on **SSH numi only**. This handoff records progress, not physical admission.

## One entry point and verified state

Read `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/checkpoint-current618-root.json` first. SHA-256:
`093da6ccb76da11c55a0eb7050be9875b07e9912bcc1e26445b52b6ea1c824d0`.

Only `current_verified_state_20261031` is authoritative; outer legacy fields are historical.

| Gate | Skin | Fat |
|---|---:|---:|
| Native quality failures remaining | **873** | **10,345** |
| Tetrahedra | 729,600 | 788,760 |
| Detailed point rows, including inactive rows | 210,298 | 253,712 |
| Nonpositive cells | 0 | 0 |
| Missing / extra complete target faces | 0 / 0 | 0 / 0 |

The combined **reference-pose** O/I/D/wall/all-185-bank audit has zero crossings, including the changed-set self audit. This does not establish motion validity. **Physical admission=false; accepted native steps=0; motion=NOT_RUN; complete current mass allocation=NOT_RUN.**

Current exact candidates:

- Skin: `outer-O-current612-canonical615-root/skin-current-O612-canonical-parent-unions.npz`, SHA `075a323111aea265b88d07eb12f83aa7902e7dd6323d86082f8a8828cbb492f9`.
- Fat: `fat-current-D-546-597-root/run011-target397-current/canonical-v3/fat-current-O612-D597-I50-canonical-parent-unions.npz`, SHA `474f98724b2c8cd6bbdbe3dd46ab6402250b1f613ecbaeec58feb1453d802d11`.
- Joint exact context: `joint-current-O612-D597-context-613-root/audit354/report.json`, SHA `4f457bbbab1d858061ac141b383cff5648ff5d2080a2a39cc4b39853b6fdb73c`. Its config pins the current typed O612 / I50 / D597 surfaces.
- Fat canonical receipt SHA `85678c5817fc928afdb18739ba9e3402cc298670487ebe78d682d8bcc362178d`.

Relative paths in this document are below:
`/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706`.

## Work efficiently: the next implementation round

**The finish line is zero quality failures in both physical layers, a complete conservative mass map, and accepted native motion. Another isolated mesh improvement or source-loader test is not completion.**

1. **Use the productive connected batch method, not repeated visual adjustment.** The last D batch transferred 152 operations and removed 249 fat failures (10,594 → 10,345), preserving coordinates and positive cells. Six safe O operations removed nine skin failures (882 → 873). One O operation caused five exact intersections; pruning that operation alone retained all safe D work and restored zero crossings. Keep this reject-by-operation workflow.

   Start from `fat-current-D-546-597-root/run011-target397-current/canonical-v3/next-D-frontier-checkpoint.json`, SHA `1bf31d99d4bdd6c09586f1c066034f9c23191a81b77cb3137e091707268a300d`. The saved measured positive-edge map is `fat-current-D-546-597-root/run003-measured-positive-subset-001/allowed-D-positive-gain-edge-map.npz`, SHA `89dc816c1fad4c5650a5b19ec0731e26999dd7f9287d4ee9b6ab4176a4b1782a`.

   The previous search exhausted 100,000 orientation trials after 22,792 edges; it has **no resumable cursor**. Rekey retained proposals by exact F32 endpoints and regenerate full endpoint-star conflicts against the new topology. Add deterministic frontier persistence before another expensive search; do not restart the same prefix. Exclude already-applied and exact-witness-rejected operations.

2. **Repair the shared I boundary as a coupled Skin/Fat problem.** Skin's remaining defects are predominantly at the I boundary; repeating the low-yield O collapse search is a poor next priority. Regenerate a current regional/witness census, then process independent I patches together with the existing paired-I operators. Preserve per-material quality nonincrease, strict combined improvement, exact paired volume accounting, positive cells, anatomical routes/rims and back412.

   Existing owners include `repair_paired_I_cross_tissue_471_root.py`, the indexed paired-I transfer/binders, and the `paired-I-local-cavity-527-root` workflow. Their hard-coded older point prefixes and blanket construction-region exclusions must be rebound to current arrays. Construction cells are not anatomical anchors. D operations already integrated into the current Fat do not require carrying obsolete reservation masks forever. Do not bypass actual anatomical constraints.

   Keep exact boundary rings, thickness, identity and source ancestry. If local topology operations cease producing useful gain, use connected cavity remeshing with the existing compiler and exact boundaries. Do not repeat larger versions of the same failed solve indefinitely.

3. **Finish the authoring/provenance adapter as one aggregate closure repair.** The source-only D authoring adapter is still incomplete. Do not confuse its checks with physical admission or silently change the unchanged actual230 gate.

   The aggregate D597 namespace has 470 active IDs, 353 active derived-D IDs, and 153 reachable parent IDs after the documented correction of 62000202. Every parent now has pinned evidence: 119 from current typed tuples, 29 BONE parents from parent366, four D parents through the 62000202 crosswalk, and 62000200 from its producer. Preserve all original tuples, mixed-source provenance and the corrupt original parent list beside its authenticated replacement.

   Remaining implementation: v9 does not yet consume the authenticated 62000200 evidence and is **uncompiled/untested**. It will still fail there. Complete all unresolved closure categories before another full loader run, then run existing230 on the **current** O612/I50/D597 roles. 62000200's registry face_count=96 disagrees with the pinned asset/producer's 190 rows; preserve and explicitly resolve that metadata discrepancy. Source layer3 is BONE, not D. Role-directed derived authoring projection must retain original ancestry and the source-to-authoring crosswalk.

4. **Finish mass mapping on the exact final topology.** Skin615 P1/143 support passed independent all-cell replay: support SHA `e73399859c1fa1e0bd47f60c3b551c13b405ea0f06a147daccf0738e3ee42f52`, replay SHA `160221eb5fa29a0c0b2d7a1f6fa908098eccd95f45eb0b8f2f0fef0127e6110c`. The replay report's edge-count label mistakenly reports 70 regions as edges; the builder reports 140 undirected edges. Correct that reporting error explicitly without claiming a new physical result.

   Fat597 run002 support is an older-topology node cache, not support for the final Fat candidate. Its geometry is `6677a4bb…35054`, support `6779cbaa…035d4`. Verify exact point-row F32 identity, reuse the valid node projections, and rebuild current cell P1/ties/ancestry/143 support. Never use whole-body averages to conceal unmapped regions.

   Emit matching final role .node/.ele and child labels; regenerate FSSUP090/FSSADJ01; rebind FSSCARR01 witness child rows and distances to final centroids while preserving source barycentric quadrature and source330 role/global IDs. Then run a bounded current paired allocator and independently replay every source/corner/child/donor/material/region/radius constraint.

   Source330 mass remains immutable: Skin 3.4525704454739734 kg; Fat 10.487108517986506 kg. Preserve M/4, existing 5e-13 relative closure, density bands, material/semantic support and declared transport radius. If a verified current cut is infeasible, retain the anatomical witness and repair its actual geometry/support problem. Do not label time/capacity exhaustion infeasible and do not relax constraints merely to pass.

   The exact receiver-signature quotient now has regressions for missing cut edges, warm-extra locking, and full-graph parity. Frozen snapshot: `skin-current-mass-550-nativepreflight/offline-receiver-signature-quotient-005/source-snapshots/cutclosure-warmrelease-v2-exact-regression/`; receipt SHA `defa7d869a60e94610f43246b4a94d744429b180d833defb3225ae0b462482de`. Historical run009's infeasibility conclusion was contradicted and is not evidence against the current model.

5. **Admit, integrate and verify real native motion.** Require **Skin quality failures=0 and Fat quality failures=0**, complete exact oriented boundaries, zero unexplained crossings, positive skin/fat cells, current conservative mass mapping, valid source identity and physical transfer. Re-run exact reference/retained-pose checks before native confirmation. Use the existing active-domain compactor and native Numi owners; preserve attachments, mass, thickness, contact, Brain, cardiac/respiratory coupling and accepted-state rollback.

   Run the integrated scene for 16 simulated seconds, then 60 seconds only if the short run passes. Check motion/strain/contact and rejected-step state restoration; do not substitute render-only skin changes, wrist locks or hidden assistance. Keep the final viewer confirmation separate from the numerical repair loop.

   The original full-Human qualification remains outstanding: 330-second baseline and matched respiratory intervention, continuous native recording, coupled traces and honest end-to-end real-time factor. These are later gates, not results already obtained.

## Resource, publication and quality rules

- Run one substantial geometry/CPU/GPU task at a time through the existing lock:
  `PY /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/run_geometry_serial.py COMMAND ARGS`.
  Here PY=`/Users/numi/numi-human-prep-venv-20261005/bin/python`. The wrapper is invoked with Python; it takes no `--` separator. Release the lock before Git push.
- Parallel workers may prepare independent code/provenance while the heavy job runs. Integrate passing batches continuously; don't let branches diverge.
- Order gates cheapest first: schema/finite/orientation/ancestry → local volume quality → typed replay → one combined exact context → target + immediate-parent canonicalization → independent mass checks → native run. Stop at the first failure and retain its deterministic witness. Preserve safe batch operations.
- Measure quality failures removed per unit work and coverage. Cache loaded arrays, exact point projections and incidence; rebuild topology-dependent constraints when required. Avoid full repeated audits of untouched regions unless the cumulative proof requires them.
- Bound an unproductive method (four iterations or ~20 minutes), record the reason and change it. Do not spend another session on low-yield tweaks.
- Disk is below the 12 GiB qualification floor. Use verified redundant outputs/cache or hash-verified lossless compression; preserve sources, current geometry, recovery evidence and parent chains. Do not repeat known no-gain compression of the 565 MB p1-anchor CSV.
- Native owner repo: `/Users/numi/Work/skin-fat-native-integration-209`, HEAD `f75955f08d6edf0f85027eac7614017ac9fbc996`, dirty. Human owner: `/Users/numi/numilab-human`, HEAD `fdab86eb28ae597e4c50e0a3b8e829a6c0e89cec`, dirty. Preserve both.
- Human publication checkout: `/Users/numi/Work/human-volume-proof-publication-20261031`. It is sparse; archive payloads remain in Git even if absent in its worktree.
- Native publication checkout: `/Users/numi/Work/numi-lab-723-publication-20261030`, clean baseline/main `d343dc2e713163c6a19d4c8886e2ced758f0e7ec` at session inspection. **Do not copy the quotient snapshot headers over this checkout wholesale.** They contain stale owner layouts and would delete newer carrier/mapping code. Port a reviewed additive patch and run owning tests before publishing to Numi2/numi-lab.
- Publish coherent tested increments to the owning repositories without paid CI. Keep geometry, replay, physical, physiological and performance claims separate.

The anatomy remains a labeled reference assembly, not measurements of one individual. Preserve the working back repair and unrelated user changes. Efficient work means delivering more verified physical capability per round, while keeping the acceptance conditions intact.

## Frozen source-closure handoff

Read `registered-authoring-view-590-D-593-root/run-010-D597-aggregate-closure-handoff/aggregate-closure.md` and its adjacent JSON/SHA256SUMS for the full authenticated 153-parent inventory and exact next command. JSON SHA: `2d5246d73c459443d2b7ad0fa0314920b7144635bf727cf4dc91d99c0f9b0b58`. The command requires a completed, pinned v10 first. Current v9 is untested; no D597 actual230 run has passed.
