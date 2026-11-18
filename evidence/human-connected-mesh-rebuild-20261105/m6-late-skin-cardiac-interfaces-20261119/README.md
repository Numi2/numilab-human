# Late skin motion and cardiac interfaces on Apple M6

The combined sixteen-second trial remains unqualified. Native control completed with
authored vascular exterior pressure. Variable pleural pressure triggered the
original geometry rejection after 5.558 s. No default assets, acceptance
thresholds or physical laws changed.

## Native motion and support

| Trial | Accepted simulated time | Sustained viewer-inclusive rate | Launch-to-exit rate | Exact skin self / skin-to-tissue pairs |
| --- | ---: | ---: | ---: | ---: |
| Previous shoulder release, authored pressure | 16.000 s | 0.1063× | 0.0981× | 359 / 16,801 |
| Wider shoulder release, authored pressure | 16.000 s | 0.1069× | 0.0987× | 55 / 14,606 |
| Wider release with local skin rebind, authored pressure | 16.000 s | 0.1073× | 0.0988× | 51 / 14,606 |
| Wider release, variable pleural pressure | 4.000 s | 0.1077× | 0.0809× | 36 / 13,597 |
| Wider release, variable pleural pressure, 16 s planned | 5.558 s, rejected | See result.json | 62.67 wall s to failure | No terminal capture |

The release initializes the existing bilateral shoulder coordinates from
0.10 to 0.18 rad. Muscle actuation, joint limits and bed support evolve
under the prior solvers. All 286 hand-to-hip or leg skin self-pairs in the
previous late capture were clear at the wider-control terminal pose. The
two 16 s controls use 8,000 2 ms steps and include three complete breaths
and 19 cardiac filling and ejection cycles. The last two seconds average
706.36 N of bed reaction against 706.32 N body weight. Sampled root
assistance in force and torque is zero. The 32 bed section rows are
byte-identical between the wider controls. Support loads use the existing
condensed skin/fat bed layers. Their reaction does not establish mutual
skin, muscle, bone or vessel contact inside the body.

At 4 s, the prior narrower variable-pressure deck had 33 skin self-pairs
and 11,833 skin-to-tissue pairs. The wider pose trades early clearance for
late hand separation, so its partial success does not qualify the release.
The rebind edits 243 collared influence rows while retaining source faces,
vertex identities, and the reference positions to within 60 nm. On frozen
4 and 16 s replays it preserves all 878 bank-pair sets and predicts 36 and
48 self-pairs. The exact native 16 s result has 51 self-pairs. Seven prior
witnesses cleared and three new ones appeared. The remainder includes
2,850 skin–bone, 11,486 skin–muscle and 270 skin–vessel pairs. The rebind
remains an opt-in experimental asset. Broader containment and continuous
motion are outside these exact snapshot checks.

Sustained timing includes GPU mechanics, physiology, Brain, observers and
native presentation at the retained cadence. It excludes startup and
terminal work. Both full runs retained accepted native geometry. This
checkpoint contains no continuous viewer video.

## Variable-pressure rejection and chamber-wall geometry

At candidate 5.560 s, the native failure latch recorded relative volume
error 0.0000113845, under the existing tolerance of 0.000020. The right
atrium coordinate of 0.0995103 exceeded box 96's upper bound of 0.0991294.
The geometry-domain gate rejected the candidate and retained accepted body
and respiratory state at 5.558 s.

The right atrium and ventricle shared partition has zero exact intersection
pairs at the three probed coordinates, excluding the 690 properly reciprocal
source facets. The corresponding interval proof leaves unresolved pairs
over the extended range. A wider seven-surface probe, including all four
chambers and registered myocardial banks, resolves why simply widening the
domain is unsafe. It finds 19 genuine triangle pairs at the old boundary,
31 at the first rejected coordinate, and 43 at the deepest sampled
excursion. The local Metal probe uses the retained compiler policy, with
bitwise verification against sequential fused Float32 accumulation. It
does not run a new accepted body trajectory.

An experimental local triangulation repair preserves all shared
interface facets and vertex bindings, with deterministic source-row
ancestry. An independent exact repeat measures 18, 24 and 28 pairs after
the repair. The right atrium–myocardium bank pair changes from two to
zero at the first failed coordinate and from seven to three at the deepest
one. Other chamber and myocardial interfaces remain implicated. The
largest bound on relative volume change is 1.36e-8. Neither this bound
nor the point checks establishes wall containment or continuous
certification. The native scene still uses its original cardiac input
and geometry gate.

## Recovery

- The adjacent `m6-resting-clearance-and-atrium-20261118` checkpoint at
  `6c94acdfc4a2f75c10531cbd0430164a19eafa97` contains the parent build,
  earlier exact capture audits, rollback results, and cardiac calibration.
- Run the trial tools from
  `/Users/numi/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-003`
  on SSH Numi with the pinned neighboring inputs and retained tooling.
  `runs/*-launch.json` records the exact native argv, environment and
  asset hashes. `runs/build-commands.json` pins the candidate host build.
  The experimental overlay is included separately and remains unpromoted.
- `prepare_common_cardiac_geometry.py` reconstructs the seven-surface probe
  from the registered source SHA-256 and common basis in the receipt.
  `right_heart_interval.cpp` can be built with Apple clang using
  `-O3 -std=c++20 -Werror -Wall -Wextra -ffp-contract=off -fno-fast-math`.
  The five interval fixtures ran on Numi, as recorded in `runs/interval-fixtures.log`.
  `common_field_probe.metal` was compiled using
  `xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math` and linked
  using `xcrun -sdk macosx metallib`. The retained determinant/volume
  authoring dependency is `tools/audit_cardiac_flip_volume.py` at the
  parent revision, staged on Numi as `retained_audit_cardiac_flip_volume.py`.
- The compressed pack receipts specify exact recovery commands and raw
  hashes. Source meshes and large captures remain on Numi. This Git
  checkpoint contains methods, source hashes, compact physiology and
  geometry witnesses, and the failed native transaction.
