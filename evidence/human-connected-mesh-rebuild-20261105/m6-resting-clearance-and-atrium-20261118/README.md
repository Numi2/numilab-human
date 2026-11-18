# M6 skin registration and variable pleural pressure trial

The combined native scene remains unqualified. This checkpoint contains a tested local
skin rebind, exact motion geometry witnesses, and the retained variable-pressure rejection.
The default assets, pressure law, chamber domain, acceptance checks, and solver owners are
unchanged. The trial uses the existing condensed bed-contact skin/fat model over runtime
`f75955f08d6edf0f85027eac7614017ac9fbc996`.

## Native motion

| Deck on SSH Numi's Apple M6 | Accepted motion | Sustained viewer-inclusive rate | Launch to exit |
| --- | ---: | ---: | ---: |
| Rebind, variable pleural pressure | 4.000 s completed | 0.10686× | 0.07847× |
| Rebind, variable pleural pressure, requested 16 s | 5.558 s, then rejection | 0.10766× accepted prefix | 62.696 wall s through failure |
| Rebind, authored constant vascular exterior pressure | 16.000 s completed | 0.10631× | 0.09811× |

The sustained timings include GPU mechanics, physiology, Brain, observers and native
viewer rendering at the retained cadence. They exclude loading and terminal work.
The accepted-prefix rate uses steps 8–2776 and the matching authoritative segment
timestamps. The 16 s control includes 3 complete breaths and 19 filling/ejection cycles.
Its mean last-two-second bed reaction is 706.28 N against 706.32 N body weight.
Root assistance sampled zero, and the 4 s bed-section measurements matched the prior
trial byte for byte.

## Exact capture audit

The collared rebind changed 165 source rows in two arm patches. Inverse rigid-basis
registration retained their initial skin positions. Topology and source row identities
were preserved. The opt-in host patch admits only the hash-pinned rebind derived from
the original thickness source. It preserves the existing bed quadrature and records
both hashes in the support identity.

| Capture | Skin self-intersection pairs | Skin–bone | Skin–muscle | Skin–vessel | Skin-to-target total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prior 4 s | 39 | 2,292 | 9,271 | 270 | 11,833 |
| Rebind, variable pressure at 4 s | 33 | 2,292 | 9,271 | 270 | 11,833 |
| Rebind, authored pressure at 16 s | 359 | 3,043 | 13,152 | 606 | 16,801 |

The pair tables cover 878 typed target banks per capture, plus skin self-intersection.
All 33 four-second skin self-pairs were already present in the prior capture.
One muscle crossing witness was removed and one was added within the same muscle
bank. The original deck has no matching 16 s exact baseline in this bundle.
The largest two late skin-fold clusters contain 268 and 33 pairs. Use the skin-source
face and owner rows in `geometry/skin-late-witness-regions.json.gz` to locate them.
These are sampled surface intersections. Complete through-volume containment and
continuous clearance remain unverified.

A bounded source-position fit was tested offline. Its area-constrained quarter-scale
trial reduced the frozen 4 s count to 11,110 but introduced 4,088 target witnesses
while removing 4,811. Larger scales introduced skin folds. No source-position fit
was admitted into the native scene.

## Right atrium

After the five-second startup continuation, the full accepted pleural gauge pressure
drives the registered thoracic compartment map. At candidate 5.560 s, the chamber
volume solve returned a relative volume error of 0.0000113845 against a tolerance of
0.000020. Its right atrium coordinate was 0.0995103, beyond box 96's upper limit of
0.0991294. The original domain gate rejected the transaction. The failure latch
retained Brain and respiratory time at 5.558 s. The separate native physiology
transaction test passed bitwise rejection rollback and replay for circulation,
respiration and Brain.

The standalone Metal calibration matched pleural pressure and all four chamber
volumes at 347 shared samples before the full-scene rejection. Over its 16 s run, 90
of 2,000 sampled states fell outside the original domain in 11 intervals. The
largest sampled atrial coordinate was 0.1227618. The first rejected pose had no
right atrium self-intersections. At the worst sampled excursion, the right atrium and
right ventricle had zero self-intersections and zero erroneous crossings across
their interface. The test retained their 690 source-declared, oppositely oriented
shared interface faces. The 90 sampled geometry evaluations also retained face
orientation. These authoring checks rounded reconstructed local points once to
Float32. Wider intervals could not be certified by the tested bounds. Continuous
clearance, cavity and wall containment, remaining interfaces and exact Metal
rounding are still required before admitting an extended domain.

## Recover and reproduce

- On the same SSH Numi workspace, restore the source and tool paths recorded in the
  launch receipts. This bundle references the pinned prior build and registered
  assets in `/Users/numi/Work/human-integrated-gaps-20261104`. Some assets are
  source-licensed inputs held locally. The applicable adjacent checkpoint is
  `m6-coupled-correction-20261118`, Git parent `582a12e75fa4a2122197ceb84a9f1e3db3371d17`.
- Regenerate the rebind from the pinned original skin (SHA-256
  `4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5`) and the
  exact source captures with `scripts/rebind_arm_witnesses.py`. Verify resulting
  SHA-256 `85c1daa2365429dd5b76538cba2ae7285086c7a4247e722be09f0a16a89e76c5`.
  A byte-for-byte recovery patch was separately verified and retained on SSH Numi
  with the licensed source. The repository contains the generator and its results.
- The recorded host overlay is relative to the exact prior
  `coupled-thumb-reference-001/NumiHumanRestingVisual.hpp`. Use
  `runs/build-commands.json` and the listed input hashes for the pinned build.
  The receipts preserve executed argv and environment. Legacy inherited `mode`
  metadata in the launch receipts describes the parent deck. The updated
  `scripts/motion.py` synchronizes that metadata for new runs.
- Trial and diagnostic scripts retain the same on-host run layout and owner tool
  dependencies. Run them from the pinned scratch directory or use the already
  staged originals there. Exact exports were losslessly compressed on Numi. Their
  independently verified recovery commands and raw SHA-256 hashes are in
  `runs/*-capture-recovery.json`. Use `result.json` for the matched timing and
  geometry comparisons. Early unsuccessful setup attempts remain in the scratch
  directory with their raw logs.
