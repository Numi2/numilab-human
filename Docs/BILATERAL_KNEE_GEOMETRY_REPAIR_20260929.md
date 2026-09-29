# Bounded bilateral knee geometry repair — 29 September 2026

The deep-crouch femur/tibia geometry mismatch now passes the existing bilateral
gate. The coupled registration search translates adjacent femur/patella, tibia
and complete-foot groups separately. Every body remains within the existing
1.5 mm interface-refinement bound; no source joint law, range, mesh topology,
rigid owner, route site or material parameter changes.

## Independent compiled-geometry result

The eight-pose audit decodes the emitted native `NHBONES1` payload. All 185
members match their exact source geometry after the prescribed FP32 round trip.
All 320 posed interface checks and 160 bilateral checks pass. The source rigid
program and joint-equality program remain unchanged.

| Deep-crouch femur/tibia difference | Before | Corrected compiled payload | Unchanged gate |
|---|---:|---:|---:|
| Minimum vertex gap | 5.425 mm | 3.111 mm | 4 mm |
| Interface patch p90 | 6.391 mm | 3.961 mm | 4 mm |

The proposal moves 32 source bone instances by translations only. The right
tibia moves 1 mm; the left femur/patella group moves 1.5 mm; the left tibia moves
1.118 mm; the left complete-foot group moves 1.5 mm. Right femur/patella and
right foot remain unchanged. Full remeasurement includes neutral/default
boundaries, all affected projected source poses and femoral-head gates. Earlier
failed proposals remain retained and unapplied; gates are not relaxed.

The audit still exits **2** for five authored dependent-coordinate range
conflicts: left translation2 in three poses, plus bilateral rotation2 in the
functional crouch. Those failures are neither suppressed nor converted into
success by the geometry correction.

## Matched tendon result and remaining failures

The paired `NHTENDON3` recompiles with the original 12 mm ordinary distance and
patch bounds, 25 mm explicit-migration bound, and force-amplification limit 4.
All 832 original source endpoint laws remain unchanged; the 18 named migrations
retain their explicit type. Native admission verifies source/bone identity and
the 632 declared bone envelopes; 10 external surfaces remain unverified.

The aggregate remains 642 envelopes and 190 source points, but two previously
admitted surface attachments fall back to source points:

- `vasint_l` insertion: surface patch conditioning fails.
- `vasmed_l` origin: surface patch conditioning fails.

`vaslat_r` and `vasmed_l` insertions gain envelope admission. These exchanges do
not establish per-endpoint preservation or improved loaded mechanics. The new
bone registration remains a provisional visual candidate; mechanical promotion
requires resolving or qualifying the two lost surface attachments. All changed
explicit migration points and attachment dispositions are recorded.

## Organ coverage correction

The five configured organ surfaces comprise four source-named organ
representations (stomach, pancreas and both kidneys) and one right atrial wall.
The mesh labelled `heart`, member `FJ2439`, has exact source type `FMA9457`
`wall of right atrium`; its `part_of` ancestry under `heart` does not establish
whole-heart coverage. Organ manifests now use exact `is_a` membership to
distinguish this component. Native surface bytes remain unchanged. Source
typing does not certify mesh completeness, clinical registration or physiology.

## Executed evidence

Local append-only evidence root:
`Build/knee-parity-registration-20260929`.

- `registration.v6`: terminal exit 0; 190.283 seconds; applied bounded proposal.
- `audit.v6`: terminal exit 2; five retained source-range failures; zero posed
  interface or bilateral failures.
- `native-neutral-right.v6`, `native-neutral-left.v6`, `native-deep-crouch.v6`:
  terminal exit 0; twelve actual four-angle native PNGs on Apple M4. The 1.4 rad
  deep-crouch capture retains the source left-translation range conflict and is
  a geometry diagnostic, not a loaded result.
- `regression.v5`: 150 passed, 1 optional historical-input test skipped,
  21 subtests passed.
- `native-paired-tests.v6`: 15 passed, 2 optional surface-layer inputs skipped.
- `compiled-geometry-tests.v6` and the source-reference follow-up checks retain
  the complete result, including the obsolete historical-failure assertion and
  its corrected regression on both input registrations.

| Artifact | SHA-256 |
|---|---|
| Input registration | `f6202fea83bb22b0d2a37ba017422e1dccd61b005909bd75efbf1596bfeceb69` |
| Corrected visual registration | `0b7129f788ae91272d03abf9f10b985359f3b9d9569941515b222787ffe71324` |
| Corrected bone payload | `2aaf0567e6a5131c88b599cd56cb605b9d585c2792084b324e84699cabbc9b33` |
| Matched tendon payload | `04f6ad8fac2ac9f9e49df39fc5c6794fa7b693dd2e2dd4d51b0c14bffc5644a8` |
| Executed native client | `a0bddf3b4fc1f685ee4fabea8c25190a742287e545ad13197d3122e2298d0499` |
| Unchanged rigid source payload | `5b756750447862643356517db9dfa6b77d0527d244b449638dca16bb9c520951` |

The retained 10-second September 28 standing video is unchanged and predates
these anatomy repairs. Anterior patellar centroid location, source-frame
agreement, test results and posed geometry do not certify cartilage-facing
orientation, articular contact, pressure, loaded force transfer, clinical anatomy
or whole-body control qualification.
