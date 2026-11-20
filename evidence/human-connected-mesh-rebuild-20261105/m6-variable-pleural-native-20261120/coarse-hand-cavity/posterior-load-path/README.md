# Native whole-shell and shoulder checks on SSH Numi

Apple M6, macOS 27.0 build 26A425. Native Metal replay with active articulated
mechanics, deformable contact, Brain, circulation and variable pleural pressure.
The nominal step is 15.625 µs. The retained scene uses a fixed bed and reports
zero root assistance. The recorded pose is a release transient.

| Scope | Result |
| --- | --- |
| Full-resolution skin/fat construction | The retained 1,448,804-cell reference clears the original 125 inversions and 154 shape failures. The [independent mass closure](../../fine-skin-fat-mass-routing/README.md) conserves 13.94458033634514 kg. |
| Coarse native skin/fat mass | The separate 28,632-cell shell conserves 13.939679838742222 kg across 628 typed pools. Its previous native audit matched the emitted cell and lumped nodal masses. |
| Native outer skin against registered bone | 2,873 start and 2,877 endpoint triangle pairs. |
| Native bonded skin/fat interface against bone | 5,265 start and 5,274 endpoint pairs. |
| Native deep fat boundary against bone | 9,712 start and 9,746 endpoint pairs. |
| Real bed support | 0.435 N across three rows at accepted step 128, against 952.864 N of body weight. |
| Bilateral superior glenohumeral attachment paths | Five samples through accepted step 128. No interior source-bone hits after classifying the registered attachment face. All 34 ligament rows passed the native terminal audit. |

The whole-shell audit tests all three cooked boundaries against 469,136
registered bone triangles. The endpoint uses rounded startup bone vertices
transformed by accepted rigid poses. The total is 17,850 start pairs and
17,897 endpoint pairs. The previous 28-pair pelvic result only covers its
changed deep boundary faces. Triangle pair counts retain side, layer, source
bone and face identities. They measure static surfaces, with no containment
or continuous-motion claim.

The shoulder audit uses the fixed published material points and all 185
registered bone mesh sources. The left humeral path hits its own triangle 645
within 4.33 µm along the rounded path at the five sampled poses. The rounded
endpoint differs by 10.8 nm from the registered barycentric point. Re-evaluating
that same path at the exact material point removes the hit at every sample.
Other triangles are audited without this classification. Loaded shoulder
endurance remains unqualified.

The bed-normal source screen sampled 523 gluteal quadrature rays. Five found
admissible chords to the pelvis without an intervening bone, totaling
12.40 cm² of projected area. None supplied an admissible femur chord. This
rejects the sampled direct load-path proposal. Oblique or continuous posterior
load paths remain eligible; the native forces and supports are unchanged.

## Solvers and acceptance

| 90-root run | Body time | Accepted | Outcome |
| --- | ---: | ---: | --- |
| Full-budget reference | 69.13 s | 90 | Comparison control |
| First-correction latch | 67.30 s | 90 | 116 force and 271 strain metrics exceed 1% |
| Independent-argument guarded stage | 39.86 s | 90 | 96 force and 221 strain metrics exceed 1% against the reference |
| Independent-argument full budget | Incomplete | 67 | Spurious certificate rejection at root 67; active solve telemetry matches reference |
| Corrected dispatch guard, full budget | 69.26 s | 90 | Identical coupled CSV; zero force and strain metrics exceed 1% |
| Corrected dispatch guard, staged budget | 28.09 s | 90 | 107 force and 246 strain metrics exceed 1% |

All completed rows reach 1.40625 ms. The final two rows use the same binary,
shader and input pack. Relaying the guard through the certificate's existing
dispatch structure restores full-budget parity. Staged control first diverges
at accepted step three. Its 59.4% body-time reduction reaches 0.0000501×
headless; process time is 42.87 s. Both the reference and corrected staged run
have one strict mechanical-domain monitor overrun. The original final
certificate remains the acceptance authority. The staged candidate is
excluded from production because force and strain differ. The separately
verified rollback restores the mechanics, pressure, circulation, tendon
adapter and Brain predecessor before retry. The runtime investigation does
not identify the exact origin of the old argument's value.

At reference root 84 the strict mechanical monitor reads 1.493e−5 against
1.250e−5. Reassembled true linear residuals track Arnoldi and no linear solve
exhausts its budget. Nonlinear cycling after a better early assembly remains
the relevant failure. Stage timings are instrumented for this root only.

## Reproduction

Use the retained native-05 checkout and toolchain on SSH Numi. The launch
receipts record input pack SHA-256
`214d94f8e953a270a2e41c47c10687ee64b710d6e4d9f9ce22b970ecf056b486`,
the active asset paths and binary/library digests. Build commands pin the
Matter override and the existing native app. The full-run comparison reuses
the same physical configuration and preserves normal acceptance, pressure
coupling and contact. With the retained source tools on `PYTHONPATH`, run
`audit_boundary_bones.py` for the complete endpoint audit and
`audit_shoulder_prefix.py` to check the published material attachments.
The private recovery receipts pin the complete witness bundle and verified
replay pack copies. The current scene uses the reference solver.

A bed-supported equilibrium and zero crossings over 16 seconds at 0.1×
under variable pleural pressure remain open.
