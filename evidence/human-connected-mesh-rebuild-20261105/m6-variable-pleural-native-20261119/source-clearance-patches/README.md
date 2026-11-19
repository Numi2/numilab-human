# Source skin patches in native resting motion

The Apple M6 run on SSH Numi accepted 8,000 steps over 16.00000076 simulated seconds under variable pleural pressure. It completed three breaths and 19 cardiac filling/ejection cycles. The measured rate was 0.10656× during integrated motion with native presentation, 0.10402× over the body timer with geometry export and 0.09857× from launch to exit. The viewer presented 252 callbacks.

Five connected source patches modify 1,067 skin vertices. They preserve face rows, weights, provenance, all 9,588 contact and seam guards, and the regional skin/fat section model. An additional proposed bone clearance was dropped when exact source testing found a fold. A subsequent sweep with additional separation axes preserved the same displacement checkpoint. The selected patches preserved the tested orientation and introduced no rest-source self-intersection. A broader joint fit also failed because it introduced new folds and crossing witnesses after conservative source-area and volume gates passed.

| Matched 16-second geometry | Connected diagonals | Source patches |
| --- | ---: | ---: |
| Skin-to-tissue triangle pairs | 14,325 | 14,230 |
| Tissue intersection curves | 220 | 217 |
| Tissue intersection length | 13.63256 m | 13.38400 m |
| Skin self-intersection pairs | 45 | 45 |

The exact native terminal audit found no newly intersecting pair identities. Muscle banks 4, 5 and 6 are clear at that capture. The additional 4.030-second accepted capture has 12,796 skin-to-tissue pairs and 32 self pairs. It is a separate sample with no matching control capture at that step. The earlier quad retriangulation removed 184 terminal tissue triangle pairs while reducing exact curve length by 0.17%. Curve length measures the extent of intersection. It does not measure tissue penetration depth or enclosed volume.

The two native runs have byte-identical mechanics, support, vascular geometry state and physiology traces. The terminal bed reaction was 705.65 N with zero recorded root assistance. The maximum observed anatomical cardiac geometry volume discrepancy was about 0.00234% under the tested pleural waveform. The physical skin and fat loads in this configuration are supplied by 32 condensed bed-contact sections.

At the terminal state, 2,898 remaining intersections have all three skin vertices in protected regions, including 588 skin-to-bone pairs. Source-only displacement with those vertices and the tissue surfaces fixed cannot clear these witnesses at the same pose. This build remains a staging candidate. Full-body internal tissue contact, continuous clearance and the long-run physiology gates remain open.

## Reproduce the asset

Use the prior connected-skin-diagonals.nhskin produced by tools/replay_resting_skin_diagonals.py. Verify its hash against source-emission.json, apply the standard binary patch with macOS /usr/bin/bspatch, then verify the resulting hash. The patch was tested by exact byte comparison against the authored asset. Atlas inputs retain their existing distribution terms.

```sh
bspatch connected-skin-diagonals.nhskin source-clearance-patches.nhskin source-clearance.bsdiff
shasum -a 256 source-clearance-patches.nhskin
```

The input and output SHA-256 values are in source-emission.json. The native build and comparison scripts reference the existing SSH Numi checkpoint. They use the original launch environment and compilation inputs recorded in the receipts.
