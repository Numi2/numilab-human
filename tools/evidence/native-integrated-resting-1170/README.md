# Integrated resting Human: closed native pair 1170

Two registered native runs completed 155,000 accepted 2 ms steps each on the SSH Mac mini: 310.000014724 s, including 300 s after initialization. The whole-body viewer, breathing mechanics, anatomical cardiac deformation, closed circulation, gas exchange, and Brain regulation ran together. **This is a completed physiology experiment, not final anatomical acceptance.** Full late-pose audits found skin and lung intersections, and a small persistent lateral COM trend remains unexplained. Repairs and diagnostic work continue separately; these sealed runs remain unchanged.

## Measured outcome

| Observation | Control | Half respiratory drive at 60–100 s |
|---|---:|---:|
| Native elapsed wall time | 2707.066 s | 2752.896 s |
| Simulated/wall real-time factor | 0.114515 | 0.112609 |
| Terminal complete breaths / heartbeats | 58 / 362 | 61 / 362 |
| Dose-window inspiratory ventilation | 6.2967 L/min | 4.9543 L/min |
| Dose-minus-pre PaCO2 | +0.325006 mmHg | +2.160251 mmHg |
| Maximum absolute blood inventory error | 0.097789 mL | 0.097789 mL |
| Decoded native movie frames | 4846 | 4846 |

The registered PaCO2 difference-in-differences is **+1.835245 mmHg**; an independent raw-window calculation matches. Late recovery passes the predeclared numerical margins: treatment-minus-control PaCO2 −0.0629 mmHg, PaO2 +0.7443 mmHg, and ventilation relative difference 0.05%. These are engineering checks, not clinical standards.

The [paired report](results/paired/paired-physiology.md), [descriptive supplement](results/descriptive/README.md), and [registered analysis](results/science/analysis.json) preserve measured cycles, outliers and conservation channels. Generic resting comparisons are not all inside their bands: control respiratory rate is about 11.27/min; some PaO2 samples exceed 100 mmHg; mean pulmonary artery pressure is about 14.77 mmHg, below the generic AACN 15–20 interval but within the cited supine review context. Reference bounds were not changed to remove deviations.

## Remaining acceptance failures

The [baseline lung audit](results/anatomy/baseline-lung-report.json) covers eight exact accepted poses and found 42 unclassified cross-surface events, all between lung owners 305 and 308, with no self intersections or degenerate triangles. Source-near-interface geometry alone does not justify classifying native events as intended contact.

The [baseline skin audit](results/anatomy/baseline-skin-summary.json) checks 859 target surfaces at each pose. It found 25,421 nonocular target/skin intersections across poses (5,587 unique surface/target-face/skin-face triples); each pose has 3,100–3,341. Skin self-intersection and invalid/degenerate target counts were zero. Early-cycle clearance does not establish late-pose clearance. Treatment anatomy audits and new correction candidates have separate evidence identities.

COM moves about 5.743 mm from 10 to 310 s, primarily during settling, but late lateral drift remains about −0.515 mm/min. Its cause is not established. A new opt-in observer is being qualified to expose per-body motion and selected contact points. [Contact attribution](results/support-contact-attribution.md) explains why static contact seed IDs cannot identify dynamic skin-support winners.

## Runtime and reproducibility

Host: Apple M4 Pro (12 CPU / 16 GPU cores, 24 GiB), macOS 26.6 build 25G72, Xcode 26.6 build 17F113. This was not an exclusive whole-host benchmark; CPU analysis overlapped treatment. [Profile metrics](postprocessing/revision-003/profile-metrics.json) show about 133 ms GPU time per sampled eight-step physical segment versus about 13 ms per rendered frame. Exact instrument scopes are retained.

The [model scope](MODEL_SCOPE.md) names existing Metal, MyoSim, Brain, circulation and anatomical owners and limitations. CPU code is not an alternate physics solver. Skin geometry participates in existing NHCNT support queries, so corrections require fresh native qualification even if masses and contact-force ownership remain unchanged.

Registered Human owner: b354949c258106d00f2bd3dd6ac91216d5a3d409. Frozen physical Lab source: d550d8ad88a76fdee5bb6e028286fd24e962571f. Viewer018, physical library014, Brain inputs, anatomy assets, the dirty-at-build viewer delta, actual argv and hashes are bound in [registration](results/science/registration.json), trial receipts, and [source revisions](provenance/source-revisions-final.json). Later publication commits do not replace execution identities.

Launch on the Mac mini:

    ssh macmini 'python3 /Users/n/numi-human-retained-delivery-20261009/launchers/launch_manual.py'

This launchable scene uses the assets with documented anatomy failures; it is not a corrected final release. Native layer/camera controls permit inspection. The recording tour was exercised; interactive mouse use on the locked console was not.

Continuous native movies and exact full traces remain in the two trial output/scene directories under:

    /Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/

Both movies decoded to completion with all seven layer snapshots and matching surface timestamps. [Control review](results/recording/control/recording-review.json) and [treatment review](results/recording/treatment/recording-review.json) bind exact hashes. The inspector validates sorted presentation timestamps, not original compressed-sample decode order. Representative frames were visually inspected; screenshots do not establish geometry clearance.

## Evidence-tool corrections

The frozen P18 package remains unchanged. Its supplemental analyzer confused Human registration/invocation revision with the Lab physical source and required an integer where the owner writes an exact decimal fingerprint string. Postprocessing revision003 corrects these contracts while preserving runtime hashes and every physiological gate; five focused tests pass.

The descriptive supplement passed string CSV rows to a numeric owner helper and tried to write its derived P–V CSV to the source path. macOS user-immutable protection rejected that write; sealed source hashes remain unchanged. Physiology revision005 uses the existing numeric owner parser and creates only a fresh requested output with source pre/post-hash verification. Eleven focused tests pass; the earlier temporary-path-alias test failure is retained.

The movie reviewer tried to hard-link protected APFS files. Recording revision002 copies receipt-verified bytes to independent review inodes while retaining source protection; four focused tests and both full reviews pass. Original attempts and logs remain for provenance.

The profile wrapper initially reused the same filename for metrics and its execution receipt. A fresh read-only extraction recovered the same metrics SHA-256, now retained as profile-metrics.json separately from the wrapper receipt.

Source-manifest.json maps copied files to retained originals and hashes. Failed attempts, short tests and historical summaries are not promoted into native acceptance. No paid CI workflow was added.
