# Physical support and the registered wrist

The full human on SSH Numi still falls toward its bed. The terminal bed force is 10.98 N for a body weighing about 953 N. The skeletal assembly accelerates downward at 9.798 m/s². All 34 registered ligament force rows and variable pleural pressure are active. Two accepted steps cover 15.625 µs, too little time to establish a breath, a heartbeat, or settling.

## A measured defect in the pelvic load path

The deep fat surface enters closed pelvic bone bank 18 by at least 3.5752 mm at one accepted point. The native contact row at practically the same point (0.0000163 mm apart) reports 3.5751 mm of closure, 0.578 kPa and 0.0071 N along the bed normal. All 131 observed pelvic rows together transmit just 1.496 N along that normal. Across the whole skeletal assembly the skin and bone rows transmit about 1.44 N. These values are actual paired contact forces. The gravity load on the skeletal assembly is 811 N.

A second static preparation tested whether protecting the shape of fragile cells locally could admit 1 mm of skeletal descent. At that target, 2,387 cells initially incurred new material failures. Translating progressively larger groups together spread the error to neighboring cells. The final admissible descent was below one nanometre. We discarded that initialization. The measured pelvic point is the first priority for rebuilding a conforming fat–bone interface and checking that contact and the bed can transmit weight to the pelvis. The reference surface, embedded cut surfaces and supported initial load must agree before longer motion can qualify.

## Matched native wrist experiment

The right distal radius accounts for all 212 exterior skin–bone pairs in this scene. The 162 affected faces meet 14 cortical triangles. Correcting only the rigid donor frame increases the exterior count to 298. A trial that cleared all 568 local outer, inner and deep wall pairs against that radius inverted tissue cells and introduced carpal crossings. Neither change entered the simulation.

We reconditioned the connected material under the same surface, with registered anatomical owners, density bounds, source masses and first moments in the reference and opening frames. Of 84 proposed local replacements, 83 passed the native geometry predicates and 42 passed the existing carrier routing directly. Compensated sums of the actual registered shipments let 11 additional replacements pass the original per-carrier tests. The resulting 53 patches replace 281 cells with 248.

| Native measurement | Previous mesh | Reconditioned wrist |
| --- | ---: | ---: |
| Newton iteration ceiling | 24 | 24 |
| Accepted motion on Apple M6 | 15.625 µs | 15.625 µs |
| Whole process | 279.37 s | 275.44 s |
| Inverted skin or fat cells | 0 | 0 |
| Local stretch failures inside the replaced regions | 3 | 0 |
| Whole-body skin / fat stretch failures | 202 / 303 | 200 / 302 |
| Whole-body low-shape / volume-ratio failures | 40 / 7 | 40 / 7 |
| Transverse wall–bone pairs | 6,568 | 6,568 |

The median improvement in the minimum shape measure across the 53 patches was 3.13× at the accepted endpoint. Elsewhere one previously passing cell fell below the minimum stretch of 2/3, and one other cell recovered. The identity of every intersecting wall–bone triangle pair was unchanged. Of the 6,568 pairs, 5,085 involved the deep fat wall. Signed depth checks used matched native body poses and closed, consistently edge-oriented cortical banks. Seven of the 41 affected banks lacked that edge certificate and retain intersection witnesses only.

With the default 16 iterations the 53-patch trial rejected its second step at a reassembled relative residual of 0.000554. Raising the ceiling to the supported 24 admitted both steps at 0.0000442. The single measured step between accepted endpoints ran at 1.41 × 10⁻⁷ real time. The entire two-step process including initialization ran at 5.67 × 10⁻⁸ real time. Sustained 0.1× and 16 seconds of crossing-free motion remain unverified.

The public default retains the full joint-stops release. Its material-check shortcut now verifies that release. The experimental wrist uses the same entry point. On SSH Numi, the following command audits the retained physical capture and writes an exact pair list. It returns status 1 while the material and crossing criteria fail.

```sh
bash scripts/run_resting_human_on_numi.sh --dense --posterior-support wrist-material \
  --material-report /tmp/numi-wrist-material.json --wall-pairs /tmp/numi-wrist-pairs.json
```

To integrate again on the same native owner, use `bash scripts/run_resting_human_on_numi.sh --dense --posterior-support wrist-material --steps 2`. The launcher verifies 597 pinned input identities. The whole-body shape comparison uses a mixed-source reference assembly. Over 1.839 m² of physical exterior, the median unsigned distance from physical face centroids to independent detailed source skin in the matched pose is 2.85 mm. The 95th percentile is 9.89 mm. Those discrepancies locate registration work and are not an accuracy estimate against a measured individual.

[Matched native data, solver traces and source identities](right-wrist-material-motion.json) · [Verified archive and restoration manifest](right-wrist-material-recovery.json) · [Connected gravity and forearm controls](connected-gravity.md)
