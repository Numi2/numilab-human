# Reference circulation volume choice

The first native integrated breath used the unchanged upstream CVSim21 LV
unstressed volume of 55 mL. That operating point produced an LV ejection fraction
around 42%, which does not match the intended resting reference adult. We retain
that source variant and introduce three explicitly named reference variants in
the existing CVSim21 authoring owner; no circulation solver is replaced.

The adopted reference value is 15 mL, with 10 and 20 mL sensitivity variants.
The 15 mL value is published for CVSim-6 in De Florio et al. (2025), Appendix A,
Table 7, DOI [10.1098/rsta.2024.0221](https://pmc.ncbi.nlm.nih.gov/articles/PMC12145504/).
Transferring that parameter to CVSim21 is a declared reference-adult modelling
choice, not an original CVSim21 parameter, patient measurement, or calibration.
The balancing allocation to the upper-body venous compartment is inferred.

For LV15, both initial and reference LV volumes decrease by 40 mL and both
upper-body venous volumes increase by 40 mL. Stressed volume and every pressure
law stay unchanged; total blood remains 5,150 mL. The native circulation owns the
new absolute volumes. They also change gas residence times. This is not a
display-only volume offset. The lowering manifest records the source, choice,
allocation, sensitivity values and limitations, while the original source
lock/initialization checks remain mandatory.

All authoring and numerical checks ran through SSH on the Apple M4 Pro Mini.
Four fresh native processes each accepted 5,000 steps of 2 ms (10.000000475 s).
The before-run numerical regression protocol and source hashes are retained.
All inputs and loaded binary/library files stayed unchanged. The native runner
explicitly makes no whole-body anatomy claim.

| LV unstressed volume | Sampled complete-cycle EF | Largest pressure difference from upstream | Largest cumulative ejection difference |
| --- | --- | --- | --- |
| Upstream 55 mL | 41.36–42.26% | reference | reference |
| Reference 10 mL | 56.20–56.88% | 0.000235 mmHg | 0.000292 mL |
| Reference 15 mL | 54.06–54.77% | 0.000228 mmHg | 0.000233 mL |
| Reference 20 mL | 52.07–52.82% | 0.000169 mmHg | 0.000175 mL |

EF uses the maximum and minimum sampled LV volumes within each of nine complete
6/7-second cardiac clock periods. It is a finite-cadence simulation diagnostic,
not echocardiography. The LV15 values fall within the male EF reference interval
52–72% in the [ASE/EACVI chamber quantification recommendations, Table 2](https://www.asecho.org/wp-content/uploads/2016/02/2015_ChamberQuantificationREV.pdf).
This single plausibility comparison does not validate other cardiac function or
the mixed-source adult. All four arms passed the declared numerical thresholds;
maximum blood-volume error was 0.005588 mL. Fourteen source-authoring tests and
eight native-launch admission tests passed; their logs are retained.

Use `config/cvsim21-resting-reference-lv15.v1.json` with the existing
`numilab_human.cvsim21` compiler. Pass its native payload explicitly to
`numilab-human resting-run --circulation PATH`. The launch receipt hashes the
selected payload. The launcher still defaults to the retained upstream variant.
Long-run coupled anatomy, gas-response and viewer acceptance remain separate.
