# Supplemental resting physiology and cycle audit

- Trial: /Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170/trials/resting-drive-half
- Arm: treatment
- CSV SHA-256: 77710f508772f23d8eab80ef52c08437a93018ee680ce8fb433079850da6d1bd
- Invocation SHA-256: 239df9c7ae0847af4a59478facbeef8a73a051e133470b1950236d05a3f64f0e
- Source revision: b354949c258106d00f2bd3dd6ac91216d5a3d409; expected frozen revision: b354949c258106d00f2bd3dd6ac91216d5a3d409; match: yes
- Completion evidence: registered exit receipt, hashed final owner observation, final native terminal, and the registered owner invocation/preflight metadata verified; the final native adapter run does not emit a Human run-metadata.json.
- Terminal accepted count exactly 155000; sampled endpoint 155000/155000, time 310.000/310.000 s, stride about 8 roots; endpoint tolerance applied only after exact accepted-count proof.
- Post-init window: [10, 310.000) s; exported samples n=18750.

Supplemental one-arm diagnostics only; this is not the registered paired treatment contrast, a population estimate, clinical calibration, or anatomy qualification. The paired estimand and numerical recovery margins remain those in this trial's registered plan. For an intervention arm, whole-window reference comparisons are descriptive and include the deliberate dose; they are not resting acceptance tests.

## Post-init physiology mean and observed range

| Observable | Mean [min, max] | Reference comparison |
|---|---:|---|
| PaCO2 | 39.905 [38.883, 41.684] (n=18750) mmHg | 35-45 mmHg; out-of-reference samples 0/18750 |
| PaO2 | 99.530 [89.538, 103.565] (n=18750) mmHg | 80-100 mmHg; out-of-reference samples 13830/18750 |
| PaO2 (MedlinePlus ABG) | 99.530 [89.538, 103.565] (n=18750) mmHg | 75-100 mmHg; below 75: 0, 75-<80: 0, above 100: 13830 samples |
| SaO2 | 97.171 [96.292, 97.466] (n=18750) % | >95% generic sea-level; below-reference samples 0/18750 |
| Aortic pressure samples (instantaneous pulse waveform) | 89.119 [68.508, 109.048] (n=18750) mmHg | descriptive only; not compared sample-by-sample with MAP |
| Pulmonary artery pressure samples (instantaneous pulse waveform) | 14.770 [9.241, 22.040] (n=18750) mmHg | descriptive only; not compared sample-by-sample with mPAP |
| Aortic sampled cycle means (MAP proxy) | 89.118 [88.933, 89.406] (n=349) mmHg | generic MAP 70-105 only if field definition is equivalent; outside cycle means 0/349 |
| Pulmonary sampled cycle means (mPAP proxy) | 14.769 [14.703, 14.830] (n=349) mmHg | AACN mPAP 15-20; outside cycle means 349/349; supine cohort 14.0+/-3.3 |
| Ledger RR | 11.900 [10.846, 13.870] (n=59) /min | average healthy resting adult 12-18/min; outside cycles 46/59 |
| Ledger VT | 0.506 [0.341, 0.554] (n=59) L | male supine cohort 0.58+/-0.28 L; descriptive, not cutoff |
| Ledger inspiratory VE | 6.003 [3.748, 7.684] (n=59) L/min | male supine cohort 8.32+/-2.78 L/min; descriptive, not cutoff |
| Aortic-ledger CO | 4.903 L/min | generic 4-8 L/min; cumulative ejection difference, not sampled q |
| LV stroke per cardiac counter interval | 70.036 [70.030, 70.098] (n=349) mL | generic SV 50-100 mL; counter/cavity semantics remain numerical |

Aortic and pulmonary artery fields above are instantaneous pulse samples. AACN MAP/mPAP bands are compared only with sampled means over complete cardiac-counter intervals, which are proxies from the exported waveform, not clinical measurements. AACN ranges are generic adult sea-level references with age/altitude variation. Mendes and Kovacs values are cohort summaries, not acceptance limits.

## Physical respiratory cycles

Complete event-ledger intervals after initialization: 59. Both positive and negative sampled airflow signs: 59/59; nonzero sampled lung-volume excursion: 59/59; positive flow with negative alveolar pressure observed: 59/59; negative flow with positive alveolar pressure observed: 59/59.
Per-cycle VT/VE use last-complete-breath and cumulative-inspired-volume ledger fields. Instantaneous sampled airflow is sign/direction evidence only; it is not integrated. Sampled volume excursion is the saved lung-volume min/max within the event interval. A sampled endpoint flow and volume increment can straddle a reversal.

Absolute difference between complete-breath volume and cumulative-ledger delta: max 0.000 mL.

## Cardiac-cycle ejection audit

Post-init counter intervals: 349. Positive LV stroke: 349/349; positive integrated aortic ejection: 349/349; positive integrated pulmonary ejection: 349/349.
Aortic ejection per counter interval: 70.038 [70.029, 70.098] (n=349) mL; pulmonary: 70.034 [70.002, 70.066] (n=349) mL.
Sampled aortic cycle-mean pressure: 89.118 [88.933, 89.406] (n=349) mmHg; pulmonary artery: 14.769 [14.703, 14.830] (n=349) mmHg.
The cycle sidecar retains per-cycle stroke/ejection and sampled pressure means/ranges. Cycle counts are numerical coverage; stroke/ejection values come from saved stroke and cumulative integrated-ejection fields, not sampled q.

## Sampled absolute residual maxima

| Field | Max absolute |
|---|---:|
| respiratory_volume_balance_ml | 0.000431626 mL |
| oxygen_balance_error_stpd_ml | 0.000232831 mL STPD |
| co2_balance_error_stpd_ml | 0.000465661 mL STPD |
| blood_error_ml | 0.097788870 mL |
| blood_continuity_residual_accum_ml | 0.096938664 mL |
| blood_residual_minus_physical_ml | 0.000169722 mL |
| blood_endpoint_minus_physical_ml | 0.001744411 mL |

These are maxima over exported samples/fields. Small conservation residuals demonstrate numerical balance only, not physiological plausibility.

## Diagnostic-field semantics

The diagnostic summary reports maximum absolute sampled values and never sums a diagnostic column across rows. O2/CO2 balance-error fields and blood_error_ml are owner-maintained running maxima. respiratory_net_volume_ml and respiratory_volume_balance_ml describe one physical step: integrated swept volume and mechanics-delta minus integrated swept volume, respectively. Blood continuity and physical-delta fields are compensated cumulative sums; their difference compares normalized with physical volume updates, while blood_endpoint_minus_physical_ml compares current endpoint volume with accumulated physical change.

## References

- [AACN Normal Ranges](https://aacn.s3-us-west-2.amazonaws.com/Courses/ecco/course-resources/resources/common-resources/Normal_Ranges.pdf): generic adult sea-level ABG/hemodynamic ranges; age and altitude vary.
- [MedlinePlus Vital Signs](https://medlineplus.gov/ency/article/002341.htm): average healthy resting adult RR 12-18/min; individual factors vary.
- [Mendes et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7253877/): male supine cohort RR 16.15+/-4.72/min, VT 0.58+/-0.28 L, VE 8.32+/-2.78 L/min; means +/- SD, not cutoffs.
- [Kovacs et al. 2009](https://pubmed.ncbi.nlm.nih.gov/19324955/): supine mPAP 14.0+/-3.3 mmHg; systematic review of 1,187 healthy-subject measurements across 47 studies, with posture-stratified data; cohort mean+/-SD, not a clinical interval.
- [Stoddart 1967](https://pubmed.ncbi.nlm.nih.gov/5183280/): controlled human alveolar-hyperventilation study; mechanism/direction context, not Numi response magnitude.
- [Dahan et al. 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC1925127/): small human series separating fast peripheral and slow central CO2-response components; not validation of exact model time constants.

Interpretation notes: /Users/n/numi-human-resting-evidence-20261005/resting-physiology-reference-review-829/README.md. Physiology does not qualify anatomy; consult the exact run-bound geometry audits separately.

## Explicit reference deviations

- PaO2: 0 below 80 and 13830 above 100 mmHg among 18750 post-init samples; observed 89.538-103.565 mmHg.
- RR: 46/59 complete event-ledger cycles outside generic 12-18/min (below=46, above=0); cycle range 10.846-13.870/min. Per-cycle times and values are in the sidecar.
- Pulmonary sampled cycle-mean proxies outside AACN 15-20 mmHg: 349/349; range 14.703-14.830 mmHg. Instantaneous samples are not classified.
