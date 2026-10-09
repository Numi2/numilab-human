# Supplemental paired physiology and support audit (final integrated study)

The registered `numi science analyze` result remains the primary-result authority; this report cross-checks that value from raw PaCO2 windows and adds physical-cycle and support diagnostics. The study plan, margins, and inputs were not changed.

## Bound inputs and completion

- Registered science verification succeeded read-only.
- Study registration SHA-256: `6e41e09d3c18ba27e0f3e130b3e9eee1098c27da7b5bd7334fa04b7d807783b2`.
- Registered science analysis SHA-256: `6f75d32da0729448a64d2e5b61284a417eccff8b62c631db3fba715514e3afeb`; registered primary difference: **1.835245 mmHg** (verdict `supported`).
- Each arm passed the completion gate: exact registered count 155000 at requested 2 ms, successful registered exit receipt/native terminal/owner observation checks, and sampled endpoint only after exact terminal count. Time labels preserve the runtime Float32 timestep (0.0020000000949949026 s); N=155000 is nominal 310.000 s and its native receipt time is about 310.000014724 s.
- Each final scene native log independently verifies the actually loaded MetalRobo library with the frozen Human loaded_metal_runtime helper, against the parent-verified path and SHA-256.
- Post-init interval analyzed: [10, 310) s (300 observed seconds); 0–10 s initialization excluded.
- Plan prediction envelope: [0.0, 40.0] mmHg; it is a sensitivity envelope, not a probability interval or clinical prediction.

## Registered primary result and response checks

| Metric | Control | Half-drive | Paired result / interpretation |
|---|---:|---:|---|
| PaCO2 pre mean, [30, 60) s | 39.2636 | 39.2636 | mmHg |
| PaCO2 dose mean, [70, 100) s | 39.5886 | 41.4238 | mmHg |
| Arm dose-minus-pre PaCO2 change | 0.325006 | 2.160251 | Registered difference-in-differences **1.835245 mmHg**; raw-trace cross-check 1.835245 mmHg (difference 0.000000) |
| Dose-window inspiratory VE (owner accepted-flow metric) | 6.2967 | 4.9543 | Treatment minus control -1.3424 L/min; required direction `< 0`: observed |
| Dose complete-breath ledger VE | 5.7684 | 4.5458 | Additional event-ledger context; see cycle reports for intervals and cycle counts |

## Late recovery comparison

Final 30 s is [280, 310) s. These are the predeclared numerical equivalence margins, not clinical standards.
- PaCO2 treatment-minus-control: -0.0629 mmHg; margin ±1.0: within.
- PaO2 treatment-minus-control: +0.7443 mmHg; margin ±5.0: within.
- Inspiratory VE treatment-minus-control: +0.0030 L/min (relative difference 0.05%); margin 10.0%: within.

## Physical cycles and conservation

- **control:** 56 complete ledger breath intervals after initialization; 56 have both positive/negative measured airflow and a nonzero sampled lung-volume excursion. 349 cardiac counter intervals; 349 have positive LV stroke and positive integrated aortic and pulmonary ejection. Per-cycle evidence is in `control-cycles.csv` and `control-physiology.md`.
- **control per-field maximum absolute numerical-diagnostic values:** {"blood_continuity_residual_accum_ml": 0.0969386638461, "blood_endpoint_minus_physical_ml": 0.001744410838, "blood_error_ml": 0.0977888703346, "blood_physical_delta_accum_ml": 0.0969084794633, "blood_residual_minus_physical_ml": 0.000169721819931, "co2_balance_error_stpd_ml": 0.000465661287308, "oxygen_balance_error_stpd_ml": 0.000232830643654, "respiratory_net_volume_ml": 1.7071191678, "respiratory_volume_balance_ml": 0.000418737045038}. This exact nine-field set includes per-step volumes, owner-maintained running maxima, and compensated cumulative blood-volume quantities. Maxima include the exact accepted terminal sample as well as post-init rows; exported rows are never summed.
- **treatment:** 59 complete ledger breath intervals after initialization; 59 have both positive/negative measured airflow and a nonzero sampled lung-volume excursion. 349 cardiac counter intervals; 349 have positive LV stroke and positive integrated aortic and pulmonary ejection. Per-cycle evidence is in `treatment-cycles.csv` and `treatment-physiology.md`.
- **treatment per-field maximum absolute numerical-diagnostic values:** {"blood_continuity_residual_accum_ml": 0.0969386638461, "blood_endpoint_minus_physical_ml": 0.001744410838, "blood_error_ml": 0.0977888703346, "blood_physical_delta_accum_ml": 0.0969084794633, "blood_residual_minus_physical_ml": 0.000169721819931, "co2_balance_error_stpd_ml": 0.000465661287308, "oxygen_balance_error_stpd_ml": 0.000232830643654, "respiratory_net_volume_ml": 1.96579435396, "respiratory_volume_balance_ml": 0.000431626290265}. This exact nine-field set includes per-step volumes, owner-maintained running maxima, and compensated cumulative blood-volume quantities. Maxima include the exact accepted terminal sample as well as post-init rows; exported rows are never summed.

## Accepted geometry captures

Each arm has five shared historical phase samples, two arm-specific late-cycle samples, and a separate terminal capture. Ordinary frame identity is its accepted step ID. Receipt time uses the runtime Float32 representation of the requested 2 ms timestep; nominal times use accepted step ID * 0.002 s. The terminal is exactly accepted N=155000 (nominal 310.000 s; native receipt time about 310.000014724 s); N-1 is not labeled terminal. The seven interior IDs were selected from prior-867 phase evidence, not fitted to these outcomes. Sparse captures do not establish whole-cycle anatomy clearance; the historical global rib-volume minimum near 65.168 s remains unsampled under the eight-frame limit.

- **control:** 8 registered frames: 47519 (95.038 s), 49151 (98.302 s), 51903 (103.806 s), 54047 (108.094 s), 55647 (111.294 s), 152191 (304.382 s), 154143 (308.286 s), 155000 (310.000 s) terminal. Terminal capture receipt and MRV pack hashes are listed in manifest.json.
- **treatment:** 8 registered frames: 47519 (95.038 s), 49151 (98.302 s), 51903 (103.806 s), 54047 (108.094 s), 55647 (111.294 s), 152447 (304.894 s), 154367 (308.734 s), 155000 (310.000 s) terminal. Terminal capture receipt and MRV pack hashes are listed in manifest.json.

## Sampled COM/support drift diagnostics

These are run-bound sampled diagnostics, not static-equilibrium or postural qualification. COM trend is a least-squares line over exported samples after 10 s. The force statistic is the exported aggregate normal impulse summed over contacts for each sampled final physical step divided by the physical timestep; it is not the largest force at one contact or a time-integrated support estimate.
- **control:** 18750 samples over [10.000, 309.984) s; endpoint COM displacement 5.743 mm (x/y/z = -2.843, -1.388, -4.793 mm); maximum excursion from first post-init sample 5.743 mm; fitted trend x/y/z = -0.41526, 0.02417, -0.24617 mm/min; sampled aggregate total normal support force at last physical step mean/range 706.32 [694.24, 872.37] N; active contacts mean/range 14.05 [14, 15].
- **treatment:** 18750 samples over [10.000, 309.984) s; endpoint COM displacement 5.743 mm (x/y/z = -2.843, -1.388, -4.793 mm); maximum excursion from first post-init sample 5.743 mm; fitted trend x/y/z = -0.41526, 0.02417, -0.24617 mm/min; sampled aggregate total normal support force at last physical step mean/range 706.32 [694.24, 872.37] N; active contacts mean/range 14.05 [14, 15].

## Per-arm generic adult physiology ranges and outliers

The two linked per-arm reports retain post-init means/ranges and explicit deviations versus generic adult intervals and posture-specific cohort summaries. They deliberately preserve outliers. These comparisons do not diagnose a person, calibrate the model, or qualify physiology.

- [Control post-init report](control-physiology.md)
- [Half-drive post-init report](treatment-physiology.md)

Pressure samples are instantaneous pulsatile waveforms; the per-arm report compares AACN MAP/mPAP reference intervals only with sampled complete-counter-cycle means and identifies those as proxies. Conservation residuals establish numerical accounting only.


## Cardiac interface limitation

The retained 911 report localizes 11,568 source-neutral right-atrial/right-ventricular intersections near the common-map tricuspid leaflet projection with a localized source-to-current RV residual up to 0.75 mm. It does not establish a 3D leaflet surface or valve-plane/orifice ownership. The reduced-order CVSim chambers and valves remain the sole functional and blood owner; this is a bounded source-interface limitation, not a whole-heart or anatomy-clearance pass. See [the exact 911 localization report](../../native-cardiac-interface-localization-911/final-localization-report.json).

Reference context: [MedlinePlus ABG](https://medlineplus.gov/lab-tests/arterial-blood-gas-abg-test/) lists PaO2 75-100 mmHg, PaCO2 35-45 mmHg and O2 saturation 95-100%; [MedlinePlus Vital Signs](https://medlineplus.gov/ency/article/002341.htm) gives 12-18 breaths/min for the average healthy resting adult and notes individual variation. The per-arm report also retains the AACN PaO2 80-100 mmHg band as a distinct generic reference; both bands and all outliers are shown without post-hoc range changes. [Kovacs et al.](https://doi.org/10.1183/09031936.00145608) review 1,187 individuals across 47 studies, of whom 882 supplied supine data (mPAP 14.0 +/- 3.3 mmHg); the 882 is a posture-specific subset, not the review total. [Mendes et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7253877/) reports male supine quiet-breathing cohort means +/- SD, not individual reference limits.
