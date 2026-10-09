# Baseline-only descriptive checks

This report reads only the closed unassisted control arm. It is not the registered pair result or a physiology qualification.

## Resting-reference observations

| Measure | Baseline observed | Frozen Human owner reference | Reading |
|---|---:|---:|---|
| PaO2 mean (mmHg) | 99.991 | [80.000, 100.000] | within owner range |
| PaCO2 mean (mmHg) | 39.683 | [35.000, 45.000] | within owner range |
| SaO2 mean (percent) | 97.219 | [95.000, open] | within owner range |
| Complete breath rate (/min) | 11.271 | [12.000, 18.000] | outside owner range |
| Complete heartbeat rate (/min) | 69.998 | [60.000, 100.000] | within owner range |
| Aortic forward output (L/min) | 4.903 | [4.000, 8.000] | within owner range |
| Pulmonary forward output (L/min) | 4.903 | [4.000, 8.000] | within owner range |
| LV stroke volume (mL) | 70.036 | [50.000, 100.000] | within owner range |
| Mean aortic pressure (mmHg) | 89.119 | [70.000, 105.000] | within owner range |
| Mean pulmonary artery pressure (mmHg) | 14.770 | [15.000, 20.000] | outside owner range |

PaO2 mean is within the 80-100 mmHg owner band, while 5551/18751 post-initialization samples exceed 100 mmHg (observed range 98.652-103.565). SaO2 samples are stored as fractions and were converted by x100 for the percent comparison; 0/18751 samples are below the owner lower bound of 95%.
The complete-breath rate is below the owner's generic 12-18/min range. Mean pulmonary artery pressure is slightly below the owner's generic 15-20 mmHg range. These are descriptive comparisons with generic references and are not model acceptance gates or clinical interpretations.

## Selected registered windows

| Window | PaO2 mean/range (mmHg) | PaCO2 mean/range (mmHg) | Lung volume mean/range (mL) | Pleural pressure range (Pa) | LV stroke mean/range (mL) |
|---|---:|---:|---:|---:|---:|
| pre | 102.97 [102.24, 103.57] | 39.26 [39.01, 39.50] | 2644.7 [2500.0, 3023.4] | [-760.3, -471.7] | 70.03 [70.03, 70.04] |
| dose | 100.68 [99.80, 101.63] | 39.59 [39.40, 39.78] | 2668.7 [2500.0, 3025.4] | [-761.8, -471.5] | 70.03 [70.03, 70.04] |
| recovery | 99.41 [99.22, 99.59] | 39.86 [39.78, 39.96] | 2658.3 [2500.0, 3031.3] | [-766.3, -470.8] | 70.04 [70.03, 70.04] |

## Mechanics and accounting checks

Existing owner pressure/volume identity checks passed over all 19375 samples; maximum residuals were 0.00030966151 mL (volume), 6.5002024e-06 Pa (airway pressure), and 7.5291403e-05 Pa (pleural compliance), with the largest at 0.0355 of the 32-Float32-epsilon allowance.
Blood volume over the post-initialization trace: 5150.002-5150.097 mL (mean 5150.057 mL). The nine exported owner diagnostics are absolute high-water/error observations, not per-row quantities to sum. The trace contains no explicit per-step VO2/VCO2 metabolic flux channel, so this report makes no metabolic-rate estimate.
Sampled COM endpoint displacement from the first post-initialization sample was 5.743 mm; maximum excursion 5.743 mm. The support-force summary is sampled endpoint output, not a static-equilibrium or postural-qualification test.
The receipt records 155,000 accepted steps at requested 2 ms, nominal 310 s and terminal accepted time 310.000014724210 s; root assistance was not observed. Counter-transition intervals had histogram {"1": 350}; the separate 855 complete-event ledger counted 349 cardiac and 56 respiratory intervals.

## Interpretation limits

The model includes perfusion-to-gas transport and gas-driven chemoreflex regulation. It does not feed respiratory pressure/volume back into cardiovascular hemodynamics, and it does not represent whole-body modal response. The registered pair and official verifier were not read by this baseline-only report. The frozen owner references and measurement methods are in the JSON.
