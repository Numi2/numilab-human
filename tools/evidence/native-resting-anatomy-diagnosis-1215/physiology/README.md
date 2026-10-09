# 1201 resting physiology reference check

This note compares the retained 40-second 1201 model trace with its pinned local configs and a few adult-at-rest references. It is a descriptive source check, not a clinical assessment or release qualification.

## Measured trace

The native run completed 20,000 accepted steps at declared dt 0.002 s, producing 2,500 observations at 16 ms cadence through 40.0000019 s. Native exit was 0 and the wrapper recorded no input changes. Binary, runtime, and respiration-metallib hashes are pinned in report.json; the trace SHA-256 is ab391d28e2caa9315d8cd0515c823debd44b8168c4f03c5ef118b49e33db343a.

The trace reports modeled PaCO2 38.883–40.000 mmHg and modeled SaO2 97.222–97.466% (the latter is a Hill-model output, not co-oximetry). Modeled PaO2 is 99.9998–103.565 mmHg, mean 102.509; a common ABG reference table gives 75–100 mmHg, but oxygen reference values depend on age and altitude, and this run configured dry atmospheric pressure at 95.06 kPa. Therefore the value is reported as a model output; it is not clinically classified. The model does not output pH or bicarbonate, which are needed for acid-base interpretation. [NCBI ABG reference](https://www.ncbi.nlm.nih.gov/books/NBK536919/), [Clinical Methods on age and altitude dependence](https://www.ncbi.nlm.nih.gov/books/NBK371/).

Seven breath-counter increments occur from 5.024 to 37.488 s. Their six measured inter-event intervals yield 11.09 breaths/min; the seven event tidal volumes average 527.8 mL and fall from 536.2 to 522.9 mL. Their combined event-based ventilation estimate is 5.85 L/min. The local config nominal resting targets are 12/min, 500 mL, and 6 L/min. A review describes about 500 mL and 12/min as typical complete-rest adult-male values while noting individual variation; these are anchors, not patient-specific thresholds. [Breathing parameters review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8672270/).

Pleural pressure spans -7.85 to -4.80 cmH2O; the config baseline is -5.00 cmH2O. This is in the same scale as the cited quiet-breathing textbook example of approximately -5 at rest and -8 near end inspiration. [NCBI respiratory mechanics reference](https://www.ncbi.nlm.nih.gov/books/NBK538183/).

The circulation config is cvsim21_supine_continuous_fixed_rate_resting_reference_lv15, explicitly qualified as source_model_variant. Its chamber period is rational 6/7 s (70/min); 46 observed cycle intervals average 0.85739 s (69.98/min). Over the full 40-second horizon, cumulative forward ejection corresponds to 4.950 L/min aortic and 4.937 L/min pulmonary flow; from 10 to 40 s the corresponding endpoint averages are 4.903 and 4.902 L/min. These are finite-window model averages, not measured cardiac output. For context, NCBI gives approximately 5 L/min at rest for a 70-kg adult, and AHA gives a broad adult resting heart-rate range of 60–100/min. [Cardiac output reference](https://www.ncbi.nlm.nih.gov/books/NBK54473/), [AHA heart-rate reference](https://www.heart.org/en/health-topics/high-blood-pressure/the-facts-about-high-blood-pressure/all-about-heart-rate-pulse).

## Internal consistency and separate anatomy result

The pinned owner helper native_respiration_trace_consistency passed on all 2,500 rows. Its maximum residuals were 0.0003000 mL for volume decomposition, 0.00000648 Pa for airway pressure, and 0.00007364 Pa for pleural compliance, each within the helper 32-Float32-epsilon allowance. Maximum absolute oxygen and carbon-dioxide balance residuals were 0.000233 mL STPD and 0.000466 mL STPD; blood endpoint-minus-physical residual was at most 0.001440 mL. These checks establish internal identities/closure for the emitted state; the owner explicitly says they do not prove causal response, anatomy, or physiological plausibility.

The terminal geometry audit is complete, but fails its skin-to-target gate: 3,112 exact Float32 triangle crossings, zero ocular crossings, zero invalid target triangles, and zero skin self-crossings. This is a separate anatomical failure; it prevents presenting this trace as an integrated qualified resting body.

## Model boundaries

The local respiration config calls itself a declared mixed-source adult reference with no individual or clinical calibration. It specifies 250 mL/min STPD O2 consumption, 200 mL/min STPD CO2 production, FRC 2.5 L, and a single perfusion-limited pulmonary equilibration. Source code uses a Hill O2 content relation and a local linear CO2-content law at fixed pH. It explicitly disclaims acid-base and Haldane-effect claims. The respiratory kernel conditions exchange on one selected pulmonary edge rather than a regional lung V/Q distribution; regional V/Q matching is a major real determinant of gas-exchange efficiency. [V/Q measurement review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8274320/). The Haldane effect depends on oxygenation changing hemoglobin CO2 carriage, which this fixed-pH linear law omits. [Mechanistic CO2/Haldane model](https://journals.physiology.org/doi/10.1152/japplphysiol.00318.2016).

Spontaneous inspiration can lower intrathoracic pressure, increase venous return, and alter left-ventricular ejection. In the pinned implementation, circulation compartment external pressures are loaded from static config values and packed as constants; the respiration kernel emits cardiac-pressure observations from those configured values but does not update them from the changing pleural-pressure trace. The model therefore includes perfusion-driven gas transport but not this dynamic pleural-pressure-to-cardiovascular feedback. [ATS cardiopulmonary interaction review](https://pmc.ncbi.nlm.nih.gov/articles/PMC5822394/).

This is a 40-second window, not a five-minute endurance result. Values are simulated and finite-window; no claim is made that instantaneous samples equal cycle averages, or that these ranges predict an individual. Exact input, output, source, and audit hashes plus the reproducible verifier command are in report.json and manifest.json.

## Reproduction

On the Mac mini, run the read-only summary with the pinned Xcode Python 3.9 interpreter:

```sh
/Applications/Xcode-26.6.0.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9 /Users/n/numi-human-retained-delivery-20261009/physiology-reference-review-1212/verify_1201_reference.py
```

The script refuses to summarize if its pinned 1201/config/source inputs have changed. It writes only report.json in this directory.
