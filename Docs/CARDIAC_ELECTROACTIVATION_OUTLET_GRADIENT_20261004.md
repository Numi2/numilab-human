# Cardiac electroactivation and outlet-gradient screen — 2026-10-04

## Result

The unchanged native continuation has advancing source-myocardial electrical
activation, but it has not produced aortic or pulmonary ejection in the retained
accepted checkpoints. At step 106 (54.272 ms), the reconstructed LV-to-aorta
pressure difference is `-6,227.874 Pa` and the RV-to-pulmonary-artery
difference is `-530.299 Pa`; both native outlet flows are exactly zero. These
sampled states do not demonstrate a valve-law failure. With a nonpositive
source-model driving gradient, zero one-way outlet flow is the expected
directional result.

The step-106 electrical observer counts 148,244/218,077 myocardial nodes above
the voltage threshold and 150,126/218,077 above the activation threshold.
Between its accepted step 105 and 106 observations, 1,426 nodes crossed the
voltage threshold and 1,413 crossed the activation threshold. Step-106 chamber
pressures are 4,653.187 Pa LV and 1,155.498 Pa RV. The voltage/activation
counts establish electrical state changes in this model; zero outlet ejection
means they do not establish a heartbeat.

## Method and evidence

The monitors read the running native process without stepping it, changing its
checkpoint, or submitting GPU work. The activation monitor resumes from its
exact step-99 per-node interval state. It validates native fields against the
trajectory row, records skipped accepted-step intervals, and keeps nodes that
were already above threshold at its first observation left-censored. The
initial attempt's mislabeled step-99 `new_*_arrivals` counts remain preserved
under `media/cardiac-electroactivation-response-20261004/attempt-001/` and are
not treated as new arrivals.

The outlet monitor decodes the accepted snapshot's vascular-state rows and the
pinned cooked compartment/connection tables. It applies the pinned linear
compliance law to the ascending aorta and pulmonary arteries, and reads LV/RV
cavity pressures and outlet flows from their accepted native rows. The
reconstructed LV/RV pressures and aortic/pulmonary flows match the corresponding
trajectory values within the preregistered `0.001 Pa` and `0.001 ml/s` checks.
The full step-106 checkpoint SHA-256 is
`6b299d8b437d142e31a9d4572dade8ccd7193db0b656b421d625b410a002ea27`; both
monitors bind the same package and native program fingerprints.

The preregistered outlet baseline is step 103 (52.736 ms): LV `4,280.354 Pa`,
ascending aorta `10,892.484 Pa`, RV `1,123.896 Pa`, and pulmonary artery
`1,689.842 Pa`. Both gradients were negative and both outlet flows were zero.
By step 106 the LV pressure had risen and the aortic pressure had fallen
slightly, but the two gradients remained negative. The current evidence
therefore narrows the next decision to the activation-to-wall-load response
while systemic pressure remains above LV pressure. It does not yet isolate
myocardial load transfer as the cause; that assessment requires the declared
continuation to reach a positive outlet gradient or its stopping horizon.

The arterial pressures are source-model reconstructions from accepted vascular
volumes and the pinned package law, not independent physiological measurements.
No material parameters, outlet laws, or controller forces were changed. The
continuation remains unqualified for ejection, repeated cycles, or a true
heartbeat.

## Live continuation

At this receipt, native PID `46480` is still running the preregistered
step-53-to-240 continuation. Read-only observers `50357` (activation) and
`50711` (outlet gradients) are running on the same host. Their local step-106
snapshots, exact plans, monitor sources, and the corrected attempt history are
retained in
[`media/cardiac-electroactivation-response-20261004/`](media/cardiac-electroactivation-response-20261004/).

Next gate: continue sampling until the native process ends. At the first
positive LV-to-aorta or RV-to-pulmonary-artery gradient, compare that accepted
state with the contemporaneous myocardial activation and outlet flow. Audit
the valve equation only if a positive driving gradient still produces zero
flow. Preserve a non-ejection result if the continuation reaches its limit.
