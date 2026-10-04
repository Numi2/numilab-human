# Cardiac electroactivation and outlet-gradient screen — 2026-10-04

## Result

The unchanged native continuation has advancing source-myocardial electrical
activation, but it has not produced aortic or pulmonary ejection in the retained
accepted checkpoints. At step 109 (55.808 ms), the reconstructed LV-to-aorta
pressure difference is `-5,795.531 Pa` and the RV-to-pulmonary-artery
difference is `-471.617 Pa`; both native outlet flows are exactly zero. The
gradients have moved toward zero since the step-103 baseline, but remain
negative. These sampled states do not demonstrate a valve-law failure. With a
nonpositive source-model driving gradient, zero one-way outlet flow is the
expected directional result.

At step 109, 152,596/218,077 myocardial nodes exceed the voltage threshold and
154,482/218,077 exceed the activation threshold. Between accepted steps 108
and 109, 1,459 nodes crossed the voltage threshold and 1,453 crossed the
activation threshold. Chamber pressures are 5,074.149 Pa LV and 1,210.173 Pa
RV. The voltage/activation counts establish electrical state changes in this
model; zero outlet ejection means they do not establish a heartbeat.

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
The full step-109 checkpoint SHA-256 is
`22edce8bf5834361b3309c73aaf5f88bd974f227acde4665fc1e774c34be8ef7`; both
monitors bind the same package and native program fingerprints. The copied
arrival-bound state SHA-256 matches the observer receipt exactly.

The preregistered outlet baseline is step 103 (52.736 ms): LV `4,280.354 Pa`,
ascending aorta `10,892.484 Pa`, RV `1,123.896 Pa`, and pulmonary artery
`1,689.842 Pa`. Both gradients were negative and both outlet flows were zero.
By step 109 the LV pressure had risen to `5,074.149 Pa` and the aortic pressure
had fallen to `10,869.680 Pa`; RV pressure had risen to `1,210.173 Pa` while
pulmonary-artery pressure was `1,681.790 Pa`. The two gradients remain
negative. A separate kinematic screen measured 6.120 mm mean myocardial-node
displacement from step 53 to 108 while chamber volume changed by less than
0.001 ml. That motion is not proof of active contraction or pressure-volume
closure. The next decision still depends on the declared continuation reaching
a positive outlet gradient or its stopping horizon.

## Active-tension source review

The current dirty Matter checkout (`fac41f55c68af622b8b06214adb7c0b876faeeb7`)
contains a native active-tension path: when an FEM object has the
active-tension flag, the element force averages `field.secondary.x` over its
four nodes, multiplies by the material's maximum fibre tension, and adds the
prescribed fibre Piola stress. The candidate source also sends electrical
boundary overrides for this run. In this checkout, `fem.metalinc` has SHA-256
`165412851278892897da11c4ad5f09f1f075b744faf9681dfed6b260e13a8caf` and
`ventricular_source_step.mm` has SHA-256
`8d47ba3460d2eb7a6380ca019c78a85dd55e9674323199c7411b0a4495797263`; both
files are modified in the working tree. The pinned build receipt identifies the
vascular source formula and binary outputs, but it does not provide a complete
transitive source fingerprint for this active-tension implementation. This is
a source-design inspection only; it does not prove that the pinned binary
applied a particular active stress at step 109. The observed motion and
activation are not treated as a causal mechanics result.

The arterial pressures are source-model reconstructions from accepted vascular
volumes and the pinned package law, not independent physiological measurements.
No material parameters, outlet laws, or controller forces were changed. The
continuation remains unqualified for ejection, repeated cycles, or a true
heartbeat.

## Live continuation

At this receipt, native PID `46480` is still running the preregistered
step-53-to-240 continuation. Read-only observers `50357` (activation) and
`50711` (outlet gradients) are running on the same host. Their local step-109
snapshots, exact plans, monitor sources, and the corrected attempt history are
retained in
[`media/cardiac-electroactivation-response-20261004/`](media/cardiac-electroactivation-response-20261004/).

Next gate: continue sampling until the native process ends. At the first
positive LV-to-aorta or RV-to-pulmonary-artery gradient, compare that accepted
state with the contemporaneous myocardial activation and outlet flow. Audit
the valve equation only if a positive driving gradient still produces zero
flow. Preserve a non-ejection result if the continuation reaches its limit.
