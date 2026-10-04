# Cardiac electroactivation and outlet-gradient screen — 2026-10-04

## Result

The earlier accepted step-109 state had advancing electrical activation but no
aortic or pulmonary ejection. Its reconstructed LV-to-aorta pressure
difference was `-5,795.531 Pa` and the RV-to-pulmonary-artery difference was
`-471.617 Pa`; both flows were zero. The gradients had moved toward zero since
the step-103 baseline, but remained negative. Those sampled states did not
demonstrate a valve-law failure: with a nonpositive source-model driving
gradient, zero one-way outlet flow is the expected directional result.

The continuation then reached accepted step 122 (62.464 ms). Pulmonary
pressure difference became positive at step 121 (`+4.195 Pa`), with flow
`5.245 ml/s`; at step 122 it was `+8.297 Pa` with flow `10.372 ml/s`. The
accepted-trajectory flow integral is `0.007996 ml` of pulmonary outflow over
the full observed interval. Aortic pressure difference remained negative at
step 122 (`-3,794.969 Pa`) and aortic flow remained zero. The exact-trajectory
analysis reports zero complete pacing cycles and maximum blood-volume error
`0.000387 ml`.

The native solver rejected the next candidate after step 122 with
`NM_STATUS_NONLINEAR_SOLVER_FAILURE` (status 10), after using the full 64-step
FGMRES budget. The run had accepted 69 additional steps from its step-53
checkpoint and stopped at 62.464 ms, well before its declared step-240 / 122.88
ms horizon. This is a recorded numerical failure, not evidence that the
predicted aortic ejection would or would not have occurred later. No automatic
retry was started. The native status diagnostics
`[0, 0, 0.000131082226, 0.00000249402251]` are retained without assigning
meanings to their individual slots.

At step 109, 152,596/218,077 myocardial nodes exceeded the voltage threshold
and 154,482/218,077 exceeded the activation threshold. At step 122, the
observer counted 171,133 voltage-threshold and 172,956 activation-threshold
nodes. Chamber pressures were 7,025.829 Pa LV and 1,673.444 Pa RV. These
voltage and activation counts show electrical-state changes in this model;
brief pulmonary outflow without aortic ejection or a complete cycle does not
establish a heartbeat.

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
closure. This remains insufficient as a causal active-contraction result.

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
applied a particular active stress at step 122. The observed motion and
activation are not treated as a causal mechanics result.

The arterial pressures are source-model reconstructions from accepted vascular
volumes and the pinned package law, not independent physiological measurements.
No material parameters, outlet laws, or controller forces were changed. The
continuation remains unqualified for ejection, repeated cycles, or a true
heartbeat.

## Stopped continuation and next gate

Native PID `46480` and read-only observer PIDs `50357` and `50711` have stopped.
The process wrapper recorded exit code 2; the last accepted checkpoint and
full native log remain on `macmini` and are bound by hashes in
[`attempt-003/run-completion.json`](media/cardiac-electroactivation-response-20261004/attempt-003/run-completion.json).
The exact forward trajectory, copied plan, observer records and independently
rerun trajectory analysis are retained beside it. The progress snapshot still
reports `native_live=true`; the later wrapper completion record and process
table establish that this flag is stale.

Next gate: analyze the first rejected candidate and the source-to-wall-pressure
transfer at the stopped state before preregistering any further native run.
The aortic driving gradient was still negative, so these data do not justify a
valve-law change. Keep the brief pulmonary ejection, zero aortic output,
zero-cycle result and nonlinear failure distinct.
