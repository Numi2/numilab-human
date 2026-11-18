# Numi M6 resting load and discrete aortic preload

The integrated native scene accepted eight 15.625 microsecond roots with the fitted aortic reference. The body still failed the settlement gate. The existing Matter contact path reported 18.70 N vertical bed reaction on the first root and zero on roots 2–8 against 952.86 N body weight. This is 0.125 ms of accepted startup motion. It does not qualify supported rest, breathing motion, or sustained viewer performance.

## Verified changes

- **Discrete aortic reference.** The analytic cylinder preload left a 3.57861 N free-node resultant in an Apple M6 frozen evaluation with the exact aortic contour, actual CVSim pressure and source wall material. Inverse elastostatics refit only the unloaded reference under the same pressure and gravity. The resulting Matter GPU norm was 0.00065794 N. Cooked current positions and nodal masses matched bit-for-bit. The 4 mm reference-change trust region stalled. The passing local fit moved an unloaded coordinate by up to 8.46 mm, or 8.59 mm in the loaded scene's registered frame. The wall correction is an inferred unloaded reference for this mesh. It includes no surrounding-tissue loads in the fit. The integrated runtime retains the actual external contact, cavity pressure and physiological state. The small fit runs once in preparation and uses the existing constitutive law. [Inverse arterial elastostatics reference](https://www.numdam.org/item/10.1051/m2an/2012057.pdf).
- **Accepted-state measurement.** The settlement gate now reads changes in the complete transaction's accepted generalized velocities. The unconstrained intermediate predictor remains telemetry. Eight-step acceleration, support reaction, FEM speed, equality and physiology gates were retained.
- **Small frozen-load arena.** The generic Matter seal now omits Metal binding padding for Numi Lab. On the same 432-node wall the old runtime exposed a fourth slot for three actual parameters and rejected the static load capture. The patched runtime used three slots and passed both the pressure and pressure-plus-gravity probes.

## Integrated observations

| Measure | Previous .25 cap, pressure ramp, first 8 roots | Fitted wall, .25 cap, full pressure, 8 roots |
| --- | ---: | ---: |
| FEM peak speed | 4.01967 m/s | 0.04607 m/s |
| Location of fastest node | Abdominal aortic wall | Skin/fat |
| Krylov columns | 4,960 | 4,160 |
| Command time | 230.69 s | 197.53 s |

The 16.1% Krylov-work and 14.4% command-time reductions compare configurations with different first-root pressure. The measured command-scope factor for the fitted run was 6.33e-7× for this prefix. The whole process took 381 s including preparation and omitted viewer presentation. A separate same-configuration shader comparison kept the contact, accepted observations, iterations and residual telemetry identical, and changed command time by 0.18%. The four-support shader remains an experiment.

Starting knee acceleration at the .25 cap was about 62–63 rad/s² on either side, down from roughly 199 and 152 at the earlier .10 cap. The source torque certificate had very small selected-row residuals while the native knees still accelerated. The static certificate includes its equality reaction estimate. The integrated trace records the real motion after the constraint and tissue forces. The .40-cap trial also failed supported rest and obtained a higher maximum articulated acceleration. A supported whole-body initialization and physical knee equilibrium remain open.

The app overlay depends on the prior tilted-bed observer and the accepted-velocity patch. It enables the prepared wall reference only with `NUMI_HUMAN_RESTING_AORTIC_DISCRETE_PRELOAD=1`. The retained launch manifests pin the source inputs, binaries, runtime libraries and exact device scope. The first overlay attempt failed in an isolated wall cook that inherited full-scene contact metadata. The tested overlay strips that metadata only from the temporary reference/mass cook and preserves all contact ownership in the actual scene. The `preload/` logs retain the wall fixture and preparatory limits.
