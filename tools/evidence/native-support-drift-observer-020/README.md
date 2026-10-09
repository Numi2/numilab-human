# Observer-on whole-body drift diagnosis: native run 020

This is a separate 310 s observer-on baseline diagnostic, not one of the registered 1170 control/treatment trials and not an anatomical or physiological acceptance result. The owner accepted 155,000 roots at 2 ms; the terminal accepted time is 310.000014724 s. The run exited 0, verified the loaded physical runtime 014 image, and reported no source files changed during execution. Build/source hashes and the run invocation are retained in the run directory.

The opt-in observer emitted 19,375 cadence-8 accepted samples for 157 body records (mass sum 71.9999994 kg) and 32 support regions. The full streams passed finite-value, cadence, identity-map, source-manifest, and profile-accounting checks. The separate run measured 0.11244 real-time factor with the observer on. CPU analysis overlapped execution, so this is descriptive rather than an exclusive-host benchmark. The observer adds measurement only; it does not establish drift cause.

Four legacy CSV channels matched the registered 1170 control by accepted-step key and parsed field values through step 155,000: 19,375 coupled rows, 19,375 COM/momentum rows, 620,000 support-impulse rows, and 4,846 surface-audit rows, with no shared-key field mismatches. This is parsed-value/key equality, not raw CSV byte identity. At all eight declared capture steps, MRVPACK sections 2–5 (vertex data, indices, primitive records, and instance records) were byte-identical. Whole-pack bytes were not required to match because pack metadata and root identity can differ.

## What the observer says about motion

World COM displacement from accepted step 125,000 to 155,000 (about 250–310 s) was [-0.514296, +0.074958, -0.002189] mm. The exact endpoint decomposition uses the root body-to-world quaternion and

    ΔCOM = Δroot_translation + (R1 − R0)c0 + R1(c1 − c0)

where c = Rᵀ(COM − root_translation). Over that interval root translation contributed +7.249424 mm on x, root rotation acting on the initial root-frame COM contributed −6.889067 mm, and the changed root-frame COM contributed −0.874654 mm. The decomposition residual is below 1e-12 m. Carrier translation alone would overstate the body's remaining lateral drift; a root-frame COM shift remains measured.

In the same sampled interval, support region 22 changed its selected skin vertex 476 times at cadence-8 observations; all 476 transitions coincided with a positive normal impulse. Its static seed label is ulna_l, while the runtime selected point is a weighted skin point and the seed is not its physical point owner. The observer reports pre-step J(q)·v_previous tangential speed at that support point: mean 0.106 mm/s, maximum 0.742 mm/s over samples with positive normal impulse. These are sparse pre-step observations, not continuous post-solve slip or proof that selector changes caused the COM shift.

A cadence-8 trapezoid of sampled COM velocity differs from the 250–310 s finite displacement by [-0.003461, +0.129661, +0.406847] mm. This is not a continuous integral and does not isolate projection, precision, contact convergence, or another cause. The remaining drift is unresolved. Known skin and lung intersections remain separate acceptance failures in the integrated 1170 evidence.

## Reproduction and evidence

Compact reports preserve the run hashes without copying multi-gigabyte CSVs, movie, or MRVPACKs. The run file-integrity manifest binds protected retained files. The publication manifest binds every file in this evidence bundle except itself. Scripts under reproduction read the retained Mac mini run and write only to fresh output folders. The original analyzer validation is retained in reports/motion-analysis-validation.json.

The 1178 body/support analyzer uses the source body-order manifest and invocation-bound rigid payload; its windows are 10–310 s and 250–310 s. Root-frame decomposition and cadence-8 endpoint consistency are separate read-only calculations. None changes the registered study, dynamics, contact settings, or qualification gates.

From the Human repository root on the Mac mini, reproduce into fresh output paths (the input run is protected and is never modified):

    python3 tools/evidence/native-support-drift-observer-020/reproduction/motion/analyze_motion.py --scene /Users/n/numi-human-retained-delivery-20261009/support-drift-baseline-020/native-run --manifest /Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json --first-step 5000 --last-step 155000 --out /tmp/human-motion-020-10-310
    python3 tools/evidence/native-support-drift-observer-020/reproduction/motion/analyze_motion.py --scene /Users/n/numi-human-retained-delivery-20261009/support-drift-baseline-020/native-run --manifest /Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json --first-step 125000 --last-step 155000 --out /tmp/human-motion-020-250-310
    python3 tools/evidence/native-support-drift-observer-020/reproduction/root-frame-decomposition/decompose.py
    python3 tools/evidence/native-support-drift-observer-020/reproduction/endpoint-velocity-consistency/endpoint_consistency.py
    python3 tools/evidence/native-support-drift-observer-020/reproduction/observer-integrity/verify_observer_streams.py

The two /tmp output directories must not already exist. The last three commands write only beside these copied scripts: the root-frame and endpoint calculations replace their adjacent report.json, while the observer-integrity check refuses an existing report.json. Run them from a fresh evidence checkout, or preserve prior reports and use a fresh copy of the reproduction directories.

See ../native-integrated-resting-1170/README.md for physiological outcomes and the still-open anatomy gates. This diagnostic does not close those gates or explain late drift.
