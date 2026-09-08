# Coupled Human support impulses

Native revision `3c15621c33238dc1ec5e51d1ddde265d99d8b95f` replaces the history-dependent support penalty with independent contact impulses in Matter's coupled Newton–FGMRES system. Loaded rigid-body checks now recover weight and Coulomb friction from cold, weight and excessive impulse guesses. This closes that bounded formulation defect; anatomical equilibrium and standing remain open.

The preceding [loaded-support diagnostic](media/costal-bvh-20260908/support-history-current.json) remains failed evidence: the old 97 kg cold-start fixture obtained only 1.222% of its required support impulse despite numerical convergence. Initializing history with body weight concealed the defect. The new law treats history as an initial guess, includes its dual residual in the Krylov operations and final certificate, and updates velocity and impulse with the same line-search fraction. Sliding retains the nonsymmetric Coulomb derivative. The native numerical contract documents the equations and the fixed-geometry approximation within each Newton linear action.

## Replay failure and repair

The batched test exposed three integration defects: immutable support queries were not replicated across environments; generic source velocities and published efforts used solver capacity in place of physical `nv`; and the runtime's advertised candidate-point capacity omitted Human support. The last defect let a six-point query exceed the world's internal Jacobian arena. Tracing found zero entries where the free body's translation Jacobian must equal one, before the first linear solve. Extra snapshots or a changed continuum could alter the symptom without fixing it.

The runtime now advertises the maximum of continuum-contact, anatomical-attachment and Human-support point counts. The borrowed-query boundary checks the internal world-point and Jacobian buffer sizes before encoding. The test deliberately understates that capacity and requires rejection. Contact argument-buffer resources are also declared on each replacement compute encoder. Temporary poisoning, broad synchronization and trace instrumentation were removed. The failed trace log and rejected full-shader-validation attempt are retained in the [evidence bundle](media/hard-support-kkt-20260908/receipt.json).

## Mac mini evidence

All final runs used Metal API validation on `ssh macmini`, with no observed NumiVivo `md-run` or `md-benchmark` workload. The receipt records the native revision, artifact hashes, commands and logs.

| Check | Result and boundary |
|---|---|
| Loaded support | Eleven cases in three environments: varied mass, timestep and history; six redundant rows; separation; sticking; sliding. Analytic impulse error is bounded by 2e-6 N s and velocity error by 2e-7 m/s. Ten consecutive runs pass bytewise q/v and impulse-history replay. |
| Internal arena admission | A deliberately undersized program is rejected before GPU execution. |
| Contact linearization | Point, sphere and ellipsoid witnesses; independent dual finite differences, sliding derivative, shared update fraction and exact history rollback pass. |
| Native regression | Nine focused tests pass, covering the new support checks, BVH/CCD, mixed MPM/FEM, cohesive mutation, shared rigid contact, production rollback and Human attachments. |
| Prepared Human | Four scenarios of four accepted roots at 100 microseconds pass, including bytewise replay and unavailable-observation suppression. Runner time: 31.781 seconds. |
| Costal fixture | The existing source-default eight-root transaction at 10 microseconds, including rejected-state isolation and retry, passes. Runner time: 28.328 seconds. |

The load oracle supplies an analytic free predictor because the generic MetalWorld device hook precedes ABA. Its fixed tetrahedron is remote from the body. It qualifies the production coupled correction service, not generic gravity ordering or anatomically loaded tissue. The prepared Human fixture uses the small admitted FEM package, not the complete registered anatomy. The costal run covers 80 microseconds of physical time; its elapsed runtime is not a sustained-behavior throughput qualification. Full GPU shader validation was rejected by host kernel-geometry admission and is not counted as a pass.

## Next acceptance target

The next useful experiment is source-compliant loaded anatomical settling on one exact prepared pose: coupled joint equalities and unilateral limits, muscle fibres, registered tissue mass/prestress, and the corrected contact law. Record the force balance, gaps, joint/fibre residuals and accepted physical time. A numerically accepted root alone cannot establish equilibrium.

Once that state is admitted, extend physical duration and measure the actual runtime bottleneck before controller tuning. Then qualify causal standing, perturbation recovery and walking against the unchanged [420-trial release matrix](NEUROMUSCULOSKELETAL_RELEASE_MATRIX.md). Regional anatomy, independent calibration and held-out validation remain required. There is no new full-Human or whole-suite completion claim.
