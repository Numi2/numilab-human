# Neuromusculoskeletal Human completion

This is the selected release target from 8 September 2026. It implements the
neuromusculoskeletal portion of [the permanent roadmap](DEVELOPMENT_ROADMAP.md).
The cumulative source target union remains intact. Systemic organ physiology
is deferred for this release; its rows must not be relabelled implemented.

Completion means that every row below passes on one published, fingerprinted
stack. A regional numerical result does not close a whole-body row. Public data
is the default; missing experimental evidence is a release dependency, with a
specified acquisition protocol rather than an invented calibration result.

| Required outcome | Owning implementation | Admission and evidence needed | Current release status |
|---|---|---|---|
| Source-backed bilateral skeleton, joints, neck, hands and feet | Human source adapters and native Core lowering | Immutable source/units/frame/laterality records; source topology, multipose kinematic and loaded-joint parity; explicit unresolved correspondence | Open: current executable composition is partial |
| Activation and compliant muscle–tendon dynamics | Human source curves; Lab muscle programs | Source force–length–velocity and activation parity, dynamic fibre/tendon equilibrium, passive loading, fatigue and energetics; no fixed global action-count ceiling | Partial: initialized-fibre activation, source-force response and tendon sensor units pass; complete dynamic source parity remains open |
| Anatomical tissue registration and mass ownership | Human binding compiler; Core/Matter compiler; joint runtime | One reference geometry, cooked nodal mass measure, conserved zeroth/first/second moments, admissible residual rigid inertia, complete COM-frame rebase, unique force owner | Partial: costal registration/mass compiler, v5 admission and eight accepted roots pass; loaded coupled qualification remains required |
| Loaded regional tissue mechanics | Matter plus source attachment/contact owners | Prestress equilibrium; time-step and mesh convergence; conservative reactions; ligament/cartilage/meniscus contact; fascia/aponeurosis/disc/skin/fat mechanics | Open: LCL prestress rejects; prescribed-boundary probes are retained separately |
| Experimental calibration | Human calibration importers and native specimen simulations | Hash-locked raw measurements, specimen identity, reproducible boundary conditions, identifiable parameters, uncertainty and sealed specimen-level validation | Open: current patellar fit is a previously inspected same-plug test-day candidate |
| Anatomical external and self-contact | Native collision/CCD/contact, authored ScenePack/RealityPack | Source colliders and exclusions, friction/compliance calibration, contact/pressure convergence, no missed crossings, conservative whole-body response | Open: current foot witnesses are insufficient |
| Causal sensing and control | SensorPack/PolicyPack, Lab and Brain | Accepted-state proprioception, tactile and visual observations with time/delay/noise/history; reflex/model-based baseline and interchangeable learned/Brain controllers; rejected futures cannot publish | Partial: source-bound spindle/periodic control and bounded native recruitment pass; calibrated anatomical closed-loop behavior remains open |
| Sustained standing, recovery and walking | TaskPack compiler, native metric producer, existing behaviour evaluator | The frozen 420-trial protocol; complete accepted-root reductions and attempt audit; exact stack/task/program bindings; assistance absent | Open: native metric producer and qualified controller remain missing |
| Broader movement and manipulation | Authored TaskPack and controller programs | Preregistered turning, running, sit/stand, terrain, reaching, grasping, carrying and bimanual outcomes with held-out loads/conditions | Open: protocols, controllers and complete native traces required |
| Personalization and population validity | Human subject compiler, calibration/UQ owners | Subject registration, parameter identifiability, uncertainty propagation and held-out subjects within declared domains | Open: no qualified population model |
| Apple runtime and delivery | Native compiler/runtime, Swift scheduler, CLI/install owners | All five PerformanceEnvelope rows fixed and passed, source-identical replay, no swap/untyped failure, real runtime visual inspection, clean installation and exact published artifact receipts | Open: interactive real-time and full-fidelity workloads unqualified |

## Integration order

The integration dependency chain is registration and ownership → prestress and
loaded contact → whole-body tissue coupling → sustained closed loop → behaviour
qualification. Data import, source completion, controller development, metric
lowering and profiling can progress alongside it. No lane waits for standing
before starting its independent work.

The [costal ownership increment](COSTAL_TISSUE_OWNERSHIP_20260908.md) consumes the
same cooked mass partition in the rigid and Matter owners, rebases local source
points and joins the accepted-root transaction. Promotion beyond this bounded
passive fixture requires loaded thorax/contact convergence, source articulation
and independent material evidence. Eight 10-microsecond accepted roots do not
close the sustained-control or behaviour gates.

## Frozen behaviour gate

Use [HUMAN_BEHAVIOR_QUALIFICATION.md](HUMAN_BEHAVIOR_QUALIFICATION.md), including
its source-bound root/trunk frames and complete native reduction contract:

- Standing: 20 distinct seeds, 60 seconds each; all pass.
- Recovery: 100 distinct seeds/nonzero impulses, five seconds each; at least 95
  recover and hold without a posture violation.
- Walking: 100 seeds at each of 0.5, 1.0 and 1.5 m/s, 120 seconds each; at least
  95 pass at each speed, with speed RMSE no greater than 0.15 m/s.
- Missing, duplicated, assisted, truncated or otherwise invalid traces
  invalidate the bundle rather than count toward permitted task failures.

Additional task thresholds must be authored from source observations and
measurement uncertainty and frozen before training/selection. No universal
height, tilt, manipulation-error or material-error threshold is invented here.

## Publication and claim rules

Retain numerical, transaction, physical, calibration, visual and performance
evidence as distinct records. Each names exact source and artifact identities,
the runtime device, invocation, attempted/accepted time, failures and measured
limits. Update owner repositories first and integration manifests from published
owners. Preserve unrelated work and historical evidence.

The active release remains **incomplete** while any required row is open.
Deferred systemic physiology and unmeasured scientific claims are never counted
as completed work for this target.

The [active muscle increment](ACTIVE_MUSCLE_CONTROL_20260908.md) adds explicit
Brain/Metal recruitment and repairs native activation continuation and sensor
units. Its short physical horizons leave the sustained behavior gate open.

The [standing/walking completion plan](STANDING_WALKING_COMPLETION_PLAN_20260908.md)
repairs recruitment-induced ground penetration and unsupported static reactions.
The current anatomically constrained posture still has an unbalanced support
wrench; source-compliant sustained behavior and the full 420-trial gate remain open.
