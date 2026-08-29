# Numi Human development roadmap

Status: evidence-led plan, 2026-08-29

Reviewed workspace snapshot:

- Numi Human: `main` at `0feea7d637c1a698c89984155f0a7098f0350e3f`
  (`origin/main` matched at review time);
- Numi runtime: MetalRobo `coupled` at
  `6d03b7a267e2743bf9cb51362ce18764ac1408a7`;
- local importer/compiler suite: `88 passed` with
  `PYTHONPATH=src /Users/home/.pyenv/versions/3.10.12/bin/python3 -m pytest -q tests`;
- `numi doctor`: Apple Silicon, Metal tools, native trainer, rollout runtime,
  robot catalog, and MLX learner reported ready.

This document is a plan, not new physical qualification. The review inspected
the live repositories and retained evidence but did not rerun a long Metal
qualification. The [completion gap ledger](HUMAN_COMPLETION_GAP_LEDGER.md)
remains the current claim boundary and must be updated when executable evidence
changes.

## Executive decision

The shortest credible path is:

```text
live truth and reproducibility
        |
        v
dynamic/source parity
        |
        v
registered anatomical contact
        |
        v
assistance-free balance
        |
        +--------------------+
        |                    |
        v                    v
closed-loop gait       two-way regional tissue
        |                    |
        +----------+---------+
                   v
       whole-body tasks and load-bearing tissues
                   |
                   v
       physical exterior, personalization, validation
```

Do not put gait training, whole-body FEM, organs, or photoreal skin on the
critical path before contact and balance. Numi Human already has unusually
strong source, force-path, transaction, and anatomy-registration foundations;
its central product gap is that the latest composition is not end-to-end
requalified and the current stand remains `balanced=false` over only a bounded
12.8 ms horizon.

Surface-envelope percentage is not a product KPI. Of the current 204 point
laws, 155 are explicitly non-bone in the source model, eight are anterior
non-rib sites, and 24 lack an anatomically correct registered bone surface.
Keeping those endpoints fail-closed is better than forcing artificial bone
attachments to reach 100% coverage.

## Product north star and boundary

The assumed product is a research-grade, Apple-native neuromusculoskeletal
digital Human that can:

1. reproduce a pinned source model and every transform, parameter, and license;
2. execute muscles, articulated dynamics, contact, sensing, and selected
   deformable regions transactionally on Apple Metal;
3. stand, recover, walk, and perform bounded whole-body tasks without hidden
   root assistance or direct-joint-torque shortcuts;
4. replay deterministically and report numerical, physical, memory, throughput,
   and visual evidence from the actual runtime; and
5. support progressively calibrated regional tissues and subject variation
   without claiming clinical validity outside held-out evidence.

It is not yet a clinical twin, injury predictor, complete organ model, finished
skin avatar, or biological theory of neural control. If the primary product is
instead an entertainment avatar, exterior and rendering work can move earlier
after Stand v2. If it is a clinical decision tool, context-of-use, subject data,
uncertainty, and validation must move ahead of general gait and appearance.

## What exists today

| Layer | Current owning evidence | Boundary still open |
| --- | --- | --- |
| Source and provenance | Pinned BodyParts3D, MyoSim, Mortensen, Rajagopal, MoBL-ARMS, and Z-Anatomy records with hashes and explicit redistribution rules | Some catalog models and motion assets have non-commercial or unresolved redistribution terms |
| Offline compiler | Python imports, classifies, registers, and compiles immutable native payloads; it is not in the live simulation loop | `model.py` is 13,310 lines, `cli.py` is 1,807 lines, and most tests are concentrated in one file |
| Native model | `NHRIGID2` 157-body/128-DoF articulation, `NHMYO2` 416 muscle routes, 1,815 sites, 143 wraps, joint equalities, and ten foot witnesses | Full held-out source parity, high-velocity behavior, complete dynamic constraints, and broad collision remain open |
| Tendon transfer | `NHTENDON3` retains all 832 endpoints: 628 distributed envelopes and 204 exact point laws; standalone Metal force/moment and replay evidence exists | The latest 628/204 payload has not been qualified through the complete persistent stand, borrowed consumer, rejection rollback, and no-direct-torque path |
| Persistent dynamics | Metal owns activation, current-pose routes, gravity/dynamics, support, state publication, and bounded replay | Current retained stand is 12.8 ms, reports residual RMS `12.5546`, and is not balanced |
| Regional tissue | Six pectoral regions drive a 326-node Matter FEM demonstration with replay and rollback | Geometry and 10% load share are assumptions; load assembly is not yet the reusable device-resident production owner; coupling is one-way |
| Control | Bounded source-muscle part coactivation and a lower-body walking action contract exist | No assistance-free posture controller, deployed Human task/policy, perturbation recovery, or held-out gait |
| Anatomy and appearance | 184 registered BodyParts3D bone meshes, source muscle/tendon presentation, torso layers, and four-angle review | Exterior is not physically skinned; muscle surfaces are presentation; organs, vessels, skin, and most tissues have no mechanics |
| Validation | Source provenance, FP64/Metal comparisons, force/moment conservation, rollback, replay, and visual evidence are retained | No subject calibration, population model, blinded internal-load prediction, or same-workload qualification of the future complete stack |

The exact evidence behind these rows is in
[Human Stand v1](HUMAN_STAND_V1.md),
[the tendon transaction](HUMAN_TENDON_STEP_TRANSACTION.md),
[the abdominal endpoint record](ABDOMINAL_SOURCE_COMPONENT_ENTHESES_V1.md),
[the pectoral fascia record](PECTORALIS_FASCIA_V1.md), and the
[gap ledger](HUMAN_COMPLETION_GAP_LEDGER.md).

## Architecture to preserve

| Owner | Responsibility | Must not own |
| --- | --- | --- |
| `numilab-human` Python | Fetch and fingerprint sources; parse; register; classify; generate review receipts and immutable payloads | Per-step simulation, hidden material inference, or runtime control |
| Core C++ | Compile and validate the Human model; own FP64 references, source lowering, fixed indices, and native ABI checks | Per-environment host stepping or untracked source substitutions |
| Metal | Persistent articulated state, muscles, contact, selected tissue kernels, sensing, inference, deterministic reductions, and rendering | Per-step host readback or an alternate Python physics path |
| Matter | Calibrated deformable regions composed through the Human transaction | Reapplying a muscle load already present through `J^T` |
| Swift/Numi scheduler | Bounded rollout submission, waits, timeouts, reset/publication, and artifact lifecycle | Physics, string lookup in the hot loop, or unbounded retained rollout copies |
| MLX | Batch learning and system identification from compact published rollouts | Owning simulation state, contact, or rollout scheduling |

Every accepted control step remains a transaction: successful state publishes;
failure restores the prior accepted state. A downstream tissue consumer may
encode against borrowed Metal objects only. It may not independently commit,
wait, retain, or replace the enclosing command buffer.

## Strategic choices

### Keep MyoSim canonical through Walk v1

The README identifies MyoSim `myofullbody` as the active mechanics owner, while
`config/numi-targets.v1.json` still describes a Rajagopal plus MoBL-ARMS target.
Resolve that ambiguity now:

- keep the pinned MyoSim 416-route body as the production foundation through
  Stand v2 and Walk v1;
- use Rajagopal/OpenSim as a comparative numerical and biomechanical oracle;
- keep Mortensen neck as an explicit registered extension; and
- admit MoBL-ARMS only as a research profile until exact bilateral composition,
  wrap parity, and commercial rights are resolved.

Changing the canonical skeleton while contact and balance are still open would
reset transforms, control, validation, and tissue attachments simultaneously.

### Preserve the 416-action physical contract

Policy implementations may use synergies, reflexes, latent actions, or a
hierarchy, but the compiled task-to-runtime contract should retain the identity
and bound of every source actuator. A reduced controller is a mapping onto that
surface, not a replacement that erases muscle provenance.

### Use regional deformable islands

Do not attempt a monolithic whole-body FEM. Use articulated dynamics for the
whole Human, distributed pressure contact for real-time feet, and selected
deformable regions where stress, strain, compliance, or load redistribution is
the actual question. Each region needs its own geometry, material, convergence,
coupling, and held-out validation receipt.

### Keep training physics and authority distinct

Smooth or simplified contact can be useful for teacher optimization and policy
training. Final selection must run on the authoritative Numi contact and tissue
path. A policy that succeeds only under training contact is not promoted.

## Milestone roadmap

Durations are planning ranges, not commitments. They assume one to two core
physics/runtime engineers, part-time biomechanics/anatomy support, and regular
access to the Apple M4 Pro evidence machine. Parallel tracks are intentional.

### M0 — Stabilize live truth and publish Baseline 0.1

Indicative duration: 1–2 weeks. This blocks every other milestone.

Deliverables:

1. Replace the stale tracked MetalRobo pin (`2aab522`) with a generated runtime
   contract tied to the exact live source and artifact fingerprints, and make
   the target catalog agree that MyoSim is the current production owner.
2. Replace the unbounded broad `git status --porcelain` audit with targeted Git
   plumbing, a timeout, and a revision-only fallback appropriate to iCloud
   File Provider checkouts.
3. Add a machine-readable Human capability/evidence registry. Every row records
   Human revision, runtime revision, source locks, payload ABI/hashes, device,
   command, duration, evidence type, and `current` versus `historical` status.
4. Rebuild and run the latest 628/204 `NHTENDON3` payload through the complete
   persistent stand transaction on the exact reviewed stack.
5. Preserve golden payloads, manifests, decoder fixtures, and replay results
   before modularizing the compiler.
6. Define artifact retention classes for the current 2.6 GB ignored `Build/`
   tree and tracked evidence media. Do not delete valid evidence merely because
   it is generated.
7. Tag the first reproducible baseline and document the supported Python/test
   entrypoint; the shell default Python currently lacks pytest.

Exit gate:

- the current Human/runtime pair, sources, and all payload hashes resolve from
  one registry entry;
- the latest persistent run proves FP64 parity, all 832 transfers, force/moment
  conservation, same-command-buffer consumer execution, rejection rollback,
  no-direct-torque identity, and byte-identical replay;
- the local 88-test suite and bounded native smoke pass; and
- no current document silently presents an older payload or runtime as the
  latest end-to-end qualification.

### M1 — Dynamic and source parity

Indicative duration: 4–6 weeks after M0.

Deliverables:

1. Freeze a held-out pose/velocity corpus spanning lower limbs, spine,
   shoulders, elbows, wrists, hands, and support-relevant joint limits.
2. Compare source transforms, mass/inertia, joint programs, route length,
   moment arm/Jacobian, activation state, compliant muscle/tendon state, passive
   force, and generalized force against FP64 and source oracles.
3. Finish and qualify persistent high-velocity bias, equality, limit, and
   passive-preload behavior across multiple timesteps. Do not infer correctness
   from the source-default pose alone.
4. Add exact wrap families only where the canonical source requires them.
   Rajagopal/MoBL features do not enter the production model without their own
   parity corpus.
5. Split source parsing, payload formats, registration, tendon compilation,
   tissue compilation, and audit code behind the golden ABI fixtures from M0.

Exit gate:

- numerical tolerances are frozen before the final runs, not fitted after
  observing errors;
- every corpus case reports CPU/Metal errors and constraint residuals;
- timestep refinement and replay do not expose state-ordering or random-stream
  changes; and
- no refactor changes a golden payload or manifest without an explicit ABI
  version and migration test.

### M2 — Registered contact and joint constraints

Indicative duration: 4–8 weeks after M1.

Deliverables:

1. Complete reviewed, provenance-pinned registrations for `calcn_r`, `toes_r`,
   `calcn_l`, and `toes_l` using the existing receipt workflow.
2. Author conservative collision proxies, contact pair exclusions, material
   identities, plane/terrain registration, and deterministic broad/narrow phase
   ordering.
3. Replace point-witness-only product contact with distributed foot pressure
   fields while retaining an exact reference path.
4. Calibrate friction, compliance, damping, and regularization against declared
   measurements or literature. Keep measured values separate from solver
   stabilization parameters.
5. Add the minimum whole-body collision and joint-limit set required for stance,
   falls, and perturbations; do not infer collision from render geometry.

Exit gate:

- foot registration passes multi-angle landmark and source-frame review;
- contact passes static load, sliding, rolling, impact, separation, pair
  exclusion, and timestep-refinement tests;
- total force, moment, centre of pressure, penetration, and energy behavior are
  reported for CPU reference and Metal; and
- a deterministic reset produces byte-identical authoritative contact replay.

### M3 — Human Stand v2

Indicative duration: 4–8 weeks after M2.

Deliverables:

1. Implement a transparent deterministic posture baseline before learning:
   centre-of-mass/centre-of-pressure feedback, task-space stabilization, and/or
   a documented reflex layer mapped only to muscle excitation.
2. Remove root assistance after initialization and keep it exactly zero during
   the qualified horizon.
3. Add push, support-surface, pose, and parameter perturbation suites with
   deterministic seeds and explicit fall/step criteria.
4. Publish device observations needed by future controllers without per-step
   readback: body state, contact/pressure, tendon state, and bounded task state.

Exit gate:

- an engineering target of at least 10 seconds of assistance-free stance is
  frozen before tuning;
- the model remains within declared support, posture, constraint, and contact
  limits and recovers from a preregistered perturbation suite;
- no direct joint torque or hidden pose drive enters the policy path;
- deterministic reset/replay, rollback, FP64 spot checks, memory, throughput,
  and GPU-counter evidence pass on the exact device; and
- four-angle runtime frames are inspected at original resolution.

Stand v2 is the first release that should be described as standing.

### D1 — Data and validation foundation

Indicative duration: starts in M0 and runs in parallel through all milestones.

Deliverables:

1. Create a dataset registry with source revision, subject/activity coverage,
   coordinate convention, measurement modality, license, allowed use, split,
   transformations, and derived-artifact lineage.
2. Use AddBiomechanics CC-BY data for broad motion/GRF generalization and
   separately licensed local/OpenCap captures for task-specific checks.
3. Adopt ISB joint-coordinate reporting and retain source coordinates alongside
   any standardized view.
4. Build train/calibration/validation/test splits by subject and activity.
   Freeze the held-out split before controller or material tuning.
5. Reserve the instrumented Grand Challenge knee-load trials as a blinded
   future gate; do not use their released targets for calibration of the trial
   designated as blind.
6. Add uncertainty and sensitivity reporting for source scale, mass, strength,
   contact, and material parameters.

Exit gate:

- every metric can be traced to a source, transform, split, and license;
- internal-load, motion, GRF, EMG, and material tests remain separate evidence
  categories; and
- a result on one subject or one activity is not presented as population
  validation.

### M4 — Human Walk v1

Indicative duration: 8–12 weeks after M3, with D1 already active.

Deliverables:

1. Compile a fingerprinted Human task, observation, action, contact, and reset
   contract. The action surface remains all 416 bounded source excitations.
2. Start from the deterministic Stand v2 controller and add a hierarchy:
   phase/foot-placement intent, reflex or synergy layer, and a small learned
   residual. Do not begin with an unconstrained monolithic gait policy.
3. Use OpenSim Moco, SCONE, Kinesis, or MuscleMimic as offline method/teacher
   references where licensing permits. They never bypass Numi physics.
4. Train in stages: weight shift, single step, repeated stepping, commanded
   speed/direction, terrain variation, and perturbation recovery.
5. Keep MLX at the batch-learning boundary and publish exact fingerprinted
   policy packs; Metal owns live inference and physics.
6. Retain every physically valid candidate and select production only from
   held-out authoritative-contact outcomes.

Exit gate:

- distance, duration, falls, speed tracking, step timing, GRF, centre of
  pressure, joint kinematics, activation, effort/metabolic proxy, and recovery
  are reported over preregistered held-out subjects/parameter sets and tasks;
- root assistance and direct torque remain absent;
- training-contact success is reproduced on authoritative contact;
- policy, world, observation, action, task, and source fingerprints match; and
- resets, replay, throughput, peak/retained memory, and GPU traces are retained.

### T1 — Production two-way regional tissue

Indicative duration: 6–10 weeks, parallel with M3/M4 after the M1 transaction
and M2 contact foundations are stable.

Deliverables:

1. Extract the pectoral Matter integration from the visual probe into a reusable
   Human runtime owner.
2. Assemble nodal loads on device through the borrowed consumer; remove the
   host-vector/shared-buffer/separate-commit demonstration boundary.
3. Return deformable reaction to the articulation by replacing the
   corresponding portion of source `J^T`, never by adding a second copy.
4. Calibrate one region with registered geometry and nonlinear material data.
   Use pectoral fascia to productionize the API, then choose plantar
   fascia/Achilles as the first contact-relevant island.
5. Compare patch tests, constitutive response, and selected regional cases with
   FEBio or another independent reference.

Exit gate:

- material source and parameter-fit receipt, mesh/timestep/nonlinear-solver
  convergence, energy and load balance, minimum Jacobian, replay, rollback,
  and held-out deformation all pass;
- rigid plus tissue generalized force does not double count the source load;
  and
- the measured cost and memory of the coupled region are reported on the same
  device and workload as the rigid baseline.

### T2 — Load-bearing joint tissues

Indicative duration: 10–16 weeks per joint family after T1.

Recommended order:

1. knee ligaments, cartilage, and menisci, because instrumented internal-load
   data provides a credible held-out target;
2. ankle/foot cartilage, plantar fascia, and Achilles interaction, because it
   directly affects contact and gait;
3. intervertebral discs and spinal ligaments; then
4. costal cartilage and thoracoabdominal fascia for the eight current anterior
   non-rib endpoints.

Each region is a separate milestone. A visually plausible tissue never closes
a mechanical row without calibrated material, contact, convergence, and
held-out response.

### M5 — Whole-body tasks and source extensions

Indicative duration: 8–12 weeks after Walk v1; can overlap T2.

Deliverables:

- integrate and parity-qualify the Mortensen cervical/hyoid extension;
- resolve exact wrap and anchor requirements for any upper-extremity source;
- keep non-commercial MoBL-ARMS in an explicit research profile;
- add reaching, carrying, sit-to-stand, turning, and recovery tasks while
  preserving lower-body balance; and
- add independent fingers/toes only when a task and source justify their new
  articulation, contact, muscles, and validation cost.

Exit gate: each task has held-out physical outcomes, authoritative contact,
deterministic resets, policy fingerprints, replay, runtime frames, and no
regression of Stand v2 or Walk v1.

### X1 — Physical exterior, sensing, and selected organs

Indicative duration: 8–16 weeks per scoped layer after M3/T1.

Separate these products:

- a high-quality skinned presentation layer for interaction and rendering;
- a physical skin/fat/fascia shell with volume control, sliding, self-contact,
  and calibrated mechanics; and
- organ/vessel models added only for a declared use case.

The skinned render may ship earlier, but must remain labeled as presentation.
Do not convert BodyParts3D proximity weights, tube-shaped solids, or attractive
meshes into physical ownership without a solver and validation receipt.

### V1 — Personalization and credibility

Indicative duration: continuous research; 12+ weeks for the first bounded
subject-specific study.

Deliverables:

- subject scaling with preserved mass, inertia, joint, route, contact, and
  tissue constraints;
- parameter priors and population distributions, not one nominal body called
  universal;
- calibration/validation separation and uncertainty propagation;
- preregistered blinded predictions for selected internal loads; and
- a risk-informed credibility report following FDA CM&S and ASME V&V guidance
  if a medical context of use is proposed.

No clinical claim is made merely because a subject mesh fits, a simulation is
stable, or a benchmark error is small.

## Research adoption matrix

| Source or method | Recommended use | Production boundary |
| --- | --- | --- |
| [OpenSim Moco](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493) | Offline optimal-control oracle, trajectory seed, inverse problem, and parameter study | Not the live runtime; validate every converted joint, path, wrap, muscle, and contact law |
| [MyoSim](https://github.com/MyoHub/myo_sim) / MyoSuite | Current source model and control-task comparator | A muscle-actuated multibody model, not whole-body tissue or clinical anatomy |
| [MuscleMimic](https://github.com/amathislab/musclemimic) and its [2026 paper](https://arxiv.org/abs/2603.25544) | Closest 416-muscle method comparator; retargeting and imitation architecture; possible research teacher | Training targets NVIDIA/Linux; code/model and motion/checkpoint licenses are separate |
| [Kinesis](https://github.com/amathislab/Kinesis) and [SCONE](https://github.com/tgeijten/scone-core) | Hierarchical imitation, optimization, reflex, and curriculum ideas | Transfer methods, not unverified checkpoints or a substitute physics path |
| [AddBiomechanics](https://addbiomechanics.org/download_data.html) | Broad CC-BY motion, force-plate, inverse-dynamics, and subject generalization data | Preserve per-dataset attribution and quality metadata; GPL software does not enter a proprietary runtime by accident |
| [OpenCap validation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462) | Accessible capture and an independently published kinematic/kinetic comparison | Markerless estimates are not an internal-force gold standard; hosted-service terms are distinct from code |
| [Grand Challenge knee loads](https://pmc.ncbi.nlm.nih.gov/articles/PMC4067494/) | Blinded medial/lateral knee-contact validation | Keep designated targets held out until predictions are frozen |
| [FEBio](https://github.com/febiosoftware/FEBio) | MIT-licensed source oracle for nonlinear tissue, contact, material, and verification cases | Offline/reference role; binary/dependency terms and constitutive assumptions still require review |
| [SOFA](https://github.com/sofa-framework/sofa) | LGPL medical-FEM comparison and rapid prototype reference | Its acceleration ecosystem is not a Metal production path; integration must respect LGPL boundaries |
| [IPC Toolkit](https://ipctk.xyz/) and [Drake hydroelastic contact](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html) | Barrier/CCD/friction and pressure-field contact design references | A pressure contact model is not automatically a converged tissue continuum |
| [ISB coordinate recommendations](https://pubmed.ncbi.nlm.nih.gov/11934426/) | Standard reporting frames and reproducible kinematic comparisons | Retain the original source frame and exact transform alongside the standardized view |
| [FDA CM&S guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/assessing-credibility-computational-modeling-and-simulation-medical-device-submissions) / [ASME V&V](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty) | Risk-informed credibility, verification, validation, and uncertainty planning | Relevant to a declared decision/context of use; not a certification inherited by using the checklist |
| [Apple Metal profiling](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics) and [MLX unified memory](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/unified_memory.rst) | Device counters, traces, memory accounting, and Apple-native batch learning | Unified memory does not make duplicate state free; performance requires same-device measurements |

External frameworks are reference implementations, not architecture templates.
CUDA/JAX/Warp research can teach algorithms, but the production Human remains a
native Metal transaction with FP64 and independent physical oracles.

## License profiles

Maintain two explicit build/training profiles:

### Free core

- MyoSim and compatible Apache-licensed code/model records;
- BodyParts3D with CC-BY attribution;
- locally owned or production-cleared motion and measurement data;
- AddBiomechanics records whose per-dataset terms are verified; and
- no model, policy, or derivative whose training lineage is unknown.

### Research-only overlays

- MoBL-ARMS non-commercial source;
- AMASS/SMPL-derived motions;
- MuscleMimic or Kinesis checkpoints trained on restricted motion;
- Z-Anatomy CC-BY-SA derivatives isolated with share-alike compliance; and
- any atlas/model with unresolved redistribution or commercial terms.

MuscleMimic's repository is Apache-2.0, but its official retargeted dataset card
states non-commercial use, no redistribution, and no commercial use of models
trained on that data. Do not let the code license launder the motion, body, or
checkpoint license. This section is an engineering license boundary, not legal
advice.

## Evidence ladder for every milestone

| Evidence class | Required proof |
| --- | --- |
| Provenance | Exact source URL/revision/hash, license, transforms, generated artifact hash, Human/runtime revisions |
| Implementation | Owning live code path and ABI; no presentation asset standing in for mechanics |
| Numerics | Analytical/FP64 comparison, residuals, constraint errors, timestep and solver convergence, failure behavior |
| Transaction | Accepted publication, injected rejection, rollback, completed-step count, deterministic replay |
| Physical outcome | Task-specific quantities such as support, GRF/CoP, distance, recovery, stress/strain, internal load, or deformation |
| Generalization | Held-out subject/activity/material/perturbation split and uncertainty/sensitivity |
| Performance | Same revision, artifact, device, workload, counters/trace, throughput, retained/peak memory, failed steps |
| Visual | Actual runtime frames from multiple angles at original resolution, with mechanics overlays separated from clean presentation |

A build, test pass, attractive image, liveness check, reward, or force-transfer
counter cannot substitute for a physical outcome in another row.

## First 30-day backlog

Ordered list:

1. Generate the live runtime/evidence registry and eliminate the stale contract.
2. Bound the audit's Git inspection and make timeout/failure typed.
3. Rebuild and end-to-end qualify the latest `NHTENDON3` composition.
4. Mark every retained report `current`, `historical`, or `component-only` from
   the registry rather than prose convention.
5. Freeze golden payload/decoder/replay fixtures and create Baseline 0.1.
6. Split the compiler only behind those fixtures; preserve unrelated generated
   artifacts and ABI bytes.
7. Freeze the multi-pose/multi-velocity source-parity corpus and tolerances.
8. Complete the four reviewed foot registration receipts.
9. Write the contact calibration and validation protocol before choosing final
   friction/compliance values.
10. Capture a same-workload baseline for step time, throughput, allocations,
    retained/peak memory, and Apple GPU counters.
11. Specify Stand v2 observations, muscle action contract, perturbations, fall
    criteria, and held-out cases before controller tuning.
12. Build the dataset/license/split registry and quarantine restricted motion
    and checkpoints from the free core.

Review bilateral EO3 and the 11 ill-conditioned attachment candidates only if
the unchanged gates pass. Do not let those localized improvements displace M0,
contact, or balance.

## Release map

| Release | Meaning |
| --- | --- |
| Baseline 0.1 | Exact live source/runtime/artifact registry and latest-stack end-to-end requalification |
| Stand 0.2 | Registered contact, seconds-long assistance-free balance, perturbation recovery, replay, and device qualification |
| Walk 0.3 | Fingerprinted closed-loop muscle policy with authoritative-contact held-out gait evidence |
| Coupled 0.4 | Reusable two-way regional tissue owner plus at least one calibrated contact-relevant tissue island |
| Whole-body 0.5 | Loaded upper/lower whole-body tasks and selected load-bearing joint tissues without regression |
| Research 1.0 | Reproducible movement/tissue releases, population/uncertainty support, held-out validation, and published evidence boundaries |

"1.0" means a credible, extensible research platform. It does not mean every
human tissue is solved or that clinical use is qualified.

## Main risks and controls

| Risk | Control |
| --- | --- |
| Evidence lineage drifts faster than code | Generated registry, tags, current/historical status, exact hashes, fail-closed docs |
| Competing source models silently mix | One canonical owner per field; explicit comparative profiles and transform receipts |
| License contamination reaches a policy | Dataset/checkpoint lineage, free-core/research profiles, quarantine by default |
| Contact is tuned to make control easy | Calibration protocol and authoritative evaluation contact frozen before training |
| Tissue force is double counted | Replace the matching `J^T` share and prove generalized-force/energy balance |
| Refactoring changes binary semantics | Golden ABI payloads, decoder migrations, manifest/replay fixtures before module splits |
| Unified memory hides duplication | Lifetime accounting, borrowed buffers, retained/peak measurement, GPU traces |
| A visually impressive layer outruns evidence | Gap ledger and evidence ladder govern public claims |

## Definition of done

A roadmap item is done only when all three are true:

1. the lowest owning layer contains the live implementation;
2. the exact compiled artifact executes through the intended Apple-native path
   with revision, device, transaction, replay, memory, and performance evidence;
   and
3. the milestone's physical result passes its preregistered held-out gate.

Anything else is a useful source, compiler feature, diagnostic, experiment, or
candidate—and should be retained and labeled as exactly that.
