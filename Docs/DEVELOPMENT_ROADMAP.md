# Numi Human frontier execution roadmap

Status: state-of-the-art-first execution specification, 2026-08-29

This roadmap is organized by permanent architecture, scientific dependencies,
and evidence convergence. It has no calendar phases, MVP ladder, or temporary
product tier. The complete target system is fixed now; every unblocked
workstream runs in parallel, and every accepted increment must inhabit the
permanent system it is intended to become.

Reviewed evidence snapshot:

- Numi Human committed baseline before the EO3 increment: `main` at
  `fda3dc5e9ae108d841bd0d300079653e6f201217`;
- Numi runtime reviewed by the preceding audit: MetalRobo `coupled` at
  `6d03b7a267e2743bf9cb51362ce18764ac1408a7`;
- importer/compiler suite at that snapshot: `88 passed` with
  `PYTHONPATH=src /Users/home/.pyenv/versions/3.10.12/bin/python3 -m pytest -q tests`;
- a live `numi context` recheck confirmed Numi Lab `0.4.0`, the current Human
  workspace, and clean runtime `coupled@6d03b7a`; a parallel `numi doctor`
  reported the robot-catalog self-check `Killed: 9`, catalog `incompatible`,
  and overall `action required`. A confirmatory call then failed to complete
  within 90 seconds. Current aggregate readiness is therefore unhealthy and
  its failure path is not yet bounded.
- the later EO3 qualification promoted MetalRobo `coupled` to
  `864a65c916c1d52318e263469f673d3336d696b6` and passed the exact 630/202
  payload through standalone transfer and a 128-step persistent transaction;
  this does not supersede the unresolved aggregate `numi doctor` health issue.

This document defines ambition and execution; it is not new physical
qualification. The [completion gap ledger](HUMAN_COMPLETION_GAP_LEDGER.md) is
the claim boundary until superseded by executable evidence on an exact stack.

## Operating law

“Best from start” means:

1. lock the final ownership, data, coupling, action, state, failure, and
   evidence contracts before adding more incompatible demonstrations;
2. implement thin vertical slices only on those final contracts—limited
   coverage is acceptable, disposable physics is not;
3. activate source, mechanics, contact, tissue, control, data, validation,
   rendering, and Apple-GPU work simultaneously;
4. prioritize by dependency fan-out, scientific risk, irreversibility, and
   evidence leverage—not by dates or release optics;
5. allow competing algorithms behind permanent interfaces and promote them by
   reproduced evidence, rather than freezing an unproven method; and
6. keep claims narrower than ambition: owning live code plus executable
   evidence determines what is implemented, not the roadmap text.

Ultra-high urgency comes from maximum safe concurrency, short proof loops,
stable interfaces, and immediate integration. It does not authorize hidden
assistance, fake materials, one-way coupling called coupled physics, alternate
host physics, unlicensed data, or performance claims without measurement.

## State-of-the-art mission

Numi Human targets an anatomically traceable, personalized, neuromuscular and
multiphysics computational human whose articulation, muscles and tendons,
anatomical contact, deformable tissues, sensing, control, appearance, and
uncertainty execute through one Apple-native transactional architecture and are
qualified against independent physical evidence.

The final Human must support:

- a whole-human source graph spanning skeleton, joints, muscles, tendons,
  aponeuroses, fascia, ligaments, cartilage, menisci, discs, skin,
  subcutaneous tissue, organs, vessels, and nerves, with an explicit mechanics
  and evidence status for every represented structure;
- source-faithful multibody and musculotendon dynamics with semantic anatomical
  identities, exact coordinate transforms, routes, wraps, activation,
  compliant fibre/tendon state, passive force, fatigue and energetics state,
  and extensible cervical, hand, foot, craniofacial, and respiratory anatomy;
- whole-human collision topology, continuous collision on every pair whose
  compiled swept-motion bound admits tunnelling, distributed pressure contact,
  full articulated coupling, friction, impact, self-contact, and contact with
  deformable anatomy;
- a multiresolution whole-human continuum in which active sparse solve domains
  refine the anatomy without changing the coupling law or creating disconnected
  tissue “islands”;
- conservative two-way exchange among articulation, muscles, contact, tissues,
  and external loads with every force owned exactly once;
- hierarchical neural control and state estimation through the full semantic
  stimulation surface, with reflex, model-based, learned, and NumiBrain-driven
  controllers interchangeable behind one contract;
- device-resident proprioception, tactile/pressure, RGB-D, event, physiological,
  and task observations, plus a mechanics-consistent physical exterior and a
  separately owned presentation surface;
- subject-specific identification, population distributions, uncertainty,
  sensitivity, identifiability, and held-out generalization; and
- deterministic replay, transactional rollback, scientific receipts, and
  same-stack Apple-GPU performance evidence.

The versioned `HumanPack.target-coverage` matrix fixes the completion scope. A
compiled run may deactivate expensive fields that are irrelevant to its task,
but the architecture must support every row and whole-Human completion requires
every cell to reach integrated qualification; absence may not be redefined as a
smaller “declared” Human.

| Fixed coverage domain | Mandatory target |
| --- | --- |
| Skeletal topology | Complete axial and appendicular skeleton; cervical/hyoid/craniofacial system; articulated hands, intrinsic fingers, feet, and intrinsic toes; source-defined constraints and collision topology |
| Neuromuscular system | All source-composed muscles and motor compartments with semantic stimulation IDs, routes/wraps, activation, fibre/tendon/aponeurosis state, entheses, fatigue, energetics, and volumetric active-muscle ownership |
| Load-bearing connective anatomy | Tendons, aponeuroses, fascia, ligaments, cartilage, menisci, intervertebral discs, joint capsules, and their rigid/deformable contact and fluid-support fields |
| Exterior and systemic anatomy | Physical skin, fat, fascia, organs, vessels, and nerves registered in the common topology/field graph, with mechanics, boundary, sensor, and validation status explicit |
| Biological sensing | Spindle, Golgi-tendon, joint, plantar/cutaneous, vestibular, visual, physiological, delay, noise, adaptation, and history programs separated from privileged simulator truth |
| Capability distribution | Equilibrium, recovery, locomotion, running, turning, manipulation, carrying, sit/stand, terrain/contact interaction, and declared physiological or intervention tasks across subject and parameter distributions |
| Personalization and evidence | Subject atlas, population priors/posteriors, identifiability, UQ, held-out physical data, competitive comparisons, and exact Apple execution evidence |

The leaf set is fixed by the content-addressed union of every stable structure,
actuator, route, tissue label, material domain, sensor, physiological
compartment, collision pair, and capability named by the pinned source
registers plus the mandatory rows above. New sources may add or supersede
leaves; they may not delete, merge away, or downgrade an existing target. Each
leaf records required representation/fidelity, owning equations, interfaces,
source/license, oracle, observable, tolerance, qualification distribution, and
evidence status. `unknown` is an honest blocked value; `not applicable` requires
a reviewed physical rationale. Until the machine-readable
`HumanPack.target-coverage.v1` materializes this exact union, architecture
convergence is blocked—the prose categories cannot be used to shrink it.

This is a frontier research target, not a clinical-validity claim. Clinical or
safety-critical claims require a declared context of use and their own risk-
informed validation; they are never inherited from model detail or visual
quality.

## Current verified baseline

| Layer | Current owning evidence | Open boundary |
| --- | --- | --- |
| Source and provenance | Pinned BodyParts3D, MyoSim, Mortensen, Rajagopal, MoBL-ARMS, and Z-Anatomy records with hashes and redistribution rules | Field ownership is not yet unified in a permanent multi-source Human representation; some sources and motions are restricted or unresolved |
| Native embodiment | `NHRIGID2` 157-body/128-DoF articulation; `NHMYO2` 416 routes, 1,815 sites, 143 wraps, joint equalities, and ten foot witnesses | Held-out source parity, high-velocity dynamics, complete topology, broad collision, and systemic physiological state remain open |
| Tendon transfer | `NHTENDON3` preserves all 832 endpoints as 630 distributed surface envelopes—628 registered BodyParts bone and two exact pinned MyoSim rib-component surfaces—and 202 exact point laws; runtime `864a65c` passes standalone Metal transfer plus 128 persistent steps, 106,496 transfers, borrowed consumption, rejection rollback, no-direct-torque identity, and bitwise replay | This is a terminal force-transfer discretization, not a deformable tendon/enthesis/cartilage continuum; the full stand remains `balanced=false` |
| Persistent dynamics | Metal owns bounded activation, current-pose routes, gravity/dynamics, support, publication, and replay | Retained stand evidence spans only 12.8 ms, has normalized residual RMS `12.5546`, and reports `balanced=false` |
| Contact | Ten source foot witnesses exercise a deterministic support path | No authoritative whole-human collision topology, full sparse articulated contact operator, distributed pressure field, or task-wide contact qualification exists |
| Deformable anatomy | Six pectoral regions drive a 326-node Matter FEM demonstration with replay and rollback | Geometry and 10% load share are assumptions; assembly crosses a host-vector/shared-buffer/separate-commit boundary; coupling is one-way and is not production tissue mechanics |
| Control | Source-muscle part coactivation and a lower-body action contract exist | No promoted assistance-free equilibrium, recovery, locomotion, manipulation, or full hierarchical Human controller exists |
| Anatomy and appearance | 184 registered BodyParts3D bone meshes, source route presentation, torso layers, and multi-angle reviews exist | Render surfaces do not own mechanics; physical skin/fat/fascia, systemic tissue mechanics, organs, vessels, and nerves are not complete |
| Validation | Source receipts, FP64/Metal comparisons, conservation checks, rollback, replay, and visual evidence are retained | No full-stack held-out subject/activity validation, blinded internal-load prediction, population UQ, or current-stack competitive benchmark exists |

The current point laws remain evidence of honest source preservation, not a
surface-coverage failure to conceal: 155 are source non-bone termini, eight are
anterior non-rib sites, 24 lack a registered correct bone surface, 11 fail
conditioning, and four are compound toe cases. Bilateral EO3 now use exact
pinned source-rib surface envelopes only after their BodyParts members reject
under unchanged gates. Visual proximity must never silently replace those
mechanics.

## Final architecture lock

The architecture below is permanent. Algorithms and discretizations may
compete inside it; ownership and scientific semantics may not drift silently.
Changing an interface requires an architecture decision record, ABI/schema
version, migration, golden-fixture update, and integrated regression evidence.

### 1. Multi-source Human representation

`HumanPack` is the locked immutable umbrella-manifest architecture. Its initial
schema must reference, rather than discard, the current `NHRIGID2`, `NHMYO2`,
`NHTENDON3`, and `NHBONES1`, plus versioned contact, tissue, sensor, material,
physiology, uncertainty, and target-coverage payloads.

Every scalar, field, topology element, and transform carries:

- a stable semantic anatomical ID and source-local ID;
- source URL/revision/hash, license, allowed use, and derivation lineage;
- units, frame, laterality, subject/population, and uncertainty;
- owning source or explicit composition rule;
- compiler/schema version and dependency hashes; and
- mechanics, presentation, calibration, validation, and qualification status.

MyoSim is the current executable source profile, not the universal schema.
Rajagopal, Mortensen, MoBL, atlas, imaging, and new subject sources enter a
field-level composition graph behind stable semantic IDs. A source bake-off can
change an owning field without forcing policy, sensor, contact, or tissue ABI
redesign. Conflicts fail closed and produce review receipts; they are never
resolved by file order.

The MyoSim adapter remains permanent-compatible, but no MyoSim field is
permanent authority merely because it runs today. Any source-owner change
updates Human, world, task, sensor, and policy compatibility fingerprints and
rejects stale compiled runs or policies even when the ABI shape is unchanged.

The action surface is therefore not globally frozen at 416. Every current
actuator remains individually addressable, while each compiled composition
publishes a fingerprinted semantic actuator table that can add cervical, hand,
foot, respiratory, or other anatomy without changing the runtime protocol.

### 2. Compiled artifact graph

```text
source records + licenses + transforms + uncertainty
                         |
                         v
                  immutable HumanPack
                         |
                         v
           Human compiler and Core lowering
                         |
                         v
 RobotPack + ScenePack + SensorPack + RealityPack
 + TaskPack + PolicyPack
 + optional TeacherPack + RunProfile
                         |
                         v
                    CompiledRun
                         |
                         v
     EvidenceRecord + replay + performance trace
```

`HumanPack` is a source/embodiment aggregate lowered into the existing Numi
pack owners; it is not a parallel Human-only runtime path. All hot-loop indices
and layouts are compiled. Python owns source acquisition, registration,
uncertainty records, and offline compilation; Core C++ owns validation,
lowering, FP64 oracles, ABI checks, and deterministic index plans; Metal owns
persistent simulation state and live execution; Swift owns bounded submission,
waits, reset, and artifact lifecycle; MLX owns batch learning and publishes
immutable policy or parameter packs.

### 3. One accepted-step transaction

One Human control step is one candidate transaction on one caller-owned or
borrowed Metal command-buffer timeline. The checkpoint covers:

- rigid `q/v`, constraints, and articulated caches;
- excitation, activation, fibre, tendon, fatigue, and energetic state;
- contact manifolds, pressure patches, impulses, warm starts, and events;
- deformable position/velocity, pressure, internal variables, active fibre,
  fluid/thermal/electrical fields defined by the permanent schema, per-run
  activation masks, and topology state;
- sensor histories, estimator/controller hidden state, task state, and RNG.

Every consumer may encode only. It may not commit, wait, retain, replace, or
read back the borrowed objects. Acceptance publishes all physical, sensor,
controller, task, and RNG state atomically; rejection restores that accepted
state. An append-only attempt counter, typed failure/rejection ledger, and
diagnostic telemetry advance outside rollback so failure evidence is never
erased. No subsystem may publish a locally successful partial step.

The causal program is explicit: reset/checkpoint → pre-step sensing and history
→ estimator/inference/delay → semantic stimulation → coupled physical candidate
→ post-step sensing → acceptance or rollback → publication. Every stage is
transactional, but causal sensor/controller transitions are not falsely treated
as nonlinear physical unknowns.

The implementation may be partitioned, but its mathematical authority is one
generalized coupled acceptance system: residual equations plus equality,
complementarity, cone, variational-inequality, impact, and discrete topology-
event conditions over the physical candidate variables and the accepted
actuation. Articulated, muscle/tendon, continuum, contact-history, and physical
physiology variables exchange energy-conjugate, equal-and-opposite quantities
and converge under one acceptance test; partitioned components are not
independently authoritative simulators.

### 4. Single force, residual, and power authority ledger

Every accepted step has a formulation-neutral ledger covering inertia and
bias, gravity, externally applied loads, line/reduced/continuum active muscle,
passive tissue and damping, pressure/field forces, equalities and joint limits,
rigid and primal/barrier contact, and all interface reactions. Each contribution
declares whether it enters as generalized force, internal stress/residual,
constraint multiplier, impulse, barrier/variational term, or state work.

When a distributed tissue region replaces a source-route `J^T` share, that
share is removed from the line-actuator owner; tissue reaction is never added
on top. A canonical fidelity-selection map assigns every anatomical load region
to exactly one of line route, reduced continuum, or full continuum for each
compiled run, with conservative activation/fibre/tendon/passive-state transfer
between representations. Direct joint torque, root wrench, pose drive, and
hidden stabilization are diagnostics only and exactly zero in promoted
behavior.

Every owner reports source, frame, application point/distribution, resultant,
moment, residual share, impulse where applicable, power/work, stored/dissipated
energy, and replacement/fidelity mask. Residual, force, momentum, work, and
energy closure are transaction acceptance criteria, not post-hoc plots.

### 5. Authoritative contact and collision

The final contact contract requires full articulated coupling, including every
off-diagonal interaction among contacts sharing a body or articulation. The
initial authoritative Metal candidate uses the sparse Delassus operator
`W = J M^-1 J^T`: MetalWorld streams articulated response columns into a
Metal-suitable layout on the borrowed timeline and device kernels consume the
full operator without CPU readback, a second queue, independent commit, or
wait. Matrix-free KKT, SAP, or other backends may compete only if they preserve
the same coupling, friction, transaction, and certificate semantics and win a
reproduced comparison.

The common contact semantic interface includes:

- deterministic broad phase, narrow phase, exclusions, and continuous
  collision detection where tunnelling changes physical outcomes;
- fingerprinted material-pair laws for dry circular-Coulomb/maximum-dissipation,
  torsional/rolling, compliant, lubricated/wet, adhesive, barrier, cartilage,
  skin, and other admitted contact, with impact, sliding, separation, and
  history/warm-start semantics;
- anatomical pressure patches and centre-of-pressure outputs rather than point
  witnesses as the production surface;
- rigid/deformable and deformable/deformable contact through the same force and
  transaction ledger; and
- solver-appropriate certificates selected from symmetry/PSD, natural
  residual, cone, complementarity, barrier feasibility, penetration, energy,
  force, and moment checks.

Rigid dry contact may realize the interface through Delassus/cone or equivalent
mixed KKT methods; deformable and mixed contact may use Matter's primal IPC or
another qualified variational backend. They share geometry IDs, friction and
material-law identities, impact semantics, force/power accounting, failure
types, transaction, and cross-formulation oracle cases. Cone/complementarity
certificates are not blindly imposed on a primal barrier, lubricated, or
adhesive formulation.

Point witnesses remain an exact diagnostic path. No controller eligible for
promotion is tuned around them as its authoritative product contact.

### 6. Multiresolution whole-human continuum

The final tissue system is one whole-human continuum architecture with active
sparse solve domains, not a collection of unrelated demonstrations. All
anatomy is registered in a common topology/field graph. Resolution and physics
activate by task, error estimator, contact, load path, and evidence need while
the interface, force law, state transaction, and conservation rules remain
unchanged.

Matter is the permanent coupled-continuum owner; its present algorithm is not
automatically the promoted backend. FEM, MPM, rods, shells, embedded/mortar or
Nitsche interfaces, domain decomposition, reduced bases, and learned reductions
may compete inside that owner; none may become a downstream sidecar with a
different transaction or force authority.

The permanent continuum semantics include:

- nonlinear mixed displacement-pressure formulations for near-incompressible
  tissue;
- anisotropic hyperelastic and viscoelastic solids, shells, rods, fibres,
  ligaments, tendons, cartilage, menisci, fascia, skin, and fat;
- active-fibre stress and consistent coupling to source activation;
- biphasic/poroelastic state where fluid support is physically material;
- thermomechanical and electrophysiological fields supported by the permanent
  schema with explicit per-run activation masks in the same transaction;
- nonlinear acceptance, determinant protection, constrained pressure and
  interface coupling, negative-curvature/failure handling, and convergence/
  error certificates; and
- deterministic, capacity-preflighted topology change for use cases that
  require cutting, puncture, separation, or deactivation.

Matrix-free Newton/Krylov block solves with pressure Schur complements,
bounded preconditioning, and merit backtracking are the initial implicit-FEM
candidate—not a ban on MPM, explicit, variational, direct, or alternative
implicit backends that satisfy the permanent semantics and benchmark better.

Every reduced-order or learned tissue model derives from a fingerprinted
full-order model and carries error certificates over frozen held-out, boundary,
out-of-distribution, and adversarial state/load envelopes. Every promoted
behavior is rerun on authoritative full-order physics over its qualification
distribution. Capacity overflow, estimator failure, or loss of validity rejects
the candidate and emits a typed growth/refinement request; it never silently
truncates the anatomy or reallocates inside a borrowed submission.

Growth occurs only at a completed-step boundary: allocate or recompile, update
the allocation fingerprint, conservatively prolong/restrict accepted state and
all history fields, validate interface traction/velocity and stored energy, then
deterministically retry. The failed attempt remains in the append-only ledger.

Adaptive regional refinement is the production execution strategy. It may
limit active high-resolution coverage in a run, but it preserves the continuous
constitutive and interface semantics, conservative prolongation/restriction,
traction/velocity continuity, history transfer, error estimates, and replay
across discretization changes. It may not create a one-way side simulation,
use placeholder load fractions, or bypass global two-way coupling. The pectoral
demonstration is evidence for inputs and rollback only; it does not define this
API.

### 7. Neural control, learning, and identification

The permanent control contract exposes the semantic stimulation table and a
physiology-rich observation schema. Controllers compose:

- fast proprioceptive/reflex and safety responses;
- device-resident state estimation and contact phase;
- reusable posture, recovery, locomotion, manipulation, respiration, and
  interaction skills; and
- task, intent, adaptation, and NumiBrain channels with explicit validity state.

Synergies and latent actions are replaceable encoders onto the full stimulation
surface, never a loss of actuator identity. Reflex, MPC/optimal-control,
imitation, reinforcement, world-model, and hybrid methods compete behind the
same ABI. Metal owns live inference. MLX owns batch learning and publishes
fingerprinted policies; no training framework owns authoritative physics.

Residuals, tangents, parameter sensitivities, and JVP/VJP boundaries are part
of the compiled offline-identification contract. The hard-contact production
step need not pretend to be globally smooth: differentiable subsystems,
smoothed offline oracles, adjoints, ensembles, and derivative-free methods can
all propose parameter or policy packs, which are promoted only on the
authoritative runtime.

Forward tangents and discrete adjoints are checked against directional
derivatives. Implicit differentiation acts on the converged residual; it may
not silently backpropagate through an arbitrary iteration count and call that a
physical gradient.

### 8. Sensing, physical exterior, and presentation

Mechanics geometry and presentation geometry have separate owners and hashes.
The physical exterior supports skin, fat, fascia, contact pressure, sliding,
volume, and self-contact under explicit activation masks. The presentation layer may add hair,
materials, subsurface appearance, and render-only detail, but never acquires
mechanical authority by proximity.

Metal produces muscle-spindle, Golgi-tendon, joint-receptor, plantar/cutaneous,
vestibular, visual, activation/fibre/tendon, contact-pressure, tactile, RGB-D,
segmentation, motion, event, physiological, and task features without per-step
host restaging. Sensor delays, noise, adaptation, histories, and failure modes
are compiled state; privileged simulator truth is a separate diagnostic schema.
Deployable policy sensors have authored cadence, timestamp, exposure, latency,
held-frame, validity, reset, and typed failure semantics in `SensorPack`; they
may not silently drop. Only the optional presentation inspector may drop frames
rather than block physics, using bounded GPU buffers. Actual runtime frames from
multiple angles are part of qualification, not a substitute for it.

### 9. Personalization, uncertainty, and evidence

Subject definitions are immutable, fingerprinted distributions—not a nominal
body called universal. A versioned `SubjectPack` separates prior/posterior
geometry and parameters from mutable online-estimator state. Scaling recomputes
mass, inertia, joint programs, route geometry, strength, contact, material
fields, and solver constraints under explicit physical constraints and
uncertainty. Calibration and validation data are disjoint by subject and
activity. Identifiability, sensitivity, posterior uncertainty, and out-of-
distribution status accompany every personalized result.

Kinematics, GRF/pressure, joint moments, muscle activation/EMG, energetics,
internal loads, tissue stress/strain, deformation, and visual accuracy remain
separate evidence categories. Success in one never silently qualifies another.

## Non-negotiable production laws

1. There is one Apple-native authoritative physics path.
2. There is one atomic accepted-step transaction.
3. Every force and energy contribution has exactly one owner.
4. Tissue coupling is conservative and two-way in any tissue-mechanics claim.
5. Stable semantic anatomical IDs survive source and topology composition.
6. All current 416 actuators remain visible; added anatomy extends rather than
   redesigns the action protocol.
7. No render surface, proximity weight, or tube-shaped visual owns mechanics.
8. No promoted behavior uses a hidden root wrench, pose drive, or direct torque.
9. A surrogate, teacher, CUDA stack, or external simulator may propose or
   compare; it never publishes authoritative Human state.
10. A provisional material/contact parameter is quarantined from promoted
    controller tuning and physical claims.
11. No command-buffer consumer commits, waits, retains, replaces, or performs
    per-step CPU readback.
12. Reduced precision, fusion, Metal 4 features, and layout changes are adopted
    only after same-workload counters and physical outcomes show no regression.
13. Build, liveness, image, reward, and diagnostic force transfer are never
    promoted into physical evidence of a different category.
14. High-throughput training may use only an authoritative model or a
    fingerprinted reduction with measured error over frozen held-out, boundary,
    OOD, and adversarial envelopes; all promotion outcomes rerun on authority.
15. Counter-based randomness and replay results are invariant to batch/chunk
    size, environment ordering, active-set scheduling, kernel fusion, and other
    semantically irrelevant dispatch changes.

## Permanent parallel workstreams

Every workstream below is active immediately. A dependency can block a claim
or a merge into the authoritative stack; it cannot justify leaving unrelated
research, data, compiler, geometry, controller, or validation work idle.
Acceptance requires every lane to have a named owner, active/blocked status,
exact dependency, current artifact, next evidence-producing action, and
integration target in the machine-readable execution registry. Until that
registry exists, this coordination gate is explicitly blocked. When a lane is
blocked, its available capacity moves to the highest-fan-out upstream dependency
rather than becoming idle.

### A. Source graph, anatomy, provenance, and licensing

- **End state:** one field-level Human composition graph covering every fixed
  target-coverage cell, with semantic IDs, uncertainty, and license-clean build
  and training profiles.
- **Current proof:** pinned source records, extensive import/registration
  receipts, 184 bone meshes, 157 bodies, 416 routes, and 832 preserved termini.
- **Active frontier:** resolve target-catalog ownership ambiguity; encode the
  `HumanPack` schema; register complete feet, hands, neck/hyoid, craniofacial,
  joint-tissue, skin, organ, vessel, and nerve topology; quarantine restricted
  sources by derivation lineage.
- **Convergence proof:** round-trip source parity, stable IDs across source
  profiles, transform/units tests, conflict rejection, license policy tests,
  and reviewed multi-angle registration.

### B. Articulation, musculotendon, and physiology

- **End state:** full source-composed articulated dynamics and musculotendon
  state over the declared Human topology, including wraps, constraints, passive
  state, fatigue, energetics, and extensible physiology fields.
- **Current proof:** `NHRIGID2`/`NHMYO2`, FP64 references, Metal route evaluation,
  joint equalities, persistent activation/fibre/tendon state, and bounded replay.
- **Active frontier:** freeze multi-pose/multi-velocity/high-load corpora and
  tolerances; complete source wrap/constraint matrices; requalify the latest
  `NHTENDON3`; expose tangents and parameter sensitivities; modularize only
  behind byte-exact fixtures.
- **Convergence proof:** source-oracle and FP64 parity, timestep refinement,
  conservation, constraint residuals, failure envelopes, replay, and no-direct-
  torque identity over held-out states.

### C. Anatomical collision and contact

- **End state:** collision topology for every fixed target-coverage cell,
  distributed anatomical pressure fields, self-contact, compiled swept-risk
  CCD, and full articulated/deformable contact authority.
- **Current proof:** deterministic ten-witness support and exact contact
  diagnostics.
- **Active frontier:** complete foot and whole-human contact registrations;
  compile material/pair/exclusion topology; implement streamed articulated
  response columns and full `W`; qualify circular friction and pressure patches
  on rigid and deformable anatomy.
- **Convergence proof:** analytical coupled-contact oracles, static/sliding/
  rolling/impact/separation cases, pressure/CoP, complementarity/cone/energy
  certificates, CCD stress cases, timestep refinement, and replay.

### D. Multiresolution deformable and multiphysics anatomy

- **End state:** a whole-human field graph with conservative two-way active
  domains for muscle, tendon, fascia, ligament, cartilage, meniscus, disc,
  skin/fat, organ, vessel, and nerve mechanics at evidence-appropriate fidelity.
- **Current proof:** pectoral meshes, Matter FEM execution, load input, replay,
  rollback, and retained demonstration boundaries.
- **Active frontier:** implement the final device-resident coupling ABI; replace
  rather than add the corresponding `J^T` share; build manufactured and FEBio
  comparison cases; qualify fascia, Achilles/plantar, knee, spine, skin, and
  organ substrate cases concurrently on the shared schema. These cases qualify
  the shared substrate only; continuum completion requires every fixed target-
  coverage cell to pass.
- **Convergence proof:** constitutive and patch tests, mesh/timestep/nonlinear
  convergence, minimum Jacobian, pressure constraint, force/work/energy closure,
  reaction consistency, rollback, held-out deformation/load, and measured cost.

### E. Control, learning, state estimation, and NumiBrain coupling

- **End state:** a transparent, robust, adaptive hierarchy controlling the full
  semantic stimulation surface across equilibrium, recovery, locomotion,
  manipulation, interaction, and physiological tasks.
- **Current proof:** part coactivation, source-muscle bounds, and a preliminary
  lower-body action contract.
- **Active frontier:** freeze the extensible observation/action/task schemas;
  compile spindle, Golgi-tendon, joint, vestibular, visual, tactile, delay, and
  noise programs; run reflex, optimal-control, imitation, RL, synergy, and
  hybrid bake-offs; establish device inference and MLX publication; define the
  NumiBrain sensor/intent boundary; train candidates against fingerprinted
  authoritative or certified-reduction physics.
- **Convergence proof:** assistance-free outcomes over preregistered task and
  perturbation distributions, authoritative contact/tissue replay, physiological
  metrics, policy fingerprints, generalization, ablations, and failure analysis.

### F. Data, personalization, system identification, and UQ

- **End state:** traceable multi-subject data, parameter distributions, offline
  differentiable/ensemble identification, and uncertainty-aware predictions.
- **Current proof:** source locks and component-level numerical comparisons.
- **Active frontier:** build the dataset/license/split registry; freeze held-out
  subjects and activities before tuning; ingest motion, GRF, pressure, EMG,
  internal-load, imaging, and material data; implement sensitivity,
  identifiability, and posterior checks.
- **Convergence proof:** calibration/validation separation, recovered synthetic
  parameters, held-out subjects/tasks, uncertainty calibration, robustness to
  parameter distributions, and blinded internal-load predictions.

### G. Sensing, physical exterior, interaction, and rendering

- **End state:** mechanics-consistent skin/contact/sensors plus a presentation
  layer that passes a fixed `VisualEnvelope` from exact live Human state. Its
  rows cover every target-coverage leaf; source/subject surface and landmark
  error, mechanics-to-render displacement, silhouette/depth/normal/material
  error, self-contact and occlusion, temporal continuity/replay, authored sensor
  cadence, multi-angle blinded review, and device cost each have a source,
  metric, tolerance, and qualification distribution. Any `unknown` tolerance
  blocks a whole-Human or visual-SOTA claim.
- **Current proof:** registered bone and soft-tissue visuals, source routes,
  torso layers, and multi-angle capture.
- **Active frontier:** compile physical skin/fat/fascia surfaces and tactile/
  pressure sensors; define RGB-D/event/proprioceptive schemas; make rendering a
  nonblocking device sidecar; populate every `VisualEnvelope` cell and improve
  it without changing mechanics ownership.
- **Convergence proof:** sensor oracle tests, zero-readback live execution,
  mechanics/render registration, self/contact behavior, temporal stability,
  original-resolution multi-angle runtime review, and GPU cost.

### H. Apple GPU runtime, scaling, and performance

- **End state:** collision, coupled physics, sensing, inference, randomness,
  rollback, and rendering remain device-resident on one transactional timeline
  and meet a fixed, versioned `PerformanceEnvelope` matrix. Its mandatory rows
  are single-Human interactive authority, complete full-fidelity qualification,
  population learning, policy/sensor inference, and nonblocking presentation.
  Every row fixes Apple
  device/SKU and OS, Human/task/tissue composition, fidelity, timestep/horizon,
  environment count, latency/throughput target, retained/peak memory ceiling,
  thermal window, failure budget, replay mode, and physical-equivalence gate.
  Interactive authority targets real-time factor at least `1.0` with p99 step
  latency no greater than its authored control period; every row forbids swap
  and untyped failure and must be Pareto-nondominated by reproduced same-device
  comparators in accepted physical steps/s, steps/J, and memory. Any unresolved
  cell blocks architecture convergence rather than weakening the target.
- **Current proof:** persistent Metal state, same-command-buffer tendon transfer,
  replay, rollback, and native Apple tooling.
- **Active frontier:** encode active sparse lists and consumer-oriented layouts;
  use environment-major physics, feature-major inference, and AoSoA/SIMD32 where
  measured; define static/dynamic/transient heaps and capacity classes; profile
  exact stack baselines; populate every fixed `PerformanceEnvelope` row;
  evaluate schedule-driven ABA, sparse response columns, fusion, precision, and
  Metal 4 paths.
- **Convergence proof:** same-revision/device/workload traces and counters,
  throughput, latency distribution, retained/peak memory, swap and thermals,
  failed steps, replay, and unchanged physical outcomes.

### I. Compiler, evidence system, validation, and claims

- **End state:** every compiled capability has an immutable evidence record and
  the repository continuously distinguishes source, implementation, numerics,
  transaction, physical outcome, generalization, performance, and visual proof.
  An append-only run/selection ledger retains failures, rejected attempts, and
  every physically valid candidate; promotion creates a new selection record
  and never overwrites competing evidence.
- **Current proof:** extensive reports, hashes, manifests, media transcripts,
  gap ledger, and 88 importer/compiler tests.
- **Active frontier:** replace the stale runtime pin with a generated live
  contract; isolate and type the robot-catalog `Killed: 9` failure; bound both
  catalog and Git audit inspection for iCloud checkouts; create the machine-
  readable capability/evidence registry; preserve golden artifacts; split the
  large compiler and test file behind ABI fixtures; implement the competitive
  decision register and integrated qualification runner.
- **Convergence proof:** exact lineage resolution, fail-closed stale evidence,
  ABI migrations, deterministic replay, injected failures, full-stack
  non-regression, and claims generated only from qualifying records.

### J. Systemic physiology and neural conduction

- **End state:** circulation/hemodynamics and perfusion, cardiac drive,
  respiratory mechanics and gas exchange, metabolism and energetic substrates,
  thermoregulation, fluid balance, electrophysiology, and peripheral neural
  conduction are registered to the same anatomy and evolve through explicit
  multi-rate fields/compartments on the accepted-step timeline.
- **Current proof:** local activation, fibre/tendon, fatigue/energetic concepts,
  and transaction fields provide partial interfaces only; no systemic owner is
  qualified today.
- **Active frontier:** freeze compartment/network/field schemas and units;
  identify source models and subject parameters; couple perfusion to active
  tissue, ventilation to thoracic mechanics, metabolic/thermal state to muscle,
  and conduction/delay to biological sensors and stimulation; define stable
  subcycling and rollback semantics.
- **Convergence proof:** mass, species, charge, momentum, and energy balance;
  manufactured and independent-oracle cases; multi-rate refinement; stable
  reaction coupling; held-out pressure/flow/gas/temperature/conduction
  observables; uncertainty; rollback; and exact-stack Apple cost.

## Critical dependency graph

```text
source graph + license policy + semantic IDs
          |              |               |
          v              v               v
 articulation/muscle   contact       continuum/material
          |              |               |
          +--------------+---------------+
                         |
                  systemic physiology
                         v
          one authoritative StepTransaction
                 |                 |
                 v                 v
        sensors/state estimate   controller/policy
                 +--------+--------+
                          v
               Human capability outcomes
                          |
                          v
      held-out validation + SOTA comparison + Apple trace
```

Dependencies govern acceptance:

- source IDs, frames, ownership, and license policy are required before an
  artifact can enter the authoritative stack;
- held-out splits and acceptance metrics are required before calibration or
  controller tuning can qualify a result;
- full force ownership, two-way reaction, and energy closure are required
  before a tissue-mechanics claim;
- authoritative contact is required before equilibrium, locomotion,
  manipulation, or recovery claims, but controller experiments run in parallel;
- exact integrated stack execution is required before performance claims;
- a reproduced external comparator and declared metric are required before
  “state of the art” is used for that capability; and
- a declared context of use, risk analysis, and corresponding validation are
  required before a clinical or safety-critical claim.

## Overlapping convergence states

These are evidence states, not periods. Workstreams can occupy different
states simultaneously and all unblocked work continues.

| Convergence state | Required evidence |
| --- | --- |
| Architecture convergence | Final owners, schemas, source composition, semantic IDs, ABIs, transaction, force ledger, failure semantics, license policy, and evidence schema are executable and versioned |
| Subsystem convergence | Each permanent solver passes analytical/FP64/independent-oracle, convergence, replay, rollback, conservation, and failure-envelope tests |
| Coupled-physics convergence | Articulation, muscles, contact, tissue reaction, physiology, sensing, and control execute in one accepted-step transaction with no unowned or duplicated force |
| Human-capability convergence | Equilibrium, recovery, locomotion, manipulation, interaction, and personalization pass preregistered outcome distributions with assistance and shortcuts absent |
| Frontier qualification | Held-out physical validation, uncertainty, reproduced competitive comparisons, exact-stack Apple counters/memory, runtime visuals, and integrated non-regression pass together |

## Maximum-concurrency execution frontier

The execution queue is continuously replenished. Each lane remains owned; when
capacity opens or a lane blocks, select the ready upstream item with the
greatest downstream fan-out and scientific risk.

| Lane | Unblocked frontier now | Evidence-producing result |
| --- | --- | --- |
| Live truth | Generate the Human/runtime/source/artifact registry; remove the stale tracked runtime pin; isolate the robot-catalog `Killed: 9`; type and bound catalog/Git audit failure | One exact stack resolves from a single immutable record; readiness fails quickly and explicitly; old evidence becomes mechanically historical |
| Permanent contracts | Encode `HumanPack`, semantic action/observation tables, force-owner masks, tissue/contact/sensor schemas, transaction checkpoint, and failure ABI | Golden schema/ABI fixtures plus migration and rejection tests |
| Current-stack qualification | Register and freeze the qualified 630/202 `NHTENDON3`, Human/runtime revisions, payload hashes, 128-step persistent transaction, rejection probe, replay, and diagnostic visuals; rerun after any dependency change | Honest end-to-end baseline resolves from the capability/evidence registry rather than prose |
| Mechanics | Freeze multi-pose/multi-velocity/high-load corpus; complete wrap, constraint, passive, tangent, and sensitivity coverage | FP64/source parity and refinement envelope over the declared model |
| Contact | Complete foot and whole-human collision registration while implementing full sparse `W`, circular friction, CCD, and pressure patches | Coupled-contact certificates and pressure/CoP evidence |
| Continuum | Implement final same-buffer two-way coupling and competing nonlinear backends; qualify substrate cases and then every fixed target-coverage cell | Force/work/energy closure, held-out deformation, complete coverage, and no double counting |
| Control | Freeze extensible schemas; run controller bake-offs and physics-fingerprinted learning continuously | Assistance-free candidate policies evaluated only on authoritative physics |
| Data/UQ | Lock dataset lineage and held-outs; build system-ID, sensitivity, uncertainty, and blinded-evaluation harnesses | Claim-specific validation envelopes and parameter distributions |
| Systemic physiology | Freeze circulation, respiration, metabolic/thermal, electrophysiology, and neural-conduction schemas; implement conservative multi-rate oracle cases | Registered field/compartment state with mass/species/charge/energy balance and rollback |
| Apple runtime | Capture exact baseline; implement active sparse schedules, heap lifetimes, response columns, and measured layout/precision candidates | Same-workload throughput/memory/counter improvements with physical equivalence |
| Sensing/visual | Compile mechanics-owned exterior/sensors and a nonblocking render sidecar | Live sensor parity and multi-angle frames from the same accepted state |
| Maintainability | Split `model.py`, `cli.py`, and concentrated tests only behind golden outputs; define artifact retention classes | Smaller ownership units with byte-exact artifacts and no evidence loss |

No lane waits for “standing” or “walking” to be declared before beginning. A
behavior candidate simply cannot be promoted until its required authoritative
mechanics and held-out evidence converge.

## State-of-the-art decision register

“State of the art” is a measured competitive requirement. Maintain a generated
register with these fields:

| Required field | Meaning |
| --- | --- |
| Domain and capability | Exact claim being compared, never “the Human” in aggregate |
| Primary source | Paper/repository/version/date and exact method configuration |
| Benchmark and split | Inputs, subjects, tasks, hardware, license, train/calibration/held-out split |
| Metric and acceptance | Units, direction, uncertainty, preregistered tolerance or target |
| Reproduction status | Unread, inspected, reproduced externally, ported, or reproduced in Numi |
| Measured Numi delta | Same metric and declared comparability limitations |
| Decision | Adopt, adapt, retain as oracle, reject, or unresolved |
| Permanent owner | Numi interface and artifact that owns any adopted method |
| Evidence and retest trigger | Artifact path plus source/runtime/benchmark changes that invalidate it |

Initial candidate register seeds are all `unreproduced`. They are research
assignments, not adopted methods or SOTA claims, until every required register
field above is populated with an exact source revision, artifact, benchmark,
Numi delta, and retest trigger.

| Domain | Candidate frontier and assigned role | Maturity | Boundary before decision |
| --- | --- | --- | --- |
| Full-body muscle embodiment | [MyoFullBody models](https://github.com/amathislab/musclemimic_models) and the 2026 [MuscleMimic preprint](https://arxiv.org/abs/2603.25544) as source/method comparators; [MS-Emulator](https://arxiv.org/abs/2603.29332) as a control-solution diversity and actuator-scale paper comparator | Unreproduced; MS-Emulator is paper-only and reports code as forthcoming | Kinematic agreement does not establish physiological activation; similar external motion can hide different internal solutions. JAX applies to the MuscleMimic training path; MS-Emulator reports MJWarp/RTX hardware. Code, data, body, and checkpoint licenses remain separate |
| Optimal control and identification | [OpenSim Moco](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008493) as direct-collocation oracle; [OpenSimAD](https://github.com/antoinefalisse/opensimAD) and [Nimble](https://nimblephysics.org/docs/intro.html) as derivative/identification comparators | Unreproduced | Moco is sensitive to discontinuities; OpenSimAD omits some constraints and SimmSplines; Nimble is beta and cautions against rigid-contact gradients for complex trajectory optimization. Every proposed pack is rerun on authoritative Numi physics |
| Contact | [MuJoCo stable computation docs](https://mujoco.readthedocs.io/en/stable/computation/index.html) for convex soft contact; [Drake SAP](https://drake.mit.edu/doxygen_cxx/group__mbp__discrete.html) and [hydroelastic pressure](https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html); [Chrono NSC](https://api.projectchrono.org/collisions.html) for nonsmooth complementarity; [IPC](https://ipc-sim.github.io/) for robust deformable collision | Unreproduced | MuJoCo does not promise exact maximum dissipation; hydroelastic pressure is not deformation and numerical modulus is not tissue modulus; robust collision alone proves neither biological fidelity nor Apple performance |
| Deformable tissue and GPU layout | [FEBio](https://pmc.ncbi.nlm.nih.gov/articles/PMC3705975/) and [SOFA](https://github.com/sofa-framework/sofa) as constitutive/nonlinear/multiphysics oracles; [NVIDIA Warp FEM](https://nvidia.github.io/warp/v1.15/api_reference/warp_fem.html) as GPU layout and active-domain comparator | Unreproduced | Agreement requires matched mesh, laws, loads, convergence, and held-out experiments. Warp is a CUDA comparator, not a production dependency or evidence of tissue fidelity |
| Neuromuscular control and sensing | [MuscleMimic](https://github.com/amathislab/musclemimic), [Kinesis](https://github.com/amathislab/Kinesis), [DEP-RL](https://arxiv.org/abs/2206.00484), [DynSyn](https://proceedings.mlr.press/v235/he24o.html), peer-reviewed [Explore-to-Learn](https://ojs.aaai.org/index.php/AAAI/article/view/39876), [LocoMuJoCo](https://github.com/robfiras/loco-mujoco), and [SMS-Human](https://arxiv.org/abs/2506.00071) as method/evaluation candidates | Unreproduced; SMS-Human is paper-only unless a code artifact is registered | No benchmark proves a universal controller; learned synergies may not erase source actuators, privileged truth may not replace biological sensing, and imitation may not replace physiological evaluation |
| Active continuum muscle | Peer-reviewed [active-stress/active-strain comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC12017808/) and the [FEBio active-muscle model](https://febiosoftware.github.io/febio-feature-manual/features/solid_material_muscle_material/) as constitutive candidates | Unreproduced | Published cases are scoped rather than whole-human qualification. Volumetric muscle replaces a line-actuator share only with identified parameters, convergence, conservation, and held-out deformation/force evidence |
| Systemic physiology | [CellML](https://www.cellml.org/) and the [Physiome Model Repository](https://models.physiomeproject.org/) as equation/source records; [SimVascular](https://github.com/SimVascular/SimVascular) as a cardiovascular flow oracle candidate | Unreproduced | Component equations do not establish a coupled whole-human physiology; units, compartments, boundary conditions, subject parameters, multi-rate conservation, licensing, and held-out observables must match before adoption |
| Data and validation | [MM-EvalKit](https://github.com/amathislab/mm-evalkit) as an implementation/metric template; [AddBiomechanics](https://pmc.ncbi.nlm.nih.gov/articles/PMC11948690/), [OpenCap validation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462), and the [blinded knee-load challenge](https://pmc.ncbi.nlm.nih.gov/articles/PMC4067494/) as data/validation candidates | Unreproduced in the Numi stack | MM-EvalKit is not an independently validated standard. Motion/GRF data do not directly measure every inferred joint/internal load, and calibration data never count as validation |
| Credibility and UQ | [ASME V&V/UQ](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty) and [FDA CM&S guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/assessing-credibility-computational-modeling-and-simulation-medical-device-submissions) as evidence-structure references | Inspected conceptually; not a Numi qualification | FDA provides official, nonbinding recommendations for medical-device submissions and is only a credibility-structure reference here unless that context of use is declared; a checklist confers neither validation nor approval |
| Apple and throughput execution | [Metal machine-learning passes](https://developer.apple.com/documentation/metal/machine-learning-passes), [Apple GPU counters](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics), and [MLX](https://github.com/ml-explore/mlx) for the native path; [MJWarp](https://github.com/google-deepmind/mujoco_warp) as NVIDIA throughput comparator | Current Numi Metal execution is partly proven; Metal 4 ML passes, MLX-to-policy export/parity, and MJWarp comparison are unreproduced | Metal 4 ML passes consume packaged Core ML networks, so MLX-trained policy export and numerical parity require evidence; unified memory is not proof of zero copies; MJWarp is not production authority |

No external stack becomes the Numi Human runtime. It is a source, comparator,
teacher, or independent oracle with an explicit license and evidence role.

## License-clean foundation

The permanent artifact graph supports the same ABI and solver semantics across
separately fingerprinted content profiles; it does not assume that different
source sets are mechanically identical:

- **free foundation:** compatible MyoSim/MyoFullBody source records, BodyParts3D
  with attribution, production-cleared measurements and motions, verified
  AddBiomechanics records, and policies with complete permissible lineage;
- **restricted research overlays:** MoBL-ARMS non-commercial material,
  AMASS/SMPL-derived motions, restricted retargeted datasets or checkpoints,
  and Z-Anatomy share-alike derivatives, with policy and publication rules
  enforced from their exact lineage;
- **quarantine/reference only:** any atlas, model, motion, body, checkpoint, or
  derivative with unresolved rights. These may be fingerprinted for review but
  cannot compile into a runnable or distributable profile.

The code license, model license, motion license, body-model license, checkpoint
license, and derived-policy license are separate fields. A permissive repository
license never launders restricted training data. This is an engineering policy,
not legal advice.

## Capability and evidence gates

Every capability and integrated composition carries all applicable rows:

| Evidence class | Required proof |
| --- | --- |
| Provenance and license | Exact source revision/hash, rights, transforms, derivation, uncertainty, Human/runtime/artifact fingerprints |
| Permanent implementation | Lowest owning live code path and final interface; no diagnostic, visual, one-way, or alternate host path standing in |
| Numerics | Analytical/manufactured/FP64/independent-oracle comparison, residuals, constraint errors, refinement, and failure envelope |
| Transaction and conservation | Acceptance, injected rejection, physical/controller rollback, advancing append-only failure ledger, completed-step/attempt counts, deterministic replay, residual/force/momentum/work/energy closure |
| Physical outcome | Claim-specific support, GRF/pressure, movement, recovery, activation, internal load, stress/strain, deformation, or physiology |
| Generalization and UQ | Frozen held-out subjects/tasks/materials/perturbations, sensitivity, identifiability, calibrated uncertainty, OOD status |
| Competitive frontier | Reproduced comparator on a declared benchmark, measured Numi delta, caveats, and retest trigger |
| Apple performance | Exact revision/artifact/device/workload, trace/counters, thermal state, latency/throughput, retained/peak memory, failures, physical equivalence |
| Visual/anatomical | Actual accepted-state runtime frames from multiple angles at original resolution, with mechanics and presentation ownership visible |
| Integrated non-regression | Exact compiled composition passes all dependent subsystem and capability gates together |

Failed runs and physically valid non-selected candidates are retained as
addressable evidence. Selection is an immutable decision record over those run
IDs; it never rewrites, deletes, or relabels the underlying outcomes.

## Main risks and controls

| Risk | Control |
| --- | --- |
| Ambition becomes prose rather than physics | Only permanent owning code plus executable evidence advances capability status |
| Maximum concurrency creates incompatible systems | Final schemas, semantic IDs, transaction, force ledger, golden ABIs, and continuous coupled integration |
| An early source model hardens into a dead end | Field-level multi-source composition behind permanent semantic interfaces |
| Limited tissue coverage becomes a permanent island model | One global continuum/coupling architecture; coverage is data, not a new solver path |
| Contact is tuned to make a policy succeed | Calibration protocol and authoritative contact; teacher physics cannot qualify behavior |
| Tissue reaction double counts muscle force | Replacement masks plus force/work/energy closure in the acceptance transaction |
| Restricted data contaminates a product policy | Per-artifact lineage, fail-closed profiles, and mechanically enforced quarantine |
| Refactoring or ABI drift destroys evidence | Golden bytes, migrations, decoder fixtures, replay, and exact dependency hashes |
| Unified memory hides duplication or stalls | Explicit heap lifetimes, capacity classes, counters, traces, and retained/peak accounting |
| Visual quality outruns mechanics | Separate geometry owners and evidence categories; multi-angle live inspection plus physical tests |
| “SOTA” becomes unmeasured branding | Generated decision register, reproduced comparator, declared metric, measured delta, and retest trigger |

## Definition of accepted progress

A capability or integrated composition is accepted only when:

1. it uses the permanent owner, semantic schema, ABI, transaction, and force
   authority defined here;
2. the exact compiled artifact executes through the intended Apple-native path
   with complete provenance and license lineage;
3. numerical, failure, rollback, replay, conservation, and dependent subsystem
   evidence pass;
4. its claimed physical outcome passes preregistered held-out and uncertainty
   gates without hidden assistance or alternate authority;
5. its competitive claim, if any, includes a reproduced comparator and measured
   delta; and
6. performance and visual claims come from the exact integrated stack.

Anything else remains a valuable source, oracle, compiler feature, diagnostic,
experiment, or candidate. It is retained, labeled precisely, and used to drive
the frontier—but it is never presented as the completed Human.
