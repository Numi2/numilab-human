# NumiLab Human completion gap ledger

This ledger defines “complete Human simulation” as a set of independently
provable workstreams. A green build or realistic image cannot close a mechanics
row, and an executable force path cannot close a material or control row.

Statuses use three terms: **proved** means the named current evidence executes
the stated boundary; **partial** means useful owning code exists but the
completion gate is still open; **open** means the end-state owner is absent.

| Workstream | Current evidence | Status | Gap that still matters | Completion gate |
| --- | --- | --- | --- | --- |
| Source foundation | BodyParts3D 4.0 plus active MyoSim 416-route body; Rajagopal and public MoBL-ARMS retained as comparative imports | partial | one source-faithful lower/upper mechanics composition is not qualified; authenticated bimanual upper source and exact non-cylinder wrap families remain separate | pinned bilateral source composition, exact wraps, mass/inertia/joint/path parity, and license receipts |
| Skeleton and articulation | 157 Core bodies, 51 joint equalities, 184 pose-bound BodyParts3D bone meshes, coherent limb/axial continuity gates | partial | several source bones share one mechanics body; no independent lesser-toe articulation; Mortensen neck merge is not active | named segment/DoF matrix with rest-pose, range, hierarchy, and multi-pose anatomical validation |
| Muscle actuation | all 416 current-pose routes, wraps, activation, compliant fibre/tendon equilibrium, `J^T`, persistent Metal stepping | partial | inferred compliant architecture and bounded recruitment are not full OpenSim-equivalent dynamic fibre/tendon state or held-out force validation | source curve/path/moment-arm parity plus held-out force-length-velocity and dynamic state tests |
| Tendon-to-bone transfer | v11 covers all 832 endpoints with 364 distributed envelopes and 468 exact point laws; 82 topology-only failures recovered with zero endpoint migration | partial | 411 distance failures, nine residual conditioning failures, 24 bodies without bone surface, four toe-distance failures, and 20 unnamed multi-bone endpoints | calibrated source-to-source registration receipts and exact per-endpoint disposition without relaxed global thresholds |
| Tendon and fascia continuum | six-region pectoral Matter FEM with transactional NHTENDON2 loads | partial | generated pectoral volume and 10% load share are assumptions; no whole-body tendon/fascia continuum or two-way bone-muscle coupling | registered regional meshes, calibrated nonlinear material receipts, two-way load coupling, convergence, replay, and held-out deformation |
| Ligaments, cartilage, and menisci | no production owning solver | open | joint constraint and visual proximity are not tissue mechanics | named geometry, nonlinear ligament laws, compliant cartilage/meniscus contact, calibration, and joint-level validation |
| Anatomical collision and contact | ten MyoSim foot witnesses and source plane run on Metal | partial | no BodyParts3D collider registration, collision exclusions, calibrated friction/compliance, or whole-body anatomical contact | conservative registered proxies, material receipts, deterministic replay, and held-out support/contact outcomes |
| Skin and exterior | exact BodyParts3D outer source sheet retained as static reference | open | no physical skin weights, material, self-contact, muscle sliding, or deformation qualification | articulated skin/fat/fascia coupling with contact, volume control, visual and mechanical validation |
| Organs, vessels, and nerves | selected exact BodyParts3D torso surfaces are pose-bound visual layers | open | no organ FEM/MPM, vessel tube mechanics, fluid coupling, or neural mechanics | named volumetric/tubular models, calibrated materials/boundaries, conservation, contact, and replay evidence |
| Balance, control, and gait | transactional part coactivation and compiled standing transaction | partial | `balanced=false`; no closed-loop posture controller, deployable walking task, learned policy, or held-out gait | assistance-free stable standing, perturbation recovery, registered foot contact, deterministic resets, and held-out gait metrics |
| Apple runtime qualification | M4 Pro executes all routes, tendon transfers, rollback, and bitwise replay | partial | no same-workload performance qualification for the complete future tissue/contact/control stack | exact revision/artifact fingerprints, counters/traces, memory accounting, throughput, replay, and physical outcomes |
| Scientific validation | source provenance and simulation limitations are explicit | partial | no subject calibration, population variability, or clinical validation | benchmark protocol with held-out anatomical and mechanical data; claims limited to measured scope |

## Current endpoint disposition

The v11 compiler retains one law for every origin and insertion:

| Disposition | Count |
| --- | ---: |
| connected BodyParts3D four-node surface envelope | 364 |
| exact source-site point law | 468 |
| endpoint migration | 0 |

The 468 point laws are not one homogeneous bug. Current fail-closed reasons are:

| Reason | Count | Correct next action |
| --- | ---: | --- |
| surface distance exceeds 12 mm | 411 | calibrated regional source-to-source registration, then rerun unchanged gates |
| conditioning still fails after topology-aware exact-surface search | 9 | retain point law; consider a source-derived mechanics surface without relaxing amplification |
| body has no registered bone surface | 24 | classify soft-tissue/aponeurosis endpoints separately; add bone geometry only when anatomically correct |
| multiple members without unique semantic identity | 20 | retain point law until the pinned source names one member or a reviewed correspondence exists |
| semantic toe representative exceeds distance gate | 4 | improve foot registration; do not move the authored hallux/digitorum sites |

## Execution order

The v11 topology-aware exact-surface solver closed 82 of the previous 91
conditioning/sparse-topology failures without changing the global thresholds.
It is complete for that admissible subset; the nine residuals remain explicit.

1. Build a calibrated, provenance-pinned regional mechanics-to-anatomy
   registration field for the direct named correspondences now rejected by
   distance, starting with the sacrum, thoracic cage, upper limbs, and feet.
2. Add registered anatomical colliders and calibrated support contact, then
   close assistance-free balance before training gait.
3. Generalize the pectoral downstream-consumer transaction into whole-body
   tendon/fascia regions with measured nonlinear materials and two-way coupling.
4. Add ligament/cartilage/meniscus mechanics at load-bearing joints, followed
   by skin and organ/vessel deformation.
5. Integrate and parity-qualify the exact Rajagopal plus upper-extremity source
   composition, then train and evaluate closed-loop standing and walking.

Every increment must update this ledger using current executable evidence.
