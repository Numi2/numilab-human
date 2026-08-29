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
| Tendon-to-bone transfer | the final upper-limb pair covers all 832 endpoints with 526 distributed envelopes and 306 exact point laws; 60 humerus-to-finger bodies, 166 candidate distances, 45 prior envelopes, and 52 continuity transitions pass | partial | 10 scapular and 80 torso/lower-body registration candidates, 13 conditioning failures, 24 missing surfaces, 20 unnamed multi-bone endpoints, and 159 soft-tissue/aponeurotic classifications | calibrated source-to-source registration receipts and exact per-endpoint disposition without relaxed global thresholds |
| Tendon and fascia continuum | six-region pectoral Matter FEM with transactional NHTENDON2 loads | partial | generated pectoral volume and 10% load share are assumptions; no whole-body tendon/fascia continuum or two-way bone-muscle coupling | registered regional meshes, calibrated nonlinear material receipts, two-way load coupling, convergence, replay, and held-out deformation |
| Ligaments, cartilage, and menisci | no production owning solver | open | joint constraint and visual proximity are not tissue mechanics | named geometry, nonlinear ligament laws, compliant cartilage/meniscus contact, calibration, and joint-level validation |
| Anatomical collision and contact | ten MyoSim foot witnesses and source plane run on Metal | partial | no BodyParts3D collider registration, collision exclusions, calibrated friction/compliance, or whole-body anatomical contact | conservative registered proxies, material receipts, deterministic replay, and held-out support/contact outcomes |
| Skin and exterior | exact BodyParts3D outer source sheet retained as static reference | open | no physical skin weights, material, self-contact, muscle sliding, or deformation qualification | articulated skin/fat/fascia coupling with contact, volume control, visual and mechanical validation |
| Organs, vessels, and nerves | selected exact BodyParts3D torso surfaces are pose-bound visual layers | open | no organ FEM/MPM, vessel tube mechanics, fluid coupling, or neural mechanics | named volumetric/tubular models, calibrated materials/boundaries, conservation, contact, and replay evidence |
| Balance, control, and gait | transactional part coactivation and compiled standing transaction | partial | `balanced=false`; no closed-loop posture controller, deployable walking task, learned policy, or held-out gait | assistance-free stable standing, perturbation recovery, registered foot contact, deterministic resets, and held-out gait metrics |
| Apple runtime qualification | M4 Pro executes all routes, tendon transfers, rollback, and bitwise replay | partial | no same-workload performance qualification for the complete future tissue/contact/control stack | exact revision/artifact fingerprints, counters/traces, memory accounting, throughput, replay, and physical outcomes |
| Scientific validation | source provenance and simulation limitations are explicit | partial | no subject calibration, population variability, or clinical validation | benchmark protocol with held-out anatomical and mechanical data; claims limited to measured scope |

## Current endpoint disposition

The current compiler retains one law for every origin and insertion:

| Disposition | Count |
| --- | ---: |
| connected BodyParts3D four-node surface envelope | 526 |
| exact source-site point law | 306 |
| endpoint migration | 0 |

The 306 point laws are not one homogeneous bug. Current fail-closed reasons are:

| Reason | Count | Correct next action |
| --- | ---: | --- |
| distance exceeds 12 mm, but source endpoint is bone-adjacent | 86 | calibrate the 10 scapular and 76 other regional correspondences, then rerun unchanged gates |
| toe semantic representative exceeds 12 mm, source endpoint is bone-adjacent | 4 | retain the exact toe compound and resolve the terminal identity/geometry without independent gap patches |
| distance exceeds 12 mm and source endpoint is not bone-adjacent | 159 | classify aponeurosis/fascia/soft-tissue ownership; never warp a bone toward the site |
| conditioning still fails after topology-aware exact-surface search | 13 | retain point law; consider a source-derived mechanics surface without relaxing amplification |
| body has no registered bone surface | 24 | classify soft-tissue/aponeurosis endpoints separately; add bone geometry only when anatomically correct |
| multiple members without unique semantic identity | 20 | retain point law until the pinned source names one member or a reviewed correspondence exists; 18 are source-bone-adjacent and 2 are not |

The 90 remaining bone-adjacent distance failures include the four semantic-toe
rows; they are separated above rather than counted twice.

## Execution order

The upper-limb source-mesh solver closed 159 intended distance failures plus
three incidental same-body endpoints without changing the global thresholds.
It is complete for the admitted 60-body subset; seven intended distal targets
remain conditioned point laws, and the ten scapular candidates fail rigid-fit
selection explicitly.

1. Resolve the ten scapular targets with pinned glenoid, acromion, coracoid,
   and medial-border landmarks, then address the remaining 80 bone-adjacent
   torso/lower-body candidates. Do not start by warping the sacrum: all 143 of
   its distance-rejected sites are non-bone-adjacent in the source model.
2. Add registered anatomical colliders and calibrated support contact, then
   close assistance-free balance before training gait.
3. Generalize the pectoral downstream-consumer transaction into whole-body
   tendon/fascia regions with measured nonlinear materials and two-way coupling.
4. Add ligament/cartilage/meniscus mechanics at load-bearing joints, followed
   by skin and organ/vessel deformation.
5. Integrate and parity-qualify the exact Rajagopal plus upper-extremity source
   composition, then train and evaluate closed-loop standing and walking.

Every increment must update this ledger using current executable evidence.
