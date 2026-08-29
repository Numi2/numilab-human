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
| Skeleton and articulation | 157 Core bodies, 51 joint equalities, 184 pose-bound BodyParts3D bone meshes; lower-limb source-mesh and 40-boundary gates plus T1-T12, costovertebral, and sacroiliac source-geometry continuity | partial | several source bones share one mechanics body; no independent lesser-toe articulation; Mortensen neck merge is not active | complete named segment/DoF matrix with multi-pose source parity and loaded joint validation |
| Muscle actuation | all 416 current-pose routes, wraps, activation, compliant fibre/tendon equilibrium, `J^T`, persistent Metal stepping | partial | inferred compliant architecture and bounded recruitment are not full OpenSim-equivalent dynamic fibre/tendon state or held-out force validation | source curve/path/moment-arm parity plus held-out force-length-velocity and dynamic state tests |
| Tendon-to-bone transfer | torso/axial-registered `NHTENDON3` covers all 832 endpoints with 620 distributed envelopes and 212 exact point laws; 18 of the envelopes use route-private exact named-bone sites with default-pose reference calibration and no added articulation | partial | ordinary single-bone distance-registration candidates are closed; four lumped-toe spreads, 11 conditioning failures, 24 missing surfaces, 20 unnamed multi-bone endpoints, and 153 soft-tissue/aponeurotic classifications remain explicitly fail-closed | exact semantic ownership or source-derived mechanics surfaces for the remaining bone candidates, followed by calibrated tissue ownership for non-bone endpoints |
| Tendon and fascia continuum | six-region pectoral Matter FEM with transactional NHTENDON2/3 loads | partial | generated pectoral volume and 10% load share are assumptions; no whole-body tendon/fascia continuum or two-way bone-muscle coupling | registered regional meshes, calibrated nonlinear material receipts, two-way load coupling, convergence, replay, and held-out deformation |
| Ligaments, cartilage, and menisci | no production owning solver | open | joint constraint and visual proximity are not tissue mechanics | named geometry, nonlinear ligament laws, compliant cartilage/meniscus contact, calibration, and joint-level validation |
| Anatomical collision and contact | ten MyoSim foot witnesses and source plane run on Metal | partial | no BodyParts3D collider registration, collision exclusions, calibrated friction/compliance, or whole-body anatomical contact | conservative registered proxies, material receipts, deterministic replay, and held-out support/contact outcomes |
| Skin and exterior | exact BodyParts3D outer source sheet retained as static reference | open | no physical skin weights, material, self-contact, muscle sliding, or deformation qualification | articulated skin/fat/fascia coupling with contact, volume control, visual and mechanical validation |
| Organs, vessels, and nerves | selected exact BodyParts3D torso surfaces are pose-bound visual layers | open | no organ FEM/MPM, vessel tube mechanics, fluid coupling, or neural mechanics | named volumetric/tubular models, calibrated materials/boundaries, conservation, contact, and replay evidence |
| Balance, control, and gait | transactional part coactivation and compiled standing transaction | partial | `balanced=false`; no closed-loop posture controller, deployable walking task, learned policy, or held-out gait | assistance-free stable standing, perturbation recovery, registered foot contact, deterministic resets, and held-out gait metrics |
| Apple runtime qualification | M4 Pro executes all routes, tendon transfers, rollback, and bitwise replay | partial | no same-workload performance qualification for the complete future tissue/contact/control stack | exact revision/artifact fingerprints, counters/traces, memory accounting, throughput, replay, and physical outcomes |
| Scientific validation | source provenance and simulation limitations are explicit | partial | no subject calibration, population variability, or clinical validation | benchmark protocol with held-out anatomical and mechanical data; claims limited to measured scope |

## Current endpoint disposition

The current compiler retains one law for every origin and insertion:

| Disposition | Value |
| --- | ---: |
| connected BodyParts3D four-node surface envelope | 602 |
| route-private migrated BodyParts3D four-node envelope | 18 |
| exact source-site point law | 212 |
| maximum endpoint migration | `17.2616479 mm` |

The 212 point laws are not one homogeneous bug. Current fail-closed reasons are:

| Reason | Count | Correct next action |
| --- | ---: | --- |
| toe semantic representative exceeds 12 mm, source endpoint is bone-adjacent | 4 | retain the exact toe compound and resolve the terminal identity/geometry without independent articulation or gap patches |
| distance exceeds 12 mm and source endpoint is not bone-adjacent | 153 | classify aponeurosis/fascia/soft-tissue ownership; never warp a bone toward the site |
| conditioning still fails after topology-aware exact-surface search | 11 | retain point law; consider a source-derived mechanics surface without relaxing amplification |
| body has no registered bone surface | 24 | classify soft-tissue/aponeurosis endpoints separately; add bone geometry only when anatomically correct |
| multiple members without unique semantic identity | 20 | retain point law until the pinned source names one member or a reviewed correspondence exists |

The ordinary single-bone distance-registration backlog is now zero. The four
toe rows remain a semantic compound problem, not permission to add independent
toe articulation or move a neighboring bone under the endpoint.

## Execution order

The upper-limb, lower-limb, thoracic, rib, and pelvis source-mesh solvers close
every ordinary single-bone distance failure without relaxing the ordinary
12 mm gate. The axial pass preserves 60 previously admitted envelopes and
recovers 56 point laws: 20 thoracic, 34 rib, and two iliacus endpoints.

1. Resolve the 20 multi-member identities and 11 conditioned patches using
   reviewed source semantics or source-derived mechanics surfaces. Classify the
   24 missing-surface and 153 non-bone rows as bone, fascia, aponeurosis, or
   other soft tissue before admitting them. Keep the four lesser-toe spreads on
   the existing bilateral rigid compounds until exact terminal identity exists.
2. Add costal cartilage, intervertebral discs, named ligaments, registered
   anatomical colliders, and calibrated support contact, then
   close assistance-free balance before training gait.
3. Generalize the pectoral downstream-consumer transaction into whole-body
   tendon/fascia regions with measured nonlinear materials and two-way coupling.
4. Add ligament/cartilage/meniscus mechanics at load-bearing joints, followed
   by skin and organ/vessel deformation.
5. Integrate and parity-qualify the exact Rajagopal plus upper-extremity source
   composition, then train and evaluate closed-loop standing and walking.

Every increment must update this ledger using current executable evidence.
