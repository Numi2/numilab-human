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
| Skeleton and articulation | 157 Core bodies, 51 joint equalities, 185 pose-bound BodyParts3D bone meshes; source-owned manubrium/sternum/clavicle continuity, lower-limb source-mesh and 40-boundary gates plus T1-T12, costovertebral, and sacroiliac source-geometry continuity | partial | several source bones share one mechanics body; no independent lesser-toe articulation; Mortensen neck merge is not active | complete named segment/DoF matrix with multi-pose source parity and loaded joint validation |
| Muscle actuation | all 416 current-pose routes, wraps, activation, compliant fibre/tendon equilibrium, `J^T`, persistent Metal stepping | partial | inferred compliant architecture and bounded recruitment are not full OpenSim-equivalent dynamic fibre/tendon state or held-out force validation | source curve/path/moment-arm parity plus held-out force-length-velocity and dynamic state tests |
| Tendon-to-bone and terminal-surface transfer | source-component-qualified `NHTENDON3` covers all 832 endpoints with 638 distributed surface envelopes and 194 exact point laws; 628 envelopes terminate on registered BodyParts3D bone, bilateral EO3 use exact source-rib fallback surfaces, and eight abdominal routes use separately typed exact anterior-thorax composite surfaces | partial | four lumped-toe spreads, 11 conditioning failures, 24 missing surfaces, and 155 source-model non-bone termini remain explicitly fail-closed; the ten source surfaces are force-transfer boundaries, not deformable tissue | retain the source surfaces until superior common-frame tissue owners pass preservation gates; add calibrated cartilage/fascia/aponeurosis volume, material, contact, and two-way force ownership |
| Tendon and fascia continuum | six-region pectoral Matter FEM with transactional NHTENDON2/3 loads | partial | generated pectoral volume and 10% load share are assumptions; no whole-body tendon/fascia continuum or two-way bone-muscle coupling | registered regional meshes, calibrated nonlinear material receipts, two-way load coupling, convergence, replay, and held-out deformation |
| Ligaments, cartilage, and menisci | exact bilateral BodyParts3D ribs 1--7 costal-cartilage FEM: 14 regions, 13,516 nodes, 46,278 positive tetrahedra, source-classified rib/sternal bands, cited pseudo-elastic starting law, M4 Pro deformation/reaction/rollback/bitwise-replay gate | partial | production owner fraction is zero pending live rib/sternum binding; material is homogeneous population-mean; articular cartilage, menisci, joint ligaments, and compliant joint contact remain absent | live non-duplicated thorax coupling and held-out costal deformation, then named joint geometry, nonlinear ligament laws, compliant cartilage/meniscus contact, calibration, and joint-level validation |
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
| connected BodyParts3D four-node surface envelope | 610 |
| route-private migrated BodyParts3D four-node envelope | 18 |
| exact pinned MyoSim rib-component four-node envelope | 2 |
| exact pinned anterior-thorax composite four-node envelope | 8 |
| exact source-site point law | 194 |
| maximum endpoint migration | `17.2616479 mm` |

The 194 point laws are not one homogeneous bug. Current fail-closed reasons are:

| Reason | Count | Correct next action |
| --- | ---: | --- |
| toe semantic representative exceeds 12 mm, source endpoint is bone-adjacent | 4 | retain the exact toe compound and resolve the terminal identity/geometry without independent articulation or gap patches |
| source endpoint is explicitly non-bone in the pinned source model | 155 | classify aponeurosis/fascia/soft-tissue ownership; never warp a bone toward the site |
| conditioning still fails after topology-aware exact-surface search | 11 | retain point law; consider a source-derived mechanics surface without relaxing amplification |
| body has no registered bone surface | 24 | classify soft-tissue/aponeurosis endpoints separately; add bone geometry only when anatomically correct |

The former 20-member semantic ambiguity is zero. Exact pinned-source topology
classifies it as 10 rib, eight anterior non-rib, and two explicit non-bone
termini. Eight rib termini pass on registered BodyParts3D ribs. The eight
anterior termini now pass unchanged force-transfer gates on their exact source
components, separately typed as unresolved anterior-thorax composite rather
than falsely labelled as bone or cartilage. Bilateral EO3
remain about 30.9 mm from those registered rib-9 surfaces, so moving either
whole rib would break already-passing entheses and costovertebral continuity.
They instead pass at 5.648 mm on their exact pinned source-rib components with
zero endpoint migration and a fail-closed BodyParts-first policy. The four toe
rows remain a semantic compound problem, not permission to add independent
articulation or move a neighboring bone under the endpoint.

## Execution order

The upper-limb, lower-limb, thoracic, rib, and pelvis source-mesh solvers close
the prior ordinary single-bone distance failures without relaxing the 12 mm
gate. The abdominal component pass then resolves every remaining multi-member
identity and admits eight more exact rib envelopes under unchanged gates.

1. Review the 11 conditioned patches without relaxing distance, amplification,
   force, or moment gates. Classify the 24 missing-surface and 155 source-model
   non-bone rows before admitting them. Keep the four lesser-toe spreads on the
   existing rigid compounds and preserve the EO3 BodyParts-first fallback plus
   separately typed anterior-thorax source surfaces.
2. Bind the compiled exact bilateral costal-cartilage volumes to their named
   live rib/sternal bodies with non-duplicated two-way reaction ownership and
   held-out deformation calibration; then resolve aponeurosis volume,
   intervertebral discs, named ligaments, registered
   anatomical colliders, and calibrated support contact, then
   close assistance-free balance before training gait.
3. Generalize the pectoral downstream-consumer transaction into whole-body
   tendon/fascia regions with measured nonlinear materials and two-way coupling.
4. Add ligament/cartilage/meniscus mechanics at load-bearing joints, followed
   by skin and organ/vessel deformation.
5. Integrate and parity-qualify the exact Rajagopal plus upper-extremity source
   composition, then train and evaluate closed-loop standing and walking.

Every increment must update this ledger using current executable evidence.
