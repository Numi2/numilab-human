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
| Tendon-to-bone and terminal-surface transfer | source-component-qualified `NHTENDON3` covers all 832 endpoints with 642 distributed surface envelopes and 190 exact point laws; 632 envelopes terminate on registered BodyParts3D bone; the final upper-limb fallback (`TRImed_l`) now has a zero-migration exact-humerus envelope; bilateral EO3 use exact source-rib fallback surfaces; eight routes use separately typed exact anterior-thorax composite surfaces, seven of which now have bounded live continuum ownership; all 14 body-7 abdominal point terminals have a separate bounded `NHFASC4` continuum owner | partial | four lumped-toe spreads, seven conditioning failures, 24 missing bone surfaces, 153 sites beyond the surface-distance gate, and two explicit source-model non-bone termini remain fail-closed in the endpoint transport; continuum ownership does not turn soft-tissue terminals into bone envelopes | retain exact source surfaces until superior common-frame owners pass preservation gates; replace point transport only when a superior regional owner preserves endpoint identity and nonduplicated force authority |
| Tendon and fascia continuum | live `NHFASC4` owns six bilateral pectoral, six latissimus-aponeurosis, eight external-oblique, and six internal-oblique routes in 12 transversely isotropic Matter objects; exact route directions, bilateral/class grouping, same-command-buffer endpoint-share replacement, per-step fixed-node reactions, replay, rollback, and current Apple M4 Pro one/four-step execution are gated | partial | the regional sheets are generated mechanics envelopes; homogeneous group frames are not per-tetrahedron histology; the matrix/fibre split and 10% load ownership are not independently calibrated; fascia sliding/contact and sustained trunk loading are absent; there is no whole-body tendon/fascia continuum | registered regional meshes, spatial fibre fields and independent biaxial calibration, nonduplicated two-way load coupling, fascia interaction, convergence, sustained replay, and held-out deformation |
| Ligaments, cartilage, and menisci | exact bilateral BodyParts3D ribs 1--7 costal-cartilage FEM plus exact Open Knee oks003 ACL/PCL/MCL/LCL/PTL/QAT topology; `NHKNEE1` ABI 2 admits the six source fibre axes, material constants, in-situ stretches, and FeBio hash; Apple M4 Pro executes the neutral-stretch source-directed ligament/PTL law with three-body reactions, rollback, and bitwise replay | partial | the knee law is a smooth source-shaped approximation rather than FEBio's `Ei`/exp-linear law; final in-situ stretch failed a bounded one-step nonlinear/performance gate and needs staged equilibrium initialization; QAT volumetric activity, production cartilage/meniscus contact, calibration, and loaded flexion remain open; costal material is population-mean and not yet live-bound | staged source prestress equilibrium, exact or validated reduced fibre-law parity, loaded knee contact/flexion, live non-duplicated costal coupling, calibration, and held-out joint-level outcomes |
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
| connected BodyParts3D four-node surface envelope | 614 |
| route-private migrated BodyParts3D four-node envelope | 18 |
| exact pinned MyoSim rib-component four-node envelope | 2 |
| exact pinned anterior-thorax composite four-node envelope | 8 |
| exact source-site point law | 190 |
| maximum endpoint migration | `17.2616479 mm` |

The 190 point laws are not one homogeneous bug. Current fail-closed reasons are:

| Reason | Count | Correct next action |
| --- | ---: | --- |
| toe semantic representative exceeds 12 mm, source endpoint is bone-adjacent | 4 | retain the exact toe compound and resolve the terminal identity/geometry without independent articulation or gap patches |
| nearest registered surface exceeds the unchanged distance gate | 153 | resolve source/common-frame ownership; never warp a bone toward the site or relax the gate globally |
| conditioning still fails after topology-aware exact-surface search | 7 | retain point law; consider a source-derived mechanics surface without relaxing amplification |
| body has no registered bone surface | 24 | ten chest and all fourteen body-7 records retain exact point transport but now have bounded regional continuum owners; upgrade those owners to registered directional tissue without relabelling soft tissue as bone |
| source endpoint is explicitly non-bone in the pinned source model | 2 | classify aponeurosis/fascia/soft-tissue ownership; never relabel it as bone |

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

1. Replace the admitted `NHFASC4` generated abdominal sheets with registered
   regional geometry and directional calibrated material frames while
   preserving all fourteen exact source terminals, the nonduplicated 10% force
   transaction during migration, and the hidden-mechanics presentation rule.
   Keep the four lesser-toe spreads on existing rigid compounds and preserve
   the EO3 and anterior-thorax fallbacks.
2. Bind the compiled exact bilateral costal-cartilage volumes to their named
   live rib/sternal bodies with non-duplicated two-way reaction ownership and
   held-out deformation calibration; then resolve aponeurosis volume,
   intervertebral discs, named ligaments, registered
   anatomical colliders, and calibrated support contact, then
   close assistance-free balance before training gait.
3. Generalize the admitted `NHFASC4` downstream-consumer transaction into
   remaining regional tendon/fascia owners with measured nonlinear materials,
   directional frames, sliding/contact, and nonduplicated two-way coupling.
4. Finish the existing Open Knee boundary: staged in-situ prestress
   equilibrium, exact-or-validated reduced FEBio fibre-law parity, loaded
   cartilage/meniscus contact and flexion. Extend the same source/material ABI
   discipline to other load-bearing joints, then skin and organ/vessel
   deformation.
5. Integrate and parity-qualify the exact Rajagopal plus upper-extremity source
   composition, then train and evaluate closed-loop standing and walking.

Every increment must update this ledger using current executable evidence.
