# Cardiac cavity reference registration — 12 September 2026

Four CVSim21 cardiac chambers now resolve to exact BodyParts3D cavity references through `numi human-circulation-anatomy`. Source geometry, topology, volume moments and native identities are bound together. The four cavities are individually embedded, closed surfaces after identifying exactly coincident source vertices. **The right atrial and right ventricular domains overlap: 42 triangle pairs intersect.** This composition is therefore not admitted as four disjoint physical blood volumes.

Both CVSim volume-coordinate variants pass a native identity-equivalence check on the physical M4 Pro. Hydraulic parameters, volume ownership and mechanical mass remain unchanged. This advances source-to-anatomy association; it does not establish body-frame registration, calibrated cardiac geometry, deforming tissue coupling, or a blood/tissue mass partition.

## Exact source geometry

The extraction verifies the pinned 64,888,505-byte `partof_BP3D_4.0_obj_99.zip`, both anatomical relationship tables and each OBJ member. These are cavity concepts, not the overlapping whole-chamber wall/septal memberships in the older 18-region template.

| CVSim chamber / source index | Cavity concept | OBJ member | Coincident vertices identified | Computed atlas cavity volume (mL) | CVSim initial volume (mL) |
| --- | --- | --- | ---: | ---: | ---: |
| Right atrium / 15 | FMA:11359 | FJ2424 | 69 | 84.551532 | 32.263950 |
| Right ventricle / 16 | FMA:9291 | FJ2423 | 12 | 116.972356 | 138.358644 |
| Left atrium / 19 | FMA:9465 | FJ2425 | 53 | 51.936799 | 47.737489 |
| Left ventricle / 20 | FMA:9466 | FJ2422 | 16 | 97.938773 | 158.755347 |

Each quotient has one connected component, Euler characteristic 2, consistent orientation, manifold vertex links and no degenerate triangles. The original coordinates and triangles remain in the artifact alongside the exact source-to-quotient vertex map. Equality uses authored decimal coordinates, with no tolerance welding, caps, moved vertices or added faces. OBJ headers explicitly supply millimetre units; the output uses metres.

The source headers report volumes of 84.2651, 116.344, 51.3647 and 97.0455 mL, respectively. Those values differ from integration of the actual selected mesh and remain preserved in the headers. The independently retained decimal-coordinate audit agrees with the new surface integrals and verifies translation invariance. First and second volume moments, centroid and inertia per unit density are available; no blood density or mass is inferred.

The atlas cardiac phase and subject correspondence are unknown. Neither geometric volume nor filling offsets are fitted to CVSim. In particular, the right atrium's initial hydraulic volume is only about 0.382 times its atlas cavity volume, whereas the left ventricle's is about 1.621 times. These are discrepancies between sources, not calibration residuals against a matched observation.

The [current official BodyParts3D license](https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html) specifies CC-BY-4.0. The downloaded OBJ headers still contain historical CC-BY-SA-2.1-Japan text; both are recorded. Attribution remains with BodyParts3D and the Database Center for Life Science.

## Embeddedness and the overlap blocker

The [intersection owner](../src/numilab_human/cardiac_cavity_intersections.py) converts the actual published Float64 metre coordinates to exact rational values and then to a shared integer denominator. AABB rejection precedes exact triangle predicates. Adjacent faces are checked too: their intersections must lie only on their common vertex or edge. Surface pairs without intersections undergo exact ray-parity containment checks, so nested solids cannot be reported as disjoint.

All four self-intersection checks pass. Five inter-cavity pairs are disjoint. The right atrium and right ventricle have 42 intersecting triangle pairs and independently observed interior witnesses. For example, source RA vertex 470 at `(-17.2741, -134.554, 1219.78) mm` lies inside the RV surface. The immutable [intersection report](media/cardiac-cavities-20260912/independent/intersections.json) retains the exact triangle indices and containment results.

No overlap is automatically trimmed, assigned to a chamber, or converted into double-counted blood mass. A subsequent geometric composition must resolve the shared atrioventricular region with explicit source correspondence and quantified geometry changes before disjoint-volume admission.

## Native integration and evidence

The existing `HumanPack.physiology-native.v3` loader receives the four cavity FMA identifiers. The other 17 compartments retain their CVSim aggregate identifiers. Stable compartment indices, unique hydraulic-volume owners, all 24 connections, constitutive laws, initial states, exact cardiac period and numerical settings remain identical. Combined source and authored identities include atlas hashes, complete cavity geometry, volume moments and the intersection audit.

The native source remains `b91fe6832813497ab532e5dbe05ed8c8233e6b1f`, clean throughout qualification, with ABI28/package13. A separate [qualification probe](../tools/cardiac_cavity_native_check.mm) links the existing compiler/runtime archives and Metal library. It does not introduce another physics owner or alter the native checkout.

For both upstream and Heldt-aligned variants, the original and anatomically associated payloads each run two environments for 64 accepted 1 ms steps. Metal API validation is enabled. All runs have zero failed steps, finite evolving state, exact accepted clock advancement, bitwise equivalent state and environment pairs, and bitwise snapshot replay. Their package fingerprints are distinct. Cross-identity restore rejects without changing accepted state. A negative control that changes the right atrial initial volume by 1% rejects as a physical-parameter change.

Final positive runs took about 5.94 and 5.45 seconds including both model initializations, package round trips and readback checks. This is bounded identity-equivalence evidence, not a throughput benchmark or new physiological calibration. The original ten-cycle/refinement receipts remain unchanged. The build, source, runtime, shader and input hashes were equal before and after every final run. Earlier successful probe runs, before finite-state/evolution assertions were added, remain on the Mac mini under `/Users/n/human-cardiac-cavities-20260912/evidence`.

## Reproduce and remaining ownership gates

```sh
numi human-circulation-anatomy --output Build/cardiac-cavities/reference.native.json
python3 tools/verify_cardiac_cavities_20260912.py
```

The command reports the 42 intersections and `disjoint_cavity_domains=false`. Payload and manifest outputs are immutable. The [configuration](../config/cvsim21-cardiac-cavities.v1.json) permits either documented source volume-coordinate variant; it rejects physiological overrides, mass extraction and calibration claims.

The [receipt](media/cardiac-cavities-20260912/receipt.json) binds the source owners, both regenerated payloads/manifests, exact geometry audit and native execution. All 163 selected Human tests pass without skips, including 36 new geometry and registration tests. All 13 receipt-verification tests also pass, including rejected scope promotion, missing owners, altered build inputs and swapped run records. They cover source tampering, topology, touching/nested/intersecting domains, strict configuration, unchanged hydraulic ownership, analytical tetrahedral moments and covariance across extreme scales.

Next requirements are an explicit overlap-resolution composition, cardiac phase and subject/body registration, and a sourced blood-to-tissue mechanical partition. Existing native donor subtraction can conserve mass and first/second moments, but needs sourced density, spatial receiver quadrature and evidence of blood inclusion in gross body mass. Changing hydraulic volumes additionally needs conservative mechanical mass/momentum transfer within the existing accepted transaction. The remaining regional vasculature, individual-organ perfusion, pressure/deformation feedback, tissue exchange, reflex/tilt behavior and standing/walking remain open.
