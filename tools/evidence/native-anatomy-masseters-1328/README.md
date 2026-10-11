# Passive superficial masseters in the integrated native human

**Orientation correction pending:** A subsequent signed-volume audit found the mirrored left masseter160 is inward-wound (−5.65116e−6 m³), although it is closed and intersection-free. This recording is retained as a bounded geometry/physiology result, not outward-orientation qualification. See orientation-erratum.json. A successor reverses that row's face order and normals while preserving positions and bindings, and will be checked in a fresh native run.

Both source-registered superficial masseters are present in the existing native resting scene (stable159 right / FMA49001, stable160 left / FMA49002). This is an incremental anatomy delivery. Whole-body anatomy and the final five-minute qualification remain incomplete; performance work is deferred.

The Z-Anatomy source is registered to the existing BodyParts3D reference with seven named head bones, a proper similarity transform, and an independent mandible check. Bone surface fit p95 is about 3.45–5.51 mm; the held-out mandible maximum is 7.31 mm. Source attachment markers are proximity proxies, not certified footprints. This mixed-source reference is not measurements of one individual. Original Z-Anatomy and upstream BodyParts3D license notices accompany the evidence.

Each 702-vertex, 1,404-face source surface had a small fold repaired by the existing harmonic owner: five vertices per side change, maximum 0.65234 mm in the source / 0.67576 mm after registration. Faces and ancestry are unchanged. Both surfaces are passive inspection geometry with one skull-body binding (core23); active jaw mechanics are not implemented. They do not add mass or independent forces.

The viewer incorrectly rejected any non-bone surface with fewer than two body bindings. Its existing renderer already supports one binding. The narrow loader fix permits a positive binding count while preserving all transform, index, weight and identity checks. Active FEM still explicitly requires its two-body endpoints. The initial rejected native attempt is retained.

At eight actual native captures across 16 simulated seconds, each masseter is closed, has zero exact self-intersections and zero crossings with all 862 surrounding surfaces, including the other masseter. All 861 previous surfaces, 157 accepted poses and 1,000 rows × 60 columns of coupled physiology are identical to the prior right-SCM scene. The 8,000-step run retains a continuous 252-frame recording and all seven inspection layers. The continuous invalid-triangle/contact audit passes; exact self/intersection checks are at eight captures, not every step.

The launcher recorded a literal DYLD_LIBRARY_PATH mismatch: new build-directory symlinks resolve to the existing pinned runtime directories. Verification records this mismatch and proves identical canonical directories, all other environment fields and argv, with unchanged inputs. No discrepancy is silently discarded.

## Reproduce

    python3 tools/evidence/native-anatomy-masseters-1328/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-masseters-1328/reproduce.py

The retained Mac mini assets and pinned runtime are required. See the native run declaration for the exact launch command and asset identities, summary.json for timings and revisions, and checks/native-verification.json for the geometry and physiology checks.

Head/neck/back coverage, eight original defective muscle rows, existing inter-tissue interfaces, and the final complete-anatomy five-minute pair still need work. No clinical validation or performance qualification is claimed.
