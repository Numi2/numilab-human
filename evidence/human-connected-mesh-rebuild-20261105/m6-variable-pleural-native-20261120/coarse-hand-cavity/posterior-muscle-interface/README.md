# Posterior gluteal and deep-fat clearance

The supported-rest target requires the bed to transfer force through the bonded skin and fat into tissue and skeletal attachments. The native reference on SSH Numi applies gravity and bed contact. Its 128-root release reached a reaction of 0.435 N against 952.864 N of weight. This increment repairs geometry in that load path and recooks the physical fat boundary and mass through the existing owner.

| Exact authoring check | Original | Gluteal fit | Combined fit |
| --- | ---: | ---: | ---: |
| Gluteus maximus against skin/fat triangle pairs | 250 | 184 | 160 |
| Gluteal volume nodes inside the bonded shell | 299 | 246 | 242 |
| Shell boundary vertices inside either gluteus | 1 | 0 | 0 |
| Mutual gluteal triangle pairs | 199 | 199 | 199 |

The independent predicate checks use the emitted Float32 lattice. The combined fit introduces no newly contained node in the tested tissue surfaces, no new off-target surface pair, and no new shell or muscle fold. The gluteal refit relocates some of the existing fat-to-muscle interface intersections. The 160 remaining triangle pairs are all at the deep fat boundary. The original bilateral muscle overlap remains present. Each gluteal volume retains its own source triangle and vertex identities and the two meshes retain positive cells. The larger absolute relative volume change is 2.41e−8. The first fit removed more pairs but introduced five folds per side. The selected right and left displacements are 2/16 and 16/16 of the subsequent protected proposals.

The shell fit moves only deep fat vertices, with its original connectivity, outer skin and bonded interface fixed. The existing density- and distance-constrained transfer closes all 628 typed mass pools over 28,632 cells. It maps 13.939679838742224 kg across 28,957 routes with a maximum distance of 59.61 mm. The maximum difference in a rigid donor's mass debit is 1.01e−11 kg relative to this fit's coarse native control. The separately repaired 1,448,804-cell reference has zero inversions and shape failures in its reference configuration and conserves 13.94458033634514 kg. Its fine-mesh motion transfer remains open.

## Skeletal binding witnesses

The existing gluteal screen reproduced five chords exactly through the registered skeletal donors. The native bone manifest identifies body and bank 1 as sacral source FJ3393. A documented gluteal origin motivated adding that source to the geometric screen. [Cadaver anatomy](https://pubmed.ncbi.nlm.nih.gov/23959791/)

The expanded screen found 79 candidate chords, 74 reaching the sacrum. The follow-up recovers the fat, gluteal and skeletal attachment barycentric weights and checks the opposing gluteal solid with the exact predicate. The sum of projected source-face areas is bounded by 31.084 cm². These rows are prepared geometric evidence. A live gluteal continuum with verified contact and skeletal load transfer remains to be integrated.

## Reproduction on SSH Numi

Use the retained shared authoring tree and source reader. Set `T` to `coarse-hand-cavity-024`, `V` to its `posterior-gluteal-continuum-041`, `B` to `$T/loaded-shell-seed-039/native-inner-fit-022`, and `R` to `resting-clearance-and-atrium-006`. The baseline lever arms, source rotations, contact configuration and all build paths are pinned in the retained launches. The pose source is the first accepted transform and records its deviation from startup. Use `$T/thickness-rebuild-034/inputs/physical-bone-initial.bin` as `BONES`.

```sh
export PYTHONPATH="$T:$T/posterior-load-interface-040:$V"
export OPENBLAS_NUM_THREADS=1
python3 fit_muscle_interface.py "$V" "$BONES" "$R" trial-02
python3 fit_muscle_interface.py "$V" "$BONES" "$R" protected-04 --protect-folds trial-02/assessment.json
python3 fit_muscle_interface.py "$V" "$BONES" "$R" bilateral-05 --proposals protected-04
python3 verify_containment.py "$V" bilateral-05 containment-05.json
python3 screen_donors.py "$T/posterior-load-path-042" "$V" bilateral-05 "$BONES" donor-screen.json
python3 bind_donor_geometry.py donor-screen.json bilateral-05 "$BONES" donor-bindings.json
python3 fit_paired_boundary.py "$V" bilateral-05 "$BONES" "$R" "$T" paired-06
python3 verify_containment.py "$V" bilateral-05 paired-06/containment.json --paired-shell paired-06/posterior-interface-fit.npz
python3 "$T/posterior-load-interface-040/map_mass.py" "$V/native-candidate-128" "$T" paired-06/posterior-interface-fit.npz paired-06/mass.json --solve-transport --mass paired-06/mass.npz
python3 "$V/balance_transfer.py" "$V/native-candidate-128" "$T" paired-06 paired-06/balanced-mass.npz
python3 "$V/native_trial.py" build "$B" "$B/candidate-fresh" paired-06 paired-native-build
python3 "$V/native_trial.py" run "$V/native-candidate-128" paired-native-build paired-native-128 --steps 128
```

The anatomy is a registered reference assembly from multiple sources. The gluteal and shell derivatives adapt BodyParts3D, Copyright © 2008 The Database Center for Life Science, under [CC Attribution–Share Alike 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/).

Seven duplicate historical cooked packs were removed after byte-for-byte recovery from the private release. The independent verification receipt records the paths, archive hash and actual filesystem space recovered.

## Native qualification result

The candidate shell ran on Apple M6, macOS 27.0 (26A425), with variable pleural pressure and zero root assistance. It accepted 128 ordered steps through 2.000 ms. The measured body stage took 98.61 s for 0.00002028×. Process wall time was 114.23 s. The recorded pleural range was −490.332581 to −490.332489 Pa.

- Native compilation matched each of the 28,632 tetrahedron masses and all 7,233 Float32 nodal masses and donor frame identities.
- Both captured endpoints had zero inverted or low-quality shell cells and zero shell self- or cross-layer triangle pairs. The minimum terminal determinant was 0.9743× its startup value across the shell.
- All 34 registered ligament rows passed the short native audit. Five accepted poses showed no unresolved bone intersection along the two fixed superior glenohumeral attachment paths.
- The whole-shell audit found 17,850 initial and 17,897 terminal layer/bone surface pairs, matching the previous control counts. This is an endpoint triangle count across all three shell layers and 469,136 registered bone triangles. The audit retains full witnesses on Numi and a compact summary here.
- Terminal bed reaction was 0.4315 N over three loaded rows against approximately 953 N of body weight. Supported equilibrium and the shoulder check under settled load remain open.

The native run exercises the physical shell and ligament owners. The two gluteal volumes remain in authoring. The installed scene remains on the prior qualified reference. Sustained end-to-end throughput, full-body clearance and 16 seconds through variable pleural pressure remain unqualified.
