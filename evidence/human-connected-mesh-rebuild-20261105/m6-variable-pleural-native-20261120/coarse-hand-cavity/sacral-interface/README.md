# Bilateral posterior tissue interfaces

The exact Float32 reference audit found 180 of 199 intersection segments between the gluteus maximus surfaces entirely outside the sacrum. Fifteen were entirely inside and four crossed the sacral surface. Thirteen percent of the summed pair-segment lengths lay inside the sacrum. Pair lengths can cover a shared geometric segment more than once.

The repair uses a common medial partition defined by the registered hip origins and conservative clearance against the registered neighboring solids. The 100 µm margin applies only to reference geometry preparation. The generated mesh records each retained source vertex and original parent face. Generated medial and obstacle facets carry distinct IDs. Six triangles on the right were contracted when their vertices emitted to identical Float32 coordinates. The receipt records the affected source faces and 2.14e-11 m² of pre-rounding area. The anatomy is a registered reference assembly. These surfaces adapt BodyParts3D, Copyright © 2008 The Database Center for Life Science, under [CC Attribution–Share Alike 2.1 Japan](https://creativecommons.org/licenses/by-sa/2.1/jp/).

## Reference results

| Assessed interface | Baseline pairs | Partition 06 pairs |
| --- | ---: | ---: |
| Right against left gluteus maximus | 199 | 0 |
| Registered bones | 1,177 | 0 |
| Skin and fat layers | 160 | 0 |
| Other muscle surfaces | 1,321 | 0 |
| Total | 2,857 | 0 |

The exact tests flagged zero self-intersections and zero opposite-side containment witnesses. Partition 11 also passed both directions of exact vertex containment against the bonded shell, the three relevant bone banks per side and five neighboring muscles per side. Partition 10 kept zero crossings against the individual physical shell layers from the initial and final replay captures with the candidate muscles held in their reference pose. The authored and native startup shell samples differed by at most 14.9 nm. The largest shell displacement over the retained native run was 22.29 µm. The test did not evolve the repaired muscles.

The cuts reduced enclosed source volume by 9.91% on the right and 7.39% on the left. No revised muscle mass, constitutive reference or attachment force was installed. The two surfaces remain authoring candidates.

## Volume and runtime status

The fixed-boundary TetGen trials retained the existing Float32 determinant floor of 1e−5 after normalization by the cube of the longest edge. Partition 06 had zero nonpositive tetrahedra with 142 right and 159 left below the floor. The strongest verified right run used partition 10, interior repositioning and bounded cavity flips. It had zero nonpositive tetrahedra, 21 below the floor and a boundary reimport mismatch. The strongest left run used partition 11 and the same sequence. It had zero nonpositive tetrahedra, 11 below the floor and exact boundary reimport. The remaining constrained cells and their coordinates are retained. The right partition 11 trial failed tetrahedralization. An earlier count-only flip trial introduced smaller determinant values and was excluded.

The latest native run for comparison is the previously published [paired skin/fat reference](../posterior-muscle-interface/README.md). It mass-matched all 28,632 skin/fat cells and 7,233 nodes. No inverted or low-quality skin/fat cells were present at its two endpoints. All 34 registered ligament rows passed and five captured poses showed no unresolved superior glenohumeral bone-path witness. That run advanced 2 ms in 98.61 s of body-stage time, or 0.00002028×, with variable pleural pressure and no root assistance. Bed reaction ended at 0.4315 N against approximately 953 N of body weight. Whole-body clearance, settled contact, sustained ligament motion and 16 s at 0.1× remain open.

## Reproduction on SSH Numi

Use the retained Python environment and the existing authoring scripts. `T` points to `coarse-hand-cavity-024`. The reader loads the exact NHTISS4 source hash recorded in `containment-11.json`. Input and result hashes are recorded in the individual receipts and `SHA256SUMS`. The capture command and binary hashes are available in the paired reference.

```sh
P="$T/posterior-muscle-interface-045"
V="$T/posterior-gluteal-continuum-041"
R="$T/../resting-clearance-and-atrium-006"
BONE="$T/thickness-rebuild-034/inputs/physical-bone-initial.bin"
RUN="$P/paired-native-128"
SHELL="$P/paired-06/posterior-interface-fit.npz"
A="$HOME/Work/numi-human-connected-surface-069/D/connected-obstacle-207/authoring-deps"
export PYTHONPATH="$A:$T:$T/posterior-load-interface-040:$P:$V"
export OPENBLAS_NUM_THREADS=1
python3 classify_sacral_conflicts.py "$P/bilateral-05" "$BONE" reproduced-classification.json
python3 build_posterior_obstacles.py "$BONE" "$SHELL" "$RUN/shell-mass.csv.tets.csv" reproduced-obstacles --native "$RUN" --reader "$R"
python3 partition_gluteal.py "$P" "$BONE" reproduced-partition --obstacles reproduced-obstacles --tissue-reader "$R"
python3 contract_interface_edges.py reproduced-partition reproduced-contract .000020 --original-source "$P/bilateral-05"
python3 partition_gluteal.py "$P" "$BONE" reproduced-audit --audit-mesh reproduced-contract --tissue-reader "$R"
python3 verify_closed_neighbors.py "$P" "$BONE" "$R" reproduced-audit reproduced-containment.json
python3 audit_emitted_shell.py "$SHELL" "$RUN" reproduced-audit reproduced-emission.json
python3 test_volume.py reproduced-audit/gluteal-29.npz reproduced-volume --polish-interior --polish-cavities
```
