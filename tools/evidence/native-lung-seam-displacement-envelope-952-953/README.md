# Selected lung/pleura seam displacement envelope (952 + 953)

This package documents two scoped, CPU-only replays of previously selected exact seam witnesses. It does not edit anatomy, rerun the simulator, or change an intersection predicate or acceptance gate. Original capture packs and source inputs remain at the hash-bound absolute paths in `retained-artifacts.json`; they are not copied here.

## Terminal pleura self-pairs (952)

At accepted step 10000, 25 exact unallowed self-intersection pairs of pleura row 310 were linked to their parent lung faces. This was a selected-pair replay, not a new whole-mesh scan. In the binary64 source-map replay, 17 pairs are disjoint, with gaps from 2.264 to 14.999 nm, and 8 are exact shared-vertex point contacts. The mapped common source vertex is exactly the intersection point on both parent triangles. These ideal intersections are single-point contacts, not residual overlap.

All 25 pairs intersect in the actual captured Float32 mesh. The maximum measured normal intrusion is 10.160 nm. The maximum reported intersection span is 127.679 um; that span is an intersection extent, not a penetration-depth measurement. No exact captured intersection is waived or marked acceptable.

For each triangle, the envelope uses the largest Euclidean displacement of its three captured vertices from their corresponding binary64 replay vertices. For any equal-barycentric point with weights `lambda_i >= 0` and `sum(lambda_i) = 1`,

`||sum(lambda_i * (captured_i - replay_i))|| <= sum(lambda_i * e_i) <= max(e_i)`.

Thus the corresponding triangle points differ by at most that triangle's maximum vertex displacement, and the distance between a pair of corresponding points can change by at most the sum of the two triangle bounds. The measured pair bounds are 16.398-30.291 nm; they cover the binary64 replay gap for all 25 selected pairs. This is an observed per-triangle capture-to-replay envelope, not a formal error bound for every shader operation.

The separate shared-contact replay records all eight common source vertices and their exact binary64 mapped point matches. The offline Float32 transcription reproduces 15 of 25 pair hits, with at most 2 ULP difference from capture; it is not bitwise equivalent to the compiled Metal execution.

## Earlier phase witnesses rechecked on captures (953)

The 93 pair/phase records were selected by an earlier exact scan and checked against the corresponding retained 931 accepted captures. This did not discover new pairs. Of the 87 same-step checks, all 87 intersect in captured Float32 geometry: 28 are also binary64 replay point contacts, while 59 are disjoint in the replay. Every observed native hit is within its measured sum-of-triangle-displacements envelope. Across those records, the replay/source relations are 31 point contacts and 62 disjoint pairs; the ideal intersections are single-point contacts, not residual overlap.

Six selected witnesses were labeled step 9999, for which the 931 set has no pack. Those six were explicitly evaluated against terminal step 10000 instead: three are captured and replay point contacts; three are disjoint in both. They are not same-step results.

The largest per-triangle captured-to-replay vertex displacement is 19.455 nm; the largest pair bound is 38.910 nm, and the largest disjoint replay gap is 12.445 nm. These values describe only the selected witnesses and listed captures.

## Interpretation and limits

The results show that the observed native intersections are within the measured geometric perturbation for these selected pairs. They do not prove that Float32 rounding is the sole cause, provide a formal bound on all GPU arithmetic, establish exact clearance for the full anatomy, or guarantee behavior between captured times. Exact native intersection counts remain visible and unallowed. No blanket seam exemption, tolerance widening, geometry change, or gate change was made.

## Reproduction

The exact scripts and their outputs are retained here. They contain absolute input paths and output-directory constants: O in 952-terminal/analyze.py and OUT in 953-phases/analyze_selected_witnesses.py. Do not rerun them over the original evidence directories. For a repeat on macmini, use disposable script copies, change only the output-directory constant to a fresh existing evidence child, and keep all input paths and hashes fixed. Use /Users/n/numi-human-prep-venv-20261005/bin/python. The 953 script still reads the pinned original 952 analysis/input records; moving its output does not change those inputs.

Compare the repeated numerical records and their input identities with the retained JSON outputs; output-path or script-identity changes can change report bytes. SHA256SUMS.txt verifies these retained publication files, and retained-artifacts.json records full external source/capture paths and hashes.
