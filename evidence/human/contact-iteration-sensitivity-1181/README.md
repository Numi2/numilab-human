# Contact-iteration sensitivity review (1181)

This bundle records a read-only comparison of two completed 310 s native runs. It is sensitivity evidence, not model qualification or an isolated performance benchmark.

The only declared physical argument change is contact iterations (64 to 32); output paths differ as expected. Both runs accepted 155,000 steps at Float32 dt 0.0020000000949949026 s, ending at 310.0000147242099 s. Loaded runtime and source-file integrity checks passed for both.

## Result

The 32-iteration run had lower process wall time (2,171.415 s vs 2,761.931 s) and higher measured real-time factor (0.142764 vs 0.112240 simulation seconds per wall second, about 1.272x). These instrumented timings are not isolated performance qualification.

The traces diverged from the first sampled step. The 32-iteration run has materially larger contact and generalized constraint residual maxima and more late COM displacement than the 64-iteration reference. Its mean support force is similar, but that value is an endpoint-sampled last-physical-step force: summed normal impulse divided by Float32(dt), not by the full simulation horizon. The formula agrees with the runtime owner field.

The corrected report separates all-run, post-initialization (accepted step >= 5000), and late (>= 250 s) windows. Tangential Jv is pre-step point-J times previous stand velocity, not post-solve sliding speed. Similar finite physiology ranges do not establish equivalent physiology because the traces diverge.

**Conclusion:** do not adopt 32 sweeps from this comparison. Faster wall time does not offset worse residuals and the changed trajectory; this sensitivity test does not explain or resolve the observed drift.

## Profile and root-frame measures

Revision 4 carries the actual observer profile and exact source-report references, plus the two retained root-frame COM decompositions. The profile is diagnostic work, not physical-loop time. The root-frame report partitions measured world COM displacement into carrier translation, root-rotation orbit, and articulated-relative COM; the sum matches the observed displacement to floating-point roundoff. These are identity decompositions, not causal explanations for settling.

## Files and reproduction

- `corrected-summary.json` and `corrected-summary-v004.json` are revision 4 (SHA-256 `649defb6e03744363087ac480dbe8084154ccb58e547aa91cf9440e07118da82`).
- `superseded-summary-v003.json` and `original-summary-v001.json` preserve earlier summaries. Revision 3 had empty profile/decomposition fields; v1 also had the superseded timing and support-force calculation.
- `revalidate.py` reruns the source-bound calculation. It requires a precreated fresh output directory and refuses to overwrite an existing report. Exact revision-4 reproduction uses `/usr/bin/python3` (Python 3.9.6); Python 3.13 changes only floating summation at about 1e-10 mL and will not produce the same report hash. Example: `mkdir -p /tmp/contact-sensitivity-rerun && /usr/bin/python3 revalidate.py /tmp/contact-sensitivity-rerun`. It reads the pinned native runs, traces, earlier comparison/observer/root-frame reports, build pins, and source checkout listed in its output.
- `revalidate-v003.py` is retained for provenance only; it writes to the fixed original evidence location and should not be used for a new run.
- `build-pins.json` binds the observer binary, compiled source, physical library, and respiration metallib.

Native logs, traces, invocations, and auxiliary reports remain in their original retained evidence locations; this bundle does not duplicate those large inputs. The summary reports their exact paths and SHA-256 hashes. `SHA256SUMS` covers all bundle payload files except itself.

Independent review reran the calculation with the pinned Python 3.9.6 interpreter and reproduced the report byte for byte. A Python 3.13.13 rerun preserved every non-mean field and input identity; 54 mean fields differed only in floating summation rounding. The exact differences and report hashes are retained in independent-reproduction.json.
