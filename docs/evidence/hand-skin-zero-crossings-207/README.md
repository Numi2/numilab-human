# Zero-crossing hand skin: rejected anatomical diagnostic checkpoint

The pinned P205/806 reference surface had **977 skin/bone triangle-pair crossings** and 125 sampled skin-node/bone inside pairs. This candidate has **0 crossings across all 185 banks, 0 exact self-intersections across 58,509 faces, and 0 inside node/bone pairs across 29,199 emitted Float32 vertices**. Exact topology checks find no nonmanifold edges, duplicate faces or zero-area faces. The original 45 head boundary edges remain unchanged and open.

**This surface fails anatomical topology acceptance:** its main component has 42 spurious handles, with two additional tiny nested cavity shells and two unused cap-center points. The pinned baseline has one genus-zero component. Zero intersections do not establish a valid skin.

This is an offline reference-pose zero-intersection diagnostic checkpoint. It is **not physically admitted** and does not change the native scene. Matching skin/fat volume, conservative mass/ancestry transfer, motion, and native mechanics are still required. Passing surface checks does not establish anatomically valid tissue thickness or physiological behavior.

## Method

Reconstruct the connected skin envelope using the existing pinned skin and a conservative polyhedral Minkowski envelope around the 18 intersecting/adjacent registered bone banks. Temporary head caps permit the Boolean construction and are removed exactly. Preserve constructive face provenance, all unchanged outer source triangles, and exact Float32 emission. A 0.2 micrometre simplification removes Boolean slivers; a final exact-predicate lattice search shifts two derived vertices by one Float32 ULP (59.6 nanometres each) to remove the measured seam folds. Original source points remain fixed by that seam repair. All 6,501 changed/removed source-face rows lie within a conservative 36.71 mm bound of the original crossing witnesses; the 13,425 definitely distant source faces remain bitwise exact.

## Inspect and reproduce

Run `python verify_checkpoint.py` as the single checkpoint integrity entrypoint. It must report `ANATOMY_REJECTED`. `python verify_checkpoint.py --require-anatomy` must fail. No acceptance path may interpret this checkpoint as admitted skin. Start with `qualification.json`, then the independently checked reports in `audit/`. `candidate.npz` uses the existing `points_world_m` and `source_O_faces` fields and stores constructive source original IDs and face IDs. `offsets/` retains the registered envelopes; `inputs.json` pins inputs and owning revisions. P205 is retained in the linked immutable Numi Lab archive. `failed-trials/` preserves the unsuccessful continuous fits.

All commands ran on SSH Numi. Source snapshots in `tools/` retain their original prepared-input paths; they require the pinned P206 prepared inputs listed in `inputs.json`, the existing exact audit owner, NumPy/SciPy, and offline authoring dependency `manifold3d==3.5.1`. The working source entrypoint is `/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/repair_envelope_stable.py`; then run `repair_quantized_seams.py`, the exact bone/self audit snapshots, and `query_final.py`. The audit CSV rows are compressed losslessly and hashed before compression.

The Float32 emission regressions pass with `python -m unittest -v test_emit_f32` from `tools/`. No visual inspection or native physical run was performed in this checkpoint. See `ATTRIBUTION.md` for asset and authoring dependency rights.

## Rejected topology experiments

A tree/cotree loop workflow reduced the main genus from 42 to 1. The resulting candidate014 still has 169 exact self-intersections, despite passing the complete 185-bank surface check. It is rejected. The fill screen found no inter-ray bridges among the 18 offset samples, but chain identities are provisional and some joint-spanning fills extend over 23.82 mm. These observations do not establish anatomical or motion validity. Whole-bank hulls, discrete directional closing, and local SDF closing were also rejected after failing topology or locality checks. They are retained privately with deterministic logs and are not promoted as replacements.

`tools/topology_check.py` is the new regression guard. It compares component count, Euler characteristic/genus, orientation, coordinate duplicates, face degeneracy and exact boundary bits against the baseline. The checkpoint verifier checks the independent topology report as well as every content hash.
