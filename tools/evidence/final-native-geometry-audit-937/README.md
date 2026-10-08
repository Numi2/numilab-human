# Current accepted-scene geometry audit

This adapter applies the retained 908 exact Float32-lattice skin/self/859-target audit to the assets and capture IDs recorded by a completed native owner run. It uses the existing intersection predicates and accepted-pack reader without changing them. The same registered source topology is checked for all 524 NHA structures; historical inventory face counts are not treated as current geometry.

The input run must have successful owner metadata, unchanged asset hashes, matching invocation/metadata environments, an unassisted native completion at the exact Float32 timestep, and valid initial/submission/true-terminal capture IDs. Every pack hash, source-anatomy binding and accepted timestamp is checked before scanning. Incomplete runs, changed inputs, missing accepted captures, off-cadence IDs and invalid geometry remain failures.

On the SSH Mac mini:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
      /Users/n/numi-human-prep-venv-20261005/bin/python ./audit_current_scene.py \
      --run /absolute/path/to/completed/owner/scene \
      --out /Users/n/numi-human-resting-evidence-20261005/fresh-geometry-audit

The main command returns 0 only when all requested captures have complete coverage, no invalid triangles, no skin self-intersections and no skin-to-target crossings. A nonzero clearance result retains the raw triangle-pair witnesses. The internal per-step command returns 0 for completed coverage even when crossings are found; only the main summary determines clearance. The --preflight-only option validates receipt and source topology but does not run intersection predicates.

The adapter retains the existing 908 schemas. Legacy hash-input names in the imported run_step function are bound to the current anatomy receipt; no 907 composition claim is made for a new skin. There is no new physical simulator, geometry repair or tolerance. All numerical physical and physiological qualification remains with the native owner and registered study.

Validation uses completed native 931 captures with current 924 anatomy and the older 907 skin. Thirteen failure/acceptance tests pass. The topology reader checks all 524 source structures in all eight captures, including true terminal 10000. The neutral-pose all-target audit reproduces the prior zero-crossing result. The moving-pose check is retained separately as a known failing anatomy case. These are reader checks, not corrected-skin or final-study acceptance.

Full pack files, per-target witnesses and topology rows are retained under the Mac mini evidence directory. Compact validation and input hashes are published here. The original v001 reader bytes remain available because the first preflight used that version; v002 added an inventory hash pin and the post-preflight immutability check.

This audit covers skin against the complete target inventory and skin self-contact at sparse captured states. It does not establish every pair of internal tissues or geometry between sampled states. Intended internal anatomical interfaces have separate source-bound audits.
