# Thirty-four repaired surfaces in the native resting human

Bilateral sartorius and tibialis posterior repairs now operate in the existing full-body Metal scene with the [verified skin correction](../native-skin-clearance-1292/README.md). **Whole-body anatomy and final five-minute qualification remain incomplete. Performance work is deferred.**

## Reference corrections

Existing exact positive-winding boundary extraction removes folded exterior regions while retaining source-face ancestry for anatomical bindings. Rejected point-junction boundaries and unsuccessful source/posed trials remain retained.

The sartorius correction separates finite junctions, collapses sub-0.1-micrometre source features that fail in Float32 poses, and moves one right-side vertex by 100.002 micrometres along its source Y axis. Bilateral tibialis posterior corrections separate point junctions, resolve residual source crossings with two one-micrometre local edits per side, and move six posed-fold vertices by at most 62.173 micrometres through a bounded local smoothing proposal. These are explicit reference inferences, not measured anatomy or source accuracy bounds. Reported bounds apply to the named local stages, not the complete positive-winding reconstruction.

Original exactly one-hot source-point proxies are retained where present: sartorius has none under this proxy definition; tibialis posterior retains seven per side. That proxy check is not measured attachment fidelity. Binding tables, physical muscle routes, mass, force ownership and physiology remain unchanged.

## Verification

Each candidate is closed, nondegenerate and self-intersection-free at source and all 24 retained early/late poses. Changed regions add no source-lineage crossings against all 859 surrounding native structures. The native 16-second run then checks the actual GPU-deformed surfaces at eight accepted captures, using actual preceding native coordinates for the comparison. All four rows remain closed and self-intersection-free, with no new changed-region crossings. Existing internal interfaces remain unqualified, including right sartorius versus rectus femoris and tibialis posterior versus nearby bone/vessel surfaces.

All 856 other rendered surfaces, including skin, and all 157 accepted body poses match the preceding run exactly. All 1,000 rows and 60 columns of the coupled trace match exactly. This is geometry integration; it does not add physiological fidelity.

The retained native framebuffer recording contains 252 frames, five timing markers, and seven anatomical inspection layers on the SSH Mac mini. Launch inputs, source revisions, exact assets, traces and achieved timing are bound in this package. No remote desktop visibility is asserted.

## Reproduce

    /usr/bin/python3 tools/evidence/native-anatomy-thirty-four-surfaces-1293/reproduce.py

The pinned Mini runtime and retained assets are required. --prepare-only verifies launch inputs. Neck/back integration, remaining muscle and foot defects, and unresolved anatomical interfaces remain open. The older five-minute baseline/intervention evidence uses older anatomy and does not qualify this updated body. Provenance remains a mixed-source reference adult.
