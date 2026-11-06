# Unchanged-geometry repeat 376

The complete native scene repeated trial 365 with identical declared geometry, executable, Metal library, solver controls, initial assets and 32 x 15.625 us steps. Only output paths and analysis bookkeeping differ. Existing resource guards remained active.

All 32 steps were accepted. All 388 captured force metrics and 282 regional strain metrics reproduce exactly; all 1,475,262 tetrahedra are identical and the actual Green-strain difference is zero. The matched-control verifier passes. This null repeat does not support run-to-run numerical variation as the explanation for trial 374's failed comparison. It does not prove general reproducibility across hardware, versions, longer horizons or other initial conditions.

Trial 374's 245-cavity batch remains unqualified and is not promoted. It passed native topology, density and mass/transport admission but changed the response beyond the declared fidelity gates. Its source mass is conservative while native lumped nodal masses change (maximum 0.423 g; L1 1.946 g); this and changed stiffness/conditioning are candidate mechanisms, not an isolated causal diagnosis.

Keep trial 365 and its compiler364 geometry as the diagnostic baseline. The earlier matched reference to 335 uses different numerical controls and cannot establish mesh equivalence. Further connectivity work must use quantitative current-operator and nodal-load comparisons before another broad native batch, rather than relaxing gates or repeating the same construction at greater scale.

Both 365 and 376 cover only 0.5 ms initialization. Sustained supported rest, real breathing/heart-cycle motion, material-energy fidelity and 0.1x end-to-end performance remain unresolved. Source code, failed evidence and raw recovery assets belong to Numi2/numilab-human; numi-lab is untouched this round.

Raw recovery: [20-file archive verification](../conditioning-repeat-recovery-378/human-conditioning374-repeat376-verification.json). Every member was independently downloaded, restored and SHA-256 checked. Local source files are retained.
