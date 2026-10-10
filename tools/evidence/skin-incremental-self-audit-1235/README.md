# Incremental exact skin self-audit

The existing multi-pose skin fitter now reuses exact pairs belonging entirely to unchanged faces. It audits every changed face against the complete candidate surface, canonicalizes global face-row pairs, and uses the existing exact intersection predicate. Complete baseline/resume checks remain; a rejected trial cannot replace the accepted cache.

On the retained step-155000 A4-to-d319 source transition, the production API took 3.701 s versus 36.541 s for the complete candidate self scan. Both returned the same empty pair table. The change touched 2,567 vertices and 5,833 faces; 774 pinned inputs remained unchanged during that comparison. Timing is offline anatomy checking on the Mac mini with another CPU fitting job active, not native simulation real-time factor.

The measured API was loaded from source SHA256 be37104adf7b99c310b8c27043dba17422eeaca84079d9a12a64170256991977, before the fitter callsite integration. Its exact bytes are retained here; the source file at the original worktree path subsequently changed. The helper body is identical in the integrated version. The current callsite is covered by the 64 focused tests recorded in validation.txt, including a complete-owner comparison across rejected backtracks and an accepted partial correction.

The prior prototype report incorrectly stated A4 forward geometry was bitwise equal to the native capture. The original is preserved alongside its correction. The production comparison independently reports the true difference: 7,098 Float32 scalars, maximum 0.697354 mm. A4 and d319 are inferred source corrections, not the captured source itself.

This is not anatomy admission: d319 still has 2,521 skin/target crossing pairs across the 17 saved poses. It is not a new native run, cardiac or gas-exchange qualification, whole-body intersection clearance, or a simulation speed measurement. The active fit remains on its earlier frozen owner. No physical forces, posture, bed, assets, or runtime state were changed by this increment.

COPY-MANIFEST.json and SHA256SUMS bind this compact copy. Full raw assets and the 774 external input pins remain in the Mac mini retained evidence paths listed in the comparison report. The executed source snapshot is reconstructed from the unchanged base plus helper insertion and verified against the exact execution hash.
