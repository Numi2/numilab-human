# Native right deep masseter coverage

The integrated resting human now includes right deep masseter 161 (FMA49004), sourced from Z-Anatomy object Deep part of masseter.r and registered with the existing head anatomy. Both deep masseters are present. They are passive anatomical representations; active jaw mechanics and certified attachment footprints remain unimplemented.

Three source-bound cuts reconcile its overlaps with temporal bone 30, mandible 35 and superficial masseter 159. The inferred 10 µm reserve removes 4.969% of registered source volume. Four source vertices receive an additional Float32 replay correction of at most 0.164082 µm. Original faces and generated boundaries retain explicit ancestry. The source-lift volume erratum corrects a zero-difference reporting mistake in the original offline report; signed-volume and intersection checks were independent and remain valid.

The fresh 16-second native run passed at all eight actual captures: one closed, positively oriented component, no exact self-intersections, and no exact intersections with all 865 other non-bed surfaces. Those 865 surfaces, all 157 body poses and the 1000-by-60 physiology trace are byte-identical to the prior left-deep scene. The continuous invalid-triangle/bed audit passes and the recording has 252 image frames with all seven inspection layers.

The existing composition owner also replays the left 162 row byte-for-byte to correct its inherited superficial 160 reconstruction metadata. No left 162 geometry changes. No MyoSim route, physical mass, forces, controller or circulation state changes. This remains the MTP0 parent configuration; the EHL/K2 sensitivity is a separate bounded diagnostic.

Reproduce on the Mac mini using pinned retained assets and runtime:

    python3 tools/evidence/native-anatomy-right-deep-masseter-1334/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-right-deep-masseter-1334/reproduce.py

Source rights and attribution are retained. This is registered mixed-source reference anatomy, not a measured individual. Remaining body-boundary and temporalis coverage work, and final five-minute qualification, are outstanding. Performance work remains deferred.
