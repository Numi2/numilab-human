# Native left temporalis anatomy increment

The integrated resting human on SSH numi (Apple M6) includes the repaired left temporalis, stable ID 164. All 157 prior tissue rows keep their exact geometry and bindings. Physical mass, muscle/tendon routes, flat-plane contact, Brain coupling and GPU physiology are unchanged.

This is source-derived passive reference anatomy with explicit skull and scalp repair patches. It uses the actual left source, not a mirrored right muscle. Its single head attachment does not model active jaw function. A 5.147 micrometre correction at the scalp cut and a 3.998 micrometre correction at the bone cut address retained Float32 reconstruction defects; these are inferred geometry changes, not source measurements. The prior bone cut removed one declared tiny cut-created shaving (22 faces, 8.53e-12 cubic metres). Temporary authoring ocular caps are absent from runtime geometry.

The first native attempt failed with three self-intersections at accepted step 8000. Its failed check and source candidate are retained. The local bone-boundary correction then passed a fresh 16-second native run with 8000 accepted steps: one closed positive component, zero self crossings and zero crossings with all 867 prior surfaces at eight actual GPU captures. All prior referenced triangle positions, all 157 body poses, and all 1000 physiological trace rows equal the parent scene exactly.

The 25-pose offline screen also passed. Its current eight captures use the prior complete scene; the other 17 are explicitly older cohorts, with an additional check against the registered right temporalis. These sampled checks are not a continuous collision proof.

The native recording includes the anatomical layer tour and synchronized measurements. End-to-end execution achieved 0.149x real time (107.100 seconds wall for 16.000001 simulated seconds). Performance remains deferred.

The launch was prepared while the final offline screen was running, so the pinned subset and declaration correctly say experimental/pending. The completed independent checks in this package establish this bounded increment without rewriting those historical inputs.

The full human deliverable remains unfinished: six back-muscle rows, inherited forearm interfaces, final five-minute matched baseline/intervention qualification, and existing physiological/force-closure limitations remain. Most muscle surfaces are passive; numerical consistency is separate from physiological plausibility and clinical validation.

Derived geometry retains Z-Anatomy attribution and CC BY-SA 4.0 in the subset manifest. BodyParts3D scene sources retain Database Center for Life Science attribution and CC BY-SA 2.1 Japan. This is a registered reference adult assembled from multiple sources, not measurements of one person.

With the retained hash-bound assets on SSH numi:

    python3 tools/evidence/native-anatomy-temporalis164-m6-1354/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-temporalis164-m6-1354/reproduce.py

The adapter uses the existing native owner and requires retained source assets. Runtime/source identities, asset hashes, command, configuration and timing remain separate.
