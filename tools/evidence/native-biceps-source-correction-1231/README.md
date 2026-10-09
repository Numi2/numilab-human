# Biceps source-correction smoke and binding-path fix

This package preserves the proof-bound two-row biceps source correction, the closed 10 s whole-scene native smoke, its two accepted geometry captures, and the focused follow-up that fixes relative parent paths in the receipt binder. The native run used the executed composer snapshot. The later path fix did not alter the already closed run.

The candidate changed only stable IDs 103 and 104, the paired long heads of the right and left biceps brachii. It is a bounded inferred source-geometry correction, not measured anatomy. The source composition retained bindings, weights, topology, and other rows. Its accepted-pose forward status before the native run was not_run.

The full native scene completed 5,000 accepted steps / 10.000000474974513 simulated seconds, with exit code 0, no changed source files during execution, and no root assistance. Owner-wrapper wall was 97.2349 s; native wall was 96.6810 s. These are distinct wall measurements. The two closed MRVPACK captures and movie remain external and hash-pinned.

The candidate's 60-column coupled physiology CSV matches the baseline prefix byte-for-byte for 625 data rows plus the header (626 raw lines). This covers the 10 s smoke only; it does not qualify later responses or the 310 s resting scene.

Exact Float32-quotient self audits report both surfaces closed/oriented, nondegenerate, and with zero unallowed self-intersections at authored candidate source, accepted step 0, and accepted step 5000. The audit covers only self intersections of surfaces 103/104. It does not test their cross-intersections, skin-target clearance, whole-body anatomical qualification, or the remaining resting geometry.

Two composition/preflight failures and two audit preflight failures are retained. Attempt 001 composition failed and is preserved. Attempt 002 is retained with its scope correction: it was incompatible with the intended current parent receipt for this run, not declared intrinsically invalid. The audit preflight failures stopped before geometry evaluation.

After the closed run, bind_anatomy_receipt was corrected to resolve a relative parent payload from the source receipt directory. The same focused change limits the new duplicate-binding rejection to the biceps-specific operation and checks any optional manifest path against the payload-inferred manifest. The exact executed composer and test sources are preserved separately from the updated sources. The focused test suite passed 21 tests on Python 3.13. No native rerun was performed after this binder-only source fix.

Reproduction of the focused suite:

    cd /Users/n/numi-human-free-apex-two-family-1178
    PYTHONPATH=src /Users/n/numi-human-prep-venv-20261005/bin/python3.13 -m pytest -q tests/test_passive_attachment_composition.py

This package does not claim 310 s acceptance, complete collision clearance, full cross-surface anatomy, or clinical validation.
