# Current motion verifier v15 compatibility

This increment publishes the frozen v15 motion verifier and its direct-O observer-route compatibility tests. The route script changes the verifier path from v14 to v15; no native execution or geometry was performed by this package. The previous v14 verifier remains in the existing evidence package and is unchanged.

## What the verifier checks

v15 accepts the existing legacy observer contract and the direct-O observer contract. For direct-O receipts it checks the observer and route schemas/status, direct-O readback agreement, disabled legacy detail bridge, native and GPU execution flags, zero exit status, and exact installed domain counts. The signed oriented-face check uses the previously audited material convention O=+1, I=-1, D=-1 and requires a face bijection while allowing face-row permutation and cyclic corner rotation.

Those orientation signs are bound to the audited 1022 shell/source convention. A future motion qualification must freshly bind the verifier to the exact current source, shell receipt, and geometry evidence; this compatibility package does not establish that binding for a new run.

## Validation

The published copies pass four route-contract test methods, including 16 negative input variants, the verifier self-test, and shell syntax validation. Original controls and the fresh publication rerun logs are included under `evidence/`.

## Limits

These are verifier and route compatibility checks only. They do not qualify a native motion run, tissue motion, physical admission, or new geometry. The recorded controls use `native_steps: 0` and `physical_admission: false`.

The shell route and verifier retain their exact Numi host paths and pinned dependencies, so this is source/evidence publication rather than a portable installation.
