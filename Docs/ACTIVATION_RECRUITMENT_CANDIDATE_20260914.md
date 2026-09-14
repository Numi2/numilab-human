# Source activation and recruitment candidate

The source-compliant preparation now has an independently executable admission
for its actual recruitment state.  It reads the frozen native transcript,
rechecks the source receipt artifact hashes, and binds all **416** route
activations, 416 fibre reference lengths, route forces, 128 generalized force
components and 18 support witnesses.

The admitted one-adult-male candidate has 237 nonzero routes, 47 routes at the
upper bound, and 179 zero routes.  It is therefore a bounded recruitment
solution rather than the previous uniform activation-1 diagnostic.  The native
search history is strictly decreasing, FP32 transport preserves the activation
vector exactly, the source generalized-force decomposition closes, and the
source equilibrium transcript remains within the pinned acceleration and force
residual gates.

This closes the source-bound recruitment hand-off.  It does not calibrate
activation to EMG or held-out force data, and it does not admit anatomical
supports/loading, a whole-body dynamic state, sustained standing, recovery or
walking.  The candidate remains `status: partial` so downstream runtime owners
cannot mistake offline recruitment evidence for behavioral qualification.

Reproduce it with:

```sh
PYTHONPATH=src python3 -m numilab_human.activation_recruitment_candidate \
  --output Docs/media/activation-recruitment-candidate-20260914/receipt-v1.json
```
