# Native regional organ blood and tissue exchange, 14 September 2026

The Matter vascular runtime now has a native Human source admission for the
regional blood/tissue amount path. On the physical Apple M4 Pro it loads the
hash-bound 21-compartment, 24-connection CVSim21 source graph, adds seven
source-labeled beds (right/left lung, right/left kidney, stomach, pancreas and
liver), advects the source blood state, and executes bidirectional oxygen
amount exchange against the seven tissue reservoirs.

The run uses the canonical `12.5 us` clock for 512 attempted steps. Candidate
37 is deliberately rejected in environment 0; 511 steps are accepted there
and 512 in environment 1. The rejected candidate leaves the accepted state and
clock unchanged. Maximum relative residuals are `5.711629397e-7` for volume
and the derived blood-mass budget, and `1.057184875e-6` for oxygen amount. A
second run from the snapshot is bitwise identical.

The executable is part of `numi-matter-vascular-check` and can be reproduced
from the isolated source worktree with:

```sh
matter/numi-matter-vascular-check --human-regional \
  matter/tools/fixtures/cvsim21.native.v3.json
```

The immutable native log is
[`native.log`](media/native-human-regional-exchange-20260914/native.log) with
SHA-256 `b1bb71ad008dc2e74b117c5a9d1838962bf5a8aca042fb5ea2954463e3149e8a`.
The source commit is `260ee02ad347a3cfb2281d6333157d084f5ba17b`; the native
binary SHA-256 is
`dd4663054a9fc52b52dfff082a8afa1d7895180ac4527fddcaccaf8d205e4fa0`.
The structured receipt is
[`receipt.json`](media/native-human-regional-exchange-20260914/receipt.json).

This closes a native source-graph amount/conservation and rejected-step
rollback subgate. The density (`1060 kg/m3`), tissue volumes, and exchange
coefficient are explicitly engineering candidates. The result does not admit
anatomical vessel tubes or capillaries, physical tissue-volume or mechanical
blood-mass ownership, organ mechanics, material or subject calibration, or
standing/walking behavior.
