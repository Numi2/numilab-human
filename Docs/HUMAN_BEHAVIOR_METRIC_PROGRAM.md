# Source-bound Human behavior metrics

`numi human-behavior-metrics` prepares an immutable `NHBHV1` metric-program
source from declared criteria, the exact Human reference manifest and its
`NHRIGID2` bytes, and the pinned MuJoCo source export. It performs no physical
steps and emits no trial results. This format does not replace the generic
TaskPack or establish its still-required lowering receipt.

```sh
numi human-behavior-metrics \
  --task config/behavior-producer-probe.v1.json \
  --human-manifest /path/to/myosim-fullbody-reference.manifest.json \
  --source-export /path/to/myosim-fullbody-export.json \
  --output /path/to/metric-source
```

The source catalog maps original names and MuJoCo IDs to the exact ordered
source records in the rigid payload. Every record's core index and source pose
must match those bytes. The original body-origin and inertial orientation are
cross-checked against independent source world poses. The native compiler then
subtracts its actual tissue-donor COM rebase from the root-origin point; an
atlas visual index or an assumed body number is never a semantic binding.

The included producer criteria are declared software-probe choices. The free
`Full Body` source origin differs from its massless inertial reference by one
metre, so the root-height criterion explicitly uses the source origin. The
trunk is `torso`, whose source local Y axis is up in its default pose. The
velocity observable is the `pelvis` body's COM. Whole-Human COM is unsupported
until the observable includes every participating rigid and deformable mass
owner with a verified nonduplicated partition.

Authoring records exact SHA-256 identities for source archive, rigid bytes,
Human manifest, MuJoCo export, authored criteria and semantic catalog. Repeat
compilation verifies existing output bytes; changed output is rejected.
Native compilation additionally binds the cooked body frames and exact clock.
There is no default conversion from unknown contact or audit coverage to zero.

The native producer measures paired high/low candidate geometry and source
body COM velocity on the existing owner submission. At the first begin-step it
also measures the exact reset pose and records `initial_posture_valid` and
`initial_settled` without advancing physics or time. Candidate measurements
remain private. Only an actual successful joint Brain–Human–Matter release,
with its exact `COMMITTED` publication fence, authorizes accumulation. The next
owner submission consumes pending records before candidate work; an explicit
final collector flushes through that same queue without a physical step or a
time advance. Rejected attempts add no accepted metric samples.

`numi.human.accepted-metric-snapshot.v1` is a privileged diagnostic snapshot,
not a Brain sensory channel or a
[`numi.human.behavior-trial.v1` qualification trace](HUMAN_BEHAVIOR_QUALIFICATION.md).
The current producer still reports unavailable forbidden-contact coverage,
native audit coverage and the complete accepted-root SHA proof schema. Its
`generic_taskpack_lowering=source_bound_metric_program` value records native
compilation of the source-bound metric program; the generic TaskPack lowering
receipt remains a separate required artifact. These gaps prevent full behavior
evidence admission. Compiler and reducer controls, or a short real producer
run, cannot establish the frozen 420-trial standing, recovery and walking
cohort.

Runtime qualification results and immutable owner revisions belong in the
current accepted-state precision qualification record; this authoring guide
makes no independent runtime-success or biological claim.
