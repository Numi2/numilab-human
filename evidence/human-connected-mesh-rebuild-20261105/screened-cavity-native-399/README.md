# Screened fixed-boundary cavity batch: failed native comparison

Human-specific preparation and evidence belong to Numi2/numilab-human. The
shared Matter constitutive/contact fixes remain in Numi2/numi-lab at 1c13906d8.

A batch of 244 disjoint, fixed-boundary skin/fat cavity retessellations preserved
source coordinates and removed 68 tetrahedra. Both the control (395) and the
candidate (399) completed 32 roots, totaling 0.5 ms of initialization, with the
same binary, runtime dependencies, timestep, solver controls and physical gates.
The candidate is **not promoted**.

| Measurement | Unchanged control | Cavity candidate |
|---|---:|---:|
| Body wall time | 209.195503 s | 235.095943 s |
| Active solves | 77 | 102 |
| Active Krylov columns | 1,232 | 1,632 |
| Budget-exhausted solves | 0 | 0 |
| Body real-time factor | 2.39011e-6x | 2.12679e-6x |

Candidate throughput was 11.02% lower (wall time 12.38% higher). The endpoint
comparison has 148 captured force metrics beyond 1% and 228 of 282 regional
strain metrics beyond 5%. These are metric counts, not defective cell counts.
Some reference signals are small; result.json retains absolute witnesses rather
than treating relative errors alone as proof of large physical deformation.
The maximum matched nodal displacement difference is 1.7072 micrometres.
That small displacement difference is amplified locally: on unchanged cell
951811 (candidate row 951378, region 23), the absolute Green-strain difference
is 0.8173. The unchanged-cell volume-weighted RMS difference is 7.55e-6. Neither
a whole-body average nor a small displacement difference clears that witness.

At the control displacement imposed on the changed topology, all regional
strain metrics pass 5%; the largest change is 1.404%. Actual native trajectories
nevertheless fail. The retained-field screen therefore does not establish mesh
fidelity or predict nonlinear/contact trajectory agreement. The two accepted
paths are evidence of execution, not evidence that the candidate is equivalent.
Total FEM mass differs by about 9.44e-11 kg in captured output; this does not
establish invariant local mass distribution.

No visual verification, sustained motion, supported resting state, complete
breath/heartbeat, energy admission, full force-owner coverage, or 0.1x performance
is claimed. The unchanged native mesh remains the comparison authority. The raw
captures, plans, exact preparation/comparison sources, proposal and failure
witnesses are retained in a separately verified Human release archive.

Recovery: [raw native evidence](https://github.com/Numi2/numilab-human/releases/download/numi-human-20261105-intermediate-recovery/human-screened-cavity-native-395-400.tar.zst).
All 84 members were independently downloaded, restored, and checked by size
and SHA-256. Sources and captures remain on Numi; only temporary archive and
download staging was removed. The 278 MB raw archive is outside Git history.
