# Square-root row equilibration on the full tissue scene

The Apple M6 native run held the complete skin/fat, bed, ligament, vascular and respiratory inputs fixed. Both arms ran the same binaries and accepted the same three 15.625 µs roots. The only launch difference was `NM_FGMRES_ROW_SCALE_SQRT`. The default retains its original row weights. Square-root scaling is opt-in and requires the coupled row-equilibration path and the fresh unscaled physical residual certificate. The physical force, contact and rollback checks stayed active. This test did not change the model's accepted timestep.

| Measure | Original weights | Square-root weights |
| --- | ---: | ---: |
| Active Krylov columns across three roots | 784 | 714 |
| Active Krylov columns on root 1 | 240 | 176 |
| Solves exhausting their budget on root 1 | 4 | 0 |
| Profiled root 1 linear GPU time | 9.176 s | 7.451 s |
| Three-root body runtime | 44.783 s | 42.949 s |
| Full process wall time | 184.974 s | 185.161 s |

Linear GPU duration decreased 18.8% on the sampled root. Body throughput across this short prefix improved by 4.3% to roughly 1.09e-6×, without a whole-process gain. Profiling was enabled. The prefix qualified neither settled throughput nor native presentation.

Accepted CSV traces matched at 214 of 231 measured values. All 17 differences are in blood-volume accounting or articulated equality diagnostics. The largest absolute differences in those channels were 6.662e-10 mL and 6.731e-11 generalized velocity units, respectively. Fifteen differences exceeded the strict relative comparison against values near zero. The comparison preserves these differences and the absolute witnesses. Regional force and strain equivalence is unqualified. The square-root branch remains experimental.

The full-tissue contact scene still released without a settled initialization. Both runs reported zero terminal bed reaction despite nonzero gravitational load. Pleural-pressure variation was enabled and remained flat over three roots. There was no root assistance. The source capture is retained by hash in its compressed canonical owner.

## Reproduction

The exact parent overlay, including the row-transform option, is published in `../fgmres-convergence-and-row-domain`. The build script replays the six existing compile and link commands on SSH Numi. The launch and execution receipts in this increment pin the inputs and binaries. Run the two invocations sequentially from this experiment directory.

```sh
python3 prepare_tempered_row_scaling.py
python3 run_native.py --weight sqrt --physical-trace --column-trace square-root-scaling
python3 run_native.py --weight original --physical-trace --column-trace matched-original
python3 check_row_weighting.py --reference matched-original square-root-scaling
```
