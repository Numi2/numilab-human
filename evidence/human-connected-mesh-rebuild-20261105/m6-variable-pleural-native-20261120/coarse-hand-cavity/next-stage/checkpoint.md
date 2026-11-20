# Native whole-body release and Newton budget comparison

2026-11-20 on SSH Numi, Apple M6. This is a 1 ms native GPU release test at a 15.625 µs root step. The subject has the terminal-contact coarse skin/fat shell, registered tendon and ligament rows, cardiac mechanics, circulation, variable pleural coupling and the Matter-owned flat bed. All runs retain the source geometry, material laws and normal acceptance gates. Launch receipts record the exact inputs. The coarse shell is distinct from the fine registered skin/fat repair.

| Trial | Accepted roots | Process wall time | Native segment wall time | Outcome |
| --- | ---: | ---: | ---: | --- |
| Paired control, 24 Newton passes | 8 | 34.934 s | 19.988 s | All steps accepted |
| Fixed 14 Newton passes | 8 | 31.913 s | 16.892 s | Diagnostic pass over 0.125 ms |
| Profile control, root 2 spans | 4 | 25.231 s | See comparison | Instrumented |
| Extra completed-cycle suppression | 4 | 25.247 s | See comparison | No material total gain |
| Full-prefix control, 24 Newton passes | 64 | 177.751 s | 157.726 s | All steps accepted |
| Staged 4/8/24 Newton passes | 64 | 125.457 s | 109.657 s | All steps accepted, regional field comparisons failed |

The staged trial used the same first-root budget and then four Newton passes for all 63 later accepted roots. Eight and 24 remain guarded fallback budgets. Neither fallback fired. In the retained implementation the caller checks that rejected trials leave accepted body, Matter, circulation, tendon, controller and presentation history byte-identical before retrying. This experiment did not exercise that path. The staged binary left the same 14,439 FEM nodes and 55,051 cells in the native test. Its 28,632 skin/fat tetrahedra had minimum determinant ratio 0.649406, with zero nonpositive cells at the endpoint. Exact nodal mass and connectivity comparisons passed. The terminal ligament certificate passed all 34 registered rows and no legacy force rows were active.

The 64-root candidate reduced native segment wall time 30.5%. Process timing additionally includes unequal preparation diagnostics, so native segment time is the cleaner comparison. Active columns changed from 713 to 608 with no linear budget exhaustions. The staged test is retained as an opt-in experiment. It failed 94 regional force-channel comparisons at the 1% gate and 126 regional strain metrics at the 5% gate. The largest resultant difference among failing channels was 0.0540 N. Spleen maximum Green-strain norm changed from 0.73683 to 0.80754. Accepted node positions differed by at most 2.02 µm. Small changes in global trace quantities do not overrule local field failures. Blood volume differed by at most 0.000466 mL and pleural pressure matched the control.

The fixed-14 control pair took 8.65% less process wall time. The only coupled trace differences were blood continuity bookkeeping up to 2.23e-9 mL. Max articulated linear velocity component difference was below 7.13e-12 m/s. No endpoint force or strain capture was requested in this eight-root comparison.

At the instrumented root 2 the full profile measured assembly 142.362 ms, linear 674.017 ms, globalization 136.837 ms and final certification 8.419 ms. The root used six Krylov columns and two active Newton starts. Extra suppression reduced the sampled linear stage by 48.629 ms. Root 1 used 56 rather than 22 active columns for that variant. Overall wall time did not improve. These encoder-split samples do not establish whole-run GPU stage shares.

The staged trial's process ratio was 0.00000797×. It began unbalanced, carried zero bed reaction, and saw pleural pressure change by a single Float32 increment. This supplies no 16-second supported motion, full pressure cycle, zero-crossing or end-to-end throughput qualification. Keep the normal budget for accepted reference work. Reproduce the fast-versus-reference comparison using the launch receipts, build patch and comparison scripts. The open numerical question is why the smaller work budget changes the local accepted force and strain fields despite passing native terminal certification.

Closed replay packs with the same byte hash and size are stored as APFS clones. Reopened original paths verified the same hash. The two storage receipts document the recovered allocation.
