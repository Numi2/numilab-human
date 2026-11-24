# Registered bed plane and coupled release on Apple M6

The physical contact witness, material-section projected area and loaded skin reconstruction now use the same plane normal and origin. The measured bed inclination is 0.143974°. The integrated regional scene uses the source-registered skeleton, 34 ligament rows, 26 fascia regions, the vascular wall, variable pleural pressure, cardiac and gas coupling and 32 bed sections. The inferred mass partition closes to 1.90 mg at 97.13195 kg. Regional bed sections carry normal skin and fat compression. Distributed skin and fat FEM is a separate, denser model.

The native material-witness selector passed flat and tilted bed tests for both current and predicted positions. The GPU reconstruction check passed 48 cases on M6. The horizontal control matched bitwise in all 16 cases. The maximum component discrepancy across all cases was 27.1 nm.

| Coupled native release | Simulated time accepted | Wall time including launch | Termination |
| --- | ---: | ---: | --- |
| Standard 2 ms first step | 0 | 17.08 s | Physical residual rejected |
| Vascular block and cavity preconditioner, 2 ms | 0 | 16.73 s | Physical residual rejected |
| 8 steps at 0.125 ms | 1.000 ms | 32.74 s | Normal exit |
| 48 steps requested at 0.125 ms | 1.250 ms | 38.50 s | Eleventh step rejected |

Every accepted sample in these releases had zero external support impulse. The initial closest material surface point has 52.25 µm of clearance. Lowering the body enough to carry the full normal weight in the fixed release witnesses produces 255.7 N·m of moment about the center of mass. A three-degree-of-freedom calculation can balance normal force and two moments with these witnesses held fixed. It leaves joint balance, contact winner reselection, static friction and motion to the native solver.

The exact source delta in native-owner.patch.gz was applied to the sources in source-identity.json and all four resulting file hashes were checked against the compiled source. joined-recovery.json identifies the independent GitHub recovery of native sources, executable, shader libraries, scene settings and failed trials. joined-inherited.json resolves duplicate files. The native diagnostics are preserved in the established formats.

On SSH Numi the eight-step command is

```sh
cd ~/Work/numi-human-supported-shoulder-085/force-route-086
./run-supported-scene --launch connected-forearms-087/registered-eight-source/launch.json
```

The coupled release does not qualify supported settling, sustained throughput, or crossing-free motion. The sixteen-second regional control and dense skin/fat construction have separate receipts.
