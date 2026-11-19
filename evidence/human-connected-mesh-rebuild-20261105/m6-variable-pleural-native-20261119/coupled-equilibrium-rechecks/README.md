# Coupled equilibrium and whole-body contact rechecks

Apple M6 Mac18,5, macOS 27.0 build 26A425. The native binaries and inputs are SHA-256 pinned in each launch. The existing regional build was based on Numi Lab checkout commit `f75955f08d6edf0f85027eac7614017ac9fbc996` with its recorded source overrides. No production code, assets, material law, acceptance threshold, physiology source, or joint-range policy changed.

| Regional trial | Accepted time | Process wall time | End-to-end factor | First rejected root or candidate |
|---|---:|---:|---:|---|
| Connected contact clock | 8 ms | 139.93 s | 0.00005717× | Root 4, microtick 12, residual 0.00346204 |
| CCD-clearance line-search gap | 6 ms | 114.29 s | 0.00005250× | Root 3, microtick 253, residual 0.000715704 |
| CCD clearance and compensated planes | 6 ms | 114.86 s | 0.00005224× | Same root, microtick, and residual as CCD clearance |
| Reset the shared fraction per Newton direction | 8 ms | 139.22 s | 0.00005747× | Same root, microtick, and residual as the connected clock |

All four failed Matter equilibrium status 10. Each used the existing 2 ms outer clock. The accepted controller timestamp remained at the final accepted time in each failure. The accepted pleural pressure changed from −490.64 Pa to −492.15 Pa in the 8 ms runs. These were regional experiments with the condensed 32-section bed support, 26 fascia regions and the registered ligaments. The two CCD trials wrote their failure receipt to the same launch-declared path. Each run's stderr summary, timings, and accepted traces are retained separately.

The full-contact launch requested 16 seconds with physical skin/fat, organs, optic nerves, aortic wall, and variable pleural pressure. It rejected before step zero during physical first-touch preparation. The loaded skin is `f554e6de…`. The compiled candidate registration requires `4fd80f44…`. The source comparison verifies different index streams and 72 kg versus 97.13195 kg launch reference masses. The prior verified registered-volume sidecars belong to the latter reference. Reassigning the payload hash would leave donor frames and embedded boundary mappings unverified.

The loaded terminal configuration from the previous condensed-bed control also failed as a fresh start. The exact pose exceeded an ankle source limit by 0.194 µrad. An explicitly authored variant projected 12 scalar values to the declared limits (largest 1.552 µrad), passed all 118 enabled joint checks and reached native first-touch alignment after adding a clearance offset to the input height. It failed nine ligament geometry checks. Interpolation by 25% and 12.5% toward the loaded pose still failed two straight reference routes, the superior glenohumeral capsuloligament and tibiocalcaneal ligament on side 0. The existing checks stayed active.

The last retained 16-second geometry is the separate condensed-bed run at 0.09937× across launch and export. At its final capture, SHA-256 `aa013b6d4c05a4504c08b03ef5ed91651fec68dff5ec2bbdad31d0e0445185d6`, the exact Float32 audit counted 14,207 skin-to-tissue triangle pairs plus 45 skin self pairs. Signed local depth is an approximate screening measurement. The independent exact audit rejected the largest muscle-bank proposal. It reduced that bank from 775 to 551 skin crossings while introducing 1,720 neighbor or self pairs. The topology-only screen left all 45 late skin self pairs. Neither edit was installed.

Full-tissue verification remains unfinished. The current skin requires joint registration with its physical skin/fat volume and mass mapping. A near-settled initialization also needs anatomical ligament routes that remain valid in the loaded neighborhood. These prerequisites precede any credible full 16-second throughput result. Disk writes on Numi failed with roughly 116 MiB reported available, so the runs and publication used bounded RAM-backed scratch.
