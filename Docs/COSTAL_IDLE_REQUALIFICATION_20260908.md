# Costal callback qualification without an observed NumiVivo job

The current source-compliant costal fixture completes eight accepted roots and
one exact rejection/retry within the unchanged **60-second per-callback deadline**.
The clean test passes in 275.104 seconds; the external runner takes 277.393 seconds.
This replaces the current fixture's missing successful deadline evidence, while
the earlier failed and interrupted runs remain historical observations. It does
not establish why those earlier runs timed out.

The [receipt](media/costal-idle-20260908/receipt.json) pins native
`1ca086546e81b00251bed0d4eeef48e8e9ecca92` and Brain
`30f138fa17378fce788701edb5c7d834ee3c7c69`, both clean. Execution used the Apple
M4 Pro on `ssh macmini`, with Metal API validation. The source-default NumanX v6
configuration contains NHEQ2, all 122 NHLIM1 limits, NHTBIND/NHTMASS costal ownership,
46,278 tetrahedra and 2,871 attachments. It uses the previous NHCNT1 support asset
and source initial pose, not the prepared NHCNT2/NHINIT1 stance.

Nine candidate tokens have generations `1,2,3,4,4,5,6,7,8`; the fourth-generation
retry token is bitwise equal to the rejected candidate. All nine native Human
and Matter outcomes report success. The eight accepted roots represent only
**80 microseconds**. Whole-run independent replay, prepared anatomical tissue
loading, sustained motion and calibration are separate requirements.

Before launch and every approximately two seconds during execution, the runner
checked for NumiVivo `md-run` and `md-benchmark` processes. No such process was
observed; the complete monitoring series is retained. This is an observed workload
boundary, not a claim that all background GPU activity was absent.

Three CPU samples show the XCTest thread waiting for the native physical
callback while a Metal submission thread is in the driver. The first sample is
already at the second root; subsequent samples reach the rejection candidate and
later accepted-root loop. These samples locate host waits but do not identify a
specific GPU kernel bottleneck. The unchanged deadline and all physical admission
tolerances remain in force.

The samples report early process footprints of 1.2G with peaks of 1.5G/1.6G, and
a late footprint of **9.0G with a 9.3G peak**. These are macOS process-footprint
display values, including driver and test allocations. They are not the native
retained-byte counter and do not yet prove an allocation leak. Memory lifetime,+driver overhead and solver throughput require dedicated profiling before any
performance claim.

The executed runner's automated sampling missed XCTest's separate process group;
the three manual samples target the verified child process. Both that exact runner
and the corrected future runner are retained. The future runner tracks descendants
for sampling and cleanup, so an interrupted job cannot leave a detached XCTest
child running merely because Swift assigned it a different process group.

```sh
python3 Docs/media/costal-idle-20260908/verify_receipt.py
python3 Docs/media/costal-idle-20260908/test_verify_receipt.py
```

The remaining completion work includes loaded compliant equilibrium, anatomical
registration at the prepared mass frames, tissue calibration, support-aware
control, sustained standing/walking and measured performance. The bounded costal
deadline pass does not promote those scopes.
