# Fine-volume replay initialization and clearance checks

The matched 24-root native replay still has no accepted step. It previously stopped at the disk watchdog during preparation. A follow-up on the same machine confirmed the recorded fine-volume skin/fat mass allocation is available and that the startup contact-history buffer has a separate, avoidable host allocation. This update leaves the original fine-volume replay, all inputs and capture paths available.

## Metal initialization

Matter now initializes empty accepted deformable contact history in the existing upload batch with a GPU fill. The typed buffers, row count and accepted-state lifetime are the same. The native fixture executed on Apple M6 with the retained 4,207,296-slot capacity, reading back all 201,950,208 bytes of each result. All six matched executions returned zero in every byte. The empty typed sentinel, three-environment allocation, both overflow cases and post-finish rejection passed.

The median isolated allocation and upload was 29.32 ms for staged host zeros and 6.01 ms for GPU fill. The sampled process footprint increase immediately after upload fell by 404,226,336 bytes (385.50 MiB). Both paths kept 201,950,208 GPU bytes. The exact owning translation unit also passed Apple Clang syntax checking against the published headers with two existing format warnings. These measurements cover just the initialization path. They do not measure sustained solver throughput or establish motion equivalence.

The Matter commit is in matter-commit.txt. Run the fixture using `bash run_probe.sh /path/to/numi-lab` with the checkout at that commit. The fixture uses the source implementations of both allocation paths in one native fixture, with separate processes for each timing.

## Offline bone clearance

The offline preparation check now screens Float32 endpoints independently against the pinned native skeletal triangles before using an endpoint as the continuation state. Its stop conditions retain the existing tissue and bed equations, force tolerance, cell shape floor, determinant limits and self-intersection audit. It reports changes to the reference pair set. Existing crossings still fail the full-scene geometry gate.

The refitted coarse shell balanced 0.953 N at 0.1% of full body weight, without additional skin/fat-to-bone pairs or invalid cells. At 0.4% and 1%, two new triangle pairs lay on the same skin triangle and bone bank that already intersect at the reference pose (outer row 2207, body 23, bone 32). Both trials replaced the pre-existing pair with bone triangle 2377 with pairs 1672 and 2045. This localizes one registration deficiency and distinguishes a change in the intersected bone triangles from a newly affected anatomical interface. The 0.7% and 0.9% ascending probes exhausted the solver budget. The source fit, material and mass were unchanged.

The coarse-shell state was used only offline. It retains prescribed deep normal coordinates, inherited crossings and a small fraction of supported load. The native replay does not consume this state.

## Storage and full-scene status

Transparent compression saved 283,103,232 allocated bytes from a generated registration map and two aliased historical influence exports. Independent data-fork size and SHA-256 rechecks passed before retiring the original allocations. Existing file paths remain usable. The machine still has insufficient headroom for the full fine-volume attempt.

The current full-volume reference has 1,448,804 cells and a 13.93967896346 kg skin/fat mass. The most recent completed native prefix retains 34 registered ligament rows. It accepted 0.250 ms of simulation, with no outer-skin self or bone crossings and 14,179 other nonocular skin-to-tissue triangle pairs, no supported equilibrium, and no resolved pleural-pressure variation. The required combined 16-second run remains open.
