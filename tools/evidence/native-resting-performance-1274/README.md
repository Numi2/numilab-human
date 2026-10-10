# Native resting performance measurements, 10 October 2026

The coupled 14-second viewer runs completed on the SSH Mac mini. Existing kinematics caching, first-SIMD contact sweeps, and one-pass ordered limits preserved the observed numerical trajectory but provided no meaningful measured gain. They remain disabled. This is not a 1x result and does not close the passive-anatomy defects.

The retained 310-second pair measures approximately 0.114x real time. These short instrumented comparisons measure about 0.104x at the launcher-owner boundary, including initialization and captures; they are not directly interchangeable benchmarks.

| GPU stage | Control median ms |
|---|---:|
| Contact / joint finish | 7.950396 |
| Kinematics | 1.2577505 |
| Constraint prework | 0.8395205 |
| Mass factor | 0.718771 |
| Projected responses | 0.524979 |
| Muscle mechanics | 0.494000 |
| Equality factor | 0.3723335 |

Four samples per stage were taken on different physical steps around 10 seconds. Absent stages are not measured zeros; sampled medians are not a complete additive profile. The physical step is approximately 2 ms. The finish counter at settled root 5000 records 15 active contact decisions per sweep and 64 sweeps; the three-contact count belongs to startup.

| Candidate | Measured stage cost ms | Decision |
|---|---:|---|
| Kinematics prepare + cached consume | 1.2802705 | Leave disabled |
| First-SIMD finish | 8.0128125 | Leave disabled |
| One-pass ordered-limit finish | 8.124125 | Leave disabled |

Each comparison retained 875 byte-identical coupled samples plus byte-identical COM/support/surface CSVs against the matched short control. The profiling-only build also produced byte-identical accepted geometry packs. This establishes equivalence at those observations, not every internal state.

Separate one-second startup probes at root 100 measured vascular phase 1.823750 ms and gas exchange plus anatomical-coordinate coupling 0.566416 ms. The twelve isolated dense-45 dispatches are a subset of vascular work and must not be added to its whole-phase timing. These startup values are not settled measurements. Physical coupled fields match at 62 shared steps; ten solver summary fields differ with observer segmentation.

The actual native window was reported on-screen during the short control/profile runs. Remote desktop screenshots and Accessibility interaction were denied, so no manual UI interaction or desktop screenshot verification is claimed. Native framebuffer recordings remain retained. This does not retroactively make the earlier locked-console 310-second runs on-screen.

## Correction to the earlier viewer observation

The earlier matched-pair-011 README said footer text was clipped in reviewed lung/final frames. Direct reinspection of the retained baseline lung and final images shows the complete footer; that observation is superseded. The lung image is byte-identical in both arms. Baseline final SHA256: 61dee771336f017cfa5049bb86f122bf06bfb3e918a9ac613bfa7962ada3cccd. Lung SHA256: f4da7056af25e800e3b49b8da382d32009e55e2012b33adf268f37f5d6f21bcb. No footer code change was justified. Historical bundle files remain unchanged.

## Retention and boundaries

Exact source/build/asset hashes, command lines, environments and execution results are in each frozen declaration and receipt. Profiling source commit: 4e0be3dfbea8b1d81478812f5fefcf29d0102b05 in Numi2/numi-lab. Device: Mac mini, Apple M4 Pro, 24 GB unified memory, macOS 26.6, Xcode 26.6. Historical preparation and launcher scripts are retained as text; original runnable paths remain on the Mini.

Large movies, geometry and raw traces remain at the absolute Mini paths in external-artifacts.json, not hosted by this repository. Failed gas startup preparation (invalid capture cadence) is retained and not counted as an executed result. Input pins stayed unchanged in successful runs.

Storage deduplication preserved all paths and hashes, replacing only verified duplicate completed geometry outputs with hardlinks. The manifest records about 6.28 GB of recovered allocation. No sole source assets were deleted.

The full requested deliverable remains open: anatomy/intersection qualification, a faster accepted integrated trajectory, and validation of new optimizations through the breathing/intervention cycle. Short numerical parity cannot establish physiological or clinical validity.

Some historical declarations copied the parent 011 source_provenance and runtime_launcher_chain fields without replacing their ancestry labels. Those inherited fields are not the executing profiling build. For the new-build runs, the actual owner_cli_preview native path and immutable_assets hashes identify /Users/n/numi-lab-performance-1274-build-001; the profiling source commit above contains the compiled change. Viewer-baseline-002 instead uses the prior1271build. The next adaptive declarations explicitly separate inherited ancestry from executing provenance.
