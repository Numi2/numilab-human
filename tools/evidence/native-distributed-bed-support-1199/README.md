# Distributed bed-support experiment 1199

## Result

This is a **failed terminal-anatomy diagnostic**, not an admitted support configuration. The completed native run advanced 10,000 accepted steps at 2 ms (20.00000095 simulated seconds) in 256.811 s wrapper wall time. It reported three breaths, 23 complete filling/ejection cycles, and zero root assistance at observed endpoints.

The 1199 terminal-only exact Float32 triangle audit scanned step 10000 across all 859 target surfaces. It found **2,601 nonocular skin-to-target crossing pairs**, zero ocular crossings, zero skin self-crossings, zero invalid target surfaces/triangles, and complete target-pair coverage. Exact skin-to-bed testing found **394 bed-facet intersection pairs**; the production query replay found six negative signed-gap vertices and a minimum signed facet gap of -2.369 micrometres, with no vertices outside the finite grid. These bed-interface intersections are reported separately from other target crossings and are not waived by unilateral support semantics. The result is not intersection-free.

Only step 10000 was geometrically scanned. The native run also retained step 0, but this audit did not scan the initial skin against targets. The evidence does not establish full-breath, continuous-time, or 300-second anatomical clearance.

## What was prepared and executed

The support payload contains 32 NHCNT1 contact records over 157 bodies and is 1,620 bytes. The source-rest partition assigned each of 54,949 full-weight skin vertices exactly once across 32 nonempty regions, with zero multiple assignments or omissions. The skin payload hash and Float32 bed grid were unchanged; the largest source-point fit error was 10.81 nm. The reported approximately 10.0015 micrometre minimum source-pose bed gap and the 1196 source-region-gap file describe source/rest partitioning, not dynamic contact clearance.

The 1199 bed lattice and complete bed primitive/index/instance data match the retained 1196 bed grid and the 1198 initial bed geometry. At step 0, the 86 registered body poses, body-state hash, respiration-state hash, and all 860 non-bed primitive records/index streams/instances/vertices match 1198. Root fingerprints differ, so this is not a claim of complete initial transaction identity. Terminal body poses later diverged.

The accepted-contact trace audit passed: 1,250 endpoint samples at an 8-step cadence; 40,000 rows for 32 contacts; nonnegative normal impulses; zero friction-cone violations; zero root assistance; and component-to-world impulse reconstruction error below 1.2e-16 Ns. The seven interior steps in each 8-step segment are not individually recorded, so this is not a per-step impulse-closure claim. Contact consistency does not override the anatomy failure.

The movie inspector retained 315 compressed image samples covering the 20-second simulation and visually checked four frames (initial, final, muscles, skeleton). Other layer frames are retained externally but were not visually qualified in that review. Presentation is separate from numerical anatomy checks.

## Audit-input recovery and retained failures

The run declaration and execution record are preserved. Execution recorded exactly two changed inputs during the run: the source-region report and its validator. They are nonruntime audit files. The exact launch-time byte streams were recovered separately as v1 files and hash-verified; the strengthened post-run v2 files are also retained. The external index rehashed all 266 declared asset pins: 264 match their launch hashes, the two audit-only inputs match the recorded post-run hashes, and all 26 native input pins match.

The preparation attempt that failed before writing candidate assets is retained alongside the successful preparation. Terminal-audit wrapper attempts that stopped before geometry scanning, and the first contact-trace adapter attempt, are recorded in failed-attempts.json. The successful scan and trace scripts are retained. Large native packs, movie, witness streams, full native log, source maps, and runtime assets are not copied into this repository; their exact external paths, sizes, and hashes are in retained-artifacts.json.

## Prior 1198 pose context

The companion 1198 diagnosis reports 3,727 terminal skin-target pairs and zero ocular/self intersections. It also records identical initial body/respiration state and all 86 body poses between 1191 and 1198, followed by terminal pose differences. It did not emit scalar joint coordinates and does not attribute the divergence solely to bed contact. The 1198 profile comparison is a run-to-run measurement, not a repeated controlled benchmark or single-kernel attribution. These records preserve the earlier hypothesis and its limits; they do not promote 1199.

## Reproduction

On the Mac mini, use the pinned Python environment and existing closed native run. The command below performs only the terminal geometry scan; it does not relaunch physics:

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 anatomy/audit-terminal-v2.py --run /Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199/native-run --out /fresh/output/path --nha-sha256 1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241 --capture-steps 0,10000 --scan --scan-only-step 10000

The audit wrapper reuses the exact 1198 predicates and permits only the two hash-verified launch-time audit-input relocations recorded in anatomy/launch-input-recovery.json.

## Source attribution

The surrounding Human scene is mixed-source reference data. Direct BodyParts3D 4.0 material requires the attribution "BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International" (CC BY 4.0). MyoSim myofullbody is Apache-2.0. The broader scene also contains references with separate terms, including Z-Anatomy CC BY-SA 4.0 and its inherited BodyParts3D CC BY-SA 2.1 Japan credit. See the repository [third-party notices](../../../THIRD_PARTY_NOTICES.md). This bundle does not redistribute upstream archives or conflate these licenses.
