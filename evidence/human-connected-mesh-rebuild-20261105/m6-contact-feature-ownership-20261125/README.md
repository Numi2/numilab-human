# Shared tissue contact on the Apple M6

At step 99, the first Newton correction reduced the pre-contact mechanical residual from 0.001723 to 0.00001236. Contact assembly raised it to 0.02807. Four different triangle pairs contributed the same edge pair. Two incident faces supplied each edge. The diagnosed fascia nodes were 1004 and 1017 on one side and 1176 and 1178 on the other. The first Newton assembly had zero active records involving this edge pair. The next had four, at a separation of 0.860 micrometres inside the 10 micrometre activation band.

The Metal change assigns one constitutive contribution to a shared, unmapped inter-object FEM feature. Incidence, exact feature node identities and topology generation determine ownership. All candidate triangles retain their swept-geometry and clearance checks. The extended variant also covers an identical vertex against the same triangle. Eighteen such duplicate classes contained 96 of the control's 143 active records before the first correction.

A second change checks currently inactive inter-object triangle pairs from the existing broadphase. It bounds the shared Newton correction when a separating axis would admit deep entry into the activation band. The physical timestep, material equations, support loads, determinant checks, final coupled residual and accepted-state transaction remain under their original owners. This predictor covers the existing candidate list of eligible reference-mode FEM surfaces. The ordinary collision checks still cover geometry outside its scope.

| Native M6 trial | Accepted steps of 0.125 ms | Outcome |
|---|---:|---|
| Reproduced control | 98 | Step 99 failed the fresh mechanical residual |
| Shorten the correction for already active reference contacts | 98 | Same witness |
| One constitutive contribution per shared edge | 109 | Step 110 failed at a different mechanical row |
| Shared edges plus prospective contact entry | 160 of 160 | Completed 20.0 ms. Timed integrated interval 0.00013357× |
| Shared edges and shared vertex-face contributions | 160 of 160 | Completed 20.0 ms. Timed integrated interval 0.00013008× |

At the original failure, prospective entry selected a common correction fraction of 0.60984. The following coupled residual fell from 0.001732 to 0.000702. The full edge-plus-entry run had zero reported body mass residual and zero root assistance. Across the first 98 common accepted steps, its relative L2 difference in bed reaction forces was 0.00653%. It exercised variable pleural pressure, 26 fascia regions and 34 ligament force rows.

Both the force and tangent components of duplicate contacts were tested on Metal. The fixture checks vertex order, scene separation, source generation, mapped features, grid nodes, the two directed material owners, repeated execution and byte-for-byte preservation of geometry. Exact final sources, binaries and matched native traces are in the recovery manifest. The compressed member archive was independently downloaded and every archived file hash checked. [Native source and evidence archive](https://github.com/Numi2/numilab-human/releases/download/human-historical-support-checkpoints-20261122/m6-feature-contact-20261125-mechanics-r1.tar.zst)

The 20 ms edge-plus-entry capture has 53 skin self-intersections and 171 skin-to-bone triangle pairs, all with the right radius. The complete skin-to-anatomy audit counted 11,762 raw pairs, including 350 ocular pairs. Individual anatomical interfaces still need classification. The regional material support path carries real bed loads, while the full detailed skin/fat volume has yet to qualify in motion. This interval contains zero full breaths and zero complete filling/ejection cycles. Sustained equilibrium, 16 seconds of clear motion, 0.1× throughput and presentation cadence remain open.

## Combined native replay

On SSH Numi, run `bash scripts/run_resting_human_on_numi.sh --check` to verify the 534 pinned inputs and the shared process monitor. Run `bash scripts/run_resting_human_on_numi.sh --current` to reproduce the 256-step case in a new output directory. The 16 and 60 second launcher arguments select 128,000 and 480,000 physical steps at the pinned 0.125 ms interval.

The combined feature ownership and prospective line search completed all 256 steps (32 ms) on the M6. The launcher took 307.355 seconds including initialization and final export, or 0.00010411×. The native timed interval measured 0.00010953×. Articulated position and velocity fingerprints and the eight compared physiological quantities matched at every one of the 252 steps also recorded before the separate diagnostic runner hit its wall-time limit.

The two heels carried 11.76–51.87 N on the right and 28.33–44.30 N on the left. Total vertical bed reaction was 951.27–955.50 N. Horizontal center-of-mass speed reached 5.21 mm/s at the endpoint. All 32 bed rows used registered material sections and skeletal force transfer. The terminal fascia audit passed 26 anatomical regions and 34 ligament force rows. Root assistance and reported body mass residual stayed zero. Pleural pressure varied from −511.93 to −490.33 Pa. The maximum absolute blood-volume residual was 0.00140 mL. No full heartbeat or breath elapsed.

The endpoint contains 37 skin self-intersections and 184 skin-to-bone triangle pairs, all at the right radius. The exact audit covered 878 anatomical banks and counted 11,747 raw skin-to-anatomy triangle pairs, including 350 ocular pairs. The 37 self-intersection witnesses yielded zero crossings on their registered source triangles, 19 under independently reconstructed skinning, and 37 on the emitted Float32 skin. These results apply to the witnessed pairs at the exported endpoint. The full deformable skin/fat volume and a settled, crossing-free body remain unqualified.

The full replay archive includes the accepted native pack, compact traces, the launcher used for recovery, and the exact geometry reports. Its recovery manifest and inherited member references document content hashes. [Verified 256-step native replay archive](https://github.com/Numi2/numilab-human/releases/download/human-historical-support-checkpoints-20261122/m6-feature-contact-20261125-integrated-r2.tar.zst)
