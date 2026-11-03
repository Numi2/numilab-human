# Native preparation and memory checkpoint 123

**Not native-motion or performance qualification. The full current skin/fat scene still has zero accepted physical steps; 0.1x remains unmet.**

The tested engine changes and complete trial evidence are in [Numi Lab at 6e55b759a4714a1929a98708f9bc9a9b0d91ca94](https://github.com/Numi2/numi-lab/tree/6e55b759a4714a1929a98708f9bc9a9b0d91ca94/tools/evidence/human-preparation-memory123-20261103). That is the single detailed entry point, including commands, device, source/binary hashes, memory controls, failed trials and current solver blockers.

All work ran on SSH Numi. Preparation releases large import/authoring copies and bounds Metal upload staging to32MiB. The current full contact arena fits at9.93GB, or10.17GB with current-pose contact-query reuse, against12.71GB recommended device working set. A single matched rejected-attempt pair improved51.835 ->31.742seconds; this is not accepted throughput. GPU parity passes for validation flags, forces, wrenches and audits.

The256-column linear solve still rejects at true residual6.04e-5. The existing regional preconditioner first exceeded its region budget; supported larger chunks then reached its dense coefficient memory limit. Sparse export currently follows dense construction. No admission or memory guard was bypassed, no contact rows removed, and no physical laws, geometry, mass or source ancestry relaxed.

The combined verified historical cleanup and user restart recovered24.7GB at the post-restart checkpoint. Later trial/OS paging consumption is recorded separately. Current inputs and mass proofs remain on Numi. Private GitHub recovery retains independently downloaded and hash/restore-verified historical archives plus556exact source/configuration files.

The OS decoder allocation began growing again after restart and remains a separate diagnostic concern; this checkpoint does not claim that leak was fixed. No native GPU benchmark remains running.
