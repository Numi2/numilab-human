# Lung source correction and native regression

The existing source preparation owner now collapses one pinned 0.2154 micrometre sliver edge on lobe 305 into its endpoint already shared with lobe 308. It deletes two triangles, preserves closed oriented topology, and retains the earlier reciprocal 307/309 repair. Respiratory effective area is bit-identical after Float32 conversion. The source changes are inferred geometric conditioning, not measurements of an individual.

The new lung candidate was composed through the existing receipt owner with the source-tracked choroid correction and the current 907 skin. The retained failed first composition attempted to skip the required 004-to-003 skin provenance link; the second attempt follows the full existing chain. Source and geometry identities are pinned in the attached records.

Native run 925 completed 10,000 accepted 2 ms steps on the SSH Mac mini using build 016 and physical runtime 014. All four physiology, joint-motion and bed-contact CSV files are byte-identical to run 914. All eight captures have identical body and respiratory state hashes, all 86 skin-owner poses, and respiratory motion. Changed input anatomy produces different root fingerprints; only root/transaction equality within each run is asserted. Runtime mesh checks report no nonfinite or zero-area triangles.

This is a source repair and state regression, not an anatomy or endurance pass. Exact transformed lung intersection checks are separate. The current 907 skin still intersects underlying tissues in moving poses. Runtime wall time was 198.084 seconds for 20 simulated seconds (0.101 times real time including initialization); performance tuning remains stopped at the user's direction.

Validation on the Mac mini: seven lung precision-conditioning unit tests and five choroid composition tests passed. The large source meshes, complete build report, native movie and traces remain in the hash-pinned external evidence directories.

Follow-up: the bounded exact native seam audit927 failed. The scanned neighborhoods contain 19/16/18/7/9/10/8/6 unallowed pairs across the eight poses, despite passing source topology/embedding. See ../native-lung-seam-failure-927. Further source regularization is required.
