# Matched native endpoints: fidelity comparison failed

Both complete native runs accept 58/58 steps at the unchanged 15.625 us timestep and physical gates. The reference reproduces every retained baseline481 physiology row exactly. Candidate495 is not promoted: independently solved trajectories exceed the existing 5% regional strain gate in 243 of 282 metrics, and 193 available force-channel metrics exceed 1%. Maximum matched displacement difference is 3.4243 micrometres. Small absolute movement does not waive the declared gates.

Applying the reference displacement to the candidate mesh leaves all 1,475,235 unchanged tetrahedra with exactly zero strain difference. Only 2/282 regional metrics exceed 5% in that counterfactual, both maxima in changed regions93/95. Actual trajectories differ across unchanged regions. This separates topology effects from trajectory sensitivity; it does not establish the solver as the sole cause. Strain squared is not material energy. Tag2 attachment force and complete material-energy coverage remain unavailable.

Reference body wall time423.544429417s; candidate447.044133125s, 5.548% slower. Active solves209 versus222 and Krylov columns3344 versus3552; zero budget exhaustion in either. This is not a performance improvement, supported-rest qualification, or proof of motion-wide anatomical validity.

Preceding attempt507 failed before physics because its shortened horizon retained a64-step observation bound. Trial509 binds that observation window to58 on both runs; physical policies are unchanged. Raw endpoint logs remain outside Git on SSH Numi, with hashes in the execution receipts.

The older364 archive receipt records independently downloaded, actually restored files before local retirement. Current495, baseline446, source inputs and the old mass-rebinding implementation remain local.
