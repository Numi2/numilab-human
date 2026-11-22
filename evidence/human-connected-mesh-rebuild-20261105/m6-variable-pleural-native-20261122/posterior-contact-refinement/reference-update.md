# Accepted displacement, contact and support on M6

The precision experiment retained two of three 15.625 µs steps. The endpoint bed force was 2,252.14 N under 952.86 N of gravity. The vertical skin/fat force delivered to the skeleton was 17.86 N under 811.45 N of articulated gravity. The tested starting configuration is out of equilibrium.

## Metal change

The matched control traced an outer skin vertex and a nearby skin triangle from donor 128. Their reference separation was 1.972 nm. At root 2, iteration 15, the first trial cleared continuous collision detection. The barrier predicted an energy decrease. Reconstruction from the high and low coordinate components showed an energy increase along the realized first trial. Rounding the velocity before accumulating displacement lost a small increment along that path.

The experimental Metal library accumulates each trial correction directly into paired displacement and recovers velocity and its inertial increment from the same state. A direct M6 test recovered 1.39699e−13 m of a specified 1.39698e−13 m displacement. The control lost the increment. Forward then reverse left a displacement error of 1.73e−18 m. The checks include ineligible and fixed nodes and the existing ordinary-coordinate path.

One native Newton iteration took its full incoming fraction of 7.63e−6 with this change. The control took 3.77e−11 at that iteration. The third physical step still rejected. The nonlinear residual changed from 0.03632484 to 0.03632436 against the mechanical acceptance limit of 1.25e−5. Full trial corrections encountered the volume bounds of adipose tetrahedra 950449 and 1035269. Accepted common coordinates and respiratory status survived rollback. The Metal patch remains experimental.

## Independent accepted endpoint

A separate two-step run reproduced the three-step trial's accepted force history. Against the control, its largest material-node displacement difference was 0.1323 nm and the terminal bed reaction differed by 0.00048 N. The 1,448,799 registered skin/fat cells had zero inversions and zero violations of the configured volume and shape floors. Thirty-seven dermal cells and sixteen fat cells exceeded their registered stretch envelope, including three severe dermal excursions.

The audit covered 878 anatomical banks at the final accepted frame. Skin self-crossings, interlayer crossings and skin–bone crossings were zero. Physical skin crossings comprised 10,027 muscle, 792 vessel and 76 tendon triangle pairs. The leading muscle banks were right extensor digitorum (915), left brachioradialis (744), right brachioradialis (617) and left extensor digitorum (575). These source muscle surfaces use articulated-body vertex blending in this viewer path. MyoSim generates muscle forces separately. A skin–muscle triangle crossing proves a mismatch of the registered envelopes. The count alone does not identify a contact force or clinical injury.

The paired-update body horizon achieved 4.86e−7×. Complete execution took 209.87 wall seconds for 31.25 µs (1.49e−7×). Pleural pressure was −490.33 Pa at both samples. This run does not establish settling, cyclic physiological operation or sustained throughput.

## Engineering decision

Keep the precision trial available for regression. The governing release problem is the equilibrium and registration of the supported assembly. A candidate must transfer measured surface load through deep tissue to the skeleton and remain stable on release under gravity. The bed traction must converge under surface refinement, with a matching tangent in the solver. The source muscle envelopes and the moving skin need a common registered interface. Keep all witnesses and the existing physical acceptance limits.

[Accepted motion](incremental-prefix-summary.json) · [Matched three-step experiments](incremental-comparison.json) · [Exact anatomical bank counts](accepted-crossing-owners.json) · [Contact source](contact-source.json) · [Reconstructed line search](contact-trajectory.json) · [Native Metal source patch](reference-update/reference-update.patch) · [Independently restored native inputs, shader, traces and captures](incremental-recovery.json)
