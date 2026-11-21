# Flat-bed preload on the registered posterior shell

Repeat the offline preload diagnostic with the exact paired skin/fat
mesh and typed masses. Compare normal-only tissue motion with unrestricted
three-dimensional motion of the same outer and bonded-layer nodes. The
deep boundary follows the same floating frame in both trials. Use the
same nonlinear energy, contact law, gravity load sequence, shape bounds
and residual tolerances. Accept a stage only when each unrestricted
force and floating-body force and moment meet the prior 1e-4 residual
criterion. Each load gets at most 65 iterations and keeps witnesses at
the first terminated stage.

The preregistered prediction is that replacing the zero tangential
displacements with lateral deformation will distribute load onto more
nodes and improve the largest converged gravity fraction. This tests
kinematic restriction. Even a successful full-load diagnostic would
require separate native verification with the actual deep attachments
and contact. The resulting state is a candidate for initial geometry
only. Neither trial is an alternate runtime.
