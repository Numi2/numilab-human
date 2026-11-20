# Variable pleural pressure with the existing abdominal aortic wall

The prior Matter guard rejected any FEM cavity when the Human
pressure callback was requested. The full registered resting model
has one existing abdominal aortic wall cavity, source ID 9007 in
stable compartment 7 (row 6). This row is outside the registered
pleural-pressure mask.

Matter still requires one environment, a vascular network without
continuum exchange or blood particles, and the accepted-state extension.
The Human extension admits only the pinned abdominal cavity and checks
that its row is outside the mask. Neither the wall pressure load nor
the constitutive, gas, contact, or rollback algorithms are changed.
The GPU pressure callback continues to stage the trial boundary from
accepted pleural pressure with the existing five-second continuation.

The native executable links a static Matter runtime in addition to
the supplemental dynamic library. The three patches reproduce the
boundary change against the pinned text of both Matter owners and
the Human stage extension. The old binary still failed with the new
dynamic library alone. The relinked binary installed the full repaired
mesh, admitted all 34 ligaments, registered the pressure mode and
initialized both Matter and respiration on SSH Numi's Apple M6.
The fourth attempt then stopped at the existing capture-storage
budget before its first physical step. These logs certify coexistence
at initialization. They do not certify dynamic load balance,
crossings in motion or sustained speed.
