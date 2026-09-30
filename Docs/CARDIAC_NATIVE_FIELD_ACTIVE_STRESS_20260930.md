# Native electrical field to non-mixed active ventricular stress

Matter [`7a16b00`](https://github.com/Numi2/numi-lab/commit/7a16b00)
adds an opt-in FEM route that reads the native candidate activation field,
forms per-cell fibre tension, and assembles its stress and tangent in the same
accepted-state transaction as mechanics. It preserves the source Guccione
passive law. The compiler requires conductive non-mixed multiphysics FEM and
the runtime rejects a simultaneous borrowed active-tension buffer.

On an Apple M4, the focused one-cell native probe accepted one 100 µs step.
Its synthetic reference cell reached free-tip activation `0.713394165` from
zero, and active stress changed the tip position by `1.5664845705e-6 m`
versus an identical field-only control. Repeated runs had byte-identical FEM
nodes and fields. The six focused source-material, frame, tension-cooking,
active-tension and field-stress tests passed.

The same probe then consumed the **exact four positions and derived material
frame of Rodero case18 ventricular cell 1** from the prior source-cell
receipt. That bounded source-geometry run accepted one 100 µs step, reached
free-tip activation `0.674182653`, changed the tip position by
`2.39174081672e-7 m` versus its field-only control, and replayed bitwise.
The [receipt](media/cardiac-field-active-stress-20260930/receipt.json) pins the
input and native binary hashes.

Both runs use **synthetic electrical conductivity, activation kinetics,
stimulus, heat capacity, density, supports and 1,000 Pa tension scale**. The
first attempt with zero heat capacity and nonzero Joule coupling failed with
`NM_STATUS_MULTIPHYSICS_FAILURE`; the accepted fixture explicitly gives the
thermal field mass and disables Joule heating. The native electric solve does
not reproduce the source CARP timing or excitation model. No full-wall native
electromechanical step, calibrated source reference, chamber loading or
heartbeat has been demonstrated. This is a working transaction and an
anatomical cell-geometry check, not a physiological beat.

The next source-bound gate is to derive a qualified electrical input and
kinetics from the published case18 data, carry it over the full ventricular
mesh in the same native transaction, then add source-consistent loading and
measured motion before any heartbeat claim.
