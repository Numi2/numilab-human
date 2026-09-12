# Shi/Hose source-faithful hydraulic authoring

`src/numilab_human/shi_hose.py` compiles the exact curated Shi/Hose 2009 CellML
closed circulation into `HumanPack.physiology-native.v2`. Its law is
`closed_periodic_elastance_orifice_v2`. The existing v1 physiology authoring,
schema, CLI and evidence pins remain unchanged.

```sh
numi human-cardiac --help
numi human-cardiac --output Build/shi-hose/closed-loop.native.json

# Equivalent explicit module entry point:
PYTHONPATH=src python3 -m numilab_human.shi_hose \
  --output Build/shi-hose/closed-loop.native.json
```

This writes a native payload and a separate source-lowering manifest. Outputs
are immutable. `--source-directory` selects a byte-identical copy of the
curated source; `--config` may select explicit numerical residual settings.
Neither option permits an unreviewed source revision or physiological override.

## Exact source ownership

The fifteen licensed, unchanged CellML files are retained under
`third_party/physiome/shi_hose_2009`, with authorship, CC BY 3.0 attribution,
immutable revision and file hashes. The compiler verifies the complete source
manifest and all fifteen file contents before reading parameters or imports.

The adapter independently parses the actual CellML unit products, imported
component aliases, hydraulic pressure/flow connections, parameter mappings and
initial values. It does not consume the research extraction JSON or import the
independent trajectory oracle. All 59 source parameters must be used in lowering.
The separate manifest retains their original lexical values, source unit names,
SI values/dimensions, parameter bindings and graph-reduction provenance.

`UnitP` is exactly **133 Pa**, as defined by `Units.cellml`; replacing it with a
different mmHg conversion changes this source model. `UnitV` is `1e-6 m3`.
Valve coefficients therefore convert with `1e-6 / sqrt(133)`. Both activation
templates retain their literal `3.14159`; this is not replaced by `math.pi`.

The source has fourteen differential states: four chamber volumes, six
vascular pressures and four inertial flows. Four valve flows and two venous
flows are algebraic. The native graph retains ten storage compartments and
ten connections. Systemic `Sat → Sar → Scp → Svn` and pulmonary
`Pat → Par → Pcp → Pvn` eliminate only the zero-storage series pressures;
individual resistance identities remain in the manifest and their exact sums
enter the corresponding connection. No compliant compartment is invented for
the arteriole or capillary resistance elements.

## Hydraulic storage and initialization

The four cardiac rows use `storage_kind=absolute_volume`. The initial dynamic
volume is the source `V0`, while `Vini` and `Pini` define the chamber
pressure–volume reference. In particular, ventricular initial `V0=500` source
volume units is not replaced by reference `Vini=5` or `10`.

The six vascular pressure states use `storage_kind=storage_displacement` and
store `C * P`. Their reference volume and pressure are zero. Both venous
initial pressures are zero, so both initial storage displacements are zero.
This is a pressure-equation coordinate, not a claim about zero blood volume.
The source does not determine absolute/unstressed vascular blood volumes.
Species, tissue reservoirs and exchanges therefore remain empty and attempts
to enable them fail compilation until compatible dilution volumes are sourced.

The compiler evaluates algebraic initial conditions only at source time zero;
it performs no integration or physical stepping. Atrial initialization retains
the activation segment wrapping across the beat boundary. Venous initial
outflows are signed pressure-driven flows, including source backflow. The
initial pulmonary valve is open. Native reset receives these derived source
values; it is not given convenient zero placeholders for nonzero algebraic
flows. The four inertial initial flows are the explicitly authored source `Q0`.

Anatomical identifiers are explicit CellML aggregate labels, such as
`CellML:shi_hose_2009:ModelSys:Svn`. They are not invented FMA mappings or
claims of registered vessel/organ geometry. Storage ownership tokens denote
source state ownership, independently of rigid/FEM mass or absolute blood
volume ownership.

## Native contract and qualification boundary

Cardiac pressure rows declare either `atrial_elastance` or
`ventricular_elastance`, exact minimum/maximum elastance, period, activation
parameters, reference state and literal source pi. For atrial rows,
`activation_end` contains source `Tpww`, the duration fraction; it is not an
absolute ending phase. Ventricular rows contain source `Ts1` and `Ts2`.
Linear rows set unused activation fields to zero. Connection rows declare
`one_way_orifice` or `resistance_inertance` with their exact source coefficients.

Native compartment IDs are sorted source aliases:
`LA, LV, Pas, Pat, Pvn, RA, RV, Sas, Sat, Svn` (stable IDs 1–10).
Connections use the corresponding `<alias>_outflow` identities (IDs 11–20).
The source manifest hash and authored numerical-config hash are separate
compatibility identities; the native loader additionally binds the entire
payload content.

Volume, flow and pressure scales and residual tolerance live in
`config/shi-hose-cardiac-source.v1.json`, explicitly marked as numerical
controls. Changing these settings does not edit physiological parameters.
Qualification tag `source_model_reproduction` identifies the requested source
comparison boundary; compilation alone does not establish numerical
reproduction. Native residual/conservation, beat-phase behavior, independent
oracle comparison, timestep refinement, replay and rollback need their own
execution evidence.

Even a reproduced hydraulic trajectory would leave absolute blood-volume
calibration, species transport, hemoglobin/gas exchange, metabolic demand,
organ deformation, autonomic control, subject fitting, and full Human
integration open. This adapter provides the actual source equations and inputs
needed for that work without filling those gaps with invented parameters.
