# Source-bound organ and circulation authoring

`HumanPack.physiology.v1` adds anatomical region references, hydraulic
compartments, blood connections, conserved species, and tissue exchange
reservoirs to the permanent Human source representation. The Python compiler
validates and lowers immutable inputs. It contains no physical stepping.

The owning implementation is `src/numilab_human/physiology.py`; the authored
shape is `schemas/humanpack-physiology.v1.schema.json`. Run:

```sh
PYTHONPATH=src python3 -m numilab_human.cli physiology-compile \
  --graph config/physiology-passive-fixture.v1.json \
  --sources Sources --output Build/physiology/passive.native.json

PYTHONPATH=src python3 -m numilab_human.cli physiology-compile \
  --graph config/physiology-organ-network-template.v1.json \
  --sources Sources --validate-only --output Build/physiology/anatomy-validation.json
```

The first command emits an explicitly synthetic executable input. The second
retains missing model parameters and overlapping source memberships. Removing
`--validate-only` from the anatomical template fails before native lowering
because its physiological and numerical parameters are unresolved.

## Anatomy identity and independent ownership

The compiler reads both BodyParts3D 4.0 element-relationship tables and checks
their bytes against `sources.lock.json`. Every region names its FMA concept,
exact source label, hierarchy, and component members. A declared complete
membership must match that source table exactly. Explicit subsets are permitted
only when labelled as subsets. Native anatomy identifiers resolve to the FMA
semantic ID, while the source graph hash binds the complete membership record.

The anatomical template names 18 regions: all four heart chambers, ascending,
arch, descending and abdominal aorta, superior and inferior vena cava, pulmonary
trunk, both lungs, both kidneys, stomach, pancreas, and liver. Each is bound to
the complete source membership in the selected hierarchy. Seven organ tissue
reservoirs are separately declared. This extends the earlier `NHANATOMY` visual
selection, whose single heart component was not complete heart geometry.

Eight component memberships overlap between the chosen heart-chamber regions.
They remain explicit in validation output. BodyParts membership is anatomical
provenance; it does not supply a watertight volume, vascular lumen, registration
to moving mechanics, mass, compliance, resistance, perfusion rate or calibration.

Every blood or tissue compartment has a separate `physical_volume_owner_id`.
Using the same token twice fails admission, including reuse across blood and
extravascular tissue. Unique tokens establish a nonduplicated ownership
declaration; they do not prove that geometrical volumes are physically disjoint.
Multiple physiological compartments may refer to one anatomical structure only
with independently declared physical ownership. No atlas-derived volume is
silently added to the existing rigid or FEM mass.

The template's aggregate circulation connections are unresolved model-authoring
proposals. They are not connectivity reconstructed from the atlas, a reproduced
source cardiovascular model, or subject-specific circulation. In particular,
the upper systemic return is an aggregate path, and portal circulation,
coronary circulation, valve laws, active chamber drive, respiratory boundaries,
and metabolic reactions remain absent. The placeholder cannot be run by
supplying an invented universal material or healthy-human parameter set.

## Physical contract and provenance

The current native-input law identifier is
`closed_linear_compliance_transport_v1`. Its admitted representation consists
of passive compliant vascular compartments, resistive/inertive connections,
well-mixed species amounts, and conservative blood-to-tissue exchanges. It does
not represent vessel wall FEM, organ deformation, cardiac contraction, gas
exchange, neural conduction, autoregulation or clinical physiology.

Each authored SI scalar carries `value`, `unit`, `provenance`, and `uncertainty`.
Source parameters reference a parameter source record with URL, revision,
SHA-256, license, allowed use and a record locator. Synthetic values require
`fixture_only`; unresolved values are null and require `uncalibrated`. Unknown
uncertainty remains explicit and cannot be relabelled as qualified. The
compiler binds source-record identities but does not independently reproduce
their experiments, confirm their contents or establish parameter calibration.

Compartment volume and species amount scales, connection flow and pressure
scales, and dimensionless residual tolerance are authored explicitly. The
compiler publishes separate amount, volume and flow tolerance fields from the
declared tolerance. This does not borrow the mechanical contact tolerance or
turn a numerical residual into biological validity. Source inertance may be
omitted only to select the explicit zero-inertance law. Exchange partition
coefficient may be omitted only to select unit partition; both templates spell
out these parameters rather than relying on those defaults.

An exchange connects exactly one blood compartment, one tissue reservoir and
one species. Its native clearance is the permeability-surface product, in
`m3/s`; its partition coefficient specifies the tissue-to-blood concentration
ratio. Both endpoints must receive opposite amount changes. Missing endpoints,
unexchanged tissue reservoirs, unsupported fields, duplicate connections,
nonfinite values, and disconnected closed hydraulic networks fail authoring.
Actual conservation, positivity, residual acceptance and rollback are native
runtime obligations, not claims made by the Python validator.

## Compiled identity and evidence boundary

Compilation sorts source IDs within each record kind and assigns a global,
positive integer sequence across species, compartments, connections, tissue
reservoirs and exchanges. Native endpoint references use those IDs. Initial
species arrays use the sorted species order. Reordering authored tables or
member lists therefore leaves compiled bytes unchanged.

The normalized authored graph hash binds all topology, parameters, provenance,
uncertainty, source and ownership metadata. The source graph hash separately
binds verified anatomy records. The native loader must retain these identities,
reject unsupported law or payload drift, and bind the input content into its
immutable world identity. Persistent physical state belongs to the native
accepted-step/checkpoint owner; a standalone native fixture does not establish
integration with the full Brain–Human–Matter transaction.

`config/physiology-passive-fixture.v1.json` contains two manufactured blood
pools, one connection, one tracer, and one tissue reservoir/exchange. Kidney
identities exercise source binding only: its volumes, compliance, initial
amounts and resistance do not model kidneys. The organ template retains 276
missing scalar parameters and cannot compile for execution.

`tests/test_physiology.py` checks deterministic lowering, identity changes,
actual source membership and template gaps, along with malformed/forged input,
duplicate volume ownership, missing species and conservation endpoints,
unsupported promotion, and topology failures. Systemic physiology also adds 15
cumulative mandatory target leaves to `HumanPack.target-coverage.v1`; earlier
targets remain present. Target compiler `.2` requires all 95 mandatory leaves;
historical compiler `.1` manifests are accepted only as historical inputs and
retain all 80 original leaf records and hashes during rematerialization. A
historical manifest cannot be used to downgrade current scope. These authoring tests are separate from native physical,
material, source-model, and full-Human qualification.
