# NumiLab Human v1

The provenance-locked foundation for a Numi Lab human built from exactly three
upstream sources:

1. **BodyParts3D 4.0** — named anatomical OBJ geometry and its `is-a` / `part-of`
   hierarchy.
2. **OpenSim RajagopalLaiUhlrich2023** — lower-body and pelvis articulated
   mechanics, inertias, muscle paths, and muscle–tendon parameters.
3. **MoBL-ARMS OpenSim Upper Extremity Dynamic Model** — bilateral shoulder,
   elbow, forearm, wrist, and upper-extremity muscle mechanics.

This repository tracks source locks and original import code only; no
third-party geometry or model data is redistributed. The build produces a
local `numi.human.v1` manifest that keeps every source value and mapping
auditable.

## Start

```sh
python3 -m venv .venv
.venv/bin/pip install -e .

# Fetches only BodyParts3D 4.0 and the pinned Rajagopal model. It never tries
# to bypass the SimTK login required for MoBL-ARMS.
numilab-human fetch --output Sources

# Download the original bimanual MoBL-ARMS archive while signed into SimTK,
# then build a local Human v1 manifest.
numilab-human build \
  --sources Sources \
  --upper-archive /path/to/MobL_ARMS_OpenSim3_bimanual_model.zip \
  --accept-upper-noncommercial-terms \
  --output Build/human-v1
```

## What the first manifest means

| Source data | NumiLab Human v1 role | Current physical boundary |
| --- | --- | --- |
| OpenSim bodies, joints, masses, inertias | articulated rigid-body specification | lower body and arms are source-authored; runtime lowering remains a Numi core extension |
| OpenSim muscle paths and Hill-type parameters | active muscle–tendon specification | source values are retained, not converted to joint torques |
| BodyParts3D bones and muscles | named geometry attached to semantic anatomy | visual/anatomical geometry, not a new independent physical source |
| BodyParts3D skin, organs, vessels, nerves | deformable/anatomical geometry candidates | no material constants or volumetric meshes are supplied upstream |
| tendons, ligaments, cartilage | nonlinear tensile / compliant-contact candidates | only OpenSim tendon parameters are active-source data; all other constitutive data needs a cited calibration |

See [the architecture](Docs/ARCHITECTURE.md), [import procedure](Docs/IMPORT.md),
and [third-party notices](THIRD_PARTY_NOTICES.md) before building or publishing
derived data.
