# Import procedure

## 1. Fetch the password-free sources

```sh
numi human fetch --output Sources
```

This fetches the six official BodyParts3D hierarchy/definition tables, the two
official 4.0 OBJ archives, and the RajagopalLaiUhlrich2023 model pinned in
`sources.lock.json`. Files with a known SHA-256 are verified before use.

## 2. Download MoBL-ARMS yourself

Sign in at the official [Upper Extremity Dynamic Model
page](https://simtk.org/projects/upexdyn/) and download its official bimanual
release, `MobL_ARMS_OpenSim3_bimanual_model.zip`. Keep the original archive
unchanged. Do not substitute a third-party GitHub mirror: this importer makes
the source URL and licence gate part of the output fingerprint.

## 3. Build an audit artifact

```sh
numi human build \
  --sources Sources \
  --upper-archive /absolute/path/MobL_ARMS_OpenSim3_bimanual_model.zip \
  --accept-upper-noncommercial-terms \
  --output Build/human-v1
```

The command emits:

- `human.v1.json` — source-preserving Numi Human v1 intermediate manifest.
- `report.json` — counts, source hashes, available parameter fields, and
  unresolved geometry registrations.

Both output directories are ignored because they are derived from source data.
Re-run the command from clean third-party archives when a new import revision
is required.

## 4. Audit every gate

```sh
numi human audit \
  --sources Sources \
  --runtime-root /absolute/path/to/MetalRobo \
  --output Build/human-v1-gates.json
```

The gate report is deliberately strict. It distinguishes a verified source
artifact from an authenticated source not yet supplied, a source manifest from
an executable Numi `RobotPack`, and a software integration from material or
physical validation. It never promotes an open gate based on a naming match or
a successful JSON build.

When `--runtime-root` is supplied, it also records whether that checkout is
clean and at the exact runtime revision whose lowering capabilities were
audited. A missing, dirty, or revision-mismatched checkout is not runtime
evidence.

## 5. Preflight every BodyParts3D OBJ member

```sh
numi human geometry-audit \
  --sources Sources \
  --output Build/bodyparts3d-topology.json
```

This writes the exact archive/member name and SHA-256 for every OBJ, with raw
vertex/face counts, bounds, and conservative edge-manifold facts. It does not
repair a mesh, establish an anatomical frame registration, create a volume
mesh, or infer a material law. Those remain separate, source-specific gates.

## 6. Compile a limited source-derived distal-leg preview

```sh
numi human preview \
  --sources Sources \
  --side right \
  --output Build/right-pin-preview
```

This output contains the right tibia, talus, calcaneus, and toes with their
Rajagopal mass/inertia data and the exact supported ankle, subtalar, and MTP
PinJoint transforms. It purposefully contains no collision geometry and no
muscle lowering, so it is only a native imported-URDF compiler preview—not a
complete Human RobotPack or physical validation.

When the matching Numi runtime is available, the bounded Metal ABA check is:

```sh
metalrobo_robot_description_cooker_probe \
  --metal Build/right-pin-preview/rajagopal-right-distal-pin-preview.urdf
```

It reports the Metal device, a successful GPU status, and a numerical payload
fingerprint. Repeat the same invocation on the same binary and device before
calling it a deterministic replay. This proves neither collision, contact,
muscle actuation, nor full-human physics.
