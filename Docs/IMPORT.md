# Import procedure

## 1. Fetch the password-free sources

```sh
numilab-human fetch --output Sources
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
numilab-human build \
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
