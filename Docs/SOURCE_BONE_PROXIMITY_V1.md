# MyoSim source-bone proximity and registration worklist v1

## Decision closed

The 468 current point fallbacks are not all BodyParts3D registration defects.
Before fitting another anatomy deformation, the importer now asks whether each
authored terminal site is within the unchanged 12 mm gate of a mesh attached
to the same body in the pinned MyoSim source model itself.

The audit uses MyoSim revision
`33c89c2bde282553dde3f526768eb3bdcfaa7649`, archive SHA-256
`280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975`,
and MuJoCo `3.12.0`. It transforms each compiled mesh geom into its authored
body frame, computes exact point-to-triangle distance, and retains the nearest
triangle, barycentric coordinates, mesh identity, body identity, and source
site identity. It emits no source mesh vertices.

The command is reproducible through the isolated source environment:

```sh
numilab-human myosim-source-bone-proximity \
  --sources Sources \
  --python .venv-myosim/bin/python \
  --maximum-distance 0.012 \
  --output Build/myosim-source-bone-proximity-v1.json

numilab-human bodyparts-registration-worklist \
  --source-audit Build/myosim-source-bone-proximity-v1.json \
  --tendon-manifest Build/numi-human-tendon-topology-v11/numi-human-tendon-attachments.manifest.json \
  --output Build/bodyparts-registration-worklist-v1.json
```

Two independent source-audit runs were byte-identical with SHA-256
`aaa2a70d1aeb528ec7bbac90a7cfe1de15b5b195a26704c4b20ca92840366125`.
The derived worklist SHA-256 is
`ae8e3c20d772d0082237b3e4081cd8713bf63758db20d314120b3983afa9f82f`.

## Exact current disposition

The worklist joins all 832 endpoints by pinned archive, exact source actuator,
muscle name, and origin/insertion identity. Original MuJoCo IDs and remapped
Core indices remain separate namespaces.

| Disposition | Count | Meaning |
| --- | ---: | --- |
| already admitted BodyParts3D envelope | 364 | no registration work |
| true BodyParts3D registration candidate | 256 | source site is already within 12 mm of its own MyoSim bone mesh |
| source-model non-bone endpoint | 159 | site is farther than 12 mm from bone before BodyParts3D is involved |
| surface-patch conditioning backlog | 9 | correct nearby surface exists but the current four-node force map remains ill-conditioned |
| BodyParts3D surface mapping missing | 24 | owning mechanics body currently has no registered BodyParts3D bone surface |
| semantic member identity unresolved | 20 | one mechanics body owns multiple BodyParts3D members without a unique reviewed target |

The 256 genuine registration candidates have a median same-source bone
distance of `0.823 mm`, mean `2.528 mm`, and maximum `11.809 mm`. Of these,
176 are in the bilateral scapula/arm/forearm/hand/finger chain, 54 are on the
torso body, and 26 are in the pelvis/lower-leg/foot chain. This makes the upper
limbs and hands the first high-yield regional registration target.

The most important negative result is the sacrum: all 143 current
distance-rejected sacrum routes are also farther than 12 mm from the sacral
mesh in MyoSim. Many are lumbar, abdominal, or aponeurotic route endpoints.
Warping the sacrum toward them would manufacture false bone entheses. The
pelvis has the same distinction at smaller scale: only two of eight current
distance failures are true registration candidates; six are source-model
non-bone endpoints.

## Efficient next gate

The next registration solve is restricted to the 176 bilateral upper-limb and
hand candidates. MyoSim's own Apache-2.0 bone meshes provide the mechanics-
aligned correspondence authority; BodyParts3D remains the rendered anatomy.
Promotion still requires:

1. exact named bone identity and bilateral consistency;
2. a proper, bounded regional transform with no reflection or endpoint move;
3. held-out source-surface residuals and joint-chain continuity gates;
4. unchanged 12 mm distance, 12 mm patch radius, force-amplification, force,
   and moment gates in the normal `NHTENDON2` compiler;
5. deterministic rebuild plus four-angle M4 Pro inspection of shoulders,
   elbows, wrists, hands, and fingers.

A coherent non-rigid method such as
[Coherent Point Drift](https://arxiv.org/abs/0905.2635) or an affine plus local
[free-form deformation](https://webdocs.cs.ualberta.ca/~vis/readingMedIm/papers/RueckertFreeForm.pdf)
may propose correspondences. It cannot supply bone semantics or admission by
itself. The upper-limb pass should begin with rigid/similarity registration of
each named source-bone pair under chain-continuity constraints; local
deformation is justified only by held-out residual improvement without folds
or anatomy breakage.

## Evidence boundary

This increment proves source-model geometric classification and produces a
provenance-pinned worklist. It does not register BodyParts3D, move a MyoSim
endpoint, admit a new tendon envelope, create a deformable tendon or fascia
material, or validate anatomy clinically. The full
[source audit](data/source-bone-proximity-v1/myosim-source-bone-proximity-v1.json),
[registration worklist](data/source-bone-proximity-v1/bodyparts-registration-worklist-v1.json),
and [checksums](data/source-bone-proximity-v1/checksums.sha256) are retained.
