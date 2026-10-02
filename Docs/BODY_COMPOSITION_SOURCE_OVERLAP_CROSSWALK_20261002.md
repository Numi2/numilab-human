# Body composition and source-overlap identity crosswalk - 2026-10-02

The new [body-composition integration receipt](media/body-composition-integration-20261002/receipt-v1.json)
binds the current whole-body source-frame overlap census alongside the existing
18-region organ-mass candidate. It does not merge their member sets or promote
any physical owner.

The crosswalk records **104** source meshes in the overlap census and **378**
members in the organ-mass candidate. They share **30** source identities; **74**
appear only in the overlap census and **348** only in the organ-mass candidate.
These are different source selections with different purposes. The 104-member
census tests source/compiled intersections for the 198 declared same-owner
crossing pairs; the 378-member inventory resolves complete membership for the
18 authored physiology regions and computes surface-moment candidates where
topology allows. A census-only identity is not thereby an organ-mass moment,
and an organ-mass identity outside the census is not thereby checked for
whole-body overlap.

The bound overlap census tested **481** exact pair geometries and retains **198**
compiled crossings. Its source/compiled crossing witnesses comprise 187 direct
matches, 10 matches after exact source-face mapping, and one documented
topology-repair mapping that removes two opposite duplicate source faces from
the raw count. The cross-domain candidate remains partial: it records zero
anatomical physical-volume owners and zero mechanical mass owners. Its
conservation and replay receipts remain scoped to their individual source
models.

The integration receipt SHA-256 is
`e274b0f491a24ccb9b7fdfd2e739e025b0273983c62674e37522f0db031b65d3`; it binds
the exact overlap census receipt SHA-256
`9a669122521a11389ea4e4f69869e400d61908b2b0b5e5d4377620f3a9ff6b26`.

This identity crosswalk does not decide whether any crossing is anatomically
intended, supply regional organ-mass moments for the 74 census-only identities,
close the eight individually unqualified source surfaces, or establish clinical
registration, tissue boundaries, mechanics, or physiology. No source geometry
was changed.

Reproduce the receipt with:

```sh
NUMI_HUMAN_PYTHON=.venv-mujoco312/bin/python \
.numi/commands/human body-composition-integration \
  --output Docs/media/body-composition-integration-20261002/receipt-v1.json
```
