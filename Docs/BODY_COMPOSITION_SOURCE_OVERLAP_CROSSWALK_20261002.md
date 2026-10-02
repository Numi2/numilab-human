# Body composition and source-overlap identity crosswalk - 2026-10-02

The new [body-composition integration receipt](media/body-composition-integration-20261002/receipt-v3.json)
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
`3603d2a0e853dcc892a8f8ae2240fbd63567df996d234a9695fef58208704f5e`; it binds
the exact overlap census receipt SHA-256
`9a669122521a11389ea4e4f69869e400d61908b2b0b5e5d4377620f3a9ff6b26`.

The [overlap-only organ moment receipt](media/whole-body-overlap-organ-moments-20261002/receipt-v2.json)
adds individual source-frame moments for **73** of those 74 identities. FJ3150
has two closed surface components with overlapping source AABBs, zero exact
triangle crossings, and exact containment of the smaller reverse-winding
component inside the larger one. The compiler retains their component moments
separately and does not aggregate them. The moments are not additive across
atlas surfaces and carry no physical-volume or mass owner.

This identity crosswalk does not decide whether any crossing is anatomically
intended, supply regional organ-mass moments for the 74 census-only identities,
close the eight individually unqualified source surfaces, or establish clinical
registration, tissue boundaries, mechanics, or physiology. No source geometry
was changed.

Reproduce the receipt with:

```sh
NUMI_HUMAN_PYTHON=.venv-mujoco312/bin/python \
.numi/commands/human whole-body-overlap-organ-moments \
  --output Docs/media/whole-body-overlap-organ-moments-20261002/receipt-v2.json
NUMI_HUMAN_PYTHON=.venv-mujoco312/bin/python \
.numi/commands/human body-composition-integration \
  --output Docs/media/body-composition-integration-20261002/receipt-v3.json
```
