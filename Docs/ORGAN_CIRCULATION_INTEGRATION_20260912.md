# Organ and circulation integration: 12 September 2026

Native **`1594f7aff5503aab96ab9de6e77b6d4f8fa4f1cd`** on `origin/coupled`
adds blood volumes, flows and conserved species to the existing Matter
transaction. Human adds a source-bound physiology compiler and an 18-region
anatomical template. **Passive transport and transaction checks pass; anatomical
organ mechanics and physiological calibration remain incomplete.**

The native authoring API composes a vascular graph with ordinary Matter objects
before `compileWorld`. Its optional association with a real FEM node region
retains identity; it does not yet couple blood pressure or volume to organ
strain. Human's public JSON compiler currently emits fixed organ reservoirs.
Neither source membership nor an anatomical name supplies a vessel lumen,
organ material, blood-to-body mass partition or physiological parameter.

## Delivered owner path

`HumanPack.physiology.v1` → source/parameter validation →
`HumanPack.physiology-native.v1` → native `readHumanPhysiologyNetwork` →
`WorldSource.vascular` → ordinary `.nmatterpack` → existing Metal
Newton/FGMRES → accepted Human/Matter publication.

The offline Human compiler verifies the pinned BodyParts3D tables, FMA identity
and selected component membership. It retains overlapping membership and
independent physical-volume ownership declarations. SI values, provenance and
uncertainty remain explicit; unresolved physiological parameters block native
compilation. The native loader rejects duplicate JSON fields, including escaped
keys, unsupported laws and qualification claims. Exact input, source and
authored graph identities survive packaging.

The persistent Metal state owns compartment volume, connection flow, blood
species amounts and tissue amounts. Pressure follows an explicit linear
compliance law; flow follows resistance/inertance; species use upstream
concentration and equal/opposite exchange. Scaled rows enter the same Newton
and FGMRES solve as mechanics. Shared line search preserves positive volume and
nonnegative amounts. Episode reset, checkpoint, rejection, snapshot/restore and
prepared publication cover every vascular state element.

Matter ABI **26**, package **11**, snapshot archive **5** and accepted-proof
manifest **5** version the change. Older Matter packages reject and must be
recooked. Earlier standing/material receipts retain their original revisions;
this increment does not turn those historical results into qualification on
the new ABI.

## Physical Mac mini results

Apple M4 Pro, 12 CPU cores, 24 GiB memory; Metal API validation enabled. The
11 selected native checks passed in 20.06 seconds. The vascular checks all
initialize the runtime from serialized Matter packages.

| Gate | Observed result |
| --- | --- |
| Two-pool forward/reverse/inertial flow and organ exchange | 16 accepted steps × 10 ms, two environments in each case |
| Independent FP64 backward-Euler elimination | maximum normalized error `2.9931e-7`; gate `8e-5` |
| Closed blood volume and species conservation | maximum relative error `8.3157e-8`; gate `3e-5` |
| Branched graph | 3 compartments, 3 edges, 2 species, 2 reservoirs, 3 exchanges; conservation error `6.0723e-8` |
| Equal-duration hydraulic refinement | 20/10/5 ms timesteps over 0.16 s; errors `4.5415e-9`, `2.2968e-9`, `1.1552e-9 m3` |
| Snapshot replay | bitwise identical |
| Rejected environment and episode reset | isolated; the other environment continues |
| Invalid state restoration | rejected without changing accepted state |
| Prepared Human/Matter proof | vascular mutation changes Matter proof; accept/reject/pending/abort/publication checks pass |
| Native JSON admission | one valid source payload; 12 malformed variants rejected |
| Existing Matter regression | stateful FEM/MPM, multiphysics, snapshot archive and production rollback pass |

Human's 28 physiology, 14 target-coverage and 18 evidence-mutation tests pass. The target compiler adds
15 systemic obligations for **95 mandatory leaves**, preserving all 80 original
leaves when migrating historical manifests. Those leaves remain obligations,
not qualification claims.

The [receipt](media/organ-circulation-20260912/receipt.json) binds source and
artifact hashes, detailed native logs, build identities and the exact synthetic
Human payload. Recheck the retained evidence with:

```sh
python3 tools/verify_organ_circulation_20260912.py \
  Docs/media/organ-circulation-20260912/receipt.json
```

The native report is
[`docs/HUMAN_VASCULAR_TRANSACTION_V1.md`](https://github.com/Numi2/numi-lab/blob/1594f7aff5503aab96ab9de6e77b6d4f8fa4f1cd/docs/HUMAN_VASCULAR_TRANSACTION_V1.md).

## Anatomical and physiological completion gates

The template binds all four heart chambers, major aortic/caval segments,
pulmonary trunk, lungs and abdominal organs to 18 source regions. It retains
eight overlapping memberships and **276 unresolved scalar parameters**.
Connections are explicit model-authoring proposals, not vascular connectivity
reconstructed or calibrated from the atlas. The executable kidney-labelled
fixture uses synthetic parameters solely to test anatomical binding and native
conservation; it does not model renal physiology.

Remaining work includes sourced/calibrated cardiac drive and valve laws,
coronary/portal and organ perfusion topology, deforming organ and vessel-wall
mechanics, pressure/volume/force and blood-mass coupling, gas binding/exchange,
metabolic reactions, thermal/fluid balance, and held-out physiological
observables. Numerical conservation of this passive law cannot close those
gates. Standing/walking, prepared-pose precision and broader Human performance
also remain separate workstreams.

The next physiology step is to pin and reproduce an explicit cardiovascular
source model and its parameter provenance, add its chamber/valve constitutive
laws to this owner, and qualify pressure/flow/perfusion against independent
observables before activating the anatomical template. See the
[authoring contract](ORGAN_CIRCULATION_AUTHORING_V1.md) for exact input rules.
