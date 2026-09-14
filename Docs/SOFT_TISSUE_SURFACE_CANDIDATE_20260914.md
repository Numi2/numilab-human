# Source muscle and soft-tissue surface candidate, 14 September 2026

This increment joins the source-complete MyoSim reference route table to the
NHTISS4 BodyParts3D surface inventory. It verifies 416 unique source actuator
routes and 150 source surface rows: 148 muscle surfaces and two tendon
surfaces. The route-to-surface map binds 178 source routes and retains 238
routes without an emitted surface as an explicit register.

The receipt also makes the missing domains visible: fat and skin surface rows
are zero, and no physical tissue-volume, mechanical-mass, or volumetric active
muscle owner is assigned. The NHTISS4 payload remains a sparse kinematic visual
binding. It is not a deformable muscle continuum, a force-transfer law, a
collision surface, a fat model, or a calibrated subject tissue model.

The immutable receipt is
[`receipt-v1.json`](media/soft-tissue-surface-candidate-20260914/receipt-v1.json),
SHA-256 `31316cc53e0ff7459720ffd14b6bf50995aad23cec739431d4728c5e85a9d11f`.
The refreshed [gap report](media/gap-execution-20260914/report-soft-tissue-surface-candidate-final2-20260914.json)
retains `integrated_qualification=not_assessed`.

Reproduce it with:

```sh
./.numi/commands/human soft-tissue-surface-candidate \
  --output Docs/media/soft-tissue-surface-candidate-20260914/receipt-v1.json
```
