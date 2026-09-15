# Vessel mass-moment transport candidate — 2026-09-15

The new `vessel-mass-moment-transport` command joins the six hash-bound
BodyParts3D vessel surfaces to the existing physical-M4 regional blood/oxygen
receipt at the canonical 12.5 µs clock. It advances each registered
surface-integral mass candidate under an explicitly labelled uniform velocity
probe and updates the first and raw second spatial moments analytically.

The published receipt is
[`Docs/media/vessel-mass-moment-candidate-20260915/transport-receipt-v1.json`](media/vessel-mass-moment-candidate-20260915/transport-receipt-v1.json).
The run attempted 512 steps, accepted 511, rejected one candidate transaction
with atomic restore, conserved the candidate mass and linear momentum, and
replayed bitwise. The source mass receipt contains six owners and
`0.25057866444098925 kg` at the explicit engineering density candidate of
`1060 kg/m³`.

This is a spatial-moment and transaction subgate. The velocity field is a
diagnostic probe rather than an anatomical flow field. The source surfaces are
not promoted to lumens or tube walls, and no pressure gradient, tissue-side
exchange, mechanical blood-mass owner, material calibration, subject
calibration, standing, or walking claim is made. Those gates still require
registered lumen/centreline/area data, deformable walls, conservative
vessel-to-tissue coupling, calibrated material and density parameters, and
held-out physiological validation.
