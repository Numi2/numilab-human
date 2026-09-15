# Registered foot proxy candidate — 2026-09-15

`foot-contact-proxy-candidate.1` is the next hand-off after the source-bound
foot registration candidate. It reads the immutable 30-member registration
receipt, transforms every BodyParts3D source AABB through its reviewed
source-OBJ-millimetre to MyoSim body-frame matrix, and emits four conservative
body-frame union bounds plus 30 per-member proxy rows.

The receipt is
[`Docs/media/foot-contact-proxy-candidate-20260915/receipt-v1.json`](media/foot-contact-proxy-candidate-20260915/receipt-v1.json).
It records four foot bodies, 30 registered source members, 30 transformed
enclosures, seven reviewed poses, 18 support witnesses and six active
witnesses. The source archive hash remains
`40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e`.

The transformed AABBs are conservative geometry candidates. They are not
collision shapes: the receipt leaves ground-frame registration, pair
exclusions, friction/compliance/restitution, swept-motion bounds, support
Jacobians, loaded equilibrium, standing, recovery and walking false. The
source registration itself remains provisional visual registration, so this
increment does not promote anatomical loading.

Validation:

- `tests/test_foot_contact_proxy_candidate.py`: 3 tests pass.
- `tests/test_foot_contact_registration_candidate.py`: 2 tests pass.
- The `human foot-contact-proxy-candidate` CLI writes an immutable canonical
  receipt and rejects non-rigid registration or profile-count drift.
