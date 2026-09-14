# Anatomical foot contact registration candidate

The lower-limb source now has an executable foot hand-off for the one-adult-male
release.  It re-reads the pinned BodyParts3D archive, derives the four source
foot groups (`calcn_r`, `toes_r`, `calcn_l`, `toes_l`), retains all 60 source-local
mesh rows as the preflight inventory, and checks the 30 reviewed `is_a` members
against the provisional source-to-MyoSim rest transforms.  The 30 unique
BodyParts3D members are hash-bound to their target bodies.  The same
candidate checks the seven-pose lower-limb continuity receipt and the 18 source
support witnesses, including the six active witnesses and static weight wrench.

The candidate is `status: partial`.  It establishes source geometry and frame
identity for the next native contact owner; it does not turn the provisional
visual transforms into colliders, choose collision exclusions, calibrate
friction/compliance, or claim dynamic anatomical support, standing, recovery or
walking.

Run it with:

```sh
PYTHONPATH=src python3 -m numilab_human.foot_contact_registration_candidate \
  --sources Sources \
  --output Docs/media/foot-contact-registration-candidate-20260914/receipt-v1.json
```
