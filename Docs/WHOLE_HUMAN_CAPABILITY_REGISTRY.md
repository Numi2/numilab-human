# Whole-Human capability registry

The cross-system registry joins source-hash-bound receipts for whole-body
anatomy, internal organs, skin, muscle, tendon, bloodflow, systemic circulation,
cardiac electrical activation, cardiac mechanics, neural physiology, and
whole-body calibration. Its profile is
`config/whole-body-capability-registry.v1.json`.

Run it from the repository root:

```sh
PYTHONPATH=src python3 -m numilab_human.whole_body_capability_registry \
  --output Build/whole-body-capability-registry-<run-id>/registry.json
```

The command rejects an incomplete profile, duplicate JSON keys, unsafe paths,
and malformed source receipts. Missing evidence and schema drift are recorded
as open gates. A subsystem is qualified only when every declared closure
condition is present and satisfied; whole-Human qualification requires every
required subsystem to close. The report retains each evidence path, schema,
and SHA-256, plus the observed facts and unmet conditions.

The registry is an evidence join, not a physiology solver. Passing it proves
that its declared source receipts satisfy their recorded conditions; it does
not independently reproduce their measurements, qualify clinical anatomy, or
turn a component candidate into whole-Human behavior. The initial profile
deliberately includes electrical conduction, tissue/organ owners, skin and
tendon closure, and calibration so a source-only geometry pass cannot imply
integrated capability.
