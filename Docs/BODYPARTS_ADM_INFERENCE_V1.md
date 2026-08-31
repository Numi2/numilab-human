# BodyParts3D abductor digiti minimi inference v1

`numilab-human bodyparts-adm-build` compiles a bilateral, nonvisual mechanics
witness for the abductor digiti minimi (ADM). It reads the exact pinned
BodyParts3D 4.0 muscle, pisiform, and fifth proximal-phalanx OBJ members, finds
surface-nearest origin/insertion clusters, and maps each bone-surface centroid
through its existing per-bone MyoSim registration. It emits:

- an auditable JSON record with member hashes, selected vertex ordinals,
  surface gaps, registration status, bilateral parity, literature parameters,
  and evidence boundaries; and
- an optional `NHADM1` native payload containing the two registered routes and
  force-capacity sensitivity range for an FP64 runtime feasibility probe.

The compiler fails closed if the archive/member hashes drift, an endpoint
cluster overlaps the other endpoint, origin-to-insertion separation is under
20 mm, a registration is missing/ambiguous, or reflected bilateral endpoint
parity exceeds 5 mm.

The checked-in source witness passes with a 1.7982 mm maximum bilateral source
endpoint residual. Right/left inferred endpoint separations are 63.31/64.46 mm;
the largest selected muscle-to-bone surface gap is 0.131 mm. These are geometry
and consistency results, not proof of actuator suitability.

## Parameter boundary

The open digital-hand dissection reports ADM PCSA 111 mm2, optimal fibre length
96 mm, zero pennation, and an 11 mm external insertion tendon. It is a
single-specimen dataset, so the artifact retains that limitation. Human muscle
specific tension is not treated as a constant: the payload carries a 22.2,
29.748, and 61.05 N low/nominal/high sensitivity band derived from 20, 26.8,
and 55 N/cm2. The nominal value follows a systematic review; the high value is
an in-vivo quadriceps measurement and is deliberately only a sensitivity
bound.

ADM insertion varies. The runtime route defaults to the observed bone surface.
The reported 31% mean extensor-expansion share (range 0-50%) remains metadata;
the compiler does not fabricate an unobserved branch in this BodyParts3D
individual.

Sources:

- https://doi.org/10.1111/joa.12877
- https://doi.org/10.1152/japplphysiol.00296.2024
- https://doi.org/10.1113/expphysiol.2009.048967
- https://pubmed.ncbi.nlm.nih.gov/11901393/

## Evidence boundary

The source surfaces and bone identities are exact. Endpoint clusters, fibre
direction, and force capacity are explicit inference/sensitivity. Both
pisiform registrations retain their documented upper-limb-chain fallback.
`NHADM1` therefore admits a feasibility probe; it is not live Hill-type
actuation, static equilibrium, or subject-specific validation.

