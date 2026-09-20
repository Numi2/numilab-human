# Native prepared support-history contract

Status: `fixed_topology_contract_qualified_production_admission_open`

## Decision

The fixed-topology NHINIT3 serialization/provenance contract and analytic-owner
transaction are implemented and published on the Human native integration
branch. The contract binds every prepared `float4` support-history row to the
SHA-256, byte extent, ABI, source-record count, and expanded row count of the
exact raw NHCNT payload. Legacy NHINIT1/2 bytes remain accepted unchanged; the
unbound decoder rejects NHINIT3.

This closes fixed-topology serialization/provenance and the analytic transaction
part of prepared-state continuity. It does not establish production-runner
admission, topology-growth migration, or improvement in the production Human.
The earlier four-grid workload used an older native revision and NHCNT1 payload;
it is historical trace-basis evidence, not a valid seeded comparator.

## Exact identity

- Native branch: `human-native-runtime-20260915`
- Implementation commit: `c785abae322a1604f09f4d509f5d2af51ac42e5e`
- Qualified head: `3b968495e05253cb4675893c3b94fec448c63361`
- Qualified tree: `ee68992948f3b884571f380498f8e9af5d78edb1`
- Physical host: Mac mini `Mac16,11`, Apple M4 Pro, 24 GB, macOS 26.6
- Matter probe SHA-256:
  `bbd86c0aefeffdcebbc049c74ea7b423381b1d0e44c9f48f6fb8be24f7d56f3e`
- NumiMatter metallib SHA-256:
  `6397674c0463683e182c41dc110944847c82a0832848ead4310bbe3e305033fc`
- Source support payload: NHCNT2, 10 source records, 18 expanded rows,
  1,044 bytes, SHA-256
  `c7712daf79cd8a589a6d23942a4df84a7da928e5911455ce19078f9b24daaaf4`
- Generated NHINIT3: 8,196 bytes, 224-byte header, 18 trailing 16-byte
  history rows, SHA-256
  `b4908f21aee3434091b4c9935fadbe276be30b142114a3dd51c1eb548ad28dad`

## What passed

The source-compliant certificate used the canonical MuJoCo-3.12-derived NHEQ2
payload rather than renaming the legacy NHEQ1 file. In support-reaction-only
mode it balanced in three accepted iterations, then emitted 18 nonnegative
normal-force rows with the exact NHCNT2 identity. Maximum normalized
acceleration was `0.00804238448447`; maximum generalized-force residual was
`0.000304879726144`. These are preparation diagnostics, not dynamic
convergence results.

The fullbody author then:

- emitted the 224-byte NHINIT3 envelope and exact raw-support identity;
- converted force rows to timestep-scaled normal impulses without using the
  standalone contact descriptor's reserved word;
- re-imported and re-emitted NHINIT3 and its Matter package byte-for-byte;
- rejected a different, valid, same-size NHCNT2 payload with the exact
  provenance diagnostic; and
- passed the focused NHINIT1/2/3 and static-support tests.

On the physical M4 Pro, the analytic Matter probe passed 11 cold, seeded,
redundant, mass, timestep, airborne, sticking, sliding, and warm-sliding cases
across three environments. Every case reported zero q, v, and normal-history
replay error. Missing or short bodies, bad history count, cone/overflow input,
undersized query capacity, and undersized, oversized, nonfinite, negative,
nontangent, or outside-cone restore state failed closed before mutation.

## Retained failure and repair

The first end-to-end import on `c785aba` exposed a fixture-tool defect: a valid
NHINIT3 without a root-translation extension was re-emitted with one. The
positive and round-trip hashes differed even though the Matter packages were
identical. The failed binaries and byte offsets remain in the receipt.

Commit `3b96849` preserves the imported root-extension representation. The
final NHINIT3, JSON receipt, and Matter package now round-trip byte-for-byte.
No failed evidence was overwritten.

## Claim boundary

This result qualifies NHINIT3 raw-NHCNT provenance, source-certificate
authoring, byte-exact import/re-emission, and the analytic Matter accepted
history, restore, rejection, and replay transaction on one physical M4 Pro.

It does not qualify the whole-Human NumanX runtime, topology-growth execution,
anatomical contact or loading, calibrated friction/compliance, pressure or
centre of pressure, force or timestep convergence, sustained standing,
recovery, walking, performance, participant generalization, biological
validity, clinical validity, or production readiness. The airborne analytic
case's retained `-7.17903e-09` impulse is inside the probe's existing tolerance
and remains visible; it is not promoted into exact unilateral proof.

## Next evidence-producing action

1. Add and qualify fail-closed NHINIT3 admission in the production stand
   runner, including exact clock, composed-source, raw-NHCNT2, package/world,
   and initial-state fingerprint checks.
2. Freeze the common native commit/tree, runner, runtime library, metallibs,
   rigid, muscle, NHEQ2, NHLIM1, and NHCNT2 identities. For each
   100/50/25/12.5 microsecond grid, author and qualify an exact-clock tuple of
   Matter-package SHA, world fingerprint, NHINIT3 SHA, and initial-state
   fingerprint.
3. Through `ssh macmini`, run matched cold-versus-seeded cases on that identical
   stack. Compare same-time q/v, first-interval and trajectory support histories,
   equality and source-limit impulses/work, tendon and dynamic force residuals,
   rollback, and replay. Retain improvement, no change, or regression.
4. If continuity is not causal, capture one paired equality/source-limit event
   and a selected-step full owner snapshot before changing solver behavior.

## Immutable artifacts

- [Receipt](media/native-prepared-support-history-v1/receipt-v1.json)
- [Independent verifier](media/native-prepared-support-history-v1/verify_receipt.py)
- [SHA-256 manifest](media/native-prepared-support-history-v1/SHA256SUMS)
- [Build/run manifest](media/native-prepared-support-history-v1/build-run-manifest.json)
- Raw certificate, authoring, negative-control, focused-test, and physical-M4
  outputs under `Docs/media/native-prepared-support-history-v1/`
