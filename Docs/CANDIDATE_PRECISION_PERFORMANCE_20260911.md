# Candidate precision, refinement and execution cost

Native `ee17f46` on `origin/coupled` and Brain `370337d` on `origin/main`
reduce measured candidate execution cost, improve prepared geometry, and expose
the next numerical blocker through exact-state timestep trials. **Timestep
convergence and full source-force consistency remain open.**

| Prepared comparison | Before | Current result |
|---|---:|---:|
| Maximum route error against native FP64, 416 muscles | 0.572 µm | 0.326 µm |
| Maximum full source-force error, 100 µs | 0.780 N | 0.385 N |
| Maximum full source-force error, 1 µs | 4.844 N | 3.290 N |
| Median candidate FK/Jacobian GPU interval | 18.073 ms | 1.395 ms |
| Matched 16-root execution, width/ancestry changes only | 30.953 s | 13.721 s |

The native geometry now accumulates body positions relative to the floating
root and groups joint anchor offsets before accumulation. It adds the world
translation at publication. Existing Jacobian, mass and body-motion operations
use translation-invariant relative differences. Independent MuJoCo 3.12 forward
kinematics, with all 57 source files verified, finds maximum initial route error
**0.316 µm** on the published four-root trace. This is source agreement at one
prepared pose, not experimental calibration.

Same-path fibre calculations retain the fixed 1e-5 normalized force/residual
and 0.501-ULP publication gates. They pass at 1 and 100 microseconds. Full source
force remains more sensitive: the largest normalized discrepancy is **0.736%**
at 100 microseconds and **4.03%** at one microsecond. These maxima differ from
the muscles with the largest absolute force error. The receipt computes the
normalization from the frozen NHMYO2 Fmax values. No fitted architecture, force
scale, risk inhibition, ground, source compliance or acceptance tolerance was
changed to reduce these errors.

The reference probe also emits native/GPU body poses and FP64 unwrapped-route
lengths at the returned GPU poses. In the baseline worst unwrapped route,
477.6 nm of the path discrepancy comes from body geometry and 94.2 nm from
subsequent route arithmetic. This separates two numerical contributors without
adding host stepping or a second physics owner.

## Measured execution change

Six GPU timestamps per candidate separately measure preparation, FK/Jacobians
and materialization. Each 16-root timing run contains 512 valid query records.
Widening the candidate dispatch to 256 lanes reduces the median FK/Jacobian
interval to 3.468 ms. A dispatch-local ancestral-DOF mask reduces it further to
1.395 ms. Preparation remains about 135 µs and materialization about 68 µs.
The mask reuses the otherwise unused factor scratch in Jacobian-only mode;
related columns still use the existing motion-column implementation. Nothing
is cached across candidates, epochs or rollback boundaries.

The width/ancestry changes preserve every byte of all 160 physical trace
records in their matched four-scenario comparison. The width-only 64-root
cohort also matches all 2,560 records of the preceding published runtime.
The approximately **13× kernel reduction** becomes **2.26× end-to-end** in the
matched 16-root runs. Kernel timings are not overall throughput.

The final published stack completes four scenarios of 64 roots, **256 accepted
roots and 6.4 ms per scenario**, in **152.513 seconds**, with bitwise
recruited replay and dropout/zero physical equality. The preceding 445.835-second
run included a brief system trace; its ratio to this run is an indicative
comparison, not a controlled performance-envelope qualification. All quoted
runs retain Metal API validation. The separate 25-µs/32-Newton failure run had
a brief overlap with an operator probe; its wall time is excluded from speedup
claims.

The broader context probe exposed an older allocation-contract failure:
generic robot contexts allocated 17 unused Human/Matter buffers. The owner now
keeps that arena lazy when no coupled Human program is present. The unchanged
32-allocation cold-context check, growth/reuse, asynchronous busy gate, discard
drain and orphan lifetime tests pass. The failed 49-allocation run is retained.
Generic floating/fixed/sphere/ellipsoid/G1 analytic, virtual-work, status,
transactional failure and replay checks pass, as do ten Human/Matter regressions.
The adapter's allocator-address-reuse subcase remains `not_observed`.

## Refinement result and first failing buffer

The fixture importer reads the original prepared NHINIT1 directly. Every
100/50/25-µs fixture has byte-identical q, v and muscle payloads; the 100-µs
fixture is identical in its entirety. Only authored world/time identity changes.
An earlier certificate-derived reconstruction produced a different muscle
payload and failed the equality assertion before entering a comparison. That
attempt remains in the bundle.

The Brain harness uses zero delivered motor commands, one-root native receptor
latency and a common initial clock origin for this cohort. It emits all ten
trace kinds after every accepted root, so a later failure cannot erase the
accepted prefix. A bootstrap timestamp underflow and a subsequent incompatible
fixed-latency attempt are retained separately from physical solver failures.
The underflow now throws a typed error; no sensory contract is bypassed.

| Timestep | Accepted roots per scenario | Accepted duration | Result |
|---|---:|---:|---|
| 100 µs | 16, zero and replay | 1.6 ms | Exact replay |
| 50 µs | 32, zero and replay | 1.6 ms | Exact replay |
| 25 µs, 16 Newton iterations | 8, zero | 0.2 ms | Root 9 rejected |
| 25 µs, 32 Newton iterations | 8, zero | 0.2 ms | Root 9 rejected |

At 25 microseconds the final norm is **0.00872637**, above the unchanged
**0.005** threshold. Support rows contribute 0.00871225 and rigid dynamics
0.00049626. Doubling Newton iterations still fails at root 9, with norm
0.0105468. The failed root publishes a zero acceptance token and does not
advance accepted physical time. The completed 100/50-µs trajectories alone
cannot establish a convergence trend.

An opt-in GPU snapshot captures all 16 contact assemblies and the final
certificate before rollback. Contact row 5 dominates the final residual; its
normal component is 0.00807643. Stored root height changes in **119.209 nm**
steps. For example, the last Newton correction requests -11.275 nm of root
vertical motion but the stored root coordinate changes by -119.209 nm. Earlier
requested movements of tens of nanometres leave that coordinate unchanged.
These are direct candidate-buffer observations. Separately, a first-order
Jacobian comparison shows gap discrepancies up to approximately 0.23 µm in
selected iterations; that comparison also contains higher-order error and is
not proof that coordinate rounding explains every residual component.

## Next mechanical gate

The next repair belongs in the native candidate and accepted-state precision
contract. Freeze its representation and ownership before adding more stance
control or longer behavior runs. A retained low displacement, if required,
must participate in the same accepted-state proof, fingerprint, checkpoint,
rollback, retry, initial-state serialization and sensory publication as the
rest of q. More precise scratch geometry cannot certify a different pose from
the one that is actually published.

Use the retained failing root as the regression: compare candidate q, pose,
point, Jacobian and contact residual at identical iterates; then require the
unchanged 25-µs nonlinear gate and exact replay over the full common duration.
Repeat at an additional refinement level and compare complete q/v, force and
contact histories at common physical times. Independently close full source
force/path error, including the small high-stiffness muscles, before promoting
loaded equilibrium. Preserve rejected roots and every failed refinement level.

The [receipt](media/candidate-precision-performance-20260911/receipt.json) binds
the source snapshots, compiled artifacts, prepared fixtures, complete traces,
failed controls and measured timestamps. Its
[verifier](media/candidate-precision-performance-20260911/verify_receipt.py)
recomputes the measurements and rejects missing frames, incomplete timestamps,
lost final iterates or promotion of a failed prefix. Registered anatomical
tissue, independent calibration, sustained standing/recovery/walking and the
five performance-envelope workloads remain open.

`python3 -m unittest discover -s tests -p 'test_*evidence.py'` passes all 22
evidence tests. Some atomic trace records follow partially flushed diagnostic
text on the same line; the verifier decodes the complete JSON record and
requires its terminating newline, exact identity and byte extent. It still
rejects missing, duplicated or truncated records. All raw logs are retained.
