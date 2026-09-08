# Prepared recruitment through Brain

The exact stationary recruitment can now be authored from the native-admitted
NHINIT1 state and executed through Brain's existing muscle controller. The
matched Mac mini comparison passes command bounds, physical activation/force
response, sensor dropout and bitwise replay. Early settling remains essentially
unchanged; this increment does not qualify standing or walking.

The [offline stance](COUPLED_EQUILIBRIUM_20260908.md) contains 261 nonzero
excitations, including two exactly equal to 1. The earlier program author
accepted only uniform tonic recruitment, while Brain limited the maximum to
0.999. Those interfaces could not represent the prepared recruitment exactly.

`numi-brain-gate-c describe-body` now includes the admitted prepared-state
SHA-256, fingerprint, world and dimensions. Human's `locomotor-program` accepts
`--prepared-state` instead of `--tonic` and checks the source archive, composed
model identity, world, clock, dimensions, exact payload extent, finite state,
unit root quaternion and static excitation/activation/fibre state. It preserves
every prepared excitation in source order. Actuator coverage follows the
admitted payload and existing 4096-channel descriptor budget.

The prepared candidate requires zero feedback gains and zero period. Source
NHMYO2 reference lengths remain in the descriptor but do not affect this tonic
candidate; they are not claimed as path lengths at the new pose. Brain now
represents excitation 1 with finite logit 10, whose tanh rounds to 1 in FP32.
Smaller excitations retain the previous conversion. Ordinary body/joint risk,
physiology, inhibition and protective motor routing continue to determine the
delivered command. Bootstrap and missing spindle observations yield zero.

## Measured result

Native `1ca086546e81b00251bed0d4eeef48e8e9ecca92`, Brain
`30f138fa` and Human authoring `f71d98c6464fd136e905cf232b20afd08f53aa28`
produce the [retained receipt](media/prepared-recruitment-20260908/receipt.json).
The prepared fixture is unchanged: NHCNT2/NHEQ2/NHLIM1, 129 q, 128 v,
416 muscles, and three tiny pelvis FEM samples. It contains no registered
anatomical tissue. All physical execution used the Apple M4 Pro on `ssh macmini`.

The comparison has four scenarios: prepared recruitment, independent replay,
zero proposal and spindle dropout. Each accepts four 100-microsecond roots,
representing **0.4 ms per scenario**. Full q/v, command, activation and applied
tendon-force bytes are retained. Recruited/replay histories are bitwise equal;
dropout and zero-command physical histories are also bitwise equal.

| Observable at the fourth root | Prepared recruitment | Zero proposal |
|---|---:|---:|
| Maximum delivered excitation | 0.2822821 | 0 |
| Right ankle velocity, rad/s | 0.1688445 | 0.1688394 |
| Root linear speed, m/s | 0.0154928 | 0.0154830 |

The maximum terminal activation difference is 0.00416493 and the maximum
applied tendon-force difference is 0.196030 N. Maximum terminal q difference
is 1.86265e-9 across mixed coordinate types. These establish a physical response
to the delivered proposal; they do not establish improved settling. The ordinary
motor path attenuates recruitment, and the body remains in motion.

Four Human authoring tests, three Brain descriptor/kernel/checkpoint tests,
the prepared comparison, the existing locomotor regression and seven evidence
tests pass. The clean comparison takes 34.9 seconds including four runtime
constructions, Brain work, sensing and qualification readbacks. A concurrent
NumiVivo `md-run` is recorded; this elapsed time is not a performance measurement.
Dirty exploratory attempts remain retained and explicitly unqualified.

## Reproduction and remaining work

Use `describe-body` with the same complete native v7 configuration, then:

```sh
numilab-human locomotor-program body.json muscle.nhmyo program.json \
  --prepared-state prepared.nhinit --length-gain 0 \
  --velocity-gain-seconds 0 --maximum-excitation 1
```

The evidence directory contains the exact description, program, compressed
NHMYO2 source payload, command requests, runtime logs and source states. NHMYO2
is derived from the pinned Apache-2.0 MyoSim source identified by the receipt.
The existing `calibrationArtifactSHA256` field binds NHINIT1 for this candidate;
its name does not establish experimental calibration.

```sh
python3 Docs/media/prepared-recruitment-20260908/verify_receipt.py
python3 Docs/media/prepared-recruitment-20260908/test_verify_receipt.py
```

The next mechanical work is source-compliant loaded equilibrium and tissue
registration at the prepared mass frames. Static NHEQ1/hard-stop reactions are
not applied as runtime assistance. A support-aware controller and calibrated
pose-specific feedback must then meet the unchanged standing/recovery/walking
protocol. Costal timeout diagnosis, tissue calibration and sustained behavior
remain open.
