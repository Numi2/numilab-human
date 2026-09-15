# Published native Human runtime source tuple, 15 September 2026

This release closes a delivery gap: the tested native runtime no longer exists
only as a local Mac mini branch and a retained patch. It is a two-repository,
immutable source tuple:

- Native runtime source: [`Numi2/numi-lab`](https://github.com/Numi2/numi-lab),
  tag `human-native-step281-rank-audit-20260915`, commit
  `337741b51bfc4a5552a837dacb4d0b5c4268d298`.
- Exact runtime input package: this repository, tag
  `human-native-runtime-source-inputs-20260915`, directory
  [`media/native-runtime-source-package-20260915`](media/native-runtime-source-package-20260915).

The input package contains the five binary inputs consumed by the public
runtime command: rigid state, muscle state, tendon attachments, support
contacts, and joint equalities. The structured, hash-locked receipt is
[`source-package-receipt-v1.json`](media/native-runtime-source-package-20260915/source-package-receipt-v1.json).
It records each filename, byte count, and SHA-256, along with the fresh
public-tag replay transcript.

## Fresh public-source build and bounded replay

On an Apple M4 Pro, a fresh clone of the public runtime tag was configured and
built with the `metalrobo_numilab_human_myosim_visual_probe` target. Its binary
SHA-256 was
`b85669fefaf414a28a9eb44531985dcd83fe549eb5b8d6c7747773bedb3e15bc`.
It was run against the published five-input tuple at `12.5 us` for 64 steps
(`0.8 ms`), with the production source passive-joint tissue, unassisted root,
and deterministic replay enabled. The retained transcript reports:

- 416 recruited muscle records;
- maximum acceleration `0.115790568292 m/s2`;
- zero penetration; and
- bitwise deterministic replay.

The command shape is:

```text
metalrobo_numilab_human_myosim_visual_probe \
  input/myosim-fullbody-core-reference.nhrigid \
  input/myosim-fullbody-muscle-reference.nhmyo OUTPUT_DIRECTORY \
  --tendon-payload input/numi-human-tendon-attachments.nhtendon \
  --support-contact-payload input/myosim-fullbody-support-contact.nhcnt \
  --joint-equality-payload input/myosim-fullbody-joint-equalities.nheq \
  --muscle-step-seconds 0.0000125 --muscle-step-count 64 \
  --muscle-activation 0.8 --persistent-metal-stand \
  --persistent-source-passive-joint-tissue --persistent-stand-trace \
  --stand-contact-iterations 64 --stand-deterministic-replay
```

The source-input receipt can be regenerated and checked without mutating an
existing receipt:

```text
numilab-human native-runtime-source-publication \
  --output Docs/media/native-runtime-source-package-20260915/source-package-receipt-v1.json
```

## Boundary

This new build is not byte-identical to the historical executable recorded by
the fibre-root receipt, so it proves a fresh public-source build and retained
bounded replay—not reproduction of that historical binary. It does not pass
the common-duration temporal force-convergence gate, sustained standing,
passive-force physiology, recovery, or walking. No solver formulation,
regularization, passive stiffness, or runtime mechanics changed in this
publication release.
