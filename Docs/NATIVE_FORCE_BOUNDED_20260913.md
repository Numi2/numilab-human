# Native full-body force-convergence boundary, 13 September 2026

This receipt closes the missing **execution boundary** for the one-adult-male
Human source package. It does not claim that the boundary is physically or
biologically qualified. The run is retained so the remaining gaps fail closed
with measured values instead of an inferred pass.

## Source and build identity

The physical Mac mini build was made from the isolated MetalRobo worktree
`human-force-convergence-20260913` at commit
`7c7513ee1859c69d647c47b5a247f4702609a3b9` (base source
`f239c6314bc3912c641db6eb909525eaa098f20a`). The Release target was built
with `BUILD_TESTING=OFF` because the repository's testing-on configure path
references the pre-existing missing target
`metalrobo_numanx_fullbody_bridge_probe`.

The produced artifacts are identified by these SHA-256 values:

| artifact | SHA-256 |
| --- | --- |
| `metalrobo_numilab_human_myosim_visual_probe` | `1d20e984789929739eecba0e6429b67267e8424651da75959cbfbd1a28c33fe8` |
| `libmetalrobo.dylib` | `bb99dc1f6d08b9c6885b6d4f1908c9b160187cd2463480d9fc171d14f364aeff` |
| Metal library | `d0532b835652395d7f0af7e34ee6a700a2db4e65402b1cbdd13dedf32486a65e` |
| build log | `3cf78e97b4f36ed3bebc8bd9cd1cfe4ee6b4381fb3c6929c03c6a0fc257d40ba` |

The source payload hashes are retained in the run notes and cover the rigid,
muscle, equality, tendon, and support inputs. The native source commit could
not be pushed from the Mac mini because its configured HTTPS remote had no
available credentials; the commit and worktree remain locally addressable.

## Full-body result

The run uses one adult-male source package with 157 bodies, 128 velocity
degrees of freedom, and 129 configuration coordinates. It executes 512
steps at exactly 12.5 microseconds (`6.4 ms`) with source support capsules,
joint equalities, tendon attachments, activation set to one, and the
persistent Metal stand owner.

The accepted large-state fallback completed all 512 steps. Its measured
boundary is:

| metric | result |
| --- | ---: |
| maximum acceleration | `46673.1992188 m/s²` |
| maximum velocity delta | `0.47243475914` |
| maximum configuration delta | `0.00158670963719` |
| maximum penetration | `1.94772340478e-7 m` |
| compiled stand balance | `false` |
| root-force residual | `776.829137423 N` |
| active support contacts | `2 / 18` |
| source CPU/Metal force delta | `0.0100077937057 N` |
| one-step replay | bitwise |
| same-horizon replay | not proved |

The exact dense ABA stage is explicitly ineligible for this package because
the existing bounded kernel contract is 32 bodies, 40 velocity degrees of
freedom, and 41 configuration coordinates. The host therefore records
`large_state_fallback`; an unguarded exact attempt is retained as a failure
receipt (`code=9`, `failing_index=7`) rather than being presented as a pass.

The machine-readable receipt is
[`media/native-force-bounded-20260913/receipt.json`](media/native-force-bounded-20260913/receipt.json).
The raw accepted run, one-step replay, rejected exact-stage outputs, and build
log are in the same directory. The receipt reports `status=partial` and
keeps `force_convergence=false` because the declared acceleration, velocity,
configuration, and static-balance gates fail.

## Remaining completion gates

The result closes only the clock/horizon execution measurement. The following
gates remain explicitly false for the one adult male:

* force convergence and assistance-free sustained standing;
* perturbation recovery and walking with held-out gait metrics;
* anatomical supports/loading with registered tissue contact and calibrated
  friction/compliance;
* activation calibration against held-out measurements and dynamic
  fibre/tendon state;
* two-way blood mass and momentum transfer through registered vessels and
  tissue exchange;
* unresolved organ, vessel, joint, and soft-tissue materials/density data;
* subject-specific anatomical, inertial, and motion calibration.

The next admissible work is to remove one declared failure at a time: first
make the full-body solver converge at the exact clock, then qualify loaded
anatomical contact and activation on the same source package, then admit blood
and material owners, and finally run the single-male standing/recovery/walking
protocol. A shorter or synthetic run cannot promote any of these claims.
