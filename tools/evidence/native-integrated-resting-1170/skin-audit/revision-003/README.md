# Registered full-cycle NHSKIN audit runner (1172)

This runner applies the existing exact Float32-lattice triangle intersection predicates to the registered 927 NHSKIN payload against the 859 inventoried target surfaces (including 17 ocular monitors), plus NHSKIN self-intersections, at every accepted geometry capture in one registered study arm. It requires an exact 155000-root terminal and exactly eight captures, with the registered receipt adapter supplying the schedule and source bindings. It keeps all witness and degenerate-triangle records; it uses no geometric tolerance or contact exemptions. It does not claim continuous-time clearance or clinical qualification.

The input audit pins are embedded in `audit_full_skin_cycle_1172.py`. In particular, it uses the 1171 registered-receipt adapter (SHA-256 `ba0b88416dbf9cab892f9daa6f954e931003f9db5075db6b758c209370efc1e8`), exact-intersection module (SHA-256 `11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb`), clearance/pack reader (SHA-256 `1321c31c22e1b1dbf0062947c6767c2cfcae9aae35d29e9ea775862feed5ddd1`), 908 audit core (SHA-256 `2eba147ca37ea80a3ed12c8dd725986d77bcd60c194077fae6ddfc5961d9dcde`), and accepted-pack validator (SHA-256 `eb9e5c762cdbab1e9b3f98c63a1580a21acba54637b105c28e2bc0a39187ba17`). The asset chain pins the 927 NHSKIN payload, its composition report and geometry registration, and the 890 complete target inventory. The NHSKIN composition code pin is checked against the retained source-history mapping in the 908 audit core.

The audit predicate modules load under a private package namespace because the P18 receipt adapter imports a different `numilab_human` checkout; the exact integration-repository predicate, clearance, geometry, model, physiology, and package-init files are all SHA-pinned. This prevents `sys.modules` from silently substituting the study-preparation checkout.

The runner uses the Mac mini Homebrew Python 3.13 interpreter and the already-cached NumPy 2.5.3 package. Reproduce its focused tests with:

```sh
ssh macmini PYTHONPATH=/Users/n/.cache/uv/archive-v0/CIJsFXWu5gIt7VCR/lib/python3.13/site-packages /opt/homebrew/bin/python3.13 -m unittest -v /Users/n/numi-human-resting-evidence-20261005/native-lung-late-skin-audit-runner-1172/revision-003/test_full_skin_cycle_1172.py
```

For a future registered arm, first run its non-writing preflight with the exact registered trial path, arm, NHA path and SHA, and a not-yet-created output path:

```sh
PYTHONPATH=/Users/n/.cache/uv/archive-v0/CIJsFXWu5gIt7VCR/lib/python3.13/site-packages /opt/homebrew/bin/python3.13 /Users/n/numi-human-resting-evidence-20261005/native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py --trial <registered-trial-path> --arm <control-or-treatment> --nha <registered-NHA-path> --nha-sha256 <registered-NHA-SHA256> --out <fresh-output-path-under-evidence-root> --validate-only
```

After reviewing the preflight, remove only `--validate-only` and use a different fresh output directory for the actual audit. Do not run against the missing 917 study or substitute the 914 trace peer. This runner is prepared for successful P18 registered arms; its full-skin scan is pending after registered arm closure.

Revision 003 also checks the exact pack surface set: all 859 pinned target surfaces, 185 bone-member supports needed by the existing pose reconstruction reader, and the NHSKIN shell. The 185 supports are not additional collision targets.
