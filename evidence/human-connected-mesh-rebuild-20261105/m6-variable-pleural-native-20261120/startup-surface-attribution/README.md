# Source registration accounts for most measured skin crossings

The exact native packed reference was audited before physical stepping and compared with the retained 16th accepted root on SSH Numi. Both use the same skin and bank-local triangle connectivity wherever a comparison is reported. The initial capture was verified against the terminal receipt's recorded base pack hash. Initial articulated transforms came from the native source-registration log and terminal transforms came from the accepted receipt. Both were composed in the existing auditor and rounded to Float32 before the same exact predicates were applied.

| Nonocular triangle pairs | Count |
| --- | ---: |
| Confirmed before physical stepping | 12,574 |
| At the accepted root 16 | 14,179 |
| Same skin triangle, same target triangle, same typed source bank | 12,567 |
| Startup only among comparable banks | 7 |
| Terminal only among comparable banks | 5 |
| Terminal pairs missing a verified startup transform | 1,607 |

At least 88.6% of the terminal pairs are already present in the native startup assembly. Both snapshots have zero skin self-pairs. The right extensor bank 137 has the same 1,794 pairs in each snapshot. Thirteen startup banks require a body transform absent from the startup registration log. All 1,607 terminal pairs without a startup match in those banks are on the right and left external oblique, banks 67 and 68, which require body 7. Zero in their startup count means no observed pairs were counted, and their actual startup status is unknown. The raw pair counts give neither penetration depth nor contact ownership. Ocular openings remain separate.

The exact same native run had zero terminal bed reaction and 0.250 ms of accepted motion. The geometry comparison establishes a source-registration problem independent of that run's solver changes. It does not establish a force-balanced position, classify intended internal interfaces, or certify 16 seconds of motion.

`audit_startup.py` uses the retained Float32 audit owner and adds an explicit startup-only path for native captures with partial authored pose records. Run it on SSH Numi as `python3 audit_startup.py NEW_DIRECTORY`. It checks raw archive hashes and uses an ephemeral staging directory. The compressed report gives the exact startup triangle-pair witnesses. `comparison.json` retains bank-specific overlap and pose-coverage results. The terminal report is pinned from the existing evidence in commit `b302ab5b8`.
