# Staged solver checkpoint

Both paired trials used the Apple M6 and the existing supported regional runtime with its 34 ligament rows, 26 fascia regions, two ACL FEM tissues, full vascular coupling and variable pleural pressure. Physical body cadence remained 2 ms with 256 tissue microticks per root. The input assets and the exact environment are pinned in each launch receipt. Both runs passed the owning terminal FEM audit and accepted 18 steps through 36 ms. The headless command included native preparation, controller and physiology. Native presentation and whole-body geometry capture were outside this measurement.

| Run | Newton / linear iteration limits by tier | Wall time | Simulated / wall | Comparison |
| --- | --- | ---: | ---: | --- |
| Retained control | 2/32, 24/32 | 308.568 s | 0.000116668× | Paired control in the parent evidence |
| Staged | 2/16, 4/32, 24/32 | 275.746 s | 0.000130555× | 1.119× speedup, below prediction, traces changed at step seven |
| Staged with original linear budget | 2/32, 8/32, 24/32 | 197.404 s | 0.000182367× | 1.563× speedup, traces changed at step 16 |

The accuracy trial matched the control physiology and body momentum traces through step 15. Step 16 rejected its two-pass attempt with byte-identical rollback. The eight-pass attempt accepted step 16, and steps 17 and 18 accepted on their first eight-pass attempts. Pleural pressure ranged from −515.839 to −490.642 Pa and root assistance remained zero. The maximum paired differences were 0.000397 mmHg PaO₂, 0.0000344 mmHg PaCO₂, 0.000109 N·s summed normal support impulse and 0.000000150 m peak bed penetration. Its final FEM determinants ranged from 0.960678 to 1.032956.

This remains an experiment. The preregistered exact trace gate failed, so these source patches are retained as reproductions without a default runtime change. The fine skin/fat mesh and its offline mass map are separate from the regional scene exercised here. Whole-body motion crossings and the 16-second duration and 0.1× speed gates remain open.
