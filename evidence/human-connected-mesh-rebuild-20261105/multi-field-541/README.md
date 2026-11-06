# Joint reference-field repair541

618 fixed-boundary interior repairs remove3823 tetrahedra and1676 low-quality cells relative495. Construction has zero native-quality failures. All coordinates and typed surfaces remain unchanged. Each repair must simultaneously satisfy the original strain5percent, cumulative force1percent and energy5percent screens at16,58,64 accepted reference roots. Independent final-field audits pass with zero force/energy failures at all three times.

Native physical transfer, conservative mass compilation, accepted-step/trajectory fidelity and throughput remain pending. These reference fields cover only a short initialization transient, not full motion or breathing qualification. The CPU analytical evaluator is preparation only; it never supplies simulation forces or state.

The compiler-source snapshot is exactly tools/simplify_human_skin_fat_boundary.py at fc4e3f64c (SHA70562ce6b14b075e97f2055152cc1c67532fd4e48e4cc1d9091a6272c6846b33). Existing material evaluator and capture owners are pinned in plan.json. Retained run_prepared.py adds joint conservative screening, without changing native equations or acceptance.
