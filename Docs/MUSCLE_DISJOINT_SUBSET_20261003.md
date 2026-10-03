# Muscle shell disjointness subset — 2026-10-03

The current compiled muscle-surface payload contains **84 closed, embedded shell candidates out of 148 source muscle surfaces**. The exact pairwise audit evaluated all **3,486** candidate pairs: 110 intersect, so the admitted set as a whole is not pairwise disjoint. A conservative subset identifies **14 candidates, representing seven paired named muscles**, each separated from every other member of the admitted 84-candidate set.

| Source member IDs | Source muscle | MyoSim route(s) |
| --- | --- | --- |
| `FJ1431`, `FJ1431M` | Psoas major, right and left | `psoas_r`, `psoas_l` |
| `FJ1506`, `FJ1506M` | Supraspinatus, right and left | `SUPSP`, `SUPSP_l` |
| `FJ1485`, `FJ1485M` | Anconeus, right and left | `ANC`, `ANC_l` |
| `FJ1496`, `FJ1496M` | Flexor carpi radialis, right and left | `FCR`, `FCR_l` |
| `FJ1473`, `FJ1473M` | Humeral head of flexor carpi ulnaris, right and left | `FCU`, `FCU_l` |
| `FJ1497`, `FJ1497M` | Flexor digitorum profundus, right and left | `FDP2–5`, `FDP2_l–FDP5_l` |
| `FJ1494`, `FJ1494M` | Extensor pollicis brevis, right and left | `EPB`, `EPB_l` |

The immutable owner-audit receipt is [receipt-v4.json](media/muscle-volume-disjointness-20261003/receipt-v4.json), SHA-256 `1181e03657181064cafee74667645d28a6b484c31f936cd39ccb3e343ea72d5c`. It binds to payload SHA-256 `7cefa97bf65aa75edddbb7ac4c0a56d5d5f41c8b0aeca4c6146e867cc45d4bd8` and source-manifest SHA-256 `bb4e9c63dd5e526fc28f26141cf53208cb4fce92228091a5f36d9a07e795f64c`.

This is a geometric separation result for source-authored compiled visual shells. It does **not** confirm clinical anatomy, create physical muscle volumes, assign mass or materials, or qualify force transmission, muscle mechanics, a whole-body partition, or standing. The other 64 source muscle surfaces were not admitted to this pairwise set; tendons, skin, bone, organs, and other tissue layers were outside its scope. All whole-body and physical-owner gates remain open.
