# Shared-source moving skin correction

The existing anatomy preparation module now corrects a single NHSKIN source across multiple accepted poses. It pulls required outward displacement through each registered source map, recomputes candidate maps, and checks the existing exact surface predicates after Float32 serialization. It preserves original source orientation, complete target inventories, ocular interfaces, fixed contacts/selectors, bed-gap floors, and cumulative displacement relative to the original baseline.

A resumed correction must reproduce its hash-bound source and pass exact self-intersection checks for every pose before further work. Collateral folded faces away from required correction seeds use zero incremental displacement at their core, with a local source-graph harmonic transition to the surrounding proposal. This cannot alter fixed anchors or active required seeds. The original full candidate gates still decide acceptance.

The 27 focused tests pass in the publication checkout. The two changed files exactly match candidate source a2eeb789e1b2f147b045e5383df1780e6ef7f3ac; publication revision 163f1c3b9409b52345029f2abbf3ec918f86bedb incorporates them on the current Human source. The raw file hashes and test command are retained here.

This is an offline inferred registration correction in the existing asset owner, not a second runtime simulator. The full native moving-skin candidate remains unfinished and unqualified. Tests and intermediate pair-count decreases do not establish anatomical clearance.
