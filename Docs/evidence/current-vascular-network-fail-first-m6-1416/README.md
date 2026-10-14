# Fail-first current vascular network gate
This CPU preparation/geometry check uses actual native007 source-zero capture and an independently reconstructed rigid source baseline. It does not run a simulation or qualify anatomy.

The bounded exact P6/Rat source-baseline collar extension on link 475 to 479 passed. The first five declared links passed their source-bound interface, closed-solid and independent caliber gates. At link six, 525 to 469, both candidate and independent baseline reported three off-patch contact pairs: (704,184), (704,185), (737,176). The check stopped there in 43.17 seconds; later links and all-target clearance did not run.

The next change is to test these three source contacts against bounded indexed collar ancestry before altering geometry or the declaration. No generic containment waiver is used. The runner explicitly stops after continuity until a source-bound junction handoff for all-target clearance is implemented.

The four integrated vascular owner suites passed 79 tests in 4.89 seconds on SSH Numi after the code was cherry-picked into main. The frozen execution source snapshot remains the commit and module hashes in the report, rather than this publication commit.
