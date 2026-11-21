# Grouped contact-force dispatch on Apple M6

The exact fine-volume scene recorded 15,449,893 FEM–body contact sample rows. The old kernel launched one GPU thread for every row. Only the first row of each contiguous slave-node group assembled forces. The new launch assigns one thread per slave-node group. It preserves row order within a group, contact laws, validity checks and certification. The Metal library and runtime must come from the same Lab revision.

The native probe uses the production cached kernels and the recorded number of sample rows. Its 86-body / 79-singleton partition is representative; the 86-body count comes from the retained native contact bank, and the fixture's forces and group membership are synthetic. Five alternating pairs on Apple M6 produced median GPU durations of 10.4971665 ms and 3.62258335 ms, a 2.90× speedup for this kernel. Every probe compared all 2,875,648 output bytes with a bitwise force oracle. Dispatch table storage fell from 61,799,572 to 2,875,648 bytes in the fixture. Other buffers and whole-scene throughput were outside this measurement.

The existing contact math, Metal triangle coupon and two-environment cache tests passed. The cache test's maximum absolute force error was 1.49e-8. The production Objective-C++ translation unit passed syntax checking. The runner documents a fresh two-library comparison from the exact Lab revisions.

The prior full-volume native prefix accepted 16 roots totaling 0.25 ms. It measured 14,179 nonocular skin–tissue triangle pairs and zero terminal bed reaction. Variable pleural pressure was enabled; the prefix was too short to observe a pressure change. These results still leave the supported equilibrium, crossing and combined motion requirements open.
