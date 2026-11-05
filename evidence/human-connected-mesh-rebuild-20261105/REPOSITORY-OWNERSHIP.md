# Numi Human repository ownership

Human-specific implementation, anatomy configuration, source/provenance records,
compiler integration and native-scene qualification evidence belong in
[Numi2/numilab-human](https://github.com/Numi2/numilab-human).
Shared Matter physics/Metal solver implementation remains in its owning
Numi2/numi-lab repository; Brain implementation remains in Numi2/numi-brain.
Do not copy those runtimes into Human or put Human trial dumps into numi-lab.

This directory migrates the 293 tracked checkpoint files from numi-lab at
b7192cda84958d51085dd4ce17b220ddf467f5a6.
REPOSITORY-MIGRATION.json verifies their exact original bytes. Historical
commands, absolute execution paths and embedded source hashes intentionally
remain unchanged; relocation does not qualify a new runtime or repeat a test.

The screened candidate341 passed source construction and conservative mass
transfer, but native344 rejected after 26 accepted microsteps. It is unpromoted.
The later conditioning trial347 also rejected; its result remains on SSH Numi
pending analysis/publication. No sustained-motion or performance success is claimed.

Continue using the existing source/compiler/runtime owners on SSH Numi.
Publish subsequent Human-specific results here; retain failed evidence compactly
and keep bulk recoverable artifacts in verified release archives.
