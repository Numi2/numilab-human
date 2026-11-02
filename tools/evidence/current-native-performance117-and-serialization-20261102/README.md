# Current Human performance and source checkpoint

Tested contact audit/body-wrench parallelization is published in Numi2/numi-lab
at 485a2467b0e7688f60d2a28c865bf8e2b73715ed. Full audit, failed trials, commands and numerical controls:
https://github.com/Numi2/numi-lab/tree/485a2467b0e7688f60d2a28c865bf8e2b73715ed/tools/evidence/human-serialization117-20261102 .

Source geometry and physical mass are unchanged. All 15,453,677 contact rows
remain. Native120 reached the parallel layout but stopped under the existing
+3 GiB swap-growth guard at 271.0 seconds during initial adapter preparation.
No physical step completed; motion and >=0.1x remain unqualified.

Cleanup114 incorrectly classified two transitive native mass-proof witnesses as
superseded. Their GitHub archive and member hashes were independently verified,
both files were restored byte-for-byte, and the two paths are now explicitly
pinned native inputs. Sources and accepted geometry were preserved.

Prioritize the normal adapter's node validation, force/reaction reductions and
stable sparse anchor incidence. Conditional topology and blood-particle paths
also contain poor-scaling work; their individual costs are unmeasured.
