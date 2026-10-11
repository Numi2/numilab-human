# Passive anatomy provenance correction

The left deep masseter162 manifest copied a reference_surface_reconstruction record for superficial masseter160. Its top-level canonical identity, source object, candidate geometry and native measurements were correct, but that inherited reconstruction record was not. The original frozen scene and evidence remain unchanged. The corrected manifest points to the actual deep-left candidate and report, preserving the explicit historical trial161 alias and the exact payload SHA.

The existing append owner now rejects a supplied reconstruction whose stable identity or final mesh counts disagree with the appended row, and rejects a conflicting external source member. It continues to accept rows without an optional reconstruction record. Thirty-one append tests pass, including the actual error pattern and an explicit historical alias. The corrected record is consumed by subsequent scene preparation; it is not a new anatomy qualification.
