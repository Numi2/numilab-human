# Native skin/fat volume capacity preflight

The current native loader accepts at most 300,000 child nodes and 200,000 child cells. A fixed triangular boundary with Euler characteristic chi requires at least (F - chi) / 2 tetrahedra. Skin and fat are counted separately; their shared I coordinates are counted once in the combined node bound.

The tested preflight now runs in the queued native-domain-217 handoff before any TetGen execution. It retains the existing bounds and pinned geometry/readiness checks. The retained fine diagnostic boundary requires at least 390,629 cells, so it cannot fit this loader before any additional quality refinement. This is a necessary bound, not physical admission.

The reduced mechanical boundary work continues. Detailed anatomy is retained for inspection and conservative physical transfer; increasing execution resolution simply to copy the detailed surface is not a qualification result.

Validation: seven focused pytest cases plus the actual handoff collar-orientation/shared-I fixture and existing typed-D/protected149 handoff tests. Runtime and accepted steps for the repaired scene remain zero.
