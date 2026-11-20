"""Build the staged scheduler with the original 32-column linear budget and an 8-pass middle tier."""
from pathlib import Path

builder = Path("build_staged_solver.py")
source = builder.read_text()
edits = (
    ('root / "staged-solver-build"', 'root / "staged-solver-accuracy-build"'),
    ('constexpr std::uint32_t middleBudget = 4u;', 'constexpr std::uint32_t middleBudget = 8u;'),
    ('                constexpr std::uint32_t shortLinearBudget = 16u;\n', ''),
    ('normalBudget > middleBudget &&\n                            normalLinearBudget > shortLinearBudget,',
     'normalBudget > middleBudget,'),
    ('{shortBudget, shortLinearBudget}', '{shortBudget, normalLinearBudget}'),
    ('staged-solver-budgets.patch', 'staged-solver-accuracy-budgets.patch'),
    ('staged-solver-build.log', 'staged-solver-accuracy-build.log'),
)
for old,new in edits:
    assert source.count(old) == 1, old
    source = source.replace(old,new,1)
exec(compile(source, str(builder), "exec"))
