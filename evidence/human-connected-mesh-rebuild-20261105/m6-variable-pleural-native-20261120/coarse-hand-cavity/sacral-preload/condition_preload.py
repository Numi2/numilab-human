"""Add an optional descent fallback to the existing offline Newton solver."""
from argparse import ArgumentParser
from difflib import unified_diff
from pathlib import Path


def edited(text):
    substitutions = (
        ("from scipy.sparse import coo_matrix\n", "from scipy.sparse import coo_matrix, diags\n"),
        ("def solve(model, v, count, weights=(.05, .15, .35, .65, 1.), penalty=None, guard=False):",
         """def solve(model, v, count, weights=(.05, .15, .35, .65, 1.), penalty=None, guard=False,
          *, regularize=False, converged=None):"""),
        ('''                row["outcome"] = "converged"
                break''',
         '''                row["outcome"] = "converged"
                if converged is not None:
                    converged(weight, v.copy())
                break'''),
        ('''            try:
                direction, guards = feasible_direction(model, tangent, v, derivative, jac, guard)
            except RuntimeError:
                row["outcome"] = "singular_tangent"
                return v, trace
            if guards:
                row["active_shape_search_rows"] = guards
            rate = derivative @ direction
''',
         '''            try:
                direction, guards = feasible_direction(model, tangent, v, derivative, jac, guard)
            except RuntimeError:
                if not regularize:
                    row["outcome"] = "singular_tangent"
                    return v, trace
                direction, guards = np.full_like(v, np.nan), 0
            rate = derivative @ direction
            if regularize and (not np.isfinite(rate) or rate >= 0.):
                # Preserve mechanical energy and its exact derivative. Modify
                # the search metric only, with the native full residual and
                # determinant guards still applied to every accepted stage.
                diagonal = diags(np.maximum(np.abs(tangent.diagonal()), 1.), format="csc")
                for shift in (1.e-3, 1.e-2, 1.e-1, 1., 10., 100.):
                    try:
                        direction, guards = feasible_direction(
                            model, tangent + shift * diagonal, v, derivative, jac, guard)
                    except RuntimeError:
                        continue
                    rate = derivative @ direction
                    if np.isfinite(rate) and rate < 0.:
                        row["search_metric_diagonal_shift"] = shift
                        break
            if guards:
                row["active_shape_search_rows"] = guards
'''),
    )
    for old, new in substitutions:
        if text.count(old) != 1:
            raise ValueError("unexpected preload solver input")
        text = text.replace(old, new, 1)
    return text


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    baseline = args.source.read_text()
    revised = edited(baseline)
    patch = "".join(unified_diff(baseline.splitlines(keepends=True), revised.splitlines(keepends=True),
                                  fromfile="a/predict.py", tofile="b/predict.py"))
    args.out.mkdir()
    (args.out / "predict.py").write_text(revised)
    (args.out / "descent-search.patch").write_text(patch)
