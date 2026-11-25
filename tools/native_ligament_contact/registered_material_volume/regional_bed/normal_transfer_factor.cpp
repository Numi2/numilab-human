// Host-side factorization for preparing a static starting field on macOS.
#include <Accelerate/Accelerate.h>
#include <cstdio>
#include <cstdlib>

extern "C" int solve_normal_transfer(int n, long* starts, int* rows, double* values, double* rhs) {
    SparseAttributes_t attributes{};
    attributes.kind = SparseSymmetric;
    attributes.triangle = SparseLowerTriangle;
    SparseMatrix_Double matrix{{n, n, starts, rows, attributes, 1}, values};
    SparseSymbolicFactorOptions order{};
    order.orderMethod = SparseOrderMetis;
    order.malloc = std::malloc;
    order.free = std::free;
    order.reportError = [](const char* message) { std::fprintf(stderr, "%s\n", message); };
    auto factor = SparseFactor(SparseFactorizationCholesky, matrix, order, SparseNumericFactorOptions{});
    const auto status = factor.status;
    if (status == SparseStatusOK) {
        SparseSolve(factor, DenseVector_Double{n, rhs});
    }
    SparseCleanup(factor);
    return static_cast<int>(status);
}
