// Preparation-only factorization of the scalar material extension. Native physics stays on Metal.
#include <Accelerate/Accelerate.h>
#include <new>

struct Factor { SparseOpaqueFactorization_Double value; };

extern "C" void* make_factor(int n, long* offsets, int* rows, double* entries) {
    SparseMatrixStructure structure{};
    structure.rowCount=n;
    structure.columnCount=n;
    structure.columnStarts=offsets;
    structure.rowIndices=rows;
    structure.attributes.kind=SparseSymmetric;
    structure.attributes.triangle=SparseLowerTriangle;
    structure.blockSize=1;
    auto factor=SparseFactor(SparseFactorizationCholesky, SparseMatrix_Double{structure,entries});
    if (factor.status!=SparseStatusOK) {
        SparseCleanup(factor);
        return nullptr;
    }
    return new Factor{factor};
}
extern "C" void solve_factor(void* handle,int n,int nrhs,double* entries) {
    DenseMatrix_Double rhs{};
    rhs.rowCount=n;
    rhs.columnCount=nrhs;
    rhs.columnStride=n;
    rhs.data=entries;
    SparseSolve(static_cast<Factor*>(handle)->value,rhs);
}
extern "C" void free_factor(void* handle) {
    auto factor=static_cast<Factor*>(handle);
    SparseCleanup(factor->value);
    delete factor;
}
