/* Reads the problem files written by `rsolvers-bench gen` (little-endian):
 *   "RSBENCH1", u64 n, u64 nnz, i32 indptr[n+1], i32 indices[nnz], f64 values[nnz], f64 b[n], f64 v0[n]
 */
#ifndef RSBENCH_COMMON_H
#define RSBENCH_COMMON_H
#include <petsc.h>
#include <stdint.h>
#include <string.h>

typedef struct {
  PetscInt n, nnz;
  int32_t *ip, *ix;
  double *val, *b, *v0;
} Prob;

static PetscErrorCode ReadProblem(const char *path, Prob *p)
{
  FILE *f = fopen(path, "rb");
  char magic[8];
  uint64_t n, nnz;
  PetscFunctionBeginUser;
  PetscCheck(f, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot open %s", path);
  PetscCheck(fread(magic, 1, 8, f) == 8 && !memcmp(magic, "RSBENCH1", 8), PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad magic");
  PetscCheck(fread(&n, 8, 1, f) == 1 && fread(&nnz, 8, 1, f) == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad header");
  p->n = (PetscInt)n;
  p->nnz = (PetscInt)nnz;
  PetscCall(PetscMalloc5(n + 1, &p->ip, nnz, &p->ix, nnz, &p->val, n, &p->b, n, &p->v0));
  PetscCheck(fread(p->ip, 4, n + 1, f) == n + 1, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "indptr");
  PetscCheck(fread(p->ix, 4, nnz, f) == nnz, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "indices");
  PetscCheck(fread(p->val, 8, nnz, f) == nnz, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "values");
  PetscCheck(fread(p->b, 8, n, f) == n, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "b");
  PetscCheck(fread(p->v0, 8, n, f) == n, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "v0");
  fclose(f);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build the parallel matrix from this rank's rows and fill vectors from global arrays. */
static PetscErrorCode BuildMat(const Prob *p, Mat *A)
{
  PetscMPIInt rank, size;
  PetscInt    m, rstart, i, *lip, *lix, nl;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (size == 1) { /* true sequential baseline: SeqAIJ, no MPI/diag-offdiag split */
    PetscInt *ip, *ix, i;
    PetscCall(PetscMalloc2(p->n + 1, &ip, p->nnz, &ix));
    for (i = 0; i <= p->n; i++) ip[i] = p->ip[i];
    for (i = 0; i < p->nnz; i++) ix[i] = p->ix[i];
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, p->n, p->n, ip, ix, p->val, A));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  m      = p->n / size + (rank < p->n % size ? 1 : 0);
  rstart = rank * (p->n / size) + (rank < p->n % size ? rank : p->n % size);
  nl     = p->ip[rstart + m] - p->ip[rstart];
  PetscCall(PetscMalloc2(m + 1, &lip, nl, &lix));
  for (i = 0; i <= m; i++) lip[i] = p->ip[rstart + i] - p->ip[rstart];
  for (i = 0; i < nl; i++) lix[i] = p->ix[p->ip[rstart] + i];
  PetscCall(MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, m, m, p->n, p->n, lip, lix, p->val + p->ip[rstart], A));
  PetscCall(PetscFree2(lip, lix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillVec(const double *g, Vec v)
{
  PetscInt     lo, hi, i;
  PetscScalar *a;
  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(v, &lo, &hi));
  PetscCall(VecGetArray(v, &a));
  for (i = lo; i < hi; i++) a[i - lo] = g[i];
  PetscCall(VecRestoreArray(v, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static int cmp_double(const void *a, const void *b)
{
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static double MaxTime(double t)
{
  double g;
  MPI_Allreduce(&t, &g, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);
  return g;
}
#endif
