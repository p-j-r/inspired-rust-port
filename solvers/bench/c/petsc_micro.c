/* PETSc kernel micro-benchmarks: MatMult, VecDot, VecAXPY (best of -reps). */
#include "common.h"

static double Best(double *t, PetscInt n)
{
  double m = t[0];
  for (PetscInt i = 1; i < n; i++) if (t[i] < m) m = t[i];
  return m;
}

int main(int argc, char **argv)
{
  Prob        p;
  Mat         A;
  Vec         x, y, z1, z2, z3, *vs;
  PC          pc[4];
  const char *pcn[4] = {PCJACOBI, PCSOR, PCILU, PCICC};
  double      tpc[4], mdot, maxpy;
  PetscScalar h[32];
  PetscInt    j;
  PetscInt    reps = 50, i;
  PetscScalar d;
  double      t0, *t, spmv, dot, axpy;
  PetscMPIInt size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  PetscCall(ReadProblem(argv[1], &p));
  PetscCall(BuildMat(&p, &A));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(FillVec(p.b, x));
  PetscCall(VecDuplicate(x, &z1));
  PetscCall(VecDuplicate(x, &z2));
  PetscCall(VecDuplicate(x, &z3));
  PetscCall(VecSet(z1, 1.0));
  PetscCall(VecSet(z2, 2.0));
  PetscCall(VecSet(z3, 3.0));
  PetscCall(PetscMalloc1(reps, &t));
#define TIME(stmt, out) \
  PetscCall(stmt); \
  for (i = 0; i < reps; i++) { PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD)); t0 = MPI_Wtime(); PetscCall(stmt); t[i] = MaxTime(MPI_Wtime() - t0); } \
  out = Best(t, reps)
  TIME(MatMult(A, x, y), spmv);
  {
    double spmv_cold;
    PetscCall(MatMult(A, x, y));
    for (i = 0; i < reps; i++) {
      PetscCall(VecAXPY(z1, 0.5, z2)); PetscCall(VecAXPY(z2, 0.5, z3)); PetscCall(VecAXPY(z3, 0.5, z1));
      PetscCall(VecAXPY(x, 1e-9, z1));
      t0 = MPI_Wtime(); PetscCall(MatMult(A, x, y)); t[i] = MPI_Wtime() - t0;
    }
    spmv_cold = Best(t, reps);
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "spmv_after_other_vector_traffic_s=%.4e\n", spmv_cold));
  }
  TIME(VecDot(x, y, &d), dot);
  TIME(VecAXPY(y, 0.5, x), axpy);
  /* Preconditioner application and 16-vector orthogonalisation kernels (GMRES/Krylov-Schur cost). */
  for (j = 0; j < 4; j++) {
    if (size > 1 && j >= 2) { tpc[j] = 0; continue; }
    PetscCall(PCCreate(PETSC_COMM_WORLD, &pc[j]));
    PetscCall(PCSetOperators(pc[j], A, A));
    PetscCall(PCSetType(pc[j], pcn[j]));
    PetscCall(PCSetUp(pc[j]));
    TIME(PCApply(pc[j], x, y), tpc[j]);
    PetscCall(PCDestroy(&pc[j]));
  }
  PetscCall(VecDuplicateVecs(x, 16, &vs));
  for (j = 0; j < 16; j++) PetscCall(VecSetRandom(vs[j], NULL));
  TIME(VecMDot(x, 16, vs, h), mdot);
  TIME(VecMAXPY(x, 16, h, vs), maxpy);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "{\"impl\":\"petsc\",\"kind\":\"micro\",\"n\":%d,\"nnz\":%d,\"threads\":%d,\"spmv_s\":%.17e,\"dot_s\":%.17e,\"axpy_s\":%.17e,\"pc_jacobi_s\":%.17e,\"pc_sor_s\":%.17e,\"pc_ilu_s\":%.17e,\"pc_icc_s\":%.17e,\"mdot16_s\":%.17e,\"maxpy16_s\":%.17e}\n", (int)p.n, (int)p.nnz, (int)size, spmv, dot, axpy, tpc[0], tpc[1], tpc[2], tpc[3], mdot, maxpy));
  PetscCall(PetscFinalize());
  return 0;
}
