/* PETSc KSP timing harness: ./petsc_ksp file.bin -solver cg|gmres -pc none|jacobi|sor|ilu|icc
 *                            -rtol 1e-5 -maxit 10000 -restart 30 -reps 5
 * Prints one JSON object per run (rank 0). */
#include "common.h"

int main(int argc, char **argv)
{
  Prob        p;
  Mat         A;
  Vec         b, x, r;
  KSP         ksp;
  PC          pc;
  char        file[PETSC_MAX_PATH_LEN], solver[32] = "cg", pcname[32] = "none";
  PetscReal   rtol = 1e-5, bnorm, rnorm;
  PetscInt    maxit = 10000, restart = 30, reps = 5, i, its = 0;
  PetscBool   flg;
  double     *times, setup, t0;
  KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
  PetscMPIInt size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscStrncpy(file, argc > 1 ? argv[1] : "", sizeof(file)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-solver", solver, sizeof(solver), &flg));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-pc", pcname, sizeof(pcname), &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &rtol, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-maxit", &maxit, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-restart", &restart, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  PetscCall(ReadProblem(file, &p));
  PetscCall(BuildMat(&p, &A));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(FillVec(p.b, b));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, !strcmp(solver, "cg") ? KSPCG : KSPGMRES));
  PetscCall(KSPGetPC(ksp, &pc));
  if (!strcmp(pcname, "none")) PetscCall(PCSetType(pc, PCNONE));
  else if (!strcmp(pcname, "jacobi")) PetscCall(PCSetType(pc, PCJACOBI));
  else if (!strcmp(pcname, "sor")) PetscCall(PCSetType(pc, PCSOR));
  else if (!strcmp(pcname, "ilu")) PetscCall(PCSetType(pc, PCILU));
  else if (!strcmp(pcname, "icc")) PetscCall(PCSetType(pc, PCICC));
  PetscCall(KSPSetTolerances(ksp, rtol, 1e-50, 1e4, maxit));
  if (!strcmp(solver, "gmres")) PetscCall(KSPGMRESSetRestart(ksp, restart));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));

  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  t0 = MPI_Wtime();
  PetscCall(KSPSetUp(ksp));
  PetscCall(PCSetUp(pc));
  setup = MaxTime(MPI_Wtime() - t0);

  PetscCall(PetscMalloc1(reps, &times));
  /* warm-up */
  PetscCall(VecZeroEntries(x));
  PetscCall(KSPSolve(ksp, b, x));
  for (i = 0; i < reps; i++) {
    PetscCall(VecZeroEntries(x));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    t0 = MPI_Wtime();
    PetscCall(KSPSolve(ksp, b, x));
    times[i] = MaxTime(MPI_Wtime() - t0);
  }
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(MatMult(A, x, r));
  PetscCall(VecAXPY(r, -1.0, b));
  PetscCall(VecNorm(r, NORM_2, &rnorm));
  PetscCall(VecNorm(b, NORM_2, &bnorm));
  qsort(times, reps, sizeof(double), cmp_double);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "{\"impl\":\"petsc\",\"kind\":\"ksp\",\"solver\":\"%s\",\"pc\":\"%s\",\"n\":%d,\"nnz\":%d,\"threads\":%d,\"setup_s\":%.17e,\"solve_min_s\":%.17e,\"solve_med_s\":%.17e,\"iterations\":%d,\"reason\":\"%s\",\"true_rel_resid\":%.17e}\n",
    solver, pcname, (int)p.n, (int)p.nnz, (int)size, setup, times[0], times[reps / 2], (int)its, KSPConvergedReasons[reason], (double)(rnorm / bnorm)));
  PetscCall(PetscFree(times));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}
