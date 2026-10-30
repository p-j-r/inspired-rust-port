/* SLEPc EPS (Krylov-Schur, Hermitian) timing harness:
 *   ./slepc_eps file.bin -nev 4 -ncv 20 -which largest|smallest|largest_real|smallest_real -tol 1e-8 -reps 3
 * The start vector is the file's v0, shared with the Rust solver. */
#include "common.h"
#include <slepceps.h>

int main(int argc, char **argv)
{
  Prob        p;
  Mat         A;
  Vec         v0, xr;
  EPS         eps;
  char        file[PETSC_MAX_PATH_LEN], which[32] = "largest";
  PetscReal   tol = 1e-8, err;
  PetscInt    nev = 1, ncv = PETSC_DEFAULT, maxit = PETSC_DEFAULT, reps = 3, i, its, nconv, lo;
  PetscMPIInt size;
  double     *times, t0;
  EPSConvergedReason reason;
  PetscScalar kr, ki;
  PetscLogEvent ev;
  PetscEventPerfInfo info;

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscStrncpy(file, argc > 1 ? argv[1] : "", sizeof(file)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-which", which, sizeof(which), NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nev", &nev, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ncv", &ncv, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-maxit", &maxit, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  PetscCall(ReadProblem(file, &p));
  PetscCall(BuildMat(&p, &A));
  PetscCall(MatCreateVecs(A, &xr, &v0));
  PetscCall(FillVec(p.v0, v0));

  PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
  PetscCall(EPSSetOperators(eps, A, NULL));
  PetscCall(EPSSetProblemType(eps, EPS_HEP));
  PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
  if (!strcmp(which, "largest")) PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE));
  else if (!strcmp(which, "smallest")) PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE));
  else if (!strcmp(which, "largest_real")) PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL));
  else PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
  PetscCall(EPSSetDimensions(eps, nev, ncv, PETSC_DEFAULT));
  PetscCall(EPSSetTolerances(eps, tol, maxit));
  PetscCall(EPSSetInitialSpace(eps, 1, &v0));

  PetscCall(PetscMalloc1(reps, &times));
  /* EPSSolve() on an already solved object returns immediately; setting the initial space
     again resets the state, so every timed solve includes EPSSetUp, as a fresh solve would. */
  PetscCall(EPSSolve(eps)); /* warm-up */
  for (i = 0; i < reps; i++) {
    PetscCall(EPSSetInitialSpace(eps, 1, &v0));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    t0 = MPI_Wtime();
    PetscCall(EPSSolve(eps));
    times[i] = MaxTime(MPI_Wtime() - t0);
  }
  PetscCall(EPSGetIterationNumber(eps, &its));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCall(EPSGetConvergedReason(eps, &reason));

  /* One extra, untimed solve with logging on to count operator applications. */
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogEventGetId("MatMult", &ev));
  PetscCall(EPSSetInitialSpace(eps, 1, &v0));
  PetscCall(EPSSolve(eps));
  PetscCall(PetscLogEventGetPerfInfo(0, ev, &info));
  (void)lo;

  qsort(times, reps, sizeof(double), cmp_double);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "{\"impl\":\"slepc\",\"kind\":\"eps\",\"n\":%d,\"nnz\":%d,\"threads\":%d,\"nev\":%d,\"which\":\"%s\",\"solve_min_s\":%.17e,\"solve_med_s\":%.17e,\"iterations\":%d,\"matvecs\":%d,\"nconv\":%d,\"reason\":\"%s\",\"values\":[",
    (int)p.n, (int)p.nnz, (int)size, (int)nev, which, times[0], times[reps / 2], (int)its, (int)info.count, (int)nconv, EPSConvergedReasons[reason]));
  for (i = 0; i < PetscMin(nev, nconv); i++) {
    PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.17e", i ? "," : "", (double)PetscRealPart(kr)));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "],\"true_rel_resid\":["));
  for (i = 0; i < PetscMin(nev, nconv); i++) {
    PetscCall(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.17e", i ? "," : "", (double)err));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]}\n"));
  PetscCall(PetscFree(times));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&v0));
  PetscCall(VecDestroy(&xr));
  PetscCall(SlepcFinalize());
  return 0;
}
