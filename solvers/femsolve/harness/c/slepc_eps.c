/* SLEPc EPS (Krylov-Schur) driver for femsolve problem bundles.
 *   ./slepc_eps <bundle-dir> -herm 0|1 -mode shift|direct -nev k -ncv c -tol t -maxit m
 *               -sigma_re x -sigma_im y -which largest|smallest|largest_real|smallest_real
 *               -reps r -definite 0|1 -block_ignored
 * Prints one JSON line with the same fields as `fs-eps`. The start vector is the bundle's v0.mtx,
 * shared with the Rust solver. */
#include "common.h"
#include <slepceps.h>

int main(int argc, char **argv)
{
  char        dir[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN + 16], mode[32] = "shift", which[32] = "largest";
  Mat         A, B = NULL;
  Vec         v0 = NULL;
  EPS         eps;
  ST          st;
  KSP         ksp;
  PC          pc;
  PetscReal   tol = 1e-8, err, sre = 0, sim = 0;
  PetscInt    ncv_used, nev = 4, ncv = PETSC_DEFAULT, maxit = PETSC_DEFAULT, reps = 3, i, its, nconv, n, nnz, herm = 0, definite = 0;
  PetscBool   flg;
  double     *times, t0, cold = 0;
  EPSConvergedReason reason;
  PetscScalar kr, ki, target;
  PetscLogEvent evst, evlu, evch;
  PetscEventPerfInfo info, infolu, infoch;
  MatInfo     minfo;
  FILE       *fcheck;

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));
  PetscCheck(argc > 1, PETSC_COMM_SELF, PETSC_ERR_USER, "usage: slepc_eps <bundle-dir> [options]");
  PetscCall(PetscStrncpy(dir, argv[1], sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mode", mode, sizeof(mode), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-which", which, sizeof(which), NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_re", &sre, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-sigma_im", &sim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nev", &nev, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ncv", &ncv, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-maxit", &maxit, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  reps = PetscMax(reps, 1);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-herm", &herm, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-definite", &definite, NULL));

  PetscCall(PetscSNPrintf(path, sizeof path, "%s/A.mtx", dir));
  PetscCall(ReadMatrix(path, &A));
  PetscCall(PetscSNPrintf(path, sizeof path, "%s/B.mtx", dir));
  if ((fcheck = fopen(path, "r"))) {
    fclose(fcheck);
    PetscCall(ReadMatrix(path, &B));
  }
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatGetInfo(A, MAT_LOCAL, &minfo));
  nnz = (PetscInt)minfo.nz_used;
  PetscCall(PetscSNPrintf(path, sizeof path, "%s/v0.mtx", dir));
  PetscCall(ReadVector(path, &v0, n));

  PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
  PetscCall(EPSSetOperators(eps, A, B));
  PetscCall(EPSSetProblemType(eps, herm ? (B ? EPS_GHEP : EPS_HEP) : (B ? EPS_GNHEP : EPS_NHEP)));
  PetscCall(EPSSetType(eps, EPSKRYLOVSCHUR));
  PetscCall(EPSSetDimensions(eps, nev, ncv, PETSC_DEFAULT));
  PetscCall(EPSSetTolerances(eps, tol, maxit));
  if (!strcmp(mode, "shift")) {
#if defined(PETSC_USE_COMPLEX)
    target = sre + sim * PETSC_i;
#else
    PetscCheck(sim == 0.0, PETSC_COMM_SELF, PETSC_ERR_SUP, "imaginary shift needs the complex build");
    target = sre;
#endif
    PetscCall(EPSSetWhichEigenpairs(eps, EPS_TARGET_MAGNITUDE));
    PetscCall(EPSSetTarget(eps, target));
    PetscCall(EPSGetST(eps, &st));
    PetscCall(STSetType(st, STSINVERT));
    PetscCall(STGetKSP(st, &ksp));
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, (definite && herm) ? PCCHOLESKY : PCLU));
  } else {
    if (!strcmp(which, "largest")) PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE));
    else if (!strcmp(which, "smallest")) PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE));
    else if (!strcmp(which, "largest_real")) PetscCall(EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL));
    else PetscCall(EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL));
  }
  PetscCall(EPSSetFromOptions(eps));
  if (v0) PetscCall(EPSSetInitialSpace(eps, 1, &v0));

  PetscCall(PetscMalloc1(reps, &times));
  t0 = MPI_Wtime();
  PetscCall(EPSSolve(eps)); /* warm-up = cold solve: includes setup and factorisation */
  cold = MPI_Wtime() - t0;
  for (i = 0; i < reps; i++) {
    if (v0) PetscCall(EPSSetInitialSpace(eps, 1, &v0));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    t0 = MPI_Wtime();
    PetscCall(EPSSolve(eps));
    times[i] = MPI_Wtime() - t0;
  }
  PetscCall(EPSGetIterationNumber(eps, &its));
  PetscCall(EPSGetConverged(eps, &nconv));
  PetscCall(EPSGetConvergedReason(eps, &reason));
  PetscCall(EPSGetDimensions(eps, NULL, &ncv_used, NULL));

  /* One extra, untimed solve with logging on: operator-application and factorisation counts. */
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogEventGetId("STApply", &evst));
  PetscCall(PetscLogEventGetId("MatLUFactorNum", &evlu));
  PetscCall(PetscLogEventGetId("MatCholFctrNum", &evch));
  if (v0) PetscCall(EPSSetInitialSpace(eps, 1, &v0));
  PetscCall(EPSSolve(eps));
  PetscCall(PetscLogEventGetPerfInfo(0, evst, &info));
  PetscCall(PetscLogEventGetPerfInfo(0, evlu, &infolu));
  PetscCall(PetscLogEventGetPerfInfo(0, evch, &infoch));

  qsort(times, reps, sizeof(double), cmp_double);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "{\"impl\":\"slepc\",\"kind\":\"eps\",\"n\":%d,\"nnz\":%d,\"scalar\":\"%s\",\"mode\":\"%s\",\"nev\":%d,\"ncv\":%d,\"restarts\":%d,\"matvecs\":%d,\"nconv\":%d,\"reason\":\"%s\",\"cold_s\":%.17e,\"total_min_s\":%.17e,\"eps_min_s\":%.17e,\"factor_s\":%.17e,\"values\":[",
    (int)n, (int)nnz,
#if defined(PETSC_USE_COMPLEX)
    "complex",
#else
    "real",
#endif
    mode, (int)nev, (int)ncv_used, (int)its, (int)info.count, (int)nconv, EPSConvergedReasons[reason], cold, times[0], times[0], infolu.time + infoch.time));
  for (i = 0; i < PetscMin(nev, nconv); i++) {
    PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s[%.17e,%.17e]", i ? "," : "", (double)PetscRealPart(kr), (double)PetscImaginaryPart(kr)));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "],\"true_rel_resid\":["));
  for (i = 0; i < PetscMin(nev, nconv); i++) {
    PetscCall(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.17e", i ? "," : "", (double)err));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]}\n"));
  (void)flg;
  PetscCall(PetscFree(times));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&v0));
  PetscCall(SlepcFinalize());
  return 0;
}
