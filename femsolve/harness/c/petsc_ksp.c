/* PETSc KSP driver for femsolve problem bundles (same schema as `fs-ksp`).
 *   ./petsc_ksp <bundle-dir> -ksp_kind cg|cocg|gmres|fgmres -pc none|jacobi -rtol r -maxit m
 *               -restart k -side left|right -orth cgs|mgs|cgs-refine -norm pc|unpc|natural -reps n
 * The right-hand side is the bundle's v0.mtx. */
#include "common.h"

int main(int argc, char **argv)
{
  char        dir[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN + 16], solver[32] = "cocg", pcn[32] = "none", side[16] = "left", orth[32] = "cgs", norm[16] = "pc";
  Mat         A;
  Vec         b, x, r;
  KSP         ksp;
  PC          pc;
  PetscReal   rtol = 1e-5, rn, bn, *hist;
  PetscInt    maxit = 10000, restart = 30, reps = 3, n, nnz, its, i, nh;
  double     *times, t0, setup = 0;
  MatInfo     minfo;
  KSPConvergedReason reason;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCheck(argc > 1, PETSC_COMM_SELF, PETSC_ERR_USER, "usage: petsc_ksp <bundle-dir> [options]");
  PetscCall(PetscStrncpy(dir, argv[1], sizeof(dir)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ksp_kind", solver, sizeof(solver), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-pc", pcn, sizeof(pcn), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-side", side, sizeof(side), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-orth", orth, sizeof(orth), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-norm", norm, sizeof(norm), NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &rtol, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-maxit", &maxit, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-restart", &restart, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  reps = PetscMax(reps, 1);

  PetscCall(PetscSNPrintf(path, sizeof path, "%s/A.mtx", dir));
  PetscCall(ReadMatrix(path, &A));
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatGetInfo(A, MAT_LOCAL, &minfo));
  nnz = (PetscInt)minfo.nz_used;
  PetscCall(PetscSNPrintf(path, sizeof path, "%s/v0.mtx", dir));
  PetscCall(ReadVector(path, &b, n));
  PetscCheck(b, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "v0.mtx needed");
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecDuplicate(b, &r));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  if (!strcmp(solver, "cg") || !strcmp(solver, "cocg")) {
    PetscCall(KSPSetType(ksp, KSPCG));
    PetscCall(KSPCGSetType(ksp, !strcmp(solver, "cg") ? KSP_CG_HERMITIAN : KSP_CG_SYMMETRIC));
  } else if (!strcmp(solver, "gmres")) {
    PetscCall(KSPSetType(ksp, KSPGMRES));
    PetscCall(KSPGMRESSetRestart(ksp, restart));
    if (!strcmp(orth, "mgs")) PetscCall(KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization));
    else if (!strcmp(orth, "cgs-refine")) {
      PetscCall(KSPGMRESSetOrthogonalization(ksp, KSPGMRESClassicalGramSchmidtOrthogonalization));
      PetscCall(KSPGMRESSetCGSRefinementType(ksp, KSP_GMRES_CGS_REFINE_IFNEEDED));
    }
  } else {
    PetscCall(KSPSetType(ksp, KSPFGMRES));
    PetscCall(KSPGMRESSetRestart(ksp, restart));
    PetscCall(KSPGMRESSetOrthogonalization(ksp, KSPGMRESModifiedGramSchmidtOrthogonalization));
  }
  PetscCall(KSPSetPCSide(ksp, !strcmp(side, "right") || !strcmp(solver, "fgmres") ? PC_RIGHT : PC_LEFT));
  PetscCall(KSPSetNormType(ksp, (!strcmp(side, "right") || !strcmp(solver, "fgmres")) && !strcmp(solver, "gmres") + !strcmp(solver, "fgmres") ? KSP_NORM_UNPRECONDITIONED : !strcmp(norm, "unpc") ? KSP_NORM_UNPRECONDITIONED : !strcmp(norm, "natural") ? KSP_NORM_NATURAL : KSP_NORM_PRECONDITIONED));
  PetscCall(KSPSetTolerances(ksp, rtol, 1e-50, 1e4, maxit));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, !strcmp(pcn, "jacobi") ? PCJACOBI : !strcmp(pcn, "amg") ? PCGAMG : PCNONE));
  PetscCall(PetscMalloc1(maxit + 10, &hist));
  PetscCall(KSPSetResidualHistory(ksp, hist, maxit + 10, PETSC_TRUE));
  t0 = MPI_Wtime();
  PetscCall(KSPSetUp(ksp));
  PetscCall(PCSetUp(pc));
  setup = MPI_Wtime() - t0;
  PetscCall(PetscMalloc1(reps, &times));
  PetscCall(VecZeroEntries(x));
  PetscCall(KSPSolve(ksp, b, x)); /* warm-up */
  for (i = 0; i < reps; i++) {
    PetscCall(VecZeroEntries(x));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
    t0 = MPI_Wtime();
    PetscCall(KSPSolve(ksp, b, x));
    times[i] = MPI_Wtime() - t0;
  }
  PetscCall(KSPGetIterationNumber(ksp, &its));
  PetscCall(KSPGetConvergedReason(ksp, &reason));
  PetscCall(KSPGetResidualHistory(ksp, NULL, &nh));
  PetscCall(MatMult(A, x, r));
  PetscCall(VecAYPX(r, -1.0, b));
  PetscCall(VecNorm(r, NORM_2, &rn));
  PetscCall(VecNorm(b, NORM_2, &bn));
  qsort(times, reps, sizeof(double), cmp_double);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    "{\"impl\":\"petsc\",\"kind\":\"ksp\",\"n\":%d,\"nnz\":%d,\"scalar\":\"%s\",\"solver\":\"%s\",\"iterations\":%d,\"reason\":\"%s\",\"setup_s\":%.17e,\"solve_min_s\":%.17e,\"true_rel_resid\":%.17e,\"history\":[",
    (int)n, (int)nnz,
#if defined(PETSC_USE_COMPLEX)
    "complex",
#else
    "real",
#endif
    solver, (int)its, KSPConvergedReasons[reason], setup, times[0], (double)(rn / bn)));
  for (i = 0; i < nh; i++) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.17e", i ? "," : "", (double)hist[i]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]}\n"));
  PetscCall(PetscFree(hist));
  PetscCall(PetscFree(times));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}
