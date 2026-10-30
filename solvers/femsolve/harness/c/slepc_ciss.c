/* SLEPc EPSCISS driver (contour integral, all eigenvalues in an ellipse) for femsolve bundles;
 * same JSON fields as `fs-contour`.
 *   ./slepc_ciss <bundle-dir> -center_re x -center_im y -radius a -vscale s -nodes N -block L -moments M -reps r */
#include "common.h"
#include <slepceps.h>

int main(int argc, char **argv)
{
  char        dir[PETSC_MAX_PATH_LEN], path[PETSC_MAX_PATH_LEN + 16];
  Mat         A, B = NULL;
  EPS         eps;
  RG          rg;
  PetscReal   cre = 0, cim = 0, radius = 1, vscale = 1, err;
  PetscInt    nodes = 24, block = 8, moments = 4, reps = 3, i, nconv, n, nnz, herm = 0;
  double     *times, t0;
  PetscScalar kr, ki, center;
  MatInfo     minfo;
  FILE       *fcheck;

  PetscCall(SlepcInitialize(&argc, &argv, NULL, NULL));
  PetscCheck(argc > 1, PETSC_COMM_SELF, PETSC_ERR_USER, "usage: slepc_ciss <bundle-dir> [options]");
  PetscCall(PetscStrncpy(dir, argv[1], sizeof(dir)));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-center_re", &cre, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-center_im", &cim, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-radius", &radius, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-vscale", &vscale, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nodes", &nodes, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-block", &block, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-moments", &moments, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-reps", &reps, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-herm", &herm, NULL));
  reps = PetscMax(reps, 1);

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
#if defined(PETSC_USE_COMPLEX)
  center = cre + cim * PETSC_i;
#else
  PetscCheck(cim == 0.0, PETSC_COMM_SELF, PETSC_ERR_SUP, "imaginary centre needs the complex build");
  center = cre;
#endif
  PetscCall(EPSCreate(PETSC_COMM_WORLD, &eps));
  PetscCall(EPSSetOperators(eps, A, B));
  PetscCall(EPSSetProblemType(eps, herm ? (B ? EPS_GHEP : EPS_HEP) : (B ? EPS_GNHEP : EPS_NHEP)));
  PetscCall(EPSSetType(eps, EPSCISS));
  PetscCall(EPSSetWhichEigenpairs(eps, EPS_ALL));
  PetscCall(EPSGetRG(eps, &rg));
  PetscCall(RGSetType(rg, RGELLIPSE));
  PetscCall(RGEllipseSetParameters(rg, center, radius, vscale));
  PetscCall(EPSCISSSetSizes(eps, nodes, block, moments, 1, moments, PETSC_TRUE));
  PetscCall(EPSCISSSetThreshold(eps, 1e-12, 1e-8));
  PetscCall(EPSSetFromOptions(eps));
  PetscCall(PetscMalloc1(reps, &times));
  PetscCall(EPSSolve(eps)); /* warm-up */
  for (i = 0; i < reps; i++) {
    PetscCall(EPSSetOperators(eps, A, B)); /* resets the solver state so that every repetition solves from scratch */
    t0 = MPI_Wtime();
    PetscCall(EPSSetUp(eps));
    PetscCall(EPSSolve(eps));
    times[i] = MPI_Wtime() - t0;
  }
  PetscCall(EPSGetConverged(eps, &nconv));
  qsort(times, reps, sizeof(double), cmp_double);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "{\"impl\":\"slepc\",\"kind\":\"contour\",\"n\":%d,\"nnz\":%d,\"count\":%d,\"total_min_s\":%.17e,\"values\":[", (int)n, (int)nnz, (int)nconv, times[0]));
  for (i = 0; i < nconv; i++) {
    PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s[%.17e,%.17e]", i ? "," : "", (double)PetscRealPart(kr), (double)PetscImaginaryPart(kr)));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "],\"true_rel_resid\":["));
  for (i = 0; i < nconv; i++) {
    PetscCall(EPSComputeError(eps, i, EPS_ERROR_RELATIVE, &err));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.17e", i ? "," : "", (double)err));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]}\n"));
  PetscCall(PetscFree(times));
  PetscCall(EPSDestroy(&eps));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(SlepcFinalize());
  return 0;
}
