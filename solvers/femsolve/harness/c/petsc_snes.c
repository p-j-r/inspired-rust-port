/* PETSc SNES driver for the nonlinear-solver comparison (same problems as `fs-snes`):
 *   ./petsc_snes -problem bratu|atan [-nx 80] [-mu 6.0] [-n 10] [-x0 2.0] -rtol 1e-10 -ls bt|basic -maxit 50 -lag 1
 * Newton with a direct (LU) linear solve; prints one JSON line. Real PETSc build only. */
#include <petsc.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  PetscInt  nx, n;
  PetscReal mu, h2inv;
  int       atan;
} Ctx;

static PetscErrorCode FormFunction(SNES snes, Vec u, Vec f, void *p)
{
  Ctx               *c = p;
  const PetscScalar *uu;
  PetscScalar       *ff;
  PetscInt           i, j, k;
  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(u, &uu));
  PetscCall(VecGetArray(f, &ff));
  if (c->atan) {
    for (i = 0; i < c->n; i++) ff[i] = atan(PetscRealPart(uu[i]));
  } else {
    for (j = 0; j < c->nx; j++)
      for (i = 0; i < c->nx; i++) {
        k = j * c->nx + i;
        ff[k] = 4.0 * uu[k] - (i > 0 ? uu[k - 1] : 0) - (i + 1 < c->nx ? uu[k + 1] : 0) - (j > 0 ? uu[k - c->nx] : 0) - (j + 1 < c->nx ? uu[k + c->nx] : 0);
        ff[k] = c->h2inv * ff[k] - c->mu * exp(PetscRealPart(uu[k]));
      }
  }
  PetscCall(VecRestoreArrayRead(u, &uu));
  PetscCall(VecRestoreArray(f, &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian(SNES snes, Vec u, Mat J, Mat P, void *p)
{
  Ctx               *c = p;
  const PetscScalar *uu;
  PetscInt           i, j, k, col[5], m;
  PetscScalar        v[5];
  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(u, &uu));
  PetscCall(MatZeroEntries(P));
  if (c->atan) {
    for (i = 0; i < c->n; i++) {
      v[0] = 1.0 / (1.0 + PetscRealPart(uu[i]) * PetscRealPart(uu[i]));
      PetscCall(MatSetValues(P, 1, &i, 1, &i, v, INSERT_VALUES));
    }
  } else {
    for (j = 0; j < c->nx; j++)
      for (i = 0; i < c->nx; i++) {
        k = j * c->nx + i;
        m = 0;
        if (j > 0) { col[m] = k - c->nx; v[m++] = -c->h2inv; }
        if (i > 0) { col[m] = k - 1; v[m++] = -c->h2inv; }
        col[m] = k; v[m++] = 4.0 * c->h2inv - c->mu * exp(PetscRealPart(uu[k]));
        if (i + 1 < c->nx) { col[m] = k + 1; v[m++] = -c->h2inv; }
        if (j + 1 < c->nx) { col[m] = k + c->nx; v[m++] = -c->h2inv; }
        PetscCall(MatSetValues(P, 1, &k, m, col, v, INSERT_VALUES));
      }
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(u, &uu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  char      problem[32] = "bratu", ls[16] = "bt";
  Ctx       c;
  Mat       J;
  Vec       u, f;
  SNES      snes;
  KSP       ksp;
  PC        pc;
  PetscReal rtol = 1e-8, x0 = 2.0, *hist;
  PetscInt  nx = 80, n = 10, maxit = 50, lag = 1, its, nh, i, lits;
  SNESConvergedReason reason;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-problem", problem, sizeof problem, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-ls", ls, sizeof ls, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &nx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-maxit", &maxit, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-lag", &lag, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rtol", &rtol, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x0", &x0, NULL));
  c.atan = !strcmp(problem, "atan");
  c.mu = 6.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &c.mu, NULL));
  c.nx = nx;
  c.h2inv = (nx + 1.0) * (nx + 1.0);
  c.n = c.atan ? n : nx * nx;

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, c.n, &u));
  PetscCall(VecDuplicate(u, &f));
  PetscCall(VecSet(u, c.atan ? x0 : 0.0));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, c.n, c.n, 5, NULL, &J));
  PetscCall(MatSetOption(J, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(SNESSetFunction(snes, f, FormFunction, &c));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &c));
  PetscCall(SNESSetType(snes, SNESNEWTONLS));
  {
    SNESLineSearch lsr;
    PetscCall(SNESGetLineSearch(snes, &lsr));
    PetscCall(SNESLineSearchSetType(lsr, !strcmp(ls, "basic") ? SNESLINESEARCHBASIC : SNESLINESEARCHBT));
  }
  PetscCall(SNESSetTolerances(snes, 1e-50, rtol, 1e-14, maxit, 10000));
  PetscCall(SNESSetLagJacobian(snes, lag));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(PetscMalloc1(maxit + 2, &hist));
  PetscCall(SNESSetConvergenceHistory(snes, hist, NULL, maxit + 2, PETSC_TRUE));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(SNESGetConvergedReason(snes, &reason));
  PetscCall(SNESGetConvergenceHistory(snes, &hist, NULL, &nh));
  PetscCall(SNESGetLinearSolveIterations(snes, &lits));
  printf("{\"impl\":\"petsc\",\"kind\":\"snes\",\"problem\":\"%s\",\"n\":%d,\"iterations\":%d,\"reason\":%d,\"residual_norms\":[", problem, (int)c.n, (int)its, (int)reason);
  for (i = 0; i < nh; i++) printf("%s%.17e", i ? "," : "", (double)hist[i]);
  printf("]}\n");
  PetscCall(PetscFree(hist));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}
