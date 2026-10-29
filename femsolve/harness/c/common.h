/* Reads problem bundles written by `fs-gen` / `femsolve::bundle::Bundle::write`:
 *   A.mtx, optional B.mtx (Matrix Market coordinate, real or complex, general/symmetric/hermitian),
 *   optional v0.mtx (Matrix Market array), meta.txt (not parsed here: run.py passes what is needed).
 * "symmetric" means a_ij = a_ji WITHOUT conjugation, also for complex files (complex symmetric
 * PML matrices); "hermitian" conjugates. Builds sequential AIJ matrices. */
#ifndef FEMSOLVE_COMMON_H
#define FEMSOLVE_COMMON_H
#include <ctype.h>
#include <petsc.h>
#include <stdio.h>
#include <string.h>

typedef struct {
  PetscInt    i, j;
  PetscScalar v;
} Trip;

static int cmp_trip(const void *a, const void *b)
{
  const Trip *x = a, *y = b;
  if (x->i != y->i) return x->i < y->i ? -1 : 1;
  if (x->j != y->j) return x->j < y->j ? -1 : 1;
  return 0;
}

static int cmp_double(const void *a, const void *b)
{
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static void lower_str(char *s)
{
  for (; *s; s++) *s = (char)tolower((unsigned char)*s);
}

/* Parses the %%MatrixMarket header; leaves the stream at the size line. */
static PetscErrorCode ReadHeader(FILE *f, char *format, char *field, char *symmetry)
{
  char line[1024], banner[64], object[64];
  PetscFunctionBeginUser;
  PetscCheck(fgets(line, sizeof line, f), PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "empty file");
  PetscCheck(sscanf(line, "%63s %63s %63s %63s %63s", banner, object, format, field, symmetry) == 5, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad header");
  lower_str(banner);
  lower_str(format);
  lower_str(field);
  lower_str(symmetry);
  PetscCheck(!strcmp(banner, "%%matrixmarket"), PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "not a Matrix Market file");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Skips comment lines and reads the next non-comment line into `line`. */
static PetscErrorCode NextDataLine(FILE *f, char *line, size_t len)
{
  PetscFunctionBeginUser;
  while (fgets(line, (int)len, f)) {
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '%' || *p == '\n' || *p == '\r' || *p == 0) continue;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "unexpected end of file");
}

static PetscErrorCode ParseScalar(const char *field, char **cursor, PetscScalar *v)
{
  double re = 0, im = 0;
  char  *end;
  PetscFunctionBeginUser;
  re = strtod(*cursor, &end);
  *cursor = end;
  if (!strcmp(field, "complex")) {
    im = strtod(*cursor, &end);
    *cursor = end;
#if !defined(PETSC_USE_COMPLEX)
    PetscCheck(im == 0.0, PETSC_COMM_SELF, PETSC_ERR_SUP, "complex file but real PETSc build");
#endif
  }
#if defined(PETSC_USE_COMPLEX)
  *v = re + im * PETSC_i;
#else
  *v = re;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadMatrix(const char *path, Mat *A)
{
  FILE   *f = fopen(path, "r");
  char    format[64], field[64], symm[64], line[4096];
  long    nr, nc, nnz, k, m = 0;
  Trip   *t;
  PetscInt *ia, *ja, i;
  PetscScalar *va;

  PetscFunctionBeginUser;
  PetscCheck(f, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot open %s", path);
  PetscCall(ReadHeader(f, format, field, symm));
  PetscCheck(!strcmp(format, "coordinate"), PETSC_COMM_SELF, PETSC_ERR_SUP, "coordinate format expected");
  PetscCall(NextDataLine(f, line, sizeof line));
  PetscCheck(sscanf(line, "%ld %ld %ld", &nr, &nc, &nnz) == 3, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad size line");
  PetscCall(PetscMalloc1(2 * nnz + 1, &t));
  for (k = 0; k < nnz; k++) {
    char *cur;
    long  r, c;
    PetscScalar v;
    PetscCall(NextDataLine(f, line, sizeof line));
    r = strtol(line, &cur, 10);
    c = strtol(cur, &cur, 10);
    PetscCall(ParseScalar(field, &cur, &v));
    t[m++] = (Trip){(PetscInt)(r - 1), (PetscInt)(c - 1), v};
    if (r != c) {
      if (!strcmp(symm, "symmetric")) t[m++] = (Trip){(PetscInt)(c - 1), (PetscInt)(r - 1), v};
      else if (!strcmp(symm, "skew-symmetric")) t[m++] = (Trip){(PetscInt)(c - 1), (PetscInt)(r - 1), -v};
      else if (!strcmp(symm, "hermitian")) t[m++] = (Trip){(PetscInt)(c - 1), (PetscInt)(r - 1), PetscConj(v)};
    }
  }
  fclose(f);
  qsort(t, m, sizeof(Trip), cmp_trip);
  /* merge duplicates, build CSR */
  PetscCall(PetscCalloc1(nr + 1, &ia));
  PetscCall(PetscMalloc2(m, &ja, m, &va));
  {
    PetscInt out = 0;
    for (k = 0; k < m; k++) {
      if (out > 0 && t[k].i == t[k - 1].i && t[k].j == ja[out - 1] && 0) { /* unreachable: dedupe below */ }
      if (k > 0 && t[k].i == t[k - 1].i && t[k].j == t[k - 1].j) {
        va[out - 1] += t[k].v;
        continue;
      }
      ja[out] = t[k].j;
      va[out] = t[k].v;
      ia[t[k].i + 1]++;
      out++;
    }
    for (i = 0; i < nr; i++) ia[i + 1] += ia[i];
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, nr, nc, ia, ja, va, A));
  }
  PetscCall(PetscFree(t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReadVector(const char *path, Vec *v, PetscInt n)
{
  FILE   *f = fopen(path, "r");
  char    format[64], field[64], symm[64], line[4096];
  long    nr, nc, k;
  PetscScalar *a;

  PetscFunctionBeginUser;
  *v = NULL;
  if (!f) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ReadHeader(f, format, field, symm));
  PetscCall(NextDataLine(f, line, sizeof line));
  PetscCheck(sscanf(line, "%ld %ld", &nr, &nc) == 2 && nr == n, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "bad vector size");
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, v));
  PetscCall(VecGetArray(*v, &a));
  for (k = 0; k < n; k++) {
    char *cur;
    PetscCall(NextDataLine(f, line, sizeof line));
    cur = line;
    PetscCall(ParseScalar(field, &cur, &a[k]));
  }
  PetscCall(VecRestoreArray(*v, &a));
  fclose(f);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
