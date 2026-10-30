| case | tier | verdict | restarts (rust/slepc) | matvecs | cold rust [s] | cold slepc [s] | warm rust [s] | warm slepc [s] | warm ratio | max resid |
|---|---|---|---|---|---|---|---|---|---|---|
| fe1d ncv=12 | A | PASS | 3 / 3 | 23 / 23 | 0.004059 | 0.004736 | 0.001929 | 0.003026 | 0.64 | 1.8e-10 |
| fe1d default | A | PASS | 1 / 1 | 20 / 20 | 0.003748 | 0.004845 | 0.001547 | 0.003118 | 0.50 | 8.0e-12 |
| fe1d nev=10 ncv=14 | A | PASS | 17 / 17 | 72 / 72 | 0.00822 | 0.008752 | 0.005674 | 0.007644 | 0.74 | 1.8e-11 |
| fe1d Cholesky (shift below spectrum) | A | PASS | 4 / 4 | 29 / 29 | 0.004654 | 0.006209 | 0.002419 | 0.003212 | 0.75 | 5.3e-05 |
| gauge1d (complex Hermitian) | A | PASS | 1 / 1 | 19 / 19 | 0.005423 | 0.00463 | 0.003233 | 0.002906 | 1.11 | 5.5e-10 |
| gauge1d ncv=8 | A | PASS | 6 / 6 | 25 / 25 | 0.005708 | 0.004341 | 0.003172 | 0.002645 | 1.20 | 2.1e-09 |
| pml1d (complex symmetric) | A | PASS | 1 / 1 | 19 / 19 | 0.005112 | 0.004927 | 0.003027 | 0.003123 | 0.97 | 1.0e-09 |
| pml1d ncv=8 | A | PASS | 6 / 6 | 25 / 25 | 0.005518 | 0.004464 | 0.003427 | 0.00298 | 1.15 | 5.0e-10 |
| pml2d pencil (complex B) | A | PASS | 11 / 11 | 119 / 119 | 0.07857 | 0.07498 | 0.07328 | 0.06503 | 1.13 | 3.1e-09 |
| lap1d direct, largest | A | PASS | 178 / 178 | 1790 / 1790 | 0 | 0.1253 | 0.03877 | 0.09051 | 0.43 | 8.9e-09 |
| lap2d (degenerate) | B | PASS | 3 / 3 | 40 / 42 | 0.1206 | 0.0633 | 0.08997 | 0.04296 | 2.09 | 5.5e-12 |
| fe1d n=20000 | A | PASS | 2 / 2 | 33 / 33 | 0.03932 | 0.05665 | 0.02914 | 0.04574 | 0.64 | 3.0e-11 |
| gauge1d n=20000 | A | PASS | 2 / 2 | 33 / 33 | 0.07236 | 0.08675 | 0.05823 | 0.07216 | 0.81 | 3.6e-12 |
| pml1d n=20000 | A | PASS | 2 / 2 | 32 / 32 | 0.07605 | 0.09224 | 0.0576 | 0.07531 | 0.76 | 1.6e-10 |
| pml2d 200x200 | B | PASS | 128 / 100 | 1407 / 1108 | 31.5 | 17.47 | 31.38 | 16.99 | 1.85 | 3.9e-09 |
| lap3d 40 Cholesky | B | PASS | 6 / 7 | 68 / 77 | 4.715 | 18.18 | 4.534 | 3.828 | 1.18 | 1.0e-07 |

cold = first solve (setup + symbolic + numeric factorisation + eigensolve); warm = minimum of the repeated solves.
For rust, a warm solve re-does the numeric factorisation (set_shift) but not the symbolic analysis; for SLEPc the
repeated EPSSolve reuses the PC factorisation (the logged factorisation time of a warm solve is 0), so warm slepc
excludes the factorisation. Compare cold with cold for an all-in number; warm rust includes factor_s, see the JSON.
ratio = rust / slepc (below 1: rust faster).
