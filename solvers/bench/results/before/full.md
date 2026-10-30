
### Linear solves (KSP): min wall time of repeated solves

| problem | cores | case | iters / restarts | petsc | petsc-native | rust | rust-native | best Rust vs best PETSc |
|---|---|---|---|---|---|---|---|---|
| p2d_256 | 1 | cg+none | 572 | 259.3 ms | 274.5 ms | 272.2 ms | 242.5 ms | 1.07× faster |
| p2d_256 | 1 | cg+jacobi | 572 | 288.7 ms | 333.6 ms | 319.6 ms | 283.2 ms | 1.02× faster |
| p2d_256 | 1 | cg+sor | 206 | 341.5 ms | 257.8 ms | 262.0 ms | 257.0 ms | 1.00× faster |
| p2d_256 | 1 | cg+icc | 175 | 191.4 ms | 236.8 ms | 208.9 ms | 187.8 ms | 1.02× faster |
| p2d_1024 | 1 | cg+none | 2161 | 36.40 s | 35.07 s | 28.37 s | 28.83 s | 1.24× faster |
| p2d_1024 | 1 | cg+jacobi | 2161 | 38.82 s | 40.10 s | 34.50 s | 32.97 s | 1.18× faster |
| p2d_1024 | 1 | cg+sor | 781 | 30.11 s | 29.72 s | 23.55 s | 22.52 s | 1.32× faster |
| p2d_1024 | 1 | cg+icc | 661 | 21.80 s | 20.63 s | 18.29 s | 18.77 s | 1.13× faster |
| p3d_100 | 1 | cg+none | 231 | 4.45 s | 4.73 s | 3.65 s | 3.70 s | 1.22× faster |
| p3d_100 | 1 | cg+jacobi | 231 | 4.74 s | 4.79 s | 4.33 s | 3.97 s | 1.19× faster |
| p3d_100 | 1 | cg+sor | 85 | 3.80 s | 3.85 s | 2.69 s | 2.94 s | 1.41× faster |
| p3d_100 | 1 | cg+icc | 77 | 2.65 s | 2.58 s | 2.51 s | 2.35 s | 1.10× faster |
| spd_rand | 1 | cg+none | 16 | 190.2 ms | 166.2 ms | 161.6 ms | 157.6 ms | 1.05× faster |
| spd_rand | 1 | cg+jacobi | 11 | 95.3 ms | 108.5 ms | 96.9 ms | 91.5 ms | 1.04× faster |
| spd_rand | 1 | cg+sor | 5 | 116.5 ms | 112.9 ms | 105.0 ms | 109.1 ms | 1.08× faster |
| spd_rand | 1 | cg+icc | 5 | 105.0 ms | 94.2 ms | 102.8 ms | 101.6 ms | 1.08× slower |
| cd2d_512 | 1 | gmres+none | 1085 | 11.20 s | 10.06 s | 10.29 s | 15.40 s | 1.02× slower |
| cd2d_512 | 1 | gmres+jacobi | 1085 | 10.76 s | 11.73 s | 10.15 s | 15.95 s | 1.06× faster |
| cd2d_512 | 1 | gmres+sor | 159 | 2.49 s | 2.28 s | 2.10 s | 2.79 s | 1.08× faster |
| cd2d_512 | 1 | gmres+ilu | 98 | 1.28 s | 1.11 s | 1.34 s | 1.66 s | 1.22× slower |
| p2d_256 | 1 | gmres+none | 2897 | 6.40 s | 5.11 s | 4.73 s | 8.70 s | 1.08× faster |
| p2d_256 | 1 | gmres+jacobi | 2897 | 5.49 s | 5.28 s | 4.87 s | 9.52 s | 1.08× faster |
| p2d_256 | 1 | gmres+sor | 493 | 1.32 s | 1.25 s | 1.44 s | 1.83 s | 1.15× slower |
| p2d_256 | 1 | gmres+ilu | 375 | 900.1 ms | 814.9 ms | 888.7 ms | 1.31 s | 1.09× slower |
| p2d_1024 | 4 | cg+none | 2161 | 9.86 s | 9.75 s | 9.00 s | 8.77 s | 1.11× faster |
| p2d_1024 | 4 | cg+jacobi | 2161 | 10.48 s | 10.36 s | 11.54 s | 11.08 s | 1.07× slower |
| p3d_100 | 4 | cg+jacobi | 231 | 1.49 s | 1.33 s | 1.42 s | 1.35 s | 1.01× slower |
| cd2d_512 | 4 | gmres+none | 1085 | 2.71 s | 3.07 s | 3.52 s | 4.69 s | 1.30× slower |
| cd2d_512 | 4 | gmres+jacobi | 1085 | 2.39 s | 2.84 s | 3.82 s | 4.94 s | 1.59× slower |

### Eigensolves (EPS Krylov–Schur): min wall time

| problem | cores | case | iters / restarts | petsc | petsc-native | rust | rust-native | best Rust vs best PETSc |
|---|---|---|---|---|---|---|---|---|
| p2d_256x250 | 1 | largest nev=4 | 341 | 6.24 s | 6.20 s | 8.11 s | 15.45 s | 1.31× slower |
| p2d_256x250 | 1 | largest nev=10 | 252/254 | 6.88 s | 6.99 s | 10.06 s | 19.25 s | 1.46× slower |
| spd_rand | 1 | largest nev=4 | 21 | 2.88 s | 3.01 s | 3.24 s | 4.66 s | 1.13× slower |
| spd_rand | 1 | largest nev=10 | 14 | 3.16 s | 3.29 s | 3.48 s | 4.96 s | 1.10× slower |
| p2d_48x40 | 1 | smallest nev=4 | 27 | 10.3 ms | 11.9 ms | 17.8 ms | 32.6 ms | 1.73× slower |
| p2d_256x250 | 4 | largest nev=10 | 252/254 | 2.13 s | 2.10 s | 5.31 s | 7.62 s | 2.53× slower |
| spd_rand | 4 | largest nev=10 | 14 | 1.04 s | 1.10 s | 1.19 s | 1.66 s | 1.14× slower |

### Kernels (best of 100)

| problem | n | nnz | kernel | petsc | petsc-native | rust | rust-native |
|---|---|---|---|---|---|---|---|
| p2d_256 | 65536 | 326656 | spmv | 0.2 ms | 0.2 ms | 0.2 ms | 0.2 ms |
| p2d_256 | 65536 | 326656 | dot | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| p2d_256 | 65536 | 326656 | axpy | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| p2d_256 | 65536 | 326656 | pc_jacobi | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| p2d_256 | 65536 | 326656 | pc_sor | 0.8 ms | 0.7 ms | 0.5 ms | 0.5 ms |
| p2d_256 | 65536 | 326656 | pc_ilu | 0.6 ms | 0.5 ms | 0.5 ms | 0.5 ms |
| p2d_256 | 65536 | 326656 | pc_icc | 0.5 ms | 0.7 ms | 0.5 ms | 0.5 ms |
| p2d_256 | 65536 | 326656 | mdot16 | 0.4 ms | 0.4 ms | 0.3 ms | 1.7 ms |
| p2d_256 | 65536 | 326656 | maxpy16 | 0.4 ms | 0.4 ms | 0.3 ms | 0.3 ms |
| p2d_1024 | 1048576 | 5238784 | spmv | 7.3 ms | 7.7 ms | 7.7 ms | 7.3 ms |
| p2d_1024 | 1048576 | 5238784 | dot | 0.5 ms | 0.5 ms | 0.6 ms | 0.5 ms |
| p2d_1024 | 1048576 | 5238784 | axpy | 0.6 ms | 0.6 ms | 0.6 ms | 0.5 ms |
| p2d_1024 | 1048576 | 5238784 | pc_jacobi | 1.4 ms | 1.4 ms | 0.9 ms | 1.1 ms |
| p2d_1024 | 1048576 | 5238784 | pc_sor | 18.0 ms | 18.4 ms | 10.8 ms | 10.7 ms |
| p2d_1024 | 1048576 | 5238784 | pc_ilu | 12.2 ms | 10.2 ms | 10.5 ms | 10.9 ms |
| p2d_1024 | 1048576 | 5238784 | pc_icc | 13.3 ms | 11.5 ms | 10.9 ms | 11.1 ms |
| p2d_1024 | 1048576 | 5238784 | mdot16 | 12.8 ms | 12.3 ms | 11.6 ms | 29.0 ms |
| p2d_1024 | 1048576 | 5238784 | maxpy16 | 12.9 ms | 12.9 ms | 10.8 ms | 11.0 ms |
| p3d_100 | 1000000 | 6940000 | spmv | 13.0 ms | 11.4 ms | 10.4 ms | 10.3 ms |
| p3d_100 | 1000000 | 6940000 | dot | 0.6 ms | 0.6 ms | 0.6 ms | 0.6 ms |
| p3d_100 | 1000000 | 6940000 | axpy | 0.6 ms | 0.6 ms | 0.7 ms | 0.5 ms |
| p3d_100 | 1000000 | 6940000 | pc_jacobi | 1.6 ms | 1.9 ms | 1.5 ms | 1.3 ms |
| p3d_100 | 1000000 | 6940000 | pc_sor | 22.6 ms | 23.9 ms | 12.4 ms | 13.2 ms |
| p3d_100 | 1000000 | 6940000 | pc_ilu | 13.2 ms | 12.8 ms | 12.0 ms | 17.4 ms |
| p3d_100 | 1000000 | 6940000 | pc_icc | 14.9 ms | 13.6 ms | 12.6 ms | 12.6 ms |
| p3d_100 | 1000000 | 6940000 | mdot16 | 12.2 ms | 11.8 ms | 12.2 ms | 27.6 ms |
| p3d_100 | 1000000 | 6940000 | maxpy16 | 12.4 ms | 11.8 ms | 11.7 ms | 10.8 ms |
| spd_rand | 200000 | 2599932 | spmv | 7.5 ms | 7.6 ms | 7.2 ms | 8.5 ms |
| spd_rand | 200000 | 2599932 | dot | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| spd_rand | 200000 | 2599932 | axpy | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| spd_rand | 200000 | 2599932 | pc_jacobi | 0.2 ms | 0.2 ms | 0.2 ms | 0.1 ms |
| spd_rand | 200000 | 2599932 | pc_sor | 12.3 ms | 11.6 ms | 9.0 ms | 9.7 ms |
| spd_rand | 200000 | 2599932 | pc_ilu | 7.3 ms | 7.5 ms | 8.2 ms | 8.2 ms |
| spd_rand | 200000 | 2599932 | pc_icc | 8.3 ms | 7.6 ms | 9.6 ms | 8.4 ms |
| spd_rand | 200000 | 2599932 | mdot16 | 2.2 ms | 1.7 ms | 2.6 ms | 5.6 ms |
| spd_rand | 200000 | 2599932 | maxpy16 | 1.7 ms | 1.6 ms | 1.5 ms | 2.0 ms |

EPS agreement p2d_256x250 largest nev=4 p=1: max rel. eigenvalue difference PETSc/Rust = 1.84e-12

EPS agreement p2d_256x250 largest nev=10 p=1: max rel. eigenvalue difference PETSc/Rust = 1.52e-12

EPS agreement spd_rand largest nev=4 p=1: max rel. eigenvalue difference PETSc/Rust = 4.63e-14

EPS agreement spd_rand largest nev=10 p=1: max rel. eigenvalue difference PETSc/Rust = 5.89e-14

EPS agreement p2d_48x40 smallest nev=4 p=1: max rel. eigenvalue difference PETSc/Rust = 9.48e-14

EPS agreement p2d_256x250 largest nev=10 p=4: max rel. eigenvalue difference PETSc/Rust = 1.51e-12

EPS agreement spd_rand largest nev=10 p=4: max rel. eigenvalue difference PETSc/Rust = 5.93e-14
