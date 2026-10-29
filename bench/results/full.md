
### Linear solves (KSP): min wall time of repeated solves

| problem | cores | case | iters / restarts | petsc | petsc-native | rust | rust-native | best Rust vs best PETSc |
|---|---|---|---|---|---|---|---|---|
| p2d_256 | 1 | cg+none | 572 | 269.5 ms | 296.1 ms | 274.9 ms | 224.0 ms | 1.20× faster |
| p2d_256 | 1 | cg+jacobi | 572 | 299.6 ms | 282.6 ms | 323.0 ms | 280.9 ms | 1.01× faster |
| p2d_256 | 1 | cg+sor | 206 | 271.4 ms | 241.6 ms | 232.9 ms | 215.0 ms | 1.12× faster |
| p2d_256 | 1 | cg+icc | 175 | 213.4 ms | 153.5 ms | 222.0 ms | 194.0 ms | 1.26× slower |
| p2d_1024 | 1 | cg+none | 2161 | 36.10 s | 37.94 s | 28.43 s | 28.29 s | 1.28× faster |
| p2d_1024 | 1 | cg+jacobi | 2161 | 39.82 s | 41.39 s | 34.75 s | 33.79 s | 1.18× faster |
| p2d_1024 | 1 | cg+sor | 781 | 31.74 s | 31.36 s | 23.13 s | 24.10 s | 1.36× faster |
| p2d_1024 | 1 | cg+icc | 661 | 20.93 s | 21.09 s | 19.23 s | 18.57 s | 1.13× faster |
| p3d_100 | 1 | cg+none | 231 | 5.09 s | 4.89 s | 4.02 s | 3.85 s | 1.27× faster |
| p3d_100 | 1 | cg+jacobi | 231 | 5.55 s | 5.11 s | 4.57 s | 4.46 s | 1.15× faster |
| p3d_100 | 1 | cg+sor | 85 | 3.95 s | 4.02 s | 2.87 s | 3.23 s | 1.38× faster |
| p3d_100 | 1 | cg+icc | 77 | 2.99 s | 2.65 s | 2.52 s | 2.45 s | 1.08× faster |
| spd_rand | 1 | cg+none | 16 | 151.3 ms | 160.7 ms | 156.4 ms | 140.5 ms | 1.08× faster |
| spd_rand | 1 | cg+jacobi | 11 | 110.7 ms | 124.4 ms | 111.4 ms | 115.2 ms | 1.01× slower |
| spd_rand | 1 | cg+sor | 5 | 143.4 ms | 211.4 ms | 120.0 ms | 143.9 ms | 1.19× faster |
| spd_rand | 1 | cg+icc | 5 | 101.3 ms | 108.0 ms | 108.1 ms | 111.0 ms | 1.07× slower |
| cd2d_512 | 1 | gmres+none | 1085 | 11.16 s | 10.75 s | 9.93 s | 10.11 s | 1.08× faster |
| cd2d_512 | 1 | gmres+jacobi | 1085 | 10.79 s | 10.64 s | 10.33 s | 10.28 s | 1.03× faster |
| cd2d_512 | 1 | gmres+sor | 159 | 2.23 s | 2.12 s | 1.83 s | 1.84 s | 1.16× faster |
| cd2d_512 | 1 | gmres+ilu | 98 | 1.11 s | 1.13 s | 1.18 s | 1.11 s | 1.00× faster |
| p2d_256 | 1 | gmres+none | 2897 | 4.89 s | 4.36 s | 4.26 s | 4.10 s | 1.06× faster |
| p2d_256 | 1 | gmres+jacobi | 2897 | 4.66 s | 4.96 s | 4.13 s | 4.29 s | 1.13× faster |
| p2d_256 | 1 | gmres+sor | 493 | 1.42 s | 1.28 s | 1.10 s | 1.13 s | 1.17× faster |
| p2d_256 | 1 | gmres+ilu | 375 | 832.7 ms | 833.3 ms | 789.6 ms | 774.4 ms | 1.08× faster |
| p2d_1024 | 4 | cg+none | 2161 | 10.16 s | 10.10 s | 10.61 s | 8.81 s | 1.15× faster |
| p2d_1024 | 4 | cg+jacobi | 2161 | 9.81 s | 9.67 s | 10.76 s | 9.81 s | 1.01× slower |
| p3d_100 | 4 | cg+jacobi | 231 | 1.12 s | 1.20 s | 1.28 s | 1.06 s | 1.06× faster |
| cd2d_512 | 4 | gmres+none | 1085 | 3.17 s | 2.94 s | 2.22 s | 2.48 s | 1.32× faster |
| cd2d_512 | 4 | gmres+jacobi | 1085 | 3.15 s | 3.27 s | 2.78 s | 2.70 s | 1.17× faster |

### Eigensolves (EPS Krylov–Schur): min wall time

| problem | cores | case | iters / restarts | petsc | petsc-native | rust | rust-native | best Rust vs best PETSc |
|---|---|---|---|---|---|---|---|---|
| p2d_256x250 | 1 | largest nev=4 | 341 | 6.34 s | 6.38 s | 7.67 s | 6.39 s | 1.01× slower |
| p2d_256x250 | 1 | largest nev=10 | 252 | 7.60 s | 7.58 s | 8.26 s | 8.41 s | 1.09× slower |
| spd_rand | 1 | largest nev=4 | 21 | 3.30 s | 3.31 s | 3.28 s | 3.10 s | 1.06× faster |
| spd_rand | 1 | largest nev=10 | 14 | 3.80 s | 3.74 s | 3.33 s | 3.82 s | 1.12× faster |
| p2d_48x40 | 1 | smallest nev=4 | 27 | 10.5 ms | 13.6 ms | 15.7 ms | 11.3 ms | 1.08× slower |
| p2d_256x250 | 4 | largest nev=10 | 252 | 2.29 s | 2.24 s | 2.11 s | 2.24 s | 1.06× faster |
| spd_rand | 4 | largest nev=10 | 14 | 1.06 s | 1.14 s | 1.19 s | 956.2 ms | 1.11× faster |

### Kernels (best of 100)

| problem | n | nnz | kernel | petsc | petsc-native | rust | rust-native |
|---|---|---|---|---|---|---|---|
| p2d_256 | 65536 | 326656 | spmv | 0.2 ms | 0.2 ms | 0.4 ms | 0.2 ms |
| p2d_256 | 65536 | 326656 | dot | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| p2d_256 | 65536 | 326656 | axpy | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| p2d_256 | 65536 | 326656 | pc_jacobi | 0.0 ms | 0.0 ms | 0.0 ms | 0.1 ms |
| p2d_256 | 65536 | 326656 | pc_sor | 0.8 ms | 0.6 ms | 0.5 ms | 0.5 ms |
| p2d_256 | 65536 | 326656 | pc_ilu | 0.7 ms | 0.5 ms | 0.5 ms | 0.5 ms |
| p2d_256 | 65536 | 326656 | pc_icc | 0.6 ms | 0.4 ms | 0.5 ms | 0.5 ms |
| p2d_256 | 65536 | 326656 | mdot16 | 0.5 ms | 0.4 ms | 0.3 ms | 0.3 ms |
| p2d_256 | 65536 | 326656 | maxpy16 | 0.4 ms | 0.4 ms | 0.3 ms | 0.3 ms |
| p2d_1024 | 1048576 | 5238784 | spmv | 8.5 ms | 7.5 ms | 7.5 ms | 7.4 ms |
| p2d_1024 | 1048576 | 5238784 | dot | 0.5 ms | 0.5 ms | 0.6 ms | 0.6 ms |
| p2d_1024 | 1048576 | 5238784 | axpy | 0.6 ms | 0.5 ms | 0.6 ms | 0.6 ms |
| p2d_1024 | 1048576 | 5238784 | pc_jacobi | 1.3 ms | 1.2 ms | 2.0 ms | 2.6 ms |
| p2d_1024 | 1048576 | 5238784 | pc_sor | 19.9 ms | 18.2 ms | 11.9 ms | 12.0 ms |
| p2d_1024 | 1048576 | 5238784 | pc_ilu | 11.8 ms | 10.1 ms | 11.0 ms | 10.8 ms |
| p2d_1024 | 1048576 | 5238784 | pc_icc | 13.0 ms | 11.1 ms | 11.1 ms | 11.4 ms |
| p2d_1024 | 1048576 | 5238784 | mdot16 | 12.6 ms | 12.9 ms | 11.4 ms | 11.4 ms |
| p2d_1024 | 1048576 | 5238784 | maxpy16 | 12.5 ms | 11.8 ms | 11.2 ms | 12.0 ms |
| p3d_100 | 1000000 | 6940000 | spmv | 10.0 ms | 10.1 ms | 9.3 ms | 9.8 ms |
| p3d_100 | 1000000 | 6940000 | dot | 0.5 ms | 0.5 ms | 0.5 ms | 0.7 ms |
| p3d_100 | 1000000 | 6940000 | axpy | 0.5 ms | 0.5 ms | 0.6 ms | 0.6 ms |
| p3d_100 | 1000000 | 6940000 | pc_jacobi | 1.8 ms | 1.3 ms | 1.0 ms | 2.5 ms |
| p3d_100 | 1000000 | 6940000 | pc_sor | 22.4 ms | 21.6 ms | 12.5 ms | 13.0 ms |
| p3d_100 | 1000000 | 6940000 | pc_ilu | 12.7 ms | 11.5 ms | 12.1 ms | 13.1 ms |
| p3d_100 | 1000000 | 6940000 | pc_icc | 13.5 ms | 12.3 ms | 11.5 ms | 17.0 ms |
| p3d_100 | 1000000 | 6940000 | mdot16 | 11.5 ms | 10.7 ms | 10.1 ms | 10.5 ms |
| p3d_100 | 1000000 | 6940000 | maxpy16 | 10.7 ms | 12.0 ms | 10.5 ms | 10.6 ms |
| spd_rand | 200000 | 2599932 | spmv | 7.2 ms | 7.1 ms | 7.3 ms | 7.3 ms |
| spd_rand | 200000 | 2599932 | dot | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| spd_rand | 200000 | 2599932 | axpy | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| spd_rand | 200000 | 2599932 | pc_jacobi | 0.2 ms | 0.2 ms | 0.1 ms | 0.4 ms |
| spd_rand | 200000 | 2599932 | pc_sor | 12.0 ms | 10.7 ms | 9.3 ms | 9.4 ms |
| spd_rand | 200000 | 2599932 | pc_ilu | 7.3 ms | 7.0 ms | 8.6 ms | 8.8 ms |
| spd_rand | 200000 | 2599932 | pc_icc | 7.9 ms | 7.7 ms | 9.0 ms | 8.8 ms |
| spd_rand | 200000 | 2599932 | mdot16 | 2.1 ms | 1.8 ms | 1.6 ms | 1.9 ms |
| spd_rand | 200000 | 2599932 | maxpy16 | 2.1 ms | 1.4 ms | 2.0 ms | 2.0 ms |

EPS agreement p2d_256x250 largest nev=4 p=1: max rel. eigenvalue difference PETSc/Rust = 1.84e-12

EPS agreement p2d_256x250 largest nev=10 p=1: max rel. eigenvalue difference PETSc/Rust = 1.49e-12

EPS agreement spd_rand largest nev=4 p=1: max rel. eigenvalue difference PETSc/Rust = 4.60e-14

EPS agreement spd_rand largest nev=10 p=1: max rel. eigenvalue difference PETSc/Rust = 5.95e-14

EPS agreement p2d_48x40 smallest nev=4 p=1: max rel. eigenvalue difference PETSc/Rust = 9.96e-14

EPS agreement p2d_256x250 largest nev=10 p=4: max rel. eigenvalue difference PETSc/Rust = 1.48e-12

EPS agreement spd_rand largest nev=10 p=4: max rel. eigenvalue difference PETSc/Rust = 5.98e-14
