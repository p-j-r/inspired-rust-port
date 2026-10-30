# inspired-rust-port

Rust ports of numerical-solver and meshing building blocks, written for porting into nyx. The repository has one folder per
topic; each folder is self-contained (own `Cargo.toml`/workspace, README, licence and notices) and is written by its own
working session.

| Folder | Content | Entry points |
|---|---|---|
| [`solvers/`](solvers/README.md) | `rsolvers` (real-only Krylov/Krylov–Schur port following PETSc/SLEPc 3.19 behaviour) and `femsolve` (generic real/complex eigen-, linear-, nonlinear- and time-stepping solvers, preconditioners, low-rank and matrix-free tools, SLEPc/PETSc comparison harness) | [`solvers/docs/STATUS.md`](solvers/docs/STATUS.md), [`solvers/docs/EXPLAINER.md`](solvers/docs/EXPLAINER.md), [`solvers/PROVENANCE.md`](solvers/PROVENANCE.md) |
| `meshers/` | meshing crate (`femmesh`), written by a separate session; not described here | its own README and notices |

Build and test the solvers: `cd solvers && cargo test --release` (add `--features parallel` for the threaded kernels in `femsolve`).

**Licence and origin of the code:** see [`NOTICES.md`](NOTICES.md) (summary), and per folder the `THIRD_PARTY_NOTICES.md`
and, for `solvers/`, [`PROVENANCE.md`](solvers/PROVENANCE.md), which says for every module whether code was taken, a third-party
source was read and re-implemented, or the code was written from a paper.
