#!/usr/bin/env python3
"""Compare femsolve against SLEPc on identical problem bundles (port item V1).

  python3 harness/run.py run    --suite quick --out harness/results/quick.jsonl
  python3 harness/run.py report harness/results/quick.jsonl
  python3 harness/run.py run-ksp --out harness/results/ksp.jsonl     # Krylov linear solvers vs PETSc KSP
  python3 harness/run.py rejudge harness/results/quick.jsonl   # recompute verdicts after changing the rules

Setup (Ubuntu 24.04):
  apt install libpetsc-real-dev libslepc-real-dev libpetsc-complex-dev libslepc-complex-dev libopenmpi-dev pkg-config
  cargo build --release -p femsolve
  make -C femsolve/harness/c

Acceptance tiers (see docs/STATUS.md):
  A  same algorithm, same start vector and options: identical restart and operator-application
     counts, eigenvalues to 1e-10 relative, residuals recomputed by each side independently
  B  different path to the same answer: eigenvalue sets to 1e-8 and residuals below tolerance
     (used when the spectrum is degenerate, where which copy emerges depends on rounding)
Timing: minimum of repeated solves after a warm-up, both sides in one session; differences
under 10 % are ties.
"""
import argparse, json, os, subprocess, sys, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUST_EPS = os.path.join(ROOT, "target", "release", "fs-eps")
RUST_GEN = os.path.join(ROOT, "target", "release", "fs-gen")
DATA = os.path.join(HERE, "data")
SLEPC = {"real": os.path.join(HERE, "c", "real", "slepc_eps"), "complex": os.path.join(HERE, "c", "complex", "slepc_eps")}

# name, spec, scalar, hermitian, args (shared), tier
def case(name, spec, scalar, herm, tier="A", **kw):
    return dict(name=name, spec=spec, scalar=scalar, herm=herm, tier=tier, **kw)

SUITES = {
    "quick": [
        case("fe1d ncv=12", "fe1d:2000", "real", 1, sigma=(0.5, 0), nev=4, ncv=12),
        case("fe1d default", "fe1d:2000", "real", 1, sigma=(0.5, 0), nev=4),
        case("fe1d nev=10 ncv=14", "fe1d:2000", "real", 1, sigma=(0.5, 0), nev=10, ncv=14, tol=1e-10),
        # SLEPc's test is ‖r‖/|θ|; eigenvalues λ ~ 1e-5 here make the relative residual of the original problem ~ tol/|λ|.
        case("fe1d Cholesky (shift below spectrum)", "fe1d:2000", "real", 1, sigma=(-1e-4, 0), nev=5, ncv=12, definite=1, resid_max=1e-2),
        case("gauge1d (complex Hermitian)", "gauge1d:2000:0.7", "complex", 1, sigma=(0.513, 0), nev=4),
        case("gauge1d ncv=8", "gauge1d:2000:0.7", "complex", 1, sigma=(0.513, 0), nev=4, ncv=8),
        case("pml1d (complex symmetric)", "pml1d:2000:0.3", "complex", 0, sigma=(0.5133, 0.3001), nev=4),
        case("pml1d ncv=8", "pml1d:2000:0.3", "complex", 0, sigma=(0.5133, 0.3001), nev=4, ncv=8),
        case("pml2d pencil (complex B)", "pml2d:60:50:3.0", "complex", 0, sigma=(1.0137, 0.0), nev=4),
        case("lap1d direct, largest", "lap1d:600", "real", 1, mode="direct", which="largest", nev=3, ncv=20, tol=1e-8, maxit=3000),
        case("lap2d (degenerate)", "lap2d:100:100", "real", 1, tier="B", sigma=(2.013, 0), nev=6),
    ],
    "full": [],
}
SUITES["full"] = SUITES["quick"] + [
    case("fe1d n=20000", "fe1d:20000", "real", 1, sigma=(0.5, 0), nev=8, reps=5),
    case("gauge1d n=20000", "gauge1d:20000:0.7", "complex", 1, sigma=(0.513, 0), nev=8, reps=5),
    case("pml1d n=20000", "pml1d:20000:0.3", "complex", 0, sigma=(0.5133, 0.3001), nev=8, reps=5),
    # ~100 restarts of a slowly converging non-Hermitian run: rounding differences between the two LU factorisations are amplified, so only tier B is meaningful
    case("pml2d 200x200", "pml2d:200:200:3.0", "complex", 0, tier="B", sigma=(1.0137, 0.0), nev=6, reps=3),
    # a cube: the spectrum is degenerate, so only tier B is meaningful
    case("lap3d 40 Cholesky", "lap3d:40", "real", 1, tier="B", sigma=(-0.1, 0), nev=6, definite=1, reps=3),
]

# ---- linear solvers (KSP) ---------------------------------------------------------------
RUST_KSP = os.path.join(ROOT, "target", "release", "fs-ksp")
PETSC_KSP = {"real": os.path.join(HERE, "c", "real", "petsc_ksp"), "complex": os.path.join(HERE, "c", "complex", "petsc_ksp")}

def kcase(name, spec, scalar, solver, pc="none", **opts):
    return dict(name=name, spec=spec, scalar=scalar, solver=solver, pc=pc, opts=opts)

# PETSc's COCG applies a real-part sign test that is wrong for complex symmetric matrices (it stops
# at iteration 2 on every complex symmetric problem tried), so COCG is compared on real SPD matrices
# (identical algorithm) and checked against a dense solve in tests/ksp.rs for the complex case.
KSP_SUITES = {
    "ksp": [
        kcase("CG complex build, real SPD", "lap2d:100:100", "complex", "cg", "jacobi"),
        kcase("COCG complex build, real SPD", "lap2d:100:100", "complex", "cocg", "jacobi"),
        kcase("GMRES(30) real nonsymmetric", "convdiff2d:200:200:5:3", "real", "gmres"),
        kcase("GMRES(30) right PC", "convdiff2d:200:200:5:3", "real", "gmres", "jacobi", side="right"),
        kcase("GMRES(30) MGS", "convdiff2d:200:200:5:3", "real", "gmres", "jacobi", orth="mgs"),
        kcase("FGMRES(30)", "convdiff2d:200:200:5:3", "real", "fgmres", "jacobi"),
        kcase("GMRES(60) complex symmetric", "helm2d:150:150:3.0:0.4", "complex", "gmres", "none", restart=60),
        kcase("GMRES(40) complex symmetric Jacobi", "helm2d:150:150:3.0:0.4", "complex", "gmres", "jacobi", restart=40),
        kcase("GMRES(40) right PC, complex", "helm2d:150:150:3.0:0.4", "complex", "gmres", "jacobi", restart=40, side="right"),
        kcase("FGMRES(40) complex", "helm2d:150:150:3.0:0.4", "complex", "fgmres", "jacobi", restart=40),
        kcase("GMRES(40) CGS+refinement, complex", "helm2d:150:150:3.0:0.4", "complex", "gmres", "jacobi", restart=40, orth="cgs-refine"),
    ],
}

def run_ksp_case(c):
    d = ensure_bundle(c["spec"])
    rust = [RUST_KSP, d, "--solver", c["solver"], "--pc", c["pc"], "--rtol", "1e-8", "--reps", "3"]
    petsc = [PETSC_KSP[c["scalar"]], d, "-ksp_kind", c["solver"], "-pc", c["pc"], "-rtol", "1e-8", "-reps", "3"]
    for k, v in c["opts"].items():
        rust += ["--" + k, str(v)]
        petsc += ["-" + k, str(v)]
    return dict(case=c["name"], spec=c["spec"], solver=c["solver"], scalar=c["scalar"], rust=run_json(rust), petsc=run_json(petsc))

def judge_ksp(r):
    a, b = r["rust"], r["petsc"]
    if "error" in a or "error" in b:
        return "ERROR", "rust: %s | petsc: %s" % (a.get("error", "ok")[:60], b.get("error", "ok")[:60])
    ha, hb = a["history"], b["history"]
    # PETSc's FGMRES logs the restart residual twice per cycle (the second copy is the recomputed
    # value, equal to ~1e-12); collapse such repeats on both sides.
    if r["solver"] == "fgmres":
        def collapse(h):
            return [v for i, v in enumerate(h) if i == 0 or abs(v - h[i - 1]) > 1e-9 * abs(h[i - 1])]
        ha, hb = collapse(ha), collapse(hb)
    n = min(len(ha), len(hb))
    dh = max((abs(x - y) / max(abs(y), 1e-300) for x, y in zip(ha[:n], hb[:n])), default=float("nan"))
    ok = a["iterations"] == b["iterations"] and dh < 1e-4 and a["true_rel_resid"] < 1e-6
    return ("PASS" if ok else "FAIL"), "iterations %d/%d, max history diff %.1e, true resid %.1e/%.1e, time %.3g/%.3g s" % (
        a["iterations"], b["iterations"], dh, a["true_rel_resid"], b["true_rel_resid"], a["solve_min_s"], b["solve_min_s"])

def cmd_run_ksp(args):
    for p in (RUST_KSP, RUST_GEN, PETSC_KSP["real"], PETSC_KSP["complex"]):
        if not os.path.exists(p):
            sys.exit("missing %s (see the setup lines at the top of this file)" % p)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for c in KSP_SUITES[args.suite]:
            r = run_ksp_case(c)
            r["verdict"], r["detail"] = judge_ksp(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print("%-4s %-36s %s" % (r["verdict"], c["name"], r["detail"]), flush=True)

def ensure_bundle(spec):
    d = os.path.join(DATA, spec.replace(":", "_"))
    if not os.path.exists(os.path.join(d, "A.mtx")):
        os.makedirs(DATA, exist_ok=True)
        subprocess.run([RUST_GEN, spec, d], check=True, stdout=subprocess.DEVNULL)
    return d

def run_json(cmd, env=None):
    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in out.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {"error": (out.stderr or out.stdout)[-500:]}

def run_case(c):
    d = ensure_bundle(c["spec"])
    reps = c.get("reps", 3)
    sre, sim = c.get("sigma", (0, 0))
    mode = c.get("mode", "shift")
    common = dict(nev=c.get("nev", 4), tol=c.get("tol", 1e-8), reps=reps)
    rust = [RUST_EPS, d, "--mode", mode, "--nev", str(common["nev"]), "--tol", str(common["tol"]), "--reps", str(reps),
            "--sigma-re", str(sre), "--sigma-im", str(sim), "--which", c.get("which", "largest")]
    slepc = [SLEPC[c["scalar"]], d, "-herm", str(c["herm"]), "-mode", mode, "-nev", str(common["nev"]), "-tol", str(common["tol"]),
             "-reps", str(reps), "-sigma_re", str(sre), "-sigma_im", str(sim), "-which", c.get("which", "largest")]
    if "ncv" in c:
        rust += ["--ncv", str(c["ncv"])]
        slepc += ["-ncv", str(c["ncv"])]
    if "maxit" in c:
        rust += ["--maxit", str(c["maxit"])]
        slepc += ["-maxit", str(c["maxit"])]
    if c.get("definite"):
        rust += ["--definite"]
        slepc += ["-definite", "1"]
    # Complex-valued PETSc cannot read an imaginary shift in the real build; the real cases use sim = 0.
    return dict(case=c["name"], spec=c["spec"], tier=c["tier"], scalar=c["scalar"], resid_max=c.get("resid_max", 100 * common["tol"]),
                rust=run_json(rust), slepc=run_json(slepc))

def rel_diff(a, b):
    za, zb = complex(*a), complex(*b)
    return abs(za - zb) / max(abs(zb), 1e-300)

def judge(r):
    a, b = r["rust"], r["slepc"]
    if "error" in a or "error" in b:
        return "ERROR", "rust: %s | slepc: %s" % (a.get("error", "ok")[:80], b.get("error", "ok")[:80])
    nev = min(len(a["values"]), len(b["values"]))
    sigma_sorted = lambda vs: vs
    dv = max((rel_diff(x, y) for x, y in zip(a["values"][:nev], b["values"][:nev])), default=float("nan"))
    res = max(a["true_rel_resid"] + b["true_rel_resid"] + [0])
    same_counts = a["restarts"] == b["restarts"] and a["matvecs"] == b["matvecs"]
    tol = a["tol"]
    res_limit = r.get("resid_max", 100 * tol)
    if r["tier"] == "A":
        ok = same_counts and dv < 1e-10 and res < res_limit
        why = "restarts %d/%d matvecs %d/%d max|dλ|/|λ|=%.1e max resid=%.1e" % (a["restarts"], b["restarts"], a["matvecs"], b["matvecs"], dv, res)
    else:
        # Tier B: degenerate spectra (or runs so long that rounding is amplified). Compare the
        # *distinct* eigenvalues: each solver returns some copies of a degenerate value, not
        # necessarily the same number, so lists are reduced to distinct values (relative gap
        # 1e-7) and compared over the length both cover.
        def distinct(vs):
            out = []
            for v in sorted(vs, key=lambda z: (round(z[0], 12), round(z[1], 12))):
                if not out or abs(complex(*v) - complex(*out[-1])) > 1e-7 * max(abs(complex(*v)), 1e-300):
                    out.append(v)
            return out
        va, vb = distinct(a["values"][:nev]), distinct(b["values"][:nev])
        m = min(len(va), len(vb))
        dv = max((rel_diff(x, y) for x, y in zip(va[:m], vb[:m])), default=float("nan"))
        ok = m > 0 and dv < 1e-8 and res < res_limit
        why = "restarts %d/%d matvecs %d/%d distinct values %d/%d max|dλ|/|λ|=%.1e max resid=%.1e" % (a["restarts"], b["restarts"], a["matvecs"], b["matvecs"], len(va), len(vb), dv, res)
    return ("PASS" if ok else "FAIL"), why

def cmd_run(args):
    for p in (RUST_EPS, RUST_GEN, SLEPC["real"], SLEPC["complex"]):
        if not os.path.exists(p):
            sys.exit("missing %s (see the setup lines at the top of this file)" % p)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for c in SUITES[args.suite]:
            r = run_case(c)
            r["verdict"], r["detail"] = judge(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print("%-4s %-34s %s" % (r["verdict"], c["name"], r["detail"]), flush=True)

def cmd_rejudge(args):
    rows = [json.loads(l) for l in open(args.file)]
    for r in rows:
        r["verdict"], r["detail"] = judge(r)
        print("%-4s %-34s %s" % (r["verdict"], r["case"], r["detail"]))
    with open(args.file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def cmd_report(args):
    rows = [json.loads(l) for l in open(args.file)]
    print("| case | tier | verdict | restarts (rust/slepc) | matvecs | cold rust [s] | cold slepc [s] | warm rust [s] | warm slepc [s] | warm ratio | max resid |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a, b = r["rust"], r["slepc"]
        if "error" in a or "error" in b:
            print("| %s | %s | ERROR | | | | | | |" % (r["case"], r["tier"]))
            continue
        ta, tb = a["total_min_s"], b["total_min_s"]
        res = max(a["true_rel_resid"] + b["true_rel_resid"] + [0])
        print("| %s | %s | %s | %d / %d | %d / %d | %.4g | %.4g | %.4g | %.4g | %.2f | %.1e |" % (
            r["case"], r["tier"], r["verdict"], a["restarts"], b["restarts"], a["matvecs"], b["matvecs"],
            a.get("cold_s", float("nan")), b.get("cold_s", float("nan")), ta, tb, ta / tb if tb else float("nan"), res))
    print()
    print("cold = first solve (setup + symbolic + numeric factorisation + eigensolve); warm = minimum of the repeated solves.")
    print("For rust, a warm solve re-does the numeric factorisation (set_shift) but not the symbolic analysis; for SLEPc the")
    print("repeated EPSSolve reuses the PC factorisation (the logged factorisation time of a warm solve is 0), so warm slepc")
    print("excludes the factorisation. Compare cold with cold for an all-in number; warm rust includes factor_s, see the JSON.")
    print("ratio = rust / slepc (below 1: rust faster).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--suite", default="quick", choices=sorted(SUITES)); r.add_argument("--out", required=True); r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report"); p.add_argument("file"); p.set_defaults(fn=cmd_report)
    k = sub.add_parser("run-ksp", help="compare linear solvers with PETSc KSP"); k.add_argument("--suite", default="ksp", choices=sorted(KSP_SUITES)); k.add_argument("--out", required=True); k.set_defaults(fn=cmd_run_ksp)
    j = sub.add_parser("rejudge", help="recompute the verdicts of a stored result file"); j.add_argument("file"); j.set_defaults(fn=cmd_rejudge)
    args = ap.parse_args(); args.fn(args)
