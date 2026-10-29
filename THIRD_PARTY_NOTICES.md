# Third-party notices

`rsolvers` is a Rust port of ideas, defaults and control flow from two C
libraries. Their sources were consulted while writing this code (PETSc 3.19.6
`src/ksp/ksp/impls/{cg,gmres}`, `src/ksp/ksp/interface/iterativ.c`,
`src/ksp/pc/impls/{jacobi,sor}`; SLEPc 3.19.2
`src/eps/impls/krylov/{epskrylov.c,krylovschur/krylovschur.c}`,
`src/eps/interface/epssetup.c`), and the algorithms' observable behaviour
(default tolerances, norm and convergence tests, restart rules, selection
order) is reproduced deliberately so results can be compared one-to-one.

Both libraries are released under permissive BSD-2-Clause licenses, which allow
derivative works provided that the copyright notice, the list of conditions and
the disclaimer are retained. They are reproduced verbatim below, and the
repository as a whole is released under BSD-2-Clause (see `LICENSE`).

The code in this repository is not a mechanical translation: data structures,
kernels (fixed-chunk deterministic reductions, fused SpMV+dot, cache-blocked
basis operations) and the Rust API are new.

---

## PETSc

Source: <https://gitlab.com/petsc/petsc/-/blob/main/LICENSE> (BSD-2-Clause)

    Copyright (c) 1991-2021, UChicago Argonne, LLC and the PETSc Development Team
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without modification,
    are permitted provided that the following conditions are met:
    
    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this
      list of conditions and the following disclaimer in the documentation and/or
      other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
    ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    
    This license DOES NOT apply to any software that may be obtained via the
    --download-package option of the PETSc configuration. Each of those packages is
    covered by its own license(s).


---

## SLEPc

Source: <https://gitlab.com/slepc/slepc/-/blob/main/LICENSE.md> (BSD-2-Clause)

    Copyright (c) 2002-2023, Universitat Politecnica de Valencia, Spain
    All rights reserved.
    
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
