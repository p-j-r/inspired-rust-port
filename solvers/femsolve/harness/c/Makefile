# Builds the SLEPc driver against the *real* and the *complex* Debian builds (3.19):
#   make            -> real/slepc_eps  complex/slepc_eps
# For other installations set PKG_CONFIG_PATH_REAL / PKG_CONFIG_PATH_COMPLEX to the directories
# that hold PETSc.pc and SLEPc.pc of the matching scalar type.
MPICC   ?= mpicc
CFLAGS  ?= -O2
PKG_REAL    ?= /usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-real/lib/pkgconfig:/usr/lib/slepcdir/slepc3.19/x86_64-linux-gnu-real/lib/pkgconfig
PKG_COMPLEX ?= /usr/lib/petscdir/petsc3.19/x86_64-linux-gnu-complex/lib/pkgconfig:/usr/lib/slepcdir/slepc3.19/x86_64-linux-gnu-complex/lib/pkgconfig

all: real/slepc_eps complex/slepc_eps real/petsc_ksp complex/petsc_ksp real/slepc_ciss complex/slepc_ciss real/petsc_snes

real/slepc_eps: slepc_eps.c common.h
	@mkdir -p real
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_REAL) pkg-config --cflags --libs SLEPc PETSc)

complex/slepc_eps: slepc_eps.c common.h
	@mkdir -p complex
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_COMPLEX) pkg-config --cflags --libs SLEPc PETSc)

real/petsc_ksp: petsc_ksp.c common.h
	@mkdir -p real
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_REAL) pkg-config --cflags --libs SLEPc PETSc)

complex/petsc_ksp: petsc_ksp.c common.h
	@mkdir -p complex
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_COMPLEX) pkg-config --cflags --libs SLEPc PETSc)

real/slepc_ciss: slepc_ciss.c common.h
	@mkdir -p real
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_REAL) pkg-config --cflags --libs SLEPc PETSc)

complex/slepc_ciss: slepc_ciss.c common.h
	@mkdir -p complex
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_COMPLEX) pkg-config --cflags --libs SLEPc PETSc)

real/petsc_snes: petsc_snes.c
	@mkdir -p real
	$(MPICC) $(CFLAGS) -o $@ $< $$(PKG_CONFIG_PATH=$(PKG_REAL) pkg-config --cflags --libs PETSc) -lm

clean:
	rm -rf real complex
