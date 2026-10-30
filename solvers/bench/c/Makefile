# Requires PETSc and SLEPc (pkg-config names PETSc, SLEPc) built with MPI.
#   distro packages:   make                       (apt install libpetsc-real-dev libslepc-real-dev)
#   custom build:      PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig:$SLEPC_DIR/$PETSC_ARCH/lib/pkgconfig make OUT=native PKGNAMES="slepc PETSc" CFLAGS="-O3 -march=native"
MPICC  ?= mpicc
CFLAGS ?= -O2
OUT    ?= distro
PKGNAMES ?= SLEPc PETSc
PKG    := $(shell pkg-config --cflags --libs $(PKGNAMES))
BINS   := $(OUT)/petsc_ksp $(OUT)/slepc_eps $(OUT)/petsc_micro

all: $(BINS)

$(OUT)/%: %.c common.h
	@mkdir -p $(OUT)
	$(MPICC) $(CFLAGS) -o $@ $< $(PKG)

clean:
	rm -rf distro native
