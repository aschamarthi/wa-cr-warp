PYTHON   ?= python3
SCRIPT    = main.py

# ── Grid sizes ────────────────────────────────────────────────────────────────
N64      = --n 64
N128     = --n 128
N256     = --n 256

# ── Default target ────────────────────────────────────────────────────────────
.PHONY: run run-cpu run-128 run-256 \
        run-viscous run-viscous-128 run-viscous-suth run-viscous-suth-128 \
        clean help

run:                           ## GPU, 64^3, inviscid (default)
	$(PYTHON) $(SCRIPT) $(N64)

run-cpu:                       ## CPU, 64^3, inviscid
	$(PYTHON) $(SCRIPT) $(N64) --cpu

run-128:                       ## GPU, 128^3, inviscid
	$(PYTHON) $(SCRIPT) $(N128)

run-256:                       ## GPU, 256^3, inviscid
	$(PYTHON) $(SCRIPT) $(N256)

run-viscous:                   ## GPU, 64^3, Re=1600, constant mu
	$(PYTHON) $(SCRIPT) $(N64) --re 1600

run-viscous-128:               ## GPU, 128^3, Re=1600, constant mu
	$(PYTHON) $(SCRIPT) $(N128) --re 1600

run-viscous-suth:              ## GPU, 64^3, Re=1600, Sutherland's law
	$(PYTHON) $(SCRIPT) $(N64) --re 1600 --viscosity sutherland

run-viscous-suth-128:          ## GPU, 128^3, Re=1600, Sutherland's law
	$(PYTHON) $(SCRIPT) $(N128) --re 1600 --viscosity sutherland

clean:                         ## Remove all generated output files
	rm -f *.npz *.png ke_tgv3d.txt

help:                          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-30s %s\n", $$1, $$2}'
