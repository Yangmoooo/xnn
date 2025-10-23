# =============================================================================
# === GENERAL CONFIGURATION
# =============================================================================
CC := clang
PYTHON := python3

# Project directories
C_DIR := nn4c
NUMPY_DIR := nn4numpy
TORCH_DIR := nn4torch

NUMPY_SCRIPT := $(NUMPY_DIR)/main.py
TORCH_SCRIPT := $(TORCH_DIR)/main.py

# C compilation flags
CFLAGS := -O3 -march=native -ffast-math -Wall -Wextra -pedantic -I$(C_DIR)/inc


# =============================================================================
# === C PROJECT: EFFICIENT SOURCE & OBJECT FILE HANDLING
# =============================================================================
# `wildcard` finds all .c files, and `patsubst` creates a corresponding list
# of .o (object) files. This allows for incremental compilation.
C_SRCS := $(wildcard $(C_DIR)/*.c)
C_OBJS := $(patsubst $(C_DIR)/%.c,$(C_DIR)/%.o,$(C_SRCS))


# =============================================================================
# === CROSS-PLATFORM CONFIGURATION
# =============================================================================
EXT := .out
RM := rm -f
LDLIBS := -lm

# Override for Windows
ifeq ($(OS),Windows_NT)
	# MinGW gcc performs much better than LLVM clang in optimization on Windows
	CC := gcc
	EXT := .exe
	RM := del /Q /F
	LDLIBS :=
	C_TARGET_WIN := $(subst /,\,$(C_DIR)\main$(EXT))
	C_OBJS_WIN := $(subst /,\,$(C_DIR)\*.o)
endif

C_TARGET := $(C_DIR)/main$(EXT)


# =============================================================================
# === PHONY TARGETS
# =============================================================================
.PHONY: all help build-c run-c clean-c clean


# =============================================================================
# === MAIN & HIGH-LEVEL TARGETS
# =============================================================================

help:
	@echo "Usage: make [TARGET]"
	@echo "--------------------------------------------------"
	@echo "High-Level Targets:"
	@echo "  all          Build all compilable projects (C, Rust)."
	@echo "  clean        Remove all build artifacts."
	@echo "  help         Show this help message."
	@echo ""
	@echo "C Project (nn4c):"
	@echo "  build-c      Compile the C project."
	@echo "  run-c        Run the C executable."
	@echo "  clean-c      Clean the C project artifacts (.o files and executable)."
	@echo ""
	@echo "Python Project (nn4numpy):"
	@echo "  run-numpy    Run the Python script."
	@echo "  clean-numpy  Clean Python cache files."
	@echo ""
	@echo "Python Project (nn4torch):"
	@echo "  run-torch    Run the Python script."
	@echo "  clean-torch  Clean Python cache files."
	@echo "--------------------------------------------------"

all: build-c

clean: clean-c clean-numpy clean-torch


# =============================================================================
# === C TARGETS
# =============================================================================

build-c: $(C_TARGET)

$(C_TARGET): $(C_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(C_DIR)/%.o: $(C_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run-c: build-c
	@echo "Running C project..."
	./$(C_TARGET)

clean-c:
	@echo "Cleaning C project..."
ifeq ($(OS),Windows_NT)
	if exist $(C_TARGET_WIN) $(RM) $(C_TARGET_WIN)
	if exist $(C_OBJS_WIN) $(RM) $(C_OBJS_WIN)
else
	$(RM) $(C_TARGET) $(C_OBJS)
endif


# =============================================================================
# === PYTHON TARGETS
# =============================================================================

# Generic rules for Python projects (e.g., numpy, torch)
PY_PROJECTS := numpy torch
RUN_PY_TARGETS := $(foreach proj,$(PY_PROJECTS),run-$(proj))
CLEAN_PY_TARGETS := $(foreach proj,$(PY_PROJECTS),clean-$(proj))

.PHONY: $(RUN_PY_TARGETS) $(CLEAN_PY_TARGETS)

$(RUN_PY_TARGETS): run-%:
	@echo "Running $* implementation..."
	@$(PYTHON) nn4$*/main.py

$(CLEAN_PY_TARGETS): clean-%:
	@echo "Cleaning $* project..."
ifeq ($(OS),Windows_NT)
	if exist nn4$*\__pycache__ rmdir /S /Q nn4$*\__pycache__
else
	find nn4$* -type d -name "__pycache__" -exec rm -rf {} +
endif
