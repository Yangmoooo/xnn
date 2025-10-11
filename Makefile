# =============================================================================
# === GENERAL CONFIGURATION
# =============================================================================
CC := clang
CARGO := cargo
PYTHON := python3

# Project directories
C_DIR := nn4c
PY_DIR := nn4py
RUST_DIR := nn4rs

PY_SCRIPT := $(PY_DIR)/main.py

# C compilation flags
CFLAGS := -O3 -march=native -ffast-math -Wall -Wextra -pedantic


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
# === PHONY TARGETS (Actions that don't create files)
# =============================================================================
.PHONY: all help build-c run-c clean-c build-py run-py clean-py build-rs run-rs clean-rs clean


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
	@echo "Python Project (nn4py):"
	@echo "  run-py       Run the Python script."
	@echo "  clean-py     Clean Python cache files."
	@echo ""
	@echo "Rust Project (nn4rs):"
	@echo "  build-rs     Build the Rust project (release mode)."
	@echo "  run-rs       Run the Rust project (release mode)."
	@echo "  clean-rs     Clean the Rust project."
	@echo "--------------------------------------------------"

all: build-c build-rs

clean: clean-c clean-py clean-rs


# =============================================================================
# === C LANGUAGE TARGETS (Incremental Compilation)
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
# === PYTHON & RUST TARGETS
# =============================================================================

run-py: $(PY_SCRIPT)
	@echo "Running Python script..."
	$(PYTHON) $(PY_SCRIPT)

clean-py:
	@echo "Cleaning Python project..."
ifeq ($(OS),Windows_NT)
	if exist $(PY_DIR)\\*.pyc $(RM) $(PY_DIR)\\*.pyc
	if exist $(PY_DIR)\\__pycache__ rmdir /S /Q $(PY_DIR)\\__pycache__
else
	find $(PY_DIR) -type f -name "*.pyc" -delete
	find $(PY_DIR) -type d -name "__pycache__" -exec rm -rf {} +
endif

build-rs:
	@echo "Building Rust project..."
	$(CARGO) build --release --manifest-path $(RUST_DIR)/Cargo.toml

run-rs:
	@echo "Running Rust project..."
	$(CARGO) run --release --manifest-path $(RUST_DIR)/Cargo.toml

clean-rs:
	@echo "Cleaning Rust project..."
	$(CARGO) clean --manifest-path $(RUST_DIR)/Cargo.toml
