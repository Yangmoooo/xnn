# =============================================================================
# === GENERAL CONFIGURATION
# =============================================================================

# --- Project Variables ---
C_DIR       := 'nn4c'
NUMPY_DIR   := 'nn4numpy'
TORCH_DIR   := 'nn4torch'

# --- Python Scripts ---
PYTHON        := 'python'
NUMPY_SCRIPT  := NUMPY_DIR + '/main.py'
TORCH_SCRIPT  := TORCH_DIR + '/main.py'

# --- C Compilation Flags ---
C_OPTIM_FLAGS  := '-O3 -march=native -ffast-math'
C_WARN_FLAGS   := '-Wall -Wextra -pedantic'
C_INC_FLAGS    := '-I' + C_DIR + '/include'
CFLAGS         := C_OPTIM_FLAGS + ' ' + C_WARN_FLAGS + ' ' + C_INC_FLAGS


# =============================================================================
# === CROSS-PLATFORM CONFIGURATION
# =============================================================================

OS := os_family()

# MinGW gcc performs much better than LLVM clang in optimization on Windows
CC     := if OS == 'windows' { 'gcc' } else { 'clang' }
EXT    := if OS == 'windows' { '.exe' } else { '.out' }
LDLIBS := if OS == 'windows' { '' } else { '-lm' }

C_TARGET  := C_DIR + '/main' + EXT


# =============================================================================
# === MAIN & HIGH-LEVEL RECIPES
# =============================================================================

# Show helpful information about available recipes.
default:
    @just --list

# Build all compilable projects.
all: build-c

# Remove all build artifacts from all projects.
clean: clean-c clean-numpy clean-torch


# =============================================================================
# === C PROJECT RECIPES
# =============================================================================

# Compile the C project.
build-c:
    @echo "Compiling C project with '{{CC}}'..."
    {{CC}} {{CFLAGS}} -o {{C_TARGET}} {{C_DIR}}/*.c {{LDLIBS}}

# Run the C executable (compiles first if needed).
run-c: build-c
    @echo "Running C project..."
    @./{{C_TARGET}}

# Clean the C project artifacts.
clean-c:
    @echo "Cleaning C project..."
    @rm -f {{C_TARGET}} {{C_DIR}}/*.o


# =============================================================================
# === PYTHON PROJECT RECIPES
# =============================================================================

# Run the NumPy Python script.
run-numpy:
    @echo "Running numpy implementation..."
    @{{PYTHON}} {{NUMPY_SCRIPT}}

# Clean Python cache files from the numpy project.
clean-numpy:
    @echo "Cleaning numpy project..."
    @find {{NUMPY_DIR}} -type d -name "__pycache__" -exec rm -rf {} +

# Run the PyTorch Python script.
run-torch:
    @echo "Running torch implementation..."
    @{{PYTHON}} {{TORCH_SCRIPT}}

# Clean Python cache files from the torch project.
clean-torch:
    @echo "Cleaning torch project..."
    @find {{TORCH_DIR}} -type d -name "__pycache__" -exec rm -rf {} +
