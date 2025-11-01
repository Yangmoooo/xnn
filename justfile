# =============================================================================
# === GENERAL CONFIGURATION
# =============================================================================

# --- Project Variables ---
C_DIR       := 'nn4c'
NUMPY_DIR   := 'nn4numpy'
TORCH_DIR   := 'nn4torch'

C_TARGET      := if os_family() == 'windows' { 'main.exe' } else { 'main.out' }
NUMPY_SCRIPT  := NUMPY_DIR + '/main.py'
TORCH_SCRIPT  := TORCH_DIR + '/main.py'


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
    @echo "Compiling C project..."
    @make

# Run the C executable (compiles first if needed).
run-c: build-c
    @echo "Running C project..."
    @./{{C_TARGET}}

# Clean the C project artifacts.
clean-c:
    @echo "Cleaning C project..."
    @rm -f {{C_DIR}}/{{C_TARGET}} {{C_DIR}}/*.o


# =============================================================================
# === PYTHON PROJECT RECIPES
# =============================================================================

# Run the NumPy Python script.
run-numpy:
    @echo "Running numpy implementation..."
    @python {{NUMPY_SCRIPT}}

# Clean Python cache files from the numpy project.
clean-numpy:
    @echo "Cleaning numpy project..."
    @find {{NUMPY_DIR}} -type d -name "__pycache__" -exec rm -rf {} +

# Run the PyTorch Python script.
run-torch:
    @echo "Running torch implementation..."
    @python {{TORCH_SCRIPT}}

# Clean Python cache files from the torch project.
clean-torch:
    @echo "Cleaning torch project..."
    @find {{TORCH_DIR}} -type d -name "__pycache__" -exec rm -rf {} +
