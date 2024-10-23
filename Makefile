# Compiler settings
NVCC        = nvcc
NVCC_FLAGS  = -arch=sm_61 -O2 -I./include -I/usr/local/cuda/include
LD_FLAGS    = -L/usr/local/cuda/lib64 -lcudart

# Directories
SRC_DIR     = ./src
OBJ_DIR     = ./obj
BIN_DIR     = ./bin
INCLUDE_DIR = ./include

# Source files
CU_SOURCES  = $(wildcard $(SRC_DIR)/*.cu)
CU_OBJECTS  = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))

# Executable
TARGET      = $(BIN_DIR)/neural_net

# Default rule
all: $(TARGET)

# Build the target executable
$(TARGET): $(CU_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LD_FLAGS)

# Compile CUDA source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Create directories if they don't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean up build files
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET)

.PHONY: all clean
