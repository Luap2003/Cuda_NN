# Compiler settings
ARCH = sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | tr -d '.')
NVCC        = nvcc
CC          = gcc
NVCC_FLAGS  = -arch=$(ARCH) -O2 -I./include -I/usr/local/cuda/include -I./tests/unity -lcublas
CC_FLAGS    = -O2 -I./include -I/usr/local/cuda/include -I./tests/unity
LD_FLAGS    = -L/usr/local/cuda/lib64 -lcudart -lcublas -lm

CC_FLAGS += $(DEBUG)
NVCC_FLAGS += $(DEBUG)

# Directories
SRC_DIR     = ./src
OBJ_DIR     = ./obj
BIN_DIR     = ./bin
INCLUDE_DIR = ./include
TEST_DIR    = ./tests
UNITY_DIR   = $(TEST_DIR)/unity

# Source files
CU_SOURCES  = $(wildcard $(SRC_DIR)/*.cu)
CU_OBJECTS  = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))
C_SOURCES  = $(wildcard $(SRC_DIR)/*.c)
C_OBJECTS  = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(C_SOURCES))

CU_OBJECTS_NO_MAIN = $(filter-out $(OBJ_DIR)/main.o, $(CU_OBJECTS))
# Unity files
UNITY_SRC   = $(UNITY_DIR)/unity.c
UNITY_OBJ   = $(OBJ_DIR)/unity.o

# Test source files
TEST_SOURCES = $(wildcard $(TEST_DIR)/test_*.c)
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_SOURCES))

# Executables
TARGET      = $(BIN_DIR)/neural_net
TEST_TARGET = $(BIN_DIR)/test_runner

# Default rule
all: $(TARGET) $(TEST_TARGET)

# Build the main project executable
$(TARGET): $(CU_OBJECTS) $(C_OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(CU_OBJECTS) $(C_OBJECTS) $(LD_FLAGS)

# Build the test runner executable
$(TEST_TARGET): $(TEST_OBJECTS) $(UNITY_OBJ) $(CU_OBJECTS_NO_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LD_FLAGS)

# Compile CUDA source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Compile C source files into object files using GCC
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile test C source files into object files using NVCC
$(OBJ_DIR)/%.o: $(TEST_DIR)/%.c | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

# Compile Unity into an object file using NVCC
$(UNITY_OBJ): $(UNITY_SRC) | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

# Create directories if they don't exist
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Run tests
test: $(TEST_TARGET)
	@echo "Running Tests..."
	$(TEST_TARGET)

# Clean up build files
clean:
	rm -rf $(OBJ_DIR)/*.o $(TARGET) $(TEST_TARGET)

.PHONY: all clean test

docs:
	doxygen Doxyfile
