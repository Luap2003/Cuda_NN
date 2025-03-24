#!/bin/bash

# Benchmark script for neural network performance comparison (C vs Python)
OUTPUT_FILE="nn_benchmark_results.md"
PYTHON_SCRIPT="python_implementation_cuda/temp.py"  # Update this with your Python script name

# Function to extract training time and accuracy from program output
extract_metrics() {
    # Use grep and awk to extract training time and accuracy
    training_time=$(echo "$1" | grep "Training time:" | awk '{print $3}')
    accuracy=$(echo "$1" | grep "Test Accuracy:" | awk '{print $3}')
    
    # Default values if not found
    if [ -z "$training_time" ]; then
        training_time="N/A"
    fi
    
    if [ -z "$accuracy" ]; then
        accuracy="N/A"
    else
        # Remove the % sign if present
        accuracy=$(echo $accuracy | sed 's/%//')
    fi
}

# Calculate speedup
calculate_speedup() {
    if [[ "$1" != "N/A" && "$2" != "N/A" && "$1" != "0" && "$1" != "0.00" ]]; then
        echo "scale=2; $2 / $1" | bc
    else
        echo "N/A"
    fi
}

# Initialize the markdown file with headers
echo "# Neural Network Benchmark Results" > $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Benchmark performed on $(date)" >> $OUTPUT_FILE
echo "System information: $(uname -a)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Part 1: Benchmarking with different epochs and fixed layer sizes
echo "## Benchmark 1: Varying Epochs (100-400)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Configuration: Fixed layer sizes [784, 256, 128, 10], batch size 16" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Epochs | C Time (s) | C Accuracy (%) | Python Time (s) | Python Accuracy (%) | Speedup (C vs Python) |" >> $OUTPUT_FILE
echo "|--------|------------|----------------|-----------------|---------------------|------------------------|" >> $OUTPUT_FILE

BATCH_SIZE=16
EPOCHS_POW2=(100 200 400)

for epochs in "${EPOCHS_POW2[@]}"; do
    echo "Running benchmark with $epochs epochs..."
    
    # Run the C neural network
    c_output=$(./bin/neural_net $BATCH_SIZE $epochs)
    extract_metrics "$c_output"
    c_training_time=$training_time
    c_accuracy=$accuracy
    
    # Run the Python neural network
    python_output=$(python $PYTHON_SCRIPT $BATCH_SIZE $epochs 256 128)
    extract_metrics "$python_output"
    python_training_time=$training_time
    python_accuracy=$accuracy
    
    # Calculate speedup (Python time / C time)
    speedup=$(calculate_speedup $c_training_time $python_training_time)
    
    # Write to markdown file
    echo "| $epochs | $c_training_time | $c_accuracy | $python_training_time | $python_accuracy | $speedup |" >> $OUTPUT_FILE
done

# Part 3: Benchmarking with different layer configurations
echo "" >> $OUTPUT_FILE
echo "## Benchmark 2: Different Layer Configurations" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Configuration: Fixed batch size 16, epochs 100" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Layer Configuration | C Time (s) | C Accuracy (%) | Python Time (s) | Python Accuracy (%) | Speedup (C vs Python) |" >> $OUTPUT_FILE
echo "|---------------------|------------|----------------|-----------------|---------------------|------------------------|" >> $OUTPUT_FILE

EPOCHS=100
LAYER_CONFIGS=(
    "128"
    "256"
    "512"
    "1024"
    "128 64"
    "256 128"
    "512 256"
    "1024 512"
    "256 128 64"
    "512 256 128"
    "1024 512 256"
)

for config in "${LAYER_CONFIGS[@]}"; do
    echo "Running benchmark with layers: $config..."
    
    # Construct the C command with variable number of arguments
    c_cmd="./bin/neural_net $BATCH_SIZE $EPOCHS $config"
    
    # Run the C neural network
    c_output=$(eval $c_cmd)
    extract_metrics "$c_output"
    c_training_time=$training_time
    c_accuracy=$accuracy
    
    # Construct the Python command
    python_cmd="python $PYTHON_SCRIPT $BATCH_SIZE $EPOCHS $config"
    
    # Run the Python neural network
    python_output=$(eval $python_cmd)
    extract_metrics "$python_output"
    python_training_time=$training_time
    python_accuracy=$accuracy
    
    # Calculate speedup
    speedup=$(calculate_speedup $c_training_time $python_training_time)
    
    # Format the layer configuration for display
    display_config="[784, $config, 10]"
    
    # Write to markdown file
    echo "| $display_config | $c_training_time | $c_accuracy | $python_training_time | $python_accuracy | $speedup |" >> $OUTPUT_FILE
done

# Part 4: Benchmarking with different batch sizes
echo "" >> $OUTPUT_FILE
echo "## Benchmark 3: Different Batch Sizes" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "Configuration: Fixed layer sizes [784, 256, 128, 10], epochs 100" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE
echo "| Batch Size | C Time (s) | C Accuracy (%) | Python Time (s) | Python Accuracy (%) | Speedup (C vs Python) |" >> $OUTPUT_FILE
echo "|------------|------------|----------------|-----------------|---------------------|------------------------|" >> $OUTPUT_FILE

EPOCHS=100
BATCH_SIZES=(8 16 32 64 128 256 512 4096)

for batch in "${BATCH_SIZES[@]}"; do
    echo "Running benchmark with batch size: $batch..."
    
    # Run the C neural network
    c_output=$(./bin/neural_net $batch $EPOCHS)
    extract_metrics "$c_output"
    c_training_time=$training_time
    c_accuracy=$accuracy
    
    # Run the Python neural network
    python_output=$(python $PYTHON_SCRIPT $batch $EPOCHS 256 128)
    extract_metrics "$python_output"
    python_training_time=$training_time
    python_accuracy=$accuracy
    
    # Calculate speedup
    speedup=$(calculate_speedup $c_training_time $python_training_time)
    
    # Write to markdown file
    echo "| $batch | $c_training_time | $c_accuracy | $python_training_time | $python_accuracy | $speedup |" >> $OUTPUT_FILE
done

echo "" >> $OUTPUT_FILE
echo "Benchmark completed. Results saved to $OUTPUT_FILE"
echo "Benchmark completed. Results saved to $OUTPUT_FILE"
