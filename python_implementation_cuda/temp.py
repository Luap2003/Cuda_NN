import numpy as np

def generate_layer_test_data(batch_size=3, input_size=4, output_size=3, output_2_size = 2):
    """
    Generate test data for neural network layer testing in C.
    
    Args:
        batch_size (int): Number of samples in the batch
        input_size (int): Size of the input layer
        output_size (int): Size of the output layer
    
    Returns:
        dict: Dictionary containing test data arrays for C code
    """
    # Seed for reproducibility
    np.random.seed(42)
    m = batch_size
    W_k_plus_1 = np.random.rand(output_2_size,output_size)
    dZ_k_plus_1 = np.random.rand(output_2_size,m)
    Z = np.random.rand(output_size,m)
    # Compute expected gradients manually

    def deriv_akt_func(Z):
        return Z > 0

    dZ = W_k_plus_1.T @ dZ_k_plus_1 * deriv_akt_func(Z)
    
    # Prepare output in column-major format (for C)
    def to_c_array_string(arr):
        return ', '.join([f'{x:.5f}f' for x in arr.flatten('F')])
    
    return {
        'W_k_plus_1': to_c_array_string(W_k_plus_1),
        'dZ_k_plus_1': to_c_array_string(dZ_k_plus_1),
        'Z': to_c_array_string(Z),
        'dZ': to_c_array_string(dZ),
        'batch_size': batch_size,
        'input_size': input_size,
        'output_size': output_size,
        'output_size_2': output_2_size
    }

def generate_c_test_code(test_data):
    """
    Generate C test function code with the computed arrays.
    
    Args:
        test_data (dict): Dictionary containing test data
    
    Returns:
        str: C test function code
    """
    print(test_data)
    

# Generate test data
test_data = generate_layer_test_data()

# Print C test code
print(generate_c_test_code(test_data))

# Optional: Print the generated arrays for verification
for key, value in test_data.items():
    if key.startswith('h_'):
        print(f"{key}: {value}")