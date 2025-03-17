// lossFunction.cu
#include "../include/lossFunction.h"
#include <cuda_runtime.h>
#include <math.h>
// main.cu
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
__global__ void compute_loss_kernel(float *d_predictions, float *d_labels, float *d_loss, int size, LossFunction loss_function) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) {
        float loss = 0.0f;
        if (loss_function == LOSS_MSE) {
            float diff = d_predictions[idx] - d_labels[idx];
            loss = diff * diff;
        }
        else if (loss_function == LOSS_CROSSENTROPY) {
            // To avoid log(0), add a small epsilon
            const float epsilon = 1e-12f;
            float pred = fmaxf(d_predictions[idx], epsilon);
            float label = d_labels[idx];
            loss = - (label * logf(pred) + (1.0f - label) * logf(1.0f - pred));
        }
        // You can add more loss functions here
        atomicAdd(d_loss, loss);
    }
}

float compute_loss(float *d_predictions, float *d_labels, int size, LossFunction loss_function) {
    float h_loss = 0.0f;
    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    compute_loss_kernel<<<blocks, threads>>>(d_predictions, d_labels, d_loss, size, loss_function);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / size;
}

// Device function for Huber loss
__device__ float huber_loss_device(float y_true, float y_pred, float delta) {
    float error = y_pred - y_true;
    float abs_error = fabsf(error);
    
    if (abs_error <= delta) {
        return 0.5f * error * error;  // MSE for small errors
    } else {
        return delta * (abs_error - 0.5f * delta);  // Linear for large errors
    }
}

// Device function for Huber loss derivative
__device__ float huber_loss_derivative(float y_true, float y_pred, float delta) {
    float error = y_pred - y_true;
    float abs_error = fabsf(error);
    
    if (abs_error <= delta) {
        return error;  // MSE derivative
    } else {
        return delta * (error > 0 ? 1.0f : -1.0f);  // MAE derivative, clamped
    }
}

// Custom loss kernel that combines Huber loss with dBa and dSr terms
__global__ void custom_loss_kernel(
    const float *predicted,    // Model predictions 
    const float *target,       // True values
    const float *scale_y,      // Scaling factors for predictions
    const float *mean_y,       // Mean values for predictions
    const float *scale_X,      // Scaling factors for inputs/targets
    const float *mean_X,       // Mean values for inputs/targets
    int batch_size,            // Number of samples
    int output_size,           // Output dimension
    float delta,               // Huber loss delta parameter
    float *loss_per_sample,    // Per-sample loss values
    float *dZ                  // Output gradient for backpropagation
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (sample_idx < batch_size) {
        float huber_loss_sum = 0.0f;
        
        // Create local storage for inverse-scaled values
        float predicted_inverse[8];  // Assuming at least 8 dimensions
        float target_inverse[8];    // Assuming at least 8 dimensions
        
        // Compute inverse scaling and basic Huber loss
        for (int j = 0; j < output_size; j++) {
            int idx = sample_idx * output_size + j;
            predicted_inverse[j] = predicted[idx] * scale_y[j] + mean_y[j];
            target_inverse[j] = target[idx] * scale_X[j] + mean_X[j];
            
            // Compute Huber loss for this dimension
            float huber = huber_loss_device(target[idx], predicted[idx], delta);
            huber_loss_sum += huber;
            
            // Store the basic gradient (Huber derivative) - scale by h1
            dZ[idx] = 0.16726490480995826 * huber_loss_derivative(target[idx], predicted[idx], delta);
        }
        
        // Add dBa term with proper weighting
        float dBa_diff = (predicted_inverse[2] + predicted_inverse[6]) - (target_inverse[2] + target_inverse[6]);
        float dBa = fabsf(dBa_diff);
        
        // Add dSr term with proper weighting
        float dSr_diff = (predicted_inverse[4] + predicted_inverse[7]) - (target_inverse[4] + target_inverse[7]);
        float dSr = fabsf(dSr_diff);
        
        // Weighted total loss
        float sample_loss = 0.16726490480995826 * huber_loss_sum + 0.5283208497548787 * dBa + 0.5099528144902471 * dSr;
        loss_per_sample[sample_idx] = sample_loss;
        
        // Add properly weighted gradients for the mass balance terms
        float sign_dBa = (dBa_diff > 0) ? 1.0f : -1.0f;
        dZ[sample_idx * output_size + 2] += 0.5283208497548787 * sign_dBa * scale_y[2];
        dZ[sample_idx * output_size + 6] += 0.5283208497548787 * sign_dBa * scale_y[6];
        
        float sign_dSr = (dSr_diff > 0) ? 1.0f : -1.0f; 
        dZ[sample_idx * output_size + 4] += 0.5099528144902471 * sign_dSr * scale_y[4];
        dZ[sample_idx * output_size + 7] += 0.5099528144902471 * sign_dSr * scale_y[7];
    }
}

// Host function to compute the custom loss
void compute_custom_loss(
    float *predicted_d,    // Predictions on device
    float *target_d,       // Targets on device
    float *scale_y_d,      // Scale factors for Y on device
    float *mean_y_d,       // Mean values for Y on device
    float *scale_X_d,      // Scale factors for X on device
    float *mean_X_d,       // Mean values for X on device
    int batch_size,        // Batch size
    int output_size,       // Output dimension
    float delta,           // Huber loss delta
    float *loss,           // Output loss value (host)
    float *dZ_d            // Output gradients (device)
) {
    // Allocate device memory for per-sample loss
    float *loss_per_sample_d;
    cudaMalloc((void**)&loss_per_sample_d, batch_size * sizeof(float));
    
    // Launch kernel
    int blocks = (batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    custom_loss_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        predicted_d, target_d,
        scale_y_d, mean_y_d,
        scale_X_d, mean_X_d,
        batch_size, output_size,
        delta,
        loss_per_sample_d, dZ_d
    );
    
    // Allocate host memory for reduction
    float *loss_per_sample_h = (float*)malloc(batch_size * sizeof(float));
    cudaMemcpy(loss_per_sample_h, loss_per_sample_d, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Sum the losses on host
    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; i++) {
        total_loss += loss_per_sample_h[i];
    }
    // Return average loss
    *loss = total_loss / batch_size;
    
    // Clean up
    cudaFree(loss_per_sample_d);
    free(loss_per_sample_h);
}