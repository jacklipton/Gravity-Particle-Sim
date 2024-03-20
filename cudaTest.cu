#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// CUDA kernel to add two arrays element-wise
__global__ void addArrays(float* a, float* b, float* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

// CPU function to add two arrays element-wise
void addArraysCPU(float* a, float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}


int main() {
    int size = 100000; // Size of the arrays
    int numBytes = size * sizeof(float);

    // Allocate memory on the host
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* h_c = new float[size];
    float* h_d = new float[size];

    // Initialize the input arrays
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Allocate memory on the device
    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, numBytes);
    cudaMalloc((void**)&d_b, numBytes);
    cudaMalloc((void**)&d_c, numBytes);

    // Copy input arrays from host to device
    cudaMemcpy(d_a, h_a, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, numBytes, cudaMemcpyHostToDevice);



    // Launch the kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
    cudaEventRecord(stop);

        // Wait for the kernel to finish
        cudaDeviceSynchronize();

        // Calculate the elapsed time between the two events
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy the result array from device to host
    cudaMemcpy(h_c, d_c, numBytes, cudaMemcpyDeviceToHost);

    // Print the elapsed time
    std::cout << "Elapsed time (GPU): " << milliseconds << " ms" << std::endl;

    // Free memory on the device
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Time the CPU function
    auto startCPU = std::chrono::high_resolution_clock::now();
    addArraysCPU(h_a, h_b, h_d, size);
    auto stopCPU = std::chrono::high_resolution_clock::now();

    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(stopCPU - startCPU);

    std::cout << "Elapsed time (CPU): "
            << durationCPU.count() / 1000.0 << " ms" << std::endl;

    // Compare the results
    bool arraysAreEqual = true;
    for (int i = 0; i < size; i++) {
        if (std::abs(h_c[i] - h_d[i]) > 1e-5) {
            arraysAreEqual = false;
            break;
        }
    }

    if (arraysAreEqual) {
        std::cout << "The arrays are equal." << std::endl;
    } else {
        std::cout << "The arrays are not equal." << std::endl;
    }

    // Free memory on the host
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_d;

    return 0;
}