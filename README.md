# PCA-EXP-5-MATRIX-MULTIPLICATION-USING-CUDA-AY-23-24

<h3>ENTER YOUR NAME : Vikamuhan reddy</h3>
<h3>ENTER YOUR REGISTER NO 212223240181</h3>
<h3>EX. NO< : 5/h3>
<h3>DATE : 18/05/25</h3>
<h1> <align=center> MATRIX MULTIPLICATION USING CUDA </h3>
  Implement Matrix Multiplication using GPU.</h3>

## AIM:
To perform Matrix Multiplication using CUDA and check its performance with nvprof.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler

## PROCEDURE:
1.	Define Constants: Define the size of the matrices (SIZE) and the size of the CUDA blocks (BLOCK_SIZE).
2.	Kernel Function: Define a CUDA kernel function matrixMultiply that performs the matrix multiplication.
3.	In the main function, perform the following steps:
4.	Initialize Matrices: Initialize the input matrices ‘a’ and ‘b’ with some values.
5.	Allocate Device Memory: Allocate memory on the GPU for the input matrices ‘a’ and ‘b’, and the output matrix ‘c’.
6.	Copy Matrices to Device: Copy the input matrices from host (CPU) memory to device (GPU) memory.
7.	Set Grid and Block Sizes: Set the grid and block sizes for the CUDA kernel launch.
8.	Start Timer: Start a timer to measure the execution time of the kernel.
9.	Launch Kernel: Launch the matrixMultiply kernel with the appropriate grid and block sizes, and the input and output matrices as arguments.
10.	Copy Result to Host: After the kernel execution, copy the result matrix from device memory to host memory.
11.	Stop Timer: Stop the timer and calculate the elapsed time.
12.	Print Result: Print the result matrix and the elapsed time.
13.	Free Device Memory: Finally, free the device memory that was allocated for the matrices.
## PROGRAM:

```cuda
%%writefile matrix_mul.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdbool.h>

#ifndef _COMMON_H
#define _COMMON_H

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif // _COMMON_H

#define SIZE 4
#define BLOCK_SIZE 2

// Host matrix multiplication for verification
void matrixMultiplyHost(int *a, int *b, int *c, int size)
{
    for (int row = 0; row < size; ++row)
    {
        for (int col = 0; col < size; ++col)
        {
            int sum = 0;
            for (int k = 0; k < size; ++k)
            {
                sum += a[row * size + k] * b[k * size + col];
            }
            c[row * size + col] = sum;
        }
    }
}

// GPU kernel
__global__ void matrixMultiply(int *a, int *b, int *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size)
    {
        int sum = 0;
        for (int k = 0; k < size; ++k)
        {
            sum += a[row * size + k] * b[k * size + col];
        }
        c[row * size + col] = sum;
    }
}

int main()
{
    int a[SIZE * SIZE], b[SIZE * SIZE], c[SIZE * SIZE], c_host[SIZE * SIZE];
    int *dev_a, *dev_b, *dev_c;
    int bytes = SIZE * SIZE * sizeof(int);

    // Initialize matrices
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            a[i * SIZE + j] = i + 1;
            b[i * SIZE + j] = j + 1;
        }
    }

    // Allocate device memory
    CHECK(cudaMalloc((void**)&dev_a, bytes));
    CHECK(cudaMalloc((void**)&dev_b, bytes));
    CHECK(cudaMalloc((void**)&dev_c, bytes));

    // Copy to device
    CHECK(cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice));

    // CUDA execution
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrixMultiply<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, SIZE);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    gettimeofday(&end, NULL);
    double elapsed_gpu = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    CHECK(cudaMemcpy(c, dev_c, bytes, cudaMemcpyDeviceToHost));

    // CPU execution
    gettimeofday(&start, NULL);
    matrixMultiplyHost(a, b, c_host, SIZE);
    gettimeofday(&end, NULL);
    double elapsed_cpu = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    // Print results
    printf("Result Matrix from GPU:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c[i * SIZE + j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix from CPU:\n");
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            printf("%d ", c_host[i * SIZE + j]);
        }
        printf("\n");
    }

    

    printf("GPU Time: %.6f s\n", elapsed_gpu);
    printf("CPU Time: %.6f s\n", elapsed_cpu);

    // Cleanup
    CHECK(cudaFree(dev_a));
    CHECK(cudaFree(dev_b));
    CHECK(cudaFree(dev_c));

    return 0;
}
```


## OUTPUT:

<img width="440" alt="Screen Shot 1947-02-28 at 22 38 33" src="https://github.com/user-attachments/assets/f7724643-da6e-4d3f-b150-06c368585146" />


## RESULT:
Thus the program has been executed by using CUDA to mulptiply two matrices. It is observed that there are variations in host and device elapsed time. Device took  0.000113 s time and host took 0.000001 s time.
