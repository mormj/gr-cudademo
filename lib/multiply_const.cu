#include <stdio.h>

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cudaerror.h"
#include <iostream>

// This is the kernel that will get launched on the device
__global__ void multiply_const_kernel(const float* in, float* out, float k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = in[i] * k;
    }
}

// Kernel wrapper so that the GNU Radio code doesn't have to compile with nvcc
void exec_multiply_const_kernel(const float* in,
                float* out,
                float k,
                int grid_size,
                int block_size,
                size_t n,
                cudaStream_t stream)
{

    multiply_const_kernel<<<grid_size, block_size, 0, stream>>>(in, out, k, n);
    check_cuda_errors(cudaGetLastError());
}

void get_block_and_grid(int* minGrid, int* minBlock)
{
   check_cuda_errors(cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, multiply_const_kernel, 0, 0));

}