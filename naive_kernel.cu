#include "kernel_classes.cuh"
__global__ void matmul_elem(int n, half* a, half* b, half* c)
{
    int column = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row < n && column < n)
    {
        half dot_prod = 0.f;
        for(int i = 0; i < n; i++)
        {
            dot_prod += a[row*n + i] * b[i*n + column];
        }
        c[row*n+column] = dot_prod;
    }
}

double NaiveKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    int BLOCK_SIZE=32;
    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    double matmul_time = measure_performance([&](){ matmul_elem<<<dimGrid, dimBlock>>>(N, a, b, output); });
    // TODO why are naive kernel differences so big compared to cublas
    // test_output(cublas_ref, N, 5e-1);

    return matmul_time;
}
