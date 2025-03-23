#include "kernel_classes.cuh"

#define TILE_WIDTH 32
__global__ void tiled_matmul(int n, half* a, half* b, half* c)
{
    __shared__ half a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ half b_tile[TILE_WIDTH][TILE_WIDTH];

    int column = blockIdx.x*TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float dot_prod = 0.f;
    for (int tile_offset = 0; tile_offset<n; tile_offset+=TILE_WIDTH)
    {
        int a_chk = tile_offset+tx < n && row < n;
        a_tile[ty][tx] = a_chk ? a[row*n + tile_offset+tx] : (half)0.f;

        int b_chk = (tile_offset+ty) < n && column < n;
        b_tile[ty][tx] = b_chk ? b[(tile_offset+ty)*n + column] : (half)0.f;

        __syncthreads();
        for(int i = 0; i < TILE_WIDTH; i++)
        {
            dot_prod += (float)a_tile[ty][i] * (float)b_tile[i][tx];
        }
        __syncthreads();
    }

    if (row < n && column < n)
    {
        c[row*n+column] = dot_prod;
    }
}

double TiledKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    dimGrid = dim3(ceil(N/(float)TILE_WIDTH), ceil(N/(float)TILE_WIDTH), 1);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);

    double tiled_time = measure_performance([&](){ tiled_matmul<<<dimGrid, dimBlock>>>(N, a, b, output); });
    // TODO why are naive kernel differences so big compared to cublas
    // test_output(cublas_ref, N, 5e-1);

    return tiled_time;
}
