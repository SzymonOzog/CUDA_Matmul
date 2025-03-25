#include "kernel_classes.cuh"

__global__ void tensor_core_matmul(int n, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc;

    nvcuda::wmma::fill_fragment(acc, 0);

    for (int32_t i = 0; i < n; i+= WMMA_MKN)
    {
        const int32_t matrix_a_row = warpM * WMMA_MKN;
        const int32_t matrix_b_col = warpN * WMMA_MKN;

        if(matrix_a_row<n && matrix_b_col<n && i<n)
        {
            nvcuda::wmma::load_matrix_sync(a_frag, a + matrix_a_row * n + i, n);
            nvcuda::wmma::load_matrix_sync(b_frag, b + i * n + matrix_b_col, n);

            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    nvcuda::wmma::store_matrix_sync(c + warpM*WMMA_MKN*n + warpN*WMMA_MKN, acc, n, nvcuda::wmma::mem_row_major);
}

double TensorCoresKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    int num_warps_x = 4;
    int num_warps_y = 4;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = (N + (WMMA_MKN*num_warps_x -1)) / (WMMA_MKN*num_warps_x);
    dimGrid.y = (N + WMMA_MKN*num_warps_y -1) / (WMMA_MKN*num_warps_y);

    double matmul_time = measure_performance([&](){ tensor_core_matmul<<<dimGrid, dimBlock>>>(N, a, b, output); });
    test_output(cublas_ref, N);

    return matmul_time;
}
