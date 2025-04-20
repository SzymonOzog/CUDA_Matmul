#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"

__global__ void tensor_core_matmul(int n, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int lane_id = threadIdx.x%32;

    // nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag;
    // nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc;
    // nvcuda::wmma::fill_fragment(acc, 0);

    mma_tile<16, 16> a_tile;
    mma_tile<16, 16> b_tile;
    mma_tile<16, 16> acc;

    for (int32_t i = 0; i < n; i+= WMMA_MKN)
    {
        const int32_t matrix_a_row = warpM * WMMA_MKN;
        const int32_t matrix_b_col = warpN * WMMA_MKN;

        if(matrix_a_row<n && matrix_b_col<n && i<n)
        {
            load_tile_a(a_tile, a + matrix_a_row*n + i, n, lane_id);

            for (int j = 0; j<4; j++)
            {
                half2 tmp;
                tmp.x = b[(i + (j%2)*8 + (lane_id%4)*2)*n + matrix_b_col + (lane_id>>2) + (j/2)*8];
                tmp.y = b[(i + (j%2)*8 + (lane_id%4)*2 + 1)*n + matrix_b_col + (lane_id>>2) + (j/2)*8];
                b_tile.x[j] = tmp;
            }
            mma(a_tile, b_tile, acc);
        }
    }

    for (int j = 0; j<4; j++)
    {
        reinterpret_cast<half2*>(&c[(warpM*WMMA_MKN + (lane_id>>2) + (j%2)*8)*n + warpN*WMMA_MKN + (j/2)*8])[lane_id%4] = acc.x[j];
    }
    // nvcuda::wmma::store_matrix_sync(c + warpM*WMMA_MKN*n + warpN*WMMA_MKN, acc, n, nvcuda::wmma::mem_row_major);
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
