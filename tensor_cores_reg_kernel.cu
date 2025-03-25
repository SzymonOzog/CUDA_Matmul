#include "kernel_classes.cuh"

template<int OUT_TILES>
__global__ void tensor_core_matmul_reg(int n, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;


    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[OUT_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];

    for(int32_t i = 0; i<OUT_TILES; i++)
        for(int32_t j = 0; j<OUT_TILES; j++)
            nvcuda::wmma::fill_fragment(acc[i][j], 0);

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    for (int32_t tile = 0; tile < n; tile+=OUT_TILES*WMMA_MKN)
    {
        for (int out_col = 0; out_col < OUT_TILES; out_col++)
        {
            for (int out_row = 0; out_row < OUT_TILES; out_row++)
            {
                int32_t a_row = matrix_a_row + out_row*WMMA_MKN;
                int32_t a_col = tile + out_col*WMMA_MKN;
                if(a_row < n && a_col < n)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag[out_row], a + a_row*n + a_col, n);
                }
            }
            for (int out_row = 0; out_row < OUT_TILES; out_row++)
            {
                int32_t b_col = matrix_b_col + (out_row)*WMMA_MKN;
                int32_t b_row = tile + out_col*WMMA_MKN;
                if (b_row < n && b_col < n)
                {
                    nvcuda::wmma::load_matrix_sync(b_frag, b + b_row*n + b_col, n);
                    for (int k = 0; k < OUT_TILES; k++)
                    {
                        nvcuda::wmma::mma_sync(acc[k][out_row], a_frag[k], b_frag, acc[k][out_row]);
                    }
                }
            }
        }
    }

    for(int32_t i = 0; i<OUT_TILES; i++)
    {
        int32_t output_row = matrix_a_row + i*WMMA_MKN;
        for(int32_t j = 0; j<OUT_TILES; j++)
        {
            int32_t output_col = matrix_b_col + j*WMMA_MKN;
            if (output_row < n && output_col < n)
            {
                nvcuda::wmma::store_matrix_sync(c + output_row * n + output_col, acc[i][j], n, nvcuda::wmma::mem_row_major);
            }
        }
    }
}

template <int OUT_TILES_REG>
double check_configuration_reg(half* a, half*b, half* output, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);
    int num_warps_x = 4;
    int num_warps_y = 4;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(OUT_TILES_REG*WMMA_MKN*num_warps_x));
    dimGrid.y = std::ceil((float)N/(OUT_TILES_REG*WMMA_MKN*num_warps_y));

    return measure_performance([&](){ tensor_core_matmul_reg<OUT_TILES_REG><<<dimGrid, dimBlock>>>(N, a, b, output); });
}
    

double TensorCoresRegKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_reg<2>(a, b, output, N));
    test_output(cublas_ref, N);

    // matmul_time = std::min(matmul_time, check_configuration_reg<3>(a, b, output, N));
    // test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_reg<4>(a, b, output, N));
    test_output(cublas_ref, N);

    return matmul_time;
}
