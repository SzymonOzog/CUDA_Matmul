#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"

template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_reg_smem(int n_elem, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;
    // const int32_t lane_id = threadIdx.x%32;

    extern __shared__ char smem[];

    half (*a_smem)[WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(smem);
    half (*b_smem)[WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(
                smem + SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[OUT_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];
    // mma_tile<16, 16> a_tile[OUT_TILES];
    // mma_tile<16, 16> b_tile;
    // mma_tile<16, 16> acc[OUT_TILES][OUT_TILES];

    for(int32_t i = 0; i<OUT_TILES; i++)
        for(int32_t j = 0; j<OUT_TILES; j++)
            nvcuda::wmma::fill_fragment(acc[i][j], 0);

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    for (int32_t tile = 0; tile < n_elem; tile+=WMMA_MKN)
    {
        half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem + tile;
        half* b_curr = b + (tile)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                i < SM_TILES*WMMA_MKN*WMMA_MKN;
                i+=blockDim.x*blockDim.y*8)
        {
            half* a_smem_curr = &a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
            half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
            reinterpret_cast<float4*>(a_smem_curr)[0]
                = reinterpret_cast<float4*>(a_gmem_curr)[0];

            half* b_smem_curr = &b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
            half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
            reinterpret_cast<float4*>(b_smem_curr)[0]
                = reinterpret_cast<float4*>(b_gmem_curr)[0];
        }
        __syncthreads();
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(a_frag[n], a_smem[laneM*OUT_TILES + n], WMMA_MKN);
            // load_tile_a(a_tile[n], a_smem[laneM*OUT_TILES + n], WMMA_MKN, lane_id);
        }
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(b_frag, b_smem[laneN*OUT_TILES + n], WMMA_MKN);
            // load_tile_b(b_tile, b_smem[laneN*OUT_TILES + n], WMMA_MKN, lane_id);
            for (int m = 0; m < OUT_TILES; m++)
            {
                nvcuda::wmma::mma_sync(acc[m][n], a_frag[m], b_frag, acc[m][n]);
                // mma(a_tile[m], b_tile, acc[m][n]);
            }
        }
        __syncthreads();
    }

    for(int32_t i = 0; i<OUT_TILES; i++)
    {
        int32_t output_row = matrix_a_row + i*WMMA_MKN;
        for(int32_t j = 0; j<OUT_TILES; j++)
        {
            int32_t output_col = matrix_b_col + j*WMMA_MKN;
            if (output_row < n_elem && output_col < n_elem)
            {
                // for (int k = 0; k<4; k++)
                // {
                //     reinterpret_cast<half2*>(&c[(output_row + (lane_id>>2) + (k%2)*8)*n_elem + output_col + (k/2)*8])[lane_id%4]
                //         = acc[i][j].x[k];
                // }
               nvcuda::wmma::store_matrix_sync(c + output_row * n_elem + output_col, acc[i][j], n_elem, nvcuda::wmma::mem_row_major);
            }
        }
    }
}

template<int SMEM_TILES, int OUT_TILES>
double check_configuration_smem(half* a, half*b, half* output, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = SMEM_TILES/OUT_TILES;
    int num_warps_y = SMEM_TILES/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    unsigned int smem_size = 2*SMEM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half);
    cudaFuncSetAttribute(tensor_core_matmul_reg_smem<SMEM_TILES, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_reg_smem<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
}

double TensorCoresSmemKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_smem<8, 2>(a, b, output, N));
    // debug_print(a, N, true); 
    // debug_print(b, N, true); 
    //
    // debug_print(cublas_ref, N, false); 
    // debug_print(output, N, true); 
    test_output(cublas_ref, N);

    // matmul_time = std::min(matmul_time, check_configuration_smem<9, 3>(a, b, output, N));
    // test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_smem<8, 4>(a, b, output, N));

    test_output(cublas_ref, N);
    return matmul_time;
}
