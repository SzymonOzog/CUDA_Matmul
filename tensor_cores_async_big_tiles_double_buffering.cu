#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"
#include "utils.cuh"

template<int BM, int BN, int BK, int OUT_TILES>
__global__ void tensor_core_matmul_async_swizzle_BT_DB(int n_elem, const half* a, const half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;
    const int32_t lane_id = threadIdx.x%32;
    constexpr const unsigned int S_BITS_A = int_log2(BK*WMMA_MKN/8);
    constexpr const unsigned int S_BITS_B = int_log2(BN*WMMA_MKN/8);
    constexpr const unsigned int A_ST_STRIDE = BM*BK*WMMA_MKN*WMMA_MKN;
    constexpr const unsigned int B_ST_STRIDE = BN*BK*WMMA_MKN*WMMA_MKN;

    extern __shared__ char smem[];

    half (*a_smem) = reinterpret_cast<half*>(smem);
    half (*b_smem) = reinterpret_cast<half*>(smem + 2*A_ST_STRIDE*sizeof(half));

    mma_tile<16, 16> a_tile[2][OUT_TILES];
    mma_tile<16, 16> b_tile[2];
    mma_tile<16, 16> acc[OUT_TILES][OUT_TILES];
    for(int i = 0; i < OUT_TILES; i++)
    {
        for (int j = 0; j < OUT_TILES; j++)
        { 
            for(int k = 0; k<acc[i][j].len; k++)
            {
                acc[i][j].x[k].x = 0.f;
                acc[i][j].x[k].y = 0.f;
            }
        }
    }

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    unsigned int stage = 0;

    const half* a_curr = a + blockIdx.x*BM*WMMA_MKN*n_elem;
    const half* b_curr = b +  blockIdx.y*BN*WMMA_MKN;
    for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
            i < BM*BK*WMMA_MKN*WMMA_MKN;
            i+=blockDim.x*blockDim.y*8)
    {
        uint32_t a_smem_curr = __cvta_generic_to_shared(&a_smem[i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)]);
        const half* a_gmem_curr = &a_curr[(i/(WMMA_MKN*BK))*n_elem + i%(WMMA_MKN*BK)];
        CP_ASYNC_CG(a_smem_curr, reinterpret_cast<const float4*>(a_gmem_curr), 16);
    }

    for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
            i < BN*BK*WMMA_MKN*WMMA_MKN;
            i+=blockDim.x*blockDim.y*8)
    {
        uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
        const half* b_gmem_curr = &b_curr[(i/(BN*WMMA_MKN))*n_elem + i%(BN*WMMA_MKN)];
        CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
    }
    CP_ASYNC_COMMIT_GROUP();
    for (int32_t tile = 0; tile < n_elem; tile+=BK*WMMA_MKN)
    {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
        if(tile + BK*WMMA_MKN < n_elem)
        {
            unsigned int ld_stage = (stage + 1)%2;
            const half* a_curr = a + blockIdx.x*BM*WMMA_MKN*n_elem + tile + BK*WMMA_MKN;
            const half* b_curr = b + (tile + BK*WMMA_MKN)*n_elem + blockIdx.y*BN*WMMA_MKN;
            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < BM*BK*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                uint32_t a_smem_curr = __cvta_generic_to_shared(&a_smem[ld_stage * A_ST_STRIDE + i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)]);
                const half* a_gmem_curr = &a_curr[(i/(WMMA_MKN*BK))*n_elem + i%(WMMA_MKN*BK)];
                CP_ASYNC_CG(a_smem_curr, reinterpret_cast<const float4*>(a_gmem_curr), 16);
            }

            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < BN*BK*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[ld_stage * B_ST_STRIDE + i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
                const half* b_gmem_curr = &b_curr[(i/(BN*WMMA_MKN))*n_elem + i%(BN*WMMA_MKN)];
                CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
        for (int k = 0; k<BK && tile + k*WMMA_MKN < n_elem; k+=2)
        {
            for (int n = 0; n < OUT_TILES; n++)
            {
                load_tile_a_shared_swizzle<S_BITS_A>(a_tile[0][n], a_smem + stage*A_ST_STRIDE, (laneM*OUT_TILES + n)*BK*WMMA_MKN*WMMA_MKN + k*WMMA_MKN, BK*WMMA_MKN, lane_id);
                load_tile_a_shared_swizzle<S_BITS_A>(a_tile[1][n], a_smem + stage*A_ST_STRIDE, (laneM*OUT_TILES + n)*BK*WMMA_MKN*WMMA_MKN + (k+1)*WMMA_MKN, BK*WMMA_MKN, lane_id);
            }
            for (int n = 0; n < OUT_TILES; n++)
            {
                load_tile_b_shared_swizzle<S_BITS_B>(b_tile[0], b_smem + stage*B_ST_STRIDE, (k*BN*WMMA_MKN + laneN*OUT_TILES + n)*WMMA_MKN, BN*WMMA_MKN, lane_id);
                load_tile_b_shared_swizzle<S_BITS_B>(b_tile[1], b_smem + stage*B_ST_STRIDE, ((k+1)*BN*WMMA_MKN + laneN*OUT_TILES + n)*WMMA_MKN, BN*WMMA_MKN, lane_id);
                for (int m = 0; m < OUT_TILES; m++)
                {
                    mma(a_tile[0][m], b_tile[0], acc[m][n]);
                }
                for (int m = 0; m < OUT_TILES; m++)
                {
                    mma(a_tile[1][m], b_tile[1], acc[m][n]);
                }
            }
        } 
        stage = (stage+1)%2;
    }

    for(int32_t i = 0; i<OUT_TILES; i++)
    {
        int32_t output_row = matrix_a_row + i*WMMA_MKN;
        for(int32_t j = 0; j<OUT_TILES; j++)
        {
            int32_t output_col = matrix_b_col + j*WMMA_MKN;
            if (output_row < n_elem && output_col < n_elem)
            {
                for (int k = 0; k<4; k++)
                {
                    reinterpret_cast<half2*>(&c[(output_row + (lane_id>>2) + (k%2)*8)*n_elem + output_col + (k/2)*8])[lane_id%4]
                        = acc[i][j].x[k];
                }
            }
        }
    }
}

template<int BM, int BN, int BK, int OUT_TILES>
double check_configuration_async_BT_DB(half* a, half*b, half* output, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = BM/OUT_TILES;
    int num_warps_y = BN/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(BM*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(BN*WMMA_MKN));
    unsigned int smem_size = 2*BM*BK*WMMA_MKN*WMMA_MKN*sizeof(half)
        + 2*BN*BK*WMMA_MKN*WMMA_MKN*sizeof(half);
    cudaFuncSetAttribute(tensor_core_matmul_async_swizzle_BT_DB<BM, BN, BK, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_async_swizzle_BT_DB<BM, BN, BK, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
}
double TensorCoresAsyncBT_DBKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB<8, 8, 2, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB<16, 8, 2, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB<8, 16, 2, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB<8, 8, 4, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    return matmul_time;
}
