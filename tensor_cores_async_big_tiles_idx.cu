#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"
#include "utils.cuh"
#include <cstdint>

template<int BM, int BN, int BK, int OUT_TILES>
__global__ void tensor_core_matmul_async_swizzle_BT_idx(int n_elem, const half* a, const half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;
    const int32_t lane_id = threadIdx.x%32;
    constexpr const unsigned int S_BITS_A = 3;
    constexpr const unsigned int S_BITS_B = 4;

    extern __shared__ char smem[];

    half (*a_smem) = reinterpret_cast<half*>(smem);
    half (*b_smem) = reinterpret_cast<half*>(smem + BM*BK*WMMA_MKN*WMMA_MKN*sizeof(half));

    mma_tile<16, 16> a_tile[OUT_TILES];
    mma_tile<16, 16> b_tile;
    mma_tile<16, 16> acc[OUT_TILES][OUT_TILES];

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    unsigned int idx = (laneM*OUT_TILES)*BK*WMMA_MKN*WMMA_MKN + (lane_id%16) * BK*WMMA_MKN + (lane_id/16)*8;
    const uint32_t a_addr_c = __cvta_generic_to_shared(a_smem + (idx^((idx&(S_MASK<<S_BITS_A))>>S_BITS_A)));

    unsigned int idx_b = (laneN*OUT_TILES)*WMMA_MKN + (lane_id%16)*BN*WMMA_MKN + (lane_id/16)*8;
    idx_b = idx_b^((idx_b&(S_MASK<<S_BITS_B))>>S_BITS_B);
    const uint32_t b_addr_c = __cvta_generic_to_shared(b_smem + idx_b);

    for (int32_t tile = 0; tile < n_elem; tile+=BK*WMMA_MKN)
    {
        const half* a_curr = a + blockIdx.x*BM*WMMA_MKN*n_elem + tile;
        const half* b_curr = b + (tile)*n_elem + blockIdx.y*BN*WMMA_MKN;
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
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
        uint32_t a_addr = a_addr_c;

        uint32_t b_addr = b_addr_c;
        // k = 0
        if (tile + 0*WMMA_MKN < n_elem)
        {
            load_tile_a_direct(a_tile[0], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[1], a_addr);
            a_addr ^= 6144;
            load_tile_a_direct(a_tile[2], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[3], a_addr);
            a_addr ^= 6176;

            // n = 0
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][0]);
            mma(a_tile[1], b_tile, acc[1][0]);
            mma(a_tile[2], b_tile, acc[2][0]);
            mma(a_tile[3], b_tile, acc[3][0]);
            b_addr ^= 32;

            // n = 1
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][1]);
            mma(a_tile[1], b_tile, acc[1][1]);
            mma(a_tile[2], b_tile, acc[2][1]);
            mma(a_tile[3], b_tile, acc[3][1]);
            b_addr ^= 96;

            // n = 2
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][2]);
            mma(a_tile[1], b_tile, acc[1][2]);
            mma(a_tile[2], b_tile, acc[2][2]);
            mma(a_tile[3], b_tile, acc[3][2]);
            b_addr ^= 32;

            // n = 3
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][3]);
            mma(a_tile[1], b_tile, acc[1][3]);
            mma(a_tile[2], b_tile, acc[2][3]);
            mma(a_tile[3], b_tile, acc[3][3]);

            b_addr ^= 4192;
        }

        // k = 1
        if (tile + 1*WMMA_MKN < n_elem)
        {
            load_tile_a_direct(a_tile[0], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[1], a_addr);
            a_addr ^= 6144;
            load_tile_a_direct(a_tile[2], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[3], a_addr);
            a_addr ^= 6240;

            // n = 0
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][0]);
            mma(a_tile[1], b_tile, acc[1][0]);
            mma(a_tile[2], b_tile, acc[2][0]);
            mma(a_tile[3], b_tile, acc[3][0]);
            b_addr ^= 32;

            // n = 1
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][1]);
            mma(a_tile[1], b_tile, acc[1][1]);
            mma(a_tile[2], b_tile, acc[2][1]);
            mma(a_tile[3], b_tile, acc[3][1]);
            b_addr ^= 96;

            // n = 2
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][2]);
            mma(a_tile[1], b_tile, acc[1][2]);
            mma(a_tile[2], b_tile, acc[2][2]);
            mma(a_tile[3], b_tile, acc[3][2]);
            b_addr ^= 32;

            // n = 3
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][3]);
            mma(a_tile[1], b_tile, acc[1][3]);
            mma(a_tile[2], b_tile, acc[2][3]);
            mma(a_tile[3], b_tile, acc[3][3]);

            b_addr ^= 12384;
        }

        // k = 2
        if (tile + 2*WMMA_MKN < n_elem)
        {
            load_tile_a_direct(a_tile[0], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[1], a_addr);
            a_addr ^= 6144;
            load_tile_a_direct(a_tile[2], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[3], a_addr);
            a_addr ^= 6176;

            // n = 0
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][0]);
            mma(a_tile[1], b_tile, acc[1][0]);
            mma(a_tile[2], b_tile, acc[2][0]);
            mma(a_tile[3], b_tile, acc[3][0]);
            b_addr ^= 32;

            // n = 1
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][1]);
            mma(a_tile[1], b_tile, acc[1][1]);
            mma(a_tile[2], b_tile, acc[2][1]);
            mma(a_tile[3], b_tile, acc[3][1]);
            b_addr ^= 96;

            // n = 2
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][2]);
            mma(a_tile[1], b_tile, acc[1][2]);
            mma(a_tile[2], b_tile, acc[2][2]);
            mma(a_tile[3], b_tile, acc[3][2]);
            b_addr ^= 32;

            // n = 3
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][3]);
            mma(a_tile[1], b_tile, acc[1][3]);
            mma(a_tile[2], b_tile, acc[2][3]);
            mma(a_tile[3], b_tile, acc[3][3]);

            b_addr ^= 4192;
        }

        // k = 3
        if (tile + 3*WMMA_MKN < n_elem)
        {
            load_tile_a_direct(a_tile[0], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[1], a_addr);
            a_addr ^= 6144;
            load_tile_a_direct(a_tile[2], a_addr);
            a_addr ^= 2048;
            load_tile_a_direct(a_tile[3], a_addr);

            // n = 0
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][0]);
            mma(a_tile[1], b_tile, acc[1][0]);
            mma(a_tile[2], b_tile, acc[2][0]);
            mma(a_tile[3], b_tile, acc[3][0]);
            b_addr ^= 32;

            // n = 1
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][1]);
            mma(a_tile[1], b_tile, acc[1][1]);
            mma(a_tile[2], b_tile, acc[2][1]);
            mma(a_tile[3], b_tile, acc[3][1]);
            b_addr ^= 96;

            // n = 2
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][2]);
            mma(a_tile[1], b_tile, acc[1][2]);
            mma(a_tile[2], b_tile, acc[2][2]);
            mma(a_tile[3], b_tile, acc[3][2]);
            b_addr ^= 32;

            // n = 3
            load_tile_b_direct(b_tile, b_addr);
            mma(a_tile[0], b_tile, acc[0][3]);
            mma(a_tile[1], b_tile, acc[1][3]);
            mma(a_tile[2], b_tile, acc[2][3]);
            mma(a_tile[3], b_tile, acc[3][3]);

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
double check_configuration_async_BT_idx(half* a, half*b, half* output, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = BM/OUT_TILES;
    int num_warps_y = BN/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(BM*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(BN*WMMA_MKN));
    unsigned int smem_size = BM*BK*WMMA_MKN*WMMA_MKN*sizeof(half)
        + BN*BK*WMMA_MKN*WMMA_MKN*sizeof(half);
    cudaFuncSetAttribute(tensor_core_matmul_async_swizzle_BT_idx<BM, BN, BK, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_async_swizzle_BT_idx<BM, BN, BK, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
}
double TensorCoresAsyncBTIdxKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_async_BT_idx<8, 8, 4, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    return matmul_time;
}
