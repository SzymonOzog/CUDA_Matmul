#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"
#include "utils.cuh"
template<int BM, int BN, int BK, int OUT_TILES>
__global__ __maxnreg__(128) void tensor_core_matmul_async_swizzle_BT_DB_FB_Reg(int n_elem, const half* a, const half* b, half* c)
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
            *reinterpret_cast<float4*>(&acc[i][j]) = float4();
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

    unsigned int idx = (laneM*OUT_TILES)*BK*WMMA_MKN*WMMA_MKN + (lane_id%16) * BK*WMMA_MKN + (lane_id/16)*8;
    idx = (idx^((idx&(S_MASK<<S_BITS_A))>>S_BITS_A));
    const uint32_t a_addr_c = __cvta_generic_to_shared(a_smem + idx);

    idx = laneN*OUT_TILES*WMMA_MKN + (lane_id%16)*BN*WMMA_MKN + (lane_id/16)*8;
    idx = idx^((idx&(S_MASK<<S_BITS_B))>>S_BITS_B);
    const uint32_t b_addr_c = __cvta_generic_to_shared(b_smem + idx);

    for (int32_t tile = 0; tile < n_elem-BK*WMMA_MKN; tile+=BK*WMMA_MKN)
    {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
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
        CP_ASYNC_COMMIT_GROUP();
        uint32_t a_addr = a_addr_c + stage*A_ST_STRIDE*sizeof(half);
        uint32_t b_addr = b_addr_c + stage*B_ST_STRIDE*sizeof(half);

        load_tile_a_direct(a_tile[0][0], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][0], a_addr);

        a_addr ^= 1056;
        load_tile_a_direct(a_tile[0][1], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][1], a_addr);

        a_addr ^= 3104;
        load_tile_a_direct(a_tile[0][2], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][2], a_addr);

        a_addr ^= 1056;
        load_tile_a_direct(a_tile[0][3], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][3], a_addr);

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        b_addr ^= 4128;
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][0]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][0]);
        }

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        b_addr ^= 4192;
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][1]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][1]);
        }

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        b_addr ^= 4128;
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][2]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][2]);
        }

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][3]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][3]);
        }
        stage = (stage+1)%2;
    }
    unsigned int ld_stage = (stage + 1)%2;
    const half* b_curr2 = b + (n_elem-BK*WMMA_MKN)*n_elem + blockIdx.y*BN*WMMA_MKN + gridDim.y * BN*WMMA_MKN;
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
            i < BN*BK*WMMA_MKN*WMMA_MKN;
            i+=blockDim.x*blockDim.y*8)
    {
        uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[ld_stage * B_ST_STRIDE + i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
        const half* b_gmem_curr = &b_curr2[(i/(BN*WMMA_MKN))*n_elem + i%(BN*WMMA_MKN)];
        CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
    }
    CP_ASYNC_COMMIT_GROUP();
    uint32_t a_addr = a_addr_c + stage*A_ST_STRIDE*sizeof(half);
    uint32_t b_addr = b_addr_c + stage*B_ST_STRIDE*sizeof(half);

    load_tile_a_direct(a_tile[0][0], a_addr);
    a_addr ^= 32;
    load_tile_a_direct(a_tile[1][0], a_addr);

    a_addr ^= 1056;
    load_tile_a_direct(a_tile[0][1], a_addr);
    a_addr ^= 32;
    load_tile_a_direct(a_tile[1][1], a_addr);

    a_addr ^= 3104;
    load_tile_a_direct(a_tile[0][2], a_addr);
    a_addr ^= 32;
    load_tile_a_direct(a_tile[1][2], a_addr);

    a_addr ^= 1056;
    load_tile_a_direct(a_tile[0][3], a_addr);
    a_addr ^= 32;
    load_tile_a_direct(a_tile[1][3], a_addr);

    load_tile_b_direct(b_tile[0], b_addr);
    b_addr ^= 4096;
    load_tile_b_direct(b_tile[1], b_addr);
    b_addr ^= 4128;
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[0][m], b_tile[0], acc[m][0]);
    }
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[1][m], b_tile[1], acc[m][0]);
    }

    load_tile_b_direct(b_tile[0], b_addr);
    b_addr ^= 4096;
    load_tile_b_direct(b_tile[1], b_addr);
    b_addr ^= 4192;
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[0][m], b_tile[0], acc[m][1]);
    }
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[1][m], b_tile[1], acc[m][1]);
    }

    load_tile_b_direct(b_tile[0], b_addr);
    b_addr ^= 4096;
    load_tile_b_direct(b_tile[1], b_addr);
    b_addr ^= 4128;
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[0][m], b_tile[0], acc[m][2]);
    }
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[1][m], b_tile[1], acc[m][2]);
    }

    load_tile_b_direct(b_tile[0], b_addr);
    b_addr ^= 4096;
    load_tile_b_direct(b_tile[1], b_addr);
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[0][m], b_tile[0], acc[m][3]);
    }
    for (int m = 0; m < OUT_TILES; m++)
    {
        mma(a_tile[1][m], b_tile[1], acc[m][3]);
    }
    stage = (stage+1)%2;

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
                int4 st;
                int row = lane_id%16;
                int col = (lane_id/16)*8;
                st.x = get_at(acc[i][j], (lane_id%8)*4 +  0, lane_id/8);
                st.y = get_at(acc[i][j], (lane_id%8)*4 +  1, lane_id/8);
                st.z = get_at(acc[i][j], (lane_id%8)*4 +  2, lane_id/8);
                st.w = get_at(acc[i][j], (lane_id%8)*4 +  3, lane_id/8);
                reinterpret_cast<int4*>(&c[(output_row + row)*n_elem + output_col + col])[0] = st;
            }
        }
    }

    for(int i = 0; i < OUT_TILES; i++)
    {
        for (int j = 0; j < OUT_TILES; j++)
        { 
            *reinterpret_cast<float4*>(&acc[i][j]) = float4();
        }
    }

    stage = (stage+1)%2;

    for (int32_t tile = n_elem - BK*WMMA_MKN; tile >= 0; tile-=BK*WMMA_MKN)
    {
        unsigned int ld_stage = (stage + 1)%2;
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
        if(tile - BK*WMMA_MKN == 0)
        {
            const half* b_curr = b + (tile - BK*WMMA_MKN)*n_elem + blockIdx.y*BN*WMMA_MKN + gridDim.y * BN*WMMA_MKN;
            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < BN*BK*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[stage * B_ST_STRIDE + i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
                const half* b_gmem_curr = &b_curr[(i/(BN*WMMA_MKN))*n_elem + i%(BN*WMMA_MKN)];
                CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
            }
        }
        else if(tile - BK*WMMA_MKN >= 0)
        {
            const half* a_curr = a + blockIdx.x*BM*WMMA_MKN*n_elem + tile - 2*BK*WMMA_MKN;
            const half* b_curr = b + (tile - BK*WMMA_MKN)*n_elem + blockIdx.y*BN*WMMA_MKN + gridDim.y * BN*WMMA_MKN;
            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < BM*BK*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                uint32_t a_smem_curr = __cvta_generic_to_shared(&a_smem[stage * A_ST_STRIDE + i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)]);
                const half* a_gmem_curr = &a_curr[(i/(WMMA_MKN*BK))*n_elem + i%(WMMA_MKN*BK)];
                CP_ASYNC_CG(a_smem_curr, reinterpret_cast<const float4*>(a_gmem_curr), 16);
            }

            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < BN*BK*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[stage * B_ST_STRIDE + i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
                const half* b_gmem_curr = &b_curr[(i/(BN*WMMA_MKN))*n_elem + i%(BN*WMMA_MKN)];
                CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
            }
        }
        CP_ASYNC_COMMIT_GROUP();
        uint32_t b_addr = b_addr_c + ld_stage*B_ST_STRIDE*sizeof(half);

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        b_addr ^= 4128;
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][0]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][0]);
        }

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        b_addr ^= 4192;
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][1]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][1]);
        }

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        b_addr ^= 4128;
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][2]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][2]);
        }

        load_tile_b_direct(b_tile[0], b_addr);
        b_addr ^= 4096;
        load_tile_b_direct(b_tile[1], b_addr);
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[0][m], b_tile[0], acc[m][3]);
        }
        for (int m = 0; m < OUT_TILES; m++)
        {
            mma(a_tile[1][m], b_tile[1], acc[m][3]);
        }
        uint32_t a_addr = a_addr_c + ld_stage*A_ST_STRIDE*sizeof(half);
        load_tile_a_direct(a_tile[0][0], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][0], a_addr);

        a_addr ^= 1056;
        load_tile_a_direct(a_tile[0][1], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][1], a_addr);

        a_addr ^= 3104;
        load_tile_a_direct(a_tile[0][2], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][2], a_addr);

        a_addr ^= 1056;
        load_tile_a_direct(a_tile[0][3], a_addr);
        a_addr ^= 32;
        load_tile_a_direct(a_tile[1][3], a_addr);

        stage = ld_stage;
    }


    const int32_t matrix_a_row2 = matrix_a_row;
    const int32_t matrix_b_col2 = matrix_b_col + gridDim.y * BN*WMMA_MKN;
    for(int32_t i = 0; i<OUT_TILES; i++)
    {
        int32_t output_row = matrix_a_row2 + i*WMMA_MKN;
        for(int32_t j = 0; j<OUT_TILES; j++)
        {
            int32_t output_col = matrix_b_col2 + j*WMMA_MKN;
            if (output_row < n_elem && output_col < n_elem)
            {
                // for (int k = 0; k<4; k++)
                // {
                //     reinterpret_cast<half2*>(&c[(output_row + (lane_id>>2) + (k%2)*8)*n_elem + output_col + (k/2)*8])[lane_id%4]
                //         = acc[i][j].x[k];
                // }
                int4 st;
                int row = lane_id%16;
                int col = (lane_id/16)*8;
                st.x = get_at(acc[i][j], (lane_id%8)*4 +  0, lane_id/8);
                st.y = get_at(acc[i][j], (lane_id%8)*4 +  1, lane_id/8);
                st.z = get_at(acc[i][j], (lane_id%8)*4 +  2, lane_id/8);
                st.w = get_at(acc[i][j], (lane_id%8)*4 +  3, lane_id/8);
                reinterpret_cast<int4*>(&c[(output_row + row)*n_elem + output_col + col])[0] = st;
            }
        }
    }
}

template<int BM, int BN, int BK, int OUT_TILES>
double check_configuration_async_BT_DB_FB_Reg(half* a, half*b, half* output, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = BM/OUT_TILES;
    int num_warps_y = BN/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(BM*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(BN*WMMA_MKN*2));
    unsigned int smem_size = 2*BM*BK*WMMA_MKN*WMMA_MKN*sizeof(half)
        + 2*BN*BK*WMMA_MKN*WMMA_MKN*sizeof(half);
    cudaFuncSetAttribute(tensor_core_matmul_async_swizzle_BT_DB_FB_Reg<BM, BN, BK, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_async_swizzle_BT_DB_FB_Reg<BM, BN, BK, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
}
double TensorCoresAsyncBT_DB_FB_RegKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB_FB_Reg<8, 8, 2, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB_FB_Reg<16, 8, 2, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async_BT_DB_FB_Reg<32, 8, 2, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    //TODO Incompatable with fast idx
    // matmul_time = std::min(matmul_time, check_configuration_async_BT_DB_FB_Reg<8, 16, 2, 4>(a, b, output, N));
    // test_output(cublas_ref, N);

    return matmul_time;
}
