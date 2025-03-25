#include "kernel_classes.cuh"

template<int SM_TILES, int N_WARPS>
__device__ inline void prefetch(half* a_gmem, half* b_gmem, float4* dest_a, float4* dest_b, int n_elem)
{
    int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
    half* a_gmem_curr = &a_gmem[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
    half* b_gmem_curr = &b_gmem[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
    if (i < SM_TILES * WMMA_MKN*WMMA_MKN)
    {
        dest_a[0] = reinterpret_cast<float4*>(a_gmem_curr)[0];
        dest_b[0] = reinterpret_cast<float4*>(b_gmem_curr)[0];
    }

    if constexpr(N_WARPS == 4)
    {
        i+=N_WARPS*32*8;
        a_gmem_curr = &a_gmem[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
        dest_a[1] = reinterpret_cast<float4*>(a_gmem_curr)[0];
        b_gmem_curr = &b_gmem[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
        dest_b[1] = reinterpret_cast<float4*>(b_gmem_curr)[0];
    }
}

template<int SM_TILES, int OUT_TILES, int N_WARPS>
__global__ void tensor_core_matmul_reg_smem_prefetch(int n_elem, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;

    extern __shared__ char smem[];

    half (*a_smem)[WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(smem);
    half (*b_smem)[WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(
                smem + SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[OUT_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];

    for(int32_t i = 0; i<OUT_TILES; i++)
        for(int32_t j = 0; j<OUT_TILES; j++)
            nvcuda::wmma::fill_fragment(acc[i][j], 0);

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    float4 prefetch_buffer[2*2];

    prefetch<SM_TILES, N_WARPS>(a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem, 
            b + blockIdx.y*SM_TILES*WMMA_MKN,
            &prefetch_buffer[0], &prefetch_buffer[2], n_elem);

    for (int32_t tile = 0; tile < n_elem; tile+=WMMA_MKN)
    {
        int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
        half* a_smem_curr = &a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
        half* b_smem_curr = &b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
        if (i < SM_TILES * WMMA_MKN*WMMA_MKN)
        {
            reinterpret_cast<float4*>(a_smem_curr)[0] = prefetch_buffer[0];
            reinterpret_cast<float4*>(b_smem_curr)[0] = prefetch_buffer[2];
        }

        if constexpr(N_WARPS == 4)
        {
            i+=N_WARPS*32*8;
            a_smem_curr = &a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
            reinterpret_cast<float4*>(a_smem_curr)[0] = prefetch_buffer[1];
            b_smem_curr = &b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
            reinterpret_cast<float4*>(b_smem_curr)[0] = prefetch_buffer[3];
        }

        __syncthreads();

        if(tile + WMMA_MKN < n_elem)
        {
            half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem + tile+WMMA_MKN;
            half* b_curr = b + (tile+WMMA_MKN)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
            prefetch<SM_TILES, N_WARPS>(a_curr, b_curr, &prefetch_buffer[0], &prefetch_buffer[2], n_elem);
        }

        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(a_frag[n], a_smem[laneM*OUT_TILES + n], WMMA_MKN);
        }
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(b_frag, b_smem[laneN*OUT_TILES + n], WMMA_MKN);
            for (int m = 0; m < OUT_TILES; m++)
            {
                nvcuda::wmma::mma_sync(acc[m][n], a_frag[m], b_frag, acc[m][n]);
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
                nvcuda::wmma::store_matrix_sync(c + output_row * n_elem + output_col, acc[i][j], n_elem, nvcuda::wmma::mem_row_major);
            }
        }
    }
}


template<int SMEM_TILES, int OUT_TILES>
double check_configuration_prefetch(half* a, half*b, half* output, int N)
{
    constexpr int N_WARPS = (SMEM_TILES/OUT_TILES) * (SMEM_TILES/OUT_TILES);
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = SMEM_TILES/OUT_TILES;
    int num_warps_y = SMEM_TILES/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    unsigned int smem_size = 2*SMEM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half);
    cudaFuncSetAttribute(tensor_core_matmul_reg_smem_prefetch<SMEM_TILES, OUT_TILES, N_WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_reg_smem_prefetch<SMEM_TILES, OUT_TILES, N_WARPS><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
}

double TensorCoresPrefetchKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_prefetch<8, 2>(a, b, output, N));
    test_output(cublas_ref, N);

    // matmul_time = std::min(matmul_time, check_configuration_prefetch<9, 3>(a, b, output, N));
    // test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_prefetch<8, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    return matmul_time;
}
