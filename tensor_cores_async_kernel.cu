#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"
#include <cuda_pipeline.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#pragma nv_diag_suppress static_var_with_dynamic_init

template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_reg_smem_async(int n_elem, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;

    extern __shared__ char smem[];

    half (*a_smem)[SM_TILES][WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[SM_TILES][WMMA_MKN*WMMA_MKN]>(smem);
    half (*b_smem)[SM_TILES][WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[SM_TILES][WMMA_MKN*WMMA_MKN]>(
                smem + 2*SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[OUT_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];

    for(int32_t i = 0; i<OUT_TILES; i++)
        for(int32_t j = 0; j<OUT_TILES; j++)
            nvcuda::wmma::fill_fragment(acc[i][j], 0);

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem;
    half* b_curr = b + blockIdx.y*SM_TILES*WMMA_MKN;
    for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
            i < SM_TILES*WMMA_MKN*WMMA_MKN;
            i+=blockDim.x*blockDim.y*8)
    {
        half* a_smem_curr = &a_smem[0][i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
        half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
        __pipeline_memcpy_async(a_smem_curr, a_gmem_curr, 16);

        half* b_smem_curr = &b_smem[0][(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
        half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
        __pipeline_memcpy_async(b_smem_curr, b_gmem_curr, 16);
    }
    __pipeline_commit();

    for (int32_t tile = 0; tile < n_elem; tile+=WMMA_MKN)
    {
        int stage = (tile/WMMA_MKN)%2;
        if (tile+WMMA_MKN<n_elem)
        {
            half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem + tile+WMMA_MKN;
            half* b_curr = b + (tile+WMMA_MKN)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < SM_TILES*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                int load_stage = (stage+1)%2;
                half* a_smem_curr = &a_smem[load_stage][i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
                half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
                __pipeline_memcpy_async(a_smem_curr, a_gmem_curr, 16);

                half* b_smem_curr = &b_smem[load_stage][(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
                half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
                __pipeline_memcpy_async(b_smem_curr, b_gmem_curr, 16);
            }
            __pipeline_commit();
        }
        __pipeline_wait_prior(0);
        //TODO Why do we need syncthreads here?
        __syncthreads();
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(a_frag[n], a_smem[stage][laneM*OUT_TILES + n], WMMA_MKN);
        }
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(b_frag, b_smem[stage][laneN*OUT_TILES + n], WMMA_MKN);
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

template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_reg_smem_async_pipeline(int n_elem, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;

    auto block = cooperative_groups::this_thread_block();
    auto grid = cooperative_groups::this_grid();

    extern __align__(128) __shared__ char smem[];

    half (*a_smem)[SM_TILES][WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[SM_TILES][WMMA_MKN*WMMA_MKN]>(smem);
    half (*b_smem)[SM_TILES][WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[SM_TILES][WMMA_MKN*WMMA_MKN]>(
                smem + 2*SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[OUT_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, 2> shared_state;

    auto pipeline = cuda::make_pipeline(block, &shared_state);


    for(int32_t i = 0; i<OUT_TILES; i++)
        for(int32_t j = 0; j<OUT_TILES; j++)
            nvcuda::wmma::fill_fragment(acc[i][j], 0);

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem;
    half* b_curr = b + blockIdx.y*SM_TILES*WMMA_MKN;
    pipeline.producer_acquire();
    for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
            i < SM_TILES*WMMA_MKN*WMMA_MKN;
            i+=blockDim.x*blockDim.y*8)
    {
        half* a_smem_curr = &a_smem[0][i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
        half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
        cuda::memcpy_async((float4*)a_smem_curr, (float4*)a_gmem_curr, sizeof(float4), pipeline);

        half* b_smem_curr = &b_smem[0][(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
        half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
        cuda::memcpy_async((float4*)b_smem_curr, (float4*)b_gmem_curr, sizeof(float4), pipeline);
    }
    pipeline.producer_commit();

    for (int32_t tile = 0; tile < n_elem; tile+=WMMA_MKN)
    {
        int stage = (tile/WMMA_MKN)%2;
        if (tile+WMMA_MKN<n_elem)
        {
            pipeline.producer_acquire();
            half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem + tile+WMMA_MKN;
            half* b_curr = b + (tile+WMMA_MKN)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
            for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                    i < SM_TILES*WMMA_MKN*WMMA_MKN;
                    i+=blockDim.x*blockDim.y*8)
            {
                int load_stage = (stage+1)%2;
                half* a_smem_curr = &a_smem[load_stage][i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)];
                half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
                cuda::memcpy_async((float4*)a_smem_curr, (float4*)a_gmem_curr, sizeof(float4), pipeline);

                half* b_smem_curr = &b_smem[load_stage][(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)];
                half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
                cuda::memcpy_async((float4*)b_smem_curr, (float4*)b_gmem_curr, sizeof(float4), pipeline);
            }
            pipeline.producer_commit();
        }
        pipeline.consumer_wait();
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(a_frag[n], a_smem[stage][laneM*OUT_TILES + n], WMMA_MKN);
        }
        for (int n = 0; n < OUT_TILES; n++)
        {
            nvcuda::wmma::load_matrix_sync(b_frag, b_smem[stage][laneN*OUT_TILES + n], WMMA_MKN);
            for (int m = 0; m < OUT_TILES; m++)
            {
                nvcuda::wmma::mma_sync(acc[m][n], a_frag[m], b_frag, acc[m][n]);
            }
        }
        pipeline.consumer_release();
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

template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_async_swizzle(int n_elem, const half* a, const half* b, half* c)
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
    half (*b_smem) = reinterpret_cast<half*>(smem + SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));

    mma_tile<16, 16> a_tile[OUT_TILES];
    mma_tile<16, 16> b_tile;
    mma_tile<16, 16> acc[OUT_TILES][OUT_TILES];

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    for (int32_t tile = 0; tile < n_elem; tile+=WMMA_MKN)
    {
        const half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem + tile;
        const half* b_curr = b + (tile)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
                i < SM_TILES*WMMA_MKN*WMMA_MKN;
                i+=blockDim.x*blockDim.y*8)
        {
            uint32_t a_smem_curr = __cvta_generic_to_shared(&a_smem[i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)]);
            const half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
            CP_ASYNC_CG(a_smem_curr, reinterpret_cast<const float4*>(a_gmem_curr), 16);

            uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
            const half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
            CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();

        for (int n = 0; n < OUT_TILES; n++)
        {
            load_tile_a_shared_swizzle<S_BITS_A>(a_tile[n], a_smem, (laneM*OUT_TILES + n)*WMMA_MKN*WMMA_MKN, WMMA_MKN, lane_id);
        }
        for (int n = 0; n < OUT_TILES; n++)
        {
            load_tile_b_shared_swizzle<S_BITS_B>(b_tile, b_smem, (laneN*OUT_TILES + n)*WMMA_MKN, SM_TILES*WMMA_MKN, lane_id);
            for (int m = 0; m < OUT_TILES; m++)
            {
                mma(a_tile[m], b_tile, acc[m][n]);
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
                for (int k = 0; k<4; k++)
                {
                    reinterpret_cast<half2*>(&c[(output_row + (lane_id>>2) + (k%2)*8)*n_elem + output_col + (k/2)*8])[lane_id%4]
                        = acc[i][j].x[k];
                }
            }
        }
    }
}

template<int SMEM_TILES, int OUT_TILES>
double check_configuration_async(half* a, half*b, half* output, int N)
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
    cudaFuncSetAttribute(tensor_core_matmul_async_swizzle<SMEM_TILES, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_async_swizzle<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
    // return std::min(measure_performance([&](){ tensor_core_matmul_reg_smem_async<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); }),
    // measure_performance([&](){ tensor_core_matmul_reg_smem_async_pipeline<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); }));
}

double TensorCoresAsyncKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    matmul_time = std::min(matmul_time, check_configuration_async<8, 2>(a, b, output, N));
    test_output(cublas_ref, N);

    // matmul_time = std::min(matmul_time, check_configuration_async<9, 3>(a, b, output, N));
    // test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async<8, 4>(a, b, output, N));
    test_output(cublas_ref, N);

    return matmul_time;
}
