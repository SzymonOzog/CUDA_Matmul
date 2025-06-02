#include "kernel_classes.cuh"
#include "ptx_helpers.cuh"
#include "utils.cuh"
template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_async_swizzle_pre_transpose(int n_elem, const half* a, const half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;
    const int32_t lane_id = threadIdx.x%32;
    constexpr const unsigned int S_BITS_A = 3;
    constexpr const unsigned int S_BITS_B = 3;

    extern __shared__ __align__(128) char smem[];

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
        const half* b_curr = b + blockIdx.y*SM_TILES*WMMA_MKN*n_elem + tile;
        // const half* b_curr = b + (tile)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
        for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8; i < SM_TILES*WMMA_MKN*WMMA_MKN; i+=blockDim.x*blockDim.y*8)
        {
            uint32_t a_smem_curr = __cvta_generic_to_shared(&a_smem[i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)]);
            const half* a_gmem_curr = &a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
            // CP_ASYNC_CG(a_smem_curr, reinterpret_cast<const float4*>(a_gmem_curr), 16);
            reinterpret_cast<float4*>(&a_smem[i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)])[0]
                = *(reinterpret_cast<const float4*>(a_gmem_curr));

            // uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[i^((i&(S_MASK<<S_BITS_B))>>S_BITS_B)]);
            uint32_t b_smem_curr = __cvta_generic_to_shared(&b_smem[i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)]);
            const half* b_gmem_curr = &b_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN];
            // const half* b_gmem_curr = &b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)];
            // CP_ASYNC_CG(b_smem_curr, reinterpret_cast<const float4*>(b_gmem_curr), 16);
            reinterpret_cast<float4*>(&b_smem[i^((i&(S_MASK<<S_BITS_A))>>S_BITS_A)])[0]
                = *(reinterpret_cast<const float4*>(b_gmem_curr));

        }
        // CP_ASYNC_COMMIT_GROUP();
        // CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();

        for (int n = 0; n < OUT_TILES; n++)
        {
            load_tile_a_shared_swizzle<S_BITS_A>(a_tile[n], a_smem, (laneM*OUT_TILES + n)*WMMA_MKN*WMMA_MKN, WMMA_MKN, lane_id);
        } for (int n = 0; n < OUT_TILES; n++) {
            load_tile_b_shared_swizzle_pre_transposed<S_BITS_B>(b_tile, b_smem, (laneN*OUT_TILES + n)*WMMA_MKN*WMMA_MKN, WMMA_MKN, lane_id);
            // half* A_h = reinterpret_cast<half*>(b_tile.x);
            // if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
            // {
            //     print_tile(b_smem, WMMA_MKN);
            // }
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

#define TILE_DIM 32
#define BLOCK_ROWS 32
__global__ void transposeCoalesced(half *odata, const half *idata)
{
  __shared__ half tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

template<int SMEM_TILES, int OUT_TILES>
double check_configuration_async_transposed(half* a, half*b, half* output, half* transposed, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = SMEM_TILES/OUT_TILES;
    int num_warps_y = SMEM_TILES/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));

    dim3 dimBlockT(1,1,1);
    dim3 dimGridT(1,1,1);
    dimBlockT.x = TILE_DIM;
    dimBlockT.y = BLOCK_ROWS;
    dimGridT.x = std::ceil((float)N/(TILE_DIM));
    dimGridT.y = std::ceil((float)N/(BLOCK_ROWS));

    unsigned int smem_size = 2*SMEM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half);
    cudaFuncSetAttribute(tensor_core_matmul_async_swizzle_pre_transpose<SMEM_TILES, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ 
            transposeCoalesced<<<dimGridT, dimBlockT>>>(transposed, b);
            tensor_core_matmul_async_swizzle_pre_transpose<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, transposed, output);
            });
}

double TensorCoresAsyncTransposeKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    // matmul_time = std::min(matmul_time, check_configuration_async_transposed<8, 2>(a, b, output, transposed, N));
    // debug_print(b,  N, true);
    // debug_print(transposed,  N, true);
    // debug_print(output,  N, true);
    // debug_print(cublas_ref,  N, false);
    // test_output(cublas_ref, N);

    matmul_time = std::min(matmul_time, check_configuration_async_transposed<8, 4>(a, b, output, transposed, N));
    test_output(cublas_ref, N);

    return matmul_time;
}

