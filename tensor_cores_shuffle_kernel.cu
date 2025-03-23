#include "kernel_classes.cuh"

template <typename T>
__device__ void store_fragment(const T& frag, half* target)
{
    const int& lane_id = threadIdx.x % 32;
    reinterpret_cast<float4*>(target)[lane_id] = 
        *reinterpret_cast<const float4*>(frag.x);
    __syncwarp();
}

template <typename T>
__device__ T load_fragment(half* source)
{
    const int& lane_id = threadIdx.x % 32;
    T ret;
    *reinterpret_cast<float4*>(ret.x) = 
        reinterpret_cast<float4*>(source)[lane_id];
    __syncwarp();
    return ret;
}


using frag_a =  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout>;
using frag_b =  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout>;
template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_reg_smem_shuffle(int n_elem, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    const int32_t laneM = threadIdx.x/32;
    const int32_t laneN = threadIdx.y;

    const int32_t warps_X = blockDim.x/32;
    const int32_t warps_total = warps_X * blockDim.y;

    extern __shared__ char smem[];

    half (*a_smem)[WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(smem);
    half (*b_smem)[WMMA_MKN*WMMA_MKN]
        = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(
                smem + SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));

    frag_a a_frag[OUT_TILES];
    frag_b b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];

    for(int32_t i = 0; i<OUT_TILES; i++)
        for(int32_t j = 0; j<OUT_TILES; j++)
            nvcuda::wmma::fill_fragment(acc[i][j], 0);

    const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
    const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;

    for (int32_t tile = 0; tile < n_elem; tile+=WMMA_MKN)
    {
        for (int i = laneM*warps_X + laneN;
                i < SM_TILES;
                i+=warps_total)
        {
            const int32_t a_row = blockIdx.x*SM_TILES*WMMA_MKN + i*WMMA_MKN;
            const int32_t a_col = tile;

            if (a_row < n_elem && a_col < n_elem)
            {
                frag_a tmp_a;
                nvcuda::wmma::load_matrix_sync(tmp_a, a + a_row*n_elem + a_col, n_elem);
                store_fragment(tmp_a, a_smem[i]);
            }
            const int32_t b_row = tile;
            const int32_t b_col = blockIdx.y*SM_TILES*WMMA_MKN + i*WMMA_MKN;
            if (b_row < n_elem && b_col < n_elem)
            {
                frag_b tmp_b;
                nvcuda::wmma::load_matrix_sync(tmp_b, b + b_row*n_elem + b_col, n_elem);
                store_fragment(tmp_b, b_smem[i]);
            }
        }
        __syncthreads();
        for (int n = 0; n < OUT_TILES && n*WMMA_MKN < n_elem; n++)
        {
            a_frag[n] = load_fragment<frag_a>(a_smem[laneM*OUT_TILES + n]);
        }
        for (int n = 0; n < OUT_TILES && n*WMMA_MKN < n_elem; n++)
        {
            b_frag = load_fragment<frag_b>(b_smem[laneN*OUT_TILES + n]);
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
double check_configuration(half* a, half*b, half* output, int N)
{
    dim3 dimBlock(1,1,1);
    dim3 dimGrid(1,1,1);

    int num_warps_x = SMEM_TILES/OUT_TILES;
    int num_warps_y = SMEM_TILES/OUT_TILES;
    dimBlock.x = num_warps_x * 32;
    dimBlock.y = num_warps_y;

    dimGrid.x = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    dimGrid.y = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
    unsigned int smem_size = 32*(SMEM_TILES*sizeof(frag_a) + SMEM_TILES*sizeof(frag_b));
    cudaFuncSetAttribute(tensor_core_matmul_reg_smem_shuffle<SMEM_TILES, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    return measure_performance([&](){ tensor_core_matmul_reg_smem_shuffle<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a, b, output); });
}

double TensorCoresShuffleKernel::run(half* a, half* b, half* cublas_ref, int N)
{
    double matmul_time = std::numeric_limits<double>::max();

    // matmul_time = std::min(matmul_time, check_configuration<8, 2>(a, b, output, N));
    // test_output(cublas_ref, N, 1e-2);
    //
    // matmul_time = std::min(matmul_time, check_configuration<9, 3>(a, b, output, N));
    // test_output(cublas_ref, N, 1e-2);
    //
    matmul_time = std::min(matmul_time, check_configuration<8, 4>(a, b, output, N));
    test_output(cublas_ref, N, 1e-2);

    return matmul_time;
}
