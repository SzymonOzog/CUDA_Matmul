#include <cmath>
#include <iostream>
#include <cassert>
#include <cublas_v2.h>
#include <mma.h>
#include <random>
#include <vector>
#include <cuda_pipeline.h>

#include "utils.cuh"

#define TILE_WIDTH 32
#define BENCH_STEPS 100
#define WARMUP_STEPS 10
#define TIMINGS 4
#define START 9

__global__ void matmul_elem(int n, half* a, half* b, half* c)
{
    int column = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row < n && column < n)
    {
        float dot_prod = 0.f;
        for(int i = 0; i < n; i++)
        {
            dot_prod += (float)a[row*n + i] * (float)b[i*n + column];
        }
        c[row*n+column] = dot_prod;
    }
}

__global__ void tiled_matmul(int n, half* a, half* b, half* c)
{
    __shared__ half a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ half b_tile[TILE_WIDTH][TILE_WIDTH];

    int column = blockIdx.x*TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float dot_prod = 0.f;
    for (int tile_offset = 0; tile_offset<n; tile_offset+=TILE_WIDTH)
    {
        int a_chk = tile_offset+tx < n && row < n;
        a_tile[ty][tx] = a_chk ? a[row*n + tile_offset+tx] : (half)0.f;

        int b_chk = (tile_offset+ty) < n && column < n;
        b_tile[ty][tx] = b_chk ? b[(tile_offset+ty)*n + column] : (half)0.f;

        __syncthreads();
        for(int i = 0; i < TILE_WIDTH; i++)
        {
            dot_prod += (float)a_tile[ty][i] * (float)b_tile[i][tx];
        }
        __syncthreads();
    }

    if (row < n && column < n)
    {
        c[row*n+column] = dot_prod;
    }
}

using layout = nvcuda::wmma::row_major;

__global__ void tensor_core_matmul(int n, half* a, half* b, half* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc;

    nvcuda::wmma::fill_fragment(acc, 0);

    for (int32_t i = 0; i < n; i+= WMMA_MKN)
    {
        const int32_t matrix_a_row = warpM * WMMA_MKN;
        const int32_t matrix_b_col = warpN * WMMA_MKN;

        if(matrix_a_row<n && matrix_b_col<n && i<n)
        {
            nvcuda::wmma::load_matrix_sync(a_frag, a + matrix_a_row * n + i, n);
            nvcuda::wmma::load_matrix_sync(b_frag, b + i * n + matrix_b_col, n);

            nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    nvcuda::wmma::store_matrix_sync(c + warpM*WMMA_MKN*n + warpN*WMMA_MKN, acc, n, nvcuda::wmma::mem_row_major);
}

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

template<int SM_TILES, int OUT_TILES>
__global__ void tensor_core_matmul_reg_smem(int n_elem, half* a, half* b, half* c)
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
        // __syncthreads();
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

    template <typename F>
double measure_performance(const F& fn)
{
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    double total_time = 0.0;

    for (int i = -WARMUP_STEPS; i<BENCH_STEPS; i++)
    {
        float run_time=0.0;
        clear_l2();
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaEventRecord(start));
        fn();
        gpuErrchk(cudaEventRecord(stop));
        gpuErrchk(cudaEventSynchronize(stop));
        gpuErrchk(cudaEventElapsedTime(&run_time, start, stop));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        if (i >= 0) // warmup
        {
            total_time += run_time;
        }
    }

    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return total_time/BENCH_STEPS;
}

int main()
{
    std::vector<half*> outputs;
    float naive_times[TIMINGS];
    float tiled_times[TIMINGS];
    float cublas_times[TIMINGS];
    float tensor_core_times[TIMINGS];
    float tensor_core_reg_times[TIMINGS];
    float tensor_core_reg_smem_times[TIMINGS];
    float tensor_core_reg_smem_shuffle_times[TIMINGS];
    half* a_d;
    half* b_d;

    long max_N = std::pow<long, long>(2, START+TIMINGS-1);
    for(int i = 0; i < 9; i++)
    {
        half* output;
        cudaMalloc((void**) &output, max_N*max_N*sizeof(half));
        cudaMemset(output, 0, max_N*max_N*sizeof(half));
        outputs.push_back(output);
        
    }

    cudaMalloc((void**) &a_d, max_N*max_N*sizeof(half));
    cudaMalloc((void**) &b_d, max_N*max_N*sizeof(half));

    half* a = new half[max_N * max_N];
    half* b = new half[max_N * max_N];
    half* c = new half[max_N * max_N];

    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, 2);

    for (int p = START; p<START+TIMINGS; p++)
    {
        long N = std::pow<long, long>(2, p);
        for (int i = 0; i<N; i++)
        {
            for (int j = 0; j<N; j++)
            {
                b[i*N + j] = dist(e2);
                a[i*N + j] = dist(e2);

            }
        }
        cudaMemcpy(a_d, a, N*N*sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, N*N*sizeof(half), cudaMemcpyHostToDevice);
        int BLOCK_SIZE=32;
        dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

        double matmul_time = measure_performance([&](){ matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[0]); });


        dimGrid = dim3(ceil(N/(float)TILE_WIDTH), ceil(N/(float)TILE_WIDTH), 1);
        dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);

        double tiled_time = measure_performance([&](){ tiled_matmul<<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[1]); });

        cublasHandle_t handle;
        cublasCreate(&handle);
        half alpha = 1.f;
        half beta = 0.f;
        double cublas_time = measure_performance([&](){ cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, b_d, N, a_d, N, &beta, outputs[2], N); });

        int num_warps_x = 4;
        int num_warps_y = 4;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = (N + (WMMA_MKN*num_warps_x -1)) / (WMMA_MKN*num_warps_x);
        dimGrid.y = (N + WMMA_MKN*num_warps_y -1) / (WMMA_MKN*num_warps_y);

        double tensor_cores_time = measure_performance([&](){ tensor_core_matmul<<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[3]); });

        constexpr int OUT_TILES_REG = 4;
        num_warps_x = 4;
        num_warps_y = 4;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(OUT_TILES_REG*WMMA_MKN*num_warps_x));
        dimGrid.y = std::ceil((float)N/(OUT_TILES_REG*WMMA_MKN*num_warps_y));
        double tensor_cores_reg_time = measure_performance([&](){ tensor_core_matmul_reg<OUT_TILES_REG><<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[5]); });
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        constexpr int SMEM_TILES = 8;
        constexpr int OUT_TILES = 2;

        num_warps_x = SMEM_TILES/OUT_TILES;
        num_warps_y = SMEM_TILES/OUT_TILES;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES*WMMA_MKN));
        unsigned int smem_size = 2*SMEM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem<SMEM_TILES, OUT_TILES>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        double tensor_cores_reg_smem_time = measure_performance([&](){ tensor_core_matmul_reg_smem<SMEM_TILES, OUT_TILES><<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[4]); });

        constexpr int SMEM_TILES2 = 9;
        constexpr int OUT_TILES2 = 3;

        num_warps_x = SMEM_TILES2/OUT_TILES2;
        num_warps_y = SMEM_TILES2/OUT_TILES2;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES2*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES2*WMMA_MKN));
        smem_size = 2*SMEM_TILES2*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem<SMEM_TILES2, OUT_TILES2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        tensor_cores_reg_smem_time = std::min(tensor_cores_reg_smem_time,
                measure_performance([&](){ tensor_core_matmul_reg_smem<SMEM_TILES2, OUT_TILES2><<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[4]); }));

        constexpr int SMEM_TILES3 = 8;
        constexpr int OUT_TILES3 = 4;

        num_warps_x = SMEM_TILES3/OUT_TILES3;
        num_warps_y = SMEM_TILES3/OUT_TILES3;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES3*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES3*WMMA_MKN));
        smem_size = 2*SMEM_TILES3*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem<SMEM_TILES3, OUT_TILES3>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        tensor_cores_reg_smem_time = std::min(tensor_cores_reg_smem_time,
                measure_performance([&](){ tensor_core_matmul_reg_smem<SMEM_TILES3, OUT_TILES3><<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[4]); }));

        smem_size = 32*(SMEM_TILES3*sizeof(frag_a) + SMEM_TILES3*sizeof(frag_b));
        // printf("smem_size = %d\n", smem_size);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem_shuffle<SMEM_TILES3, OUT_TILES3>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        double tensor_cores_reg_smem_shuffle_time = measure_performance([&](){ tensor_core_matmul_reg_smem_shuffle<SMEM_TILES3, OUT_TILES3><<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[6]); });

        constexpr int SMEM_TILES4 = 8;
        constexpr int OUT_TILES4 = 4;
        constexpr int N_WARPS = (SMEM_TILES4/OUT_TILES4) * (SMEM_TILES4/OUT_TILES4);

        num_warps_x = SMEM_TILES4/OUT_TILES4;
        num_warps_y = SMEM_TILES4/OUT_TILES4;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES4*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES4*WMMA_MKN));
        smem_size = 2*SMEM_TILES4*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem_prefetch<SMEM_TILES4, OUT_TILES4, N_WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        double tensor_cores_reg_smem_prefetch_time = measure_performance([&](){ tensor_core_matmul_reg_smem_prefetch<SMEM_TILES4, OUT_TILES4, N_WARPS>
                <<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[7]); });

        constexpr int SMEM_TILES5 = 8;
        constexpr int OUT_TILES5 = 2;
        constexpr int N_WARPS2 = (SMEM_TILES5/OUT_TILES5) * (SMEM_TILES5/OUT_TILES5);

        num_warps_x = SMEM_TILES5/OUT_TILES5;
        num_warps_y = SMEM_TILES5/OUT_TILES5;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES5*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES5*WMMA_MKN));
        smem_size = 2*SMEM_TILES5*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem_prefetch<SMEM_TILES5, OUT_TILES5, N_WARPS2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        tensor_cores_reg_smem_prefetch_time = std::min(tensor_cores_reg_smem_prefetch_time,
                measure_performance([&](){ tensor_core_matmul_reg_smem_prefetch<SMEM_TILES5, OUT_TILES5, N_WARPS2>
                    <<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[7]); }));


        constexpr int SMEM_TILES6 = 8;
        constexpr int OUT_TILES6 = 4;

        num_warps_x = SMEM_TILES6/OUT_TILES6;
        num_warps_y = SMEM_TILES6/OUT_TILES6;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES6*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES6*WMMA_MKN));
        smem_size = 2*2*SMEM_TILES6*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem_async<SMEM_TILES6, OUT_TILES6>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        double tensor_cores_reg_smem_async_time = measure_performance([&](){ tensor_core_matmul_reg_smem_async<SMEM_TILES6, OUT_TILES6>
                <<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[8]); });

        constexpr int SMEM_TILES7 = 8;
        constexpr int OUT_TILES7 = 2;

        num_warps_x = SMEM_TILES7/OUT_TILES7;
        num_warps_y = SMEM_TILES7/OUT_TILES7;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = std::ceil((float)N/(SMEM_TILES7*WMMA_MKN));
        dimGrid.y = std::ceil((float)N/(SMEM_TILES7*WMMA_MKN));
        smem_size = 2*2*SMEM_TILES7*WMMA_MKN*WMMA_MKN*sizeof(half);
        cudaFuncSetAttribute(tensor_core_matmul_reg_smem_prefetch<SMEM_TILES7, OUT_TILES7, N_WARPS2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

        tensor_cores_reg_smem_async_time = std::min(tensor_cores_reg_smem_async_time,
                measure_performance([&](){ tensor_core_matmul_reg_smem_prefetch<SMEM_TILES7, OUT_TILES7, N_WARPS2>
                    <<<dimGrid, dimBlock, smem_size>>>(N, a_d, b_d, outputs[8]); }));

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        long ops = 2*std::pow(N, 3);
        std::cout<<"n = "<<N<<" matmul time: "<<matmul_time<< " gflops: " <<(double)ops/(matmul_time*1e6) <<
            "\n tiled time: "<<tiled_time<< " gflops: " <<(double)ops/(tiled_time*1e6) <<
            "\n tensor cores time: "<<tensor_cores_time<< " gflops: " <<(double)ops/(tensor_cores_time*1e6) <<
            "\n tensor cores reg time: "<<tensor_cores_reg_time<< " gflops: " <<(double)ops/(tensor_cores_reg_time*1e6) <<
            "\n tensor cores reg smem time: "<<tensor_cores_reg_smem_time<< " gflops: " <<(double)ops/(tensor_cores_reg_smem_time*1e6) <<
            "\n tensor cores reg smem shuffle time: "<<tensor_cores_reg_smem_shuffle_time<< " gflops: " <<(double)ops/(tensor_cores_reg_smem_shuffle_time*1e6) <<
            "\n tensor cores reg smem prefetch time: "<<tensor_cores_reg_smem_prefetch_time<< " gflops: " <<(double)ops/(tensor_cores_reg_smem_prefetch_time*1e6) <<
            "\n tensor cores reg smem async time: "<<tensor_cores_reg_smem_async_time<< " gflops: " <<(double)ops/(tensor_cores_reg_smem_async_time*1e6) <<
            "\n cublas time: "<<cublas_time<< " gflops: " <<(double)ops/(cublas_time*1e6) <<
            "\n -------------------------------------------------------------------------------------" <<
            std::endl;

        naive_times[p-START] = matmul_time;
        tiled_times[p-START] = tiled_time;
        cublas_times[p-START] = cublas_time;
        tensor_core_times[p-START] = tensor_cores_time;
        tensor_core_reg_times[p-START] = tensor_cores_reg_time;
        tensor_core_reg_smem_times[p-START] = tensor_cores_reg_smem_time;
    }
    half* compare = new half[max_N*max_N];
    half* d_h = new half[max_N*max_N];
    cudaMemcpy(compare, outputs[2], max_N*max_N*sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 3; i < outputs.size(); i++)
    {
        cudaMemcpy(d_h, outputs[i], max_N*max_N*sizeof(half), cudaMemcpyDeviceToHost);
        float tolerance = 1e-3;
        for (int j = 0; j < max_N*max_N; j++)
        {
            float relative_difference = abs(1 - ((float)compare[j] / (float)d_h[j]));
            ASSERT(relative_difference < tolerance, "failed at output %d, index %d, %f, %f, rdiff; %f\n", i, j, (float)d_h[j], (float)compare[j], relative_difference);
        } 
        cudaFree(outputs[i]);
    }
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(compare);

    std::cout<<"normal_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<naive_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    std::cout<<"tiled_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<tiled_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    std::cout<<"cublas_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<cublas_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    std::cout<<"tensor_core_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<tensor_core_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    std::cout<<"tensor_core_reg_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<tensor_core_reg_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    std::cout<<"tensor_core_reg_smem_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<tensor_core_reg_smem_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;

    return 0;
}
