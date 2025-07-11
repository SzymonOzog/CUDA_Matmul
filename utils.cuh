#pragma once

#include <iomanip>
#include <cassert>
#include <cmath>
#include <iostream>
#include <cuda_fp16.h>
#include <mma.h>
#include "ptx_helpers.cuh"

#define WMMA_MKN 16
#define BENCH_STEPS 100
#define WARMUP_STEPS 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


inline void debug_print(half* matrix, int N, bool device)
{
    half* host_ptr;
    if (device)
    {
        host_ptr = new half[N*N];
        cudaMemcpy(host_ptr, matrix, N*N*sizeof(half), cudaMemcpyDeviceToHost);
    }
    else
    {
        host_ptr = matrix;
    }

    const int col_width = 5;

    std::cout << std::endl;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << std::setw(col_width)
                      << std::fixed << std::setprecision(1)
                      << static_cast<float>(host_ptr[i*N + j]) << " ";

            if (j % WMMA_MKN == WMMA_MKN - 1)
                std::cout << " | ";
        }
        std::cout << std::endl;

        if (i % WMMA_MKN == WMMA_MKN - 1)
        {
            std::cout << std::string(N * (col_width + 1) + (N / WMMA_MKN) * 3, '_') << std::endl;
        }
    }
    std::cout << std::endl;
    std::cout << std::endl;

    if (device)
        delete[] host_ptr;
}

// Credit: https://github.com/alexarmbr/matmul-playground
constexpr unsigned int int_log2(unsigned int x)
{
    unsigned int result = 0;
    while (x >>= 1)
    {
        result++;
    }
    return result;
}

inline void clear_l2() 
{
    // Get actual L2 size via CUDA on first call of this function
    static int l2_clear_size = 0;
    static unsigned char* gpu_scratch_l2_clear = NULL;
    if (!gpu_scratch_l2_clear) {
        cudaDeviceGetAttribute(&l2_clear_size, cudaDevAttrL2CacheSize, 0);
        l2_clear_size *= 2; // just to be extra safe (cache is not necessarily strict LRU)
        gpuErrchk(cudaMalloc(&gpu_scratch_l2_clear, l2_clear_size));
    }
    // Clear L2 cache (this is run on every call unlike the above code)
    gpuErrchk(cudaMemset(gpu_scratch_l2_clear, 0, l2_clear_size));
}

__device__ inline void print_tile(half* tile,  int stride)
{
    printf("-------------------------------------------------------------------------------\n");
    for (int i = 0; i < WMMA_MKN; i++)
    {
        for (int j = 0; j < WMMA_MKN; j++)
        {
            printf("%f, ", (float)tile[i*stride + j]);
        }
        printf("\n");
    }
    printf("-------------------------------------------------------------------------------\n");
}

static __device__ __forceinline__ int get_at(const mma_tile<16, 16>& t, int target_thread, int pos)
{
    int ret;
    const int* x = reinterpret_cast<const int*>(t.x);
    for(int i = 0; i < t.len; i++)
    {
        int val = __shfl_sync(0xFFFFFFFF, x[i], target_thread, 32);
        if (pos == i)
            ret = val;
    }
    return ret;
}

template <typename F>
inline double measure_performance(const F& fn)
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

