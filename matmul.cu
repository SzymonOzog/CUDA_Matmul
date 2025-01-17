#include <cmath>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <cublas_v2.h>
#include <mma.h>
#include <random>
#include <vector>

#define TILE_WIDTH 32
#define BENCH_STEPS 100
#define WARMUP_STEPS 25
#define TIMINGS 2
#define START 11

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

using datatype = half;
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void debug_print(datatype* matrix, int N, bool device)
{
    datatype* host_ptr;
    if (device)
    {
        host_ptr = new datatype[N*N];
        cudaMemcpy(host_ptr, matrix, N*N*sizeof(datatype), cudaMemcpyDeviceToHost);
    }
    else
    {
        host_ptr = matrix;
    }
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout<<std::setprecision(3)<<(float)host_ptr[i*N + j]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
    if (device)
        delete[] host_ptr;
}

void clear_l2() 
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

__global__ void matmul_elem(int n, datatype* a, datatype* b, datatype* c)
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

__global__ void tiled_matmul(int n, datatype* a, datatype* b, datatype* c)
{
    __shared__ datatype a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ datatype b_tile[TILE_WIDTH][TILE_WIDTH];

    int column = blockIdx.x*TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y*TILE_WIDTH + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float dot_prod = 0.f;
    for (int tile_offset = 0; tile_offset<n; tile_offset+=TILE_WIDTH)
    {
        int a_chk = tile_offset+tx < n && row < n;
        a_tile[ty][tx] = a_chk ? a[row*n + tile_offset+tx] : (datatype)0.f;

        int b_chk = (tile_offset+ty) < n && column < n;
        b_tile[ty][tx] = b_chk ? b[(tile_offset+ty)*n + column] : (datatype)0.f;

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
#define WMMA_MKN 16

__global__ void tensor_core_matmul(int n, datatype* a, datatype* b, datatype* c)
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

template<int WMMA_TILE_SIZE, int REG_TILES>
__global__ void tensor_core_matmul_smem(int n, datatype* a, datatype* b, datatype* c)
{
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    __shared__ datatype smem[WMMA_TILE_SIZE*WMMA_MKN*WMMA_MKN];

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[REG_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag[REG_TILES];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc;

    nvcuda::wmma::fill_fragment(acc, 0);
    const int32_t matrix_a_row = warpM * WMMA_MKN;

    for (int32_t tile = 0; tile < ceilf((float)n/(WMMA_MKN*WMMA_TILE_SIZE)); tile+=1)
    {
        for(int32_t i = threadIdx.x; i < WMMA_TILE_SIZE*WMMA_MKN*WMMA_MKN; i+=blockDim.x)
        {
            int32_t row = tile * WMMA_TILE_SIZE*WMMA_MKN + i/WMMA_MKN;
            int32_t column = warpN*WMMA_MKN + i%WMMA_MKN;
            if (row<n && column < n)
                smem[i] =  b[row*n + column];
        }
        __syncthreads();
        for (int32_t i = 0; i < WMMA_TILE_SIZE; i+=REG_TILES)
        {
            #pragma unroll REG_TILES
            for (int j = 0; j < REG_TILES; j++)
            {
                int idx = i + j;
                int mat_a_col = (tile*WMMA_TILE_SIZE+idx)*WMMA_MKN;
                if(matrix_a_row<n && mat_a_col<n)
                {
                    nvcuda::wmma::load_matrix_sync(a_frag[j], a + matrix_a_row * n + mat_a_col, n);
                    nvcuda::wmma::load_matrix_sync(b_frag[j], smem+idx*WMMA_MKN*WMMA_MKN, WMMA_MKN);

                    nvcuda::wmma::mma_sync(acc, a_frag[j], b_frag[j], acc);
                }
            }
        }
        __syncthreads();
    }

    if (warpM * WMMA_MKN < n && warpN*WMMA_MKN < n)
        nvcuda::wmma::store_matrix_sync(c + warpM*WMMA_MKN*n + warpN*WMMA_MKN, acc, n, nvcuda::wmma::mem_row_major);
}


void cpu_matmul(int n, datatype* a, datatype* b, datatype*c)
{
    for (int i = 0; i<n; i++)
    {
        for (int j = 0; j<n; j++)
        {
            datatype dot_product = 0.f;
            for (int k = 0; k<n; k++)
            {
                dot_product += a[i*n + k] * b[k*n + j];
            }
            c[i*n+j] = dot_product; 
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
    std::vector<datatype*> outputs;
    float naive_times[TIMINGS];
    float tiled_times[TIMINGS];
    float cublas_times[TIMINGS];
    float tensor_core_times[TIMINGS];
    float tensor_core_smem_times[TIMINGS];
    datatype* a_d;
    datatype* b_d;

    long max_N = std::pow<long, long>(2, START+TIMINGS-1);
    for(int i = 0; i < 5; i++)
    {
        datatype* output;
        cudaMalloc((void**) &output, max_N*max_N*sizeof(datatype));
        cudaMemset(output, 0, max_N*max_N*sizeof(datatype));
        outputs.push_back(output);
        
    }

    cudaMalloc((void**) &a_d, max_N*max_N*sizeof(datatype));
    cudaMalloc((void**) &b_d, max_N*max_N*sizeof(datatype));

    datatype* a = new datatype[max_N * max_N];
    datatype* b = new datatype[max_N * max_N];
    datatype* c = new datatype[max_N * max_N];

    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(-10, 10);

    for (int p = START; p<START+TIMINGS; p++)
    {
        long N = std::pow<long, long>(2, p);
        for (int i = 0; i<N; i++)
        {
            for (int j = 0; j<N; j++)
            {
                a[i*N + j] = 0;
                b[i*N + j] = dist(e2);
            }
            a[i*N + i] = dist(e2);
        }
        cudaMemcpy(a_d, a, N*N*sizeof(datatype), cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, N*N*sizeof(datatype), cudaMemcpyHostToDevice);
        int BLOCK_SIZE=32;

        dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

        double matmul_time = measure_performance([&](){ matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[0]); });


        dimGrid = dim3(ceil(N/(float)TILE_WIDTH), ceil(N/(float)TILE_WIDTH), 1);
        dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);

        double tiled_time = measure_performance([&](){ tiled_matmul<<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[1]); });

        cublasHandle_t handle;
        cublasCreate(&handle);
        datatype alpha = 1.f;
        datatype beta = 0.f;
        double cublas_time = measure_performance([&](){ cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, b_d, N, a_d, N, &beta, outputs[2], N); });
        ;
        int num_warps_x = 4;
        int num_warps_y = 4;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = (N + (WMMA_MKN*num_warps_x -1)) / (WMMA_MKN*num_warps_x);
        dimGrid.y = (N + WMMA_MKN*num_warps_y -1) / (WMMA_MKN*num_warps_y);

        double tensor_cores_time = measure_performance([&](){ tensor_core_matmul<<<dimGrid, dimBlock>>>(N, a_d, b_d, outputs[3]); });

        num_warps_x = 32;
        num_warps_y = 1;
        dimBlock.x = num_warps_x * 32;
        dimBlock.y = num_warps_y;

        dimGrid.x = (N + (WMMA_MKN*num_warps_x -1)) / (WMMA_MKN*num_warps_x);
        dimGrid.y = (N + WMMA_MKN*num_warps_y -1) / (WMMA_MKN*num_warps_y);

        double tensor_cores_smem_time = measure_performance([&](){ tensor_core_matmul_smem<32, 8><<<dimGrid, dimBlock>>>(N, b_d, b_d, outputs[4]); });

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        std::cout<<"n = "<<N<<" matmul time: "<<matmul_time<<
            " tiled time: "<<tiled_time<<
            " tensor cores time: "<<tensor_cores_time<<
            " tensor cores smem time: "<<tensor_cores_smem_time<<
            " cublas time: "<<cublas_time<<
            std::endl;


        naive_times[p-START] = matmul_time;
        tiled_times[p-START] = tiled_time;
        cublas_times[p-START] = cublas_time;
        tensor_core_times[p-START] = tensor_cores_time;
        tensor_core_smem_times[p-START] = tensor_cores_smem_time;
    }
    datatype* compare = new datatype[max_N*max_N];
    datatype* d_h = new datatype[max_N*max_N];
    cudaMemcpy(compare, outputs[0], max_N*max_N*sizeof(datatype), cudaMemcpyDeviceToHost);
    for(int i = 1; i < outputs.size(); i++)
    {
        cudaMemcpy(d_h, outputs[i], max_N*max_N*sizeof(datatype), cudaMemcpyDeviceToHost);
        float tolerance = 1e-8;
        for (int j = 0; j < max_N*max_N; j++)
        {
            ASSERT(abs((float)compare[j] - (float)d_h[j]) < tolerance, "failed at output %d, index %d, %f, %f\n", i, j, (float)d_h[j], (float)compare[j]);
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

    std::cout<<"tensor_core_smem_times = [";
    for (int i = 0; i<TIMINGS; i++)
    {
        std::cout<<tensor_core_smem_times[i]<<", ";
    }
    std::cout<<"]"<<std::endl;
    return 0;
}
