#include <iostream>
#include <cassert>
#include <cublas_v2.h>

#define TILE_WIDTH 32
#define BENCH_STEPS 100
#define WARMUP_STEPS 25
#define TIMINGS 6
#define START 8

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

int main()
{
  float naive_times[TIMINGS];
  float tiled_times[TIMINGS];
  float cublas_times[TIMINGS];
  datatype* a_d;
  datatype* b_d;
  datatype* c_d;
  datatype* d_d;
  datatype* e_d;


  long max_N = std::pow<long, long>(2, START+TIMINGS-1);
  cudaMalloc((void**) &a_d, max_N*max_N*sizeof(datatype));
  cudaMalloc((void**) &b_d, max_N*max_N*sizeof(datatype));
  cudaMalloc((void**) &c_d, max_N*max_N*sizeof(datatype));
  cudaMalloc((void**) &d_d, max_N*max_N*sizeof(datatype));
  cudaMalloc((void**) &e_d, max_N*max_N*sizeof(datatype));

  datatype* a = new datatype[max_N * max_N];
  datatype* b = new datatype[max_N * max_N];
  datatype* c = new datatype[max_N * max_N];

  cudaEvent_t start, stop;
  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  for (int p = START; p<START+TIMINGS; p++)
  {
    long N = std::pow<long, long>(2, p);
    for (int i = 0; i<N; i++)
    {
        for (int j = 0; j<N; j++)
        {
            a[i*N + j] = 0;
            if (i == j)
            {
                a[i*N + j] = 2;
            }
            b[i*N + j] = i+j;
        }
    }
    cudaMemcpy(a_d, a, N*N*sizeof(datatype), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, N*N*sizeof(datatype), cudaMemcpyHostToDevice);
    int BLOCK_SIZE=32;

    dim3 dimGrid(ceil(N/(float)BLOCK_SIZE), ceil(N/(float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    double matmul_time=0.0;
    for (int i = -1; i<BENCH_STEPS; i++)
    {
      float run_time=0.0;
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      matmul_elem<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d);
      gpuErrchk(cudaEventRecord(stop));
      gpuErrchk(cudaEventSynchronize(stop));
      gpuErrchk(cudaEventElapsedTime(&run_time, start, stop));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      if (i >= 0) // warmup
      {
        matmul_time += run_time;
      }
    }

    dimGrid = dim3(ceil(N/(float)TILE_WIDTH), ceil(N/(float)TILE_WIDTH), 1);
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);

    double tiled_time=0.0;
    for (int i = -WARMUP_STEPS; i<BENCH_STEPS; i++)
    {
      float run_time=0.0;
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      tiled_matmul<<<dimGrid, dimBlock>>>(N, a_d, b_d, d_d);
      gpuErrchk(cudaEventRecord(stop));
      gpuErrchk(cudaEventSynchronize(stop));
      gpuErrchk(cudaEventElapsedTime(&run_time, start, stop));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      if (i >= 0) // warmup
      {
        tiled_time += run_time;
      }
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    datatype alpha = 1.f;
    datatype beta = 0.f;
    double cublas_time=0.0;
    for (int i = -WARMUP_STEPS; i<BENCH_STEPS; i++)
    {
      float run_time=0.0;
      clear_l2();
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaEventRecord(start));
      cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, a_d, N, b_d, N, &beta, e_d, N);
      gpuErrchk(cudaEventRecord(stop));
      gpuErrchk(cudaEventSynchronize(stop));
      gpuErrchk(cudaEventElapsedTime(&run_time, start, stop));
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
      if (i >= 0) // warmup
      {
        cublas_time += run_time;
      }
    }

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    std::cout<<"n = "<<N<<" matmul time: "<<matmul_time/BENCH_STEPS<<" tiled time: "<<tiled_time/BENCH_STEPS<<" cublas time: "<<cublas_time/BENCH_STEPS<<std::endl;


    naive_times[p-START] = matmul_time/BENCH_STEPS;
    tiled_times[p-START] = tiled_time/BENCH_STEPS;
    cublas_times[p-START] = cublas_time/BENCH_STEPS;
  }
  datatype* c_h = new datatype[max_N*max_N];
  datatype* d_h = new datatype[max_N*max_N];
  datatype* e_h = new datatype[max_N*max_N];
  cudaMemcpy(c_h, c_d, max_N*max_N*sizeof(datatype), cudaMemcpyDeviceToHost);
  cudaMemcpy(d_h, d_d, max_N*max_N*sizeof(datatype), cudaMemcpyDeviceToHost);
  cudaMemcpy(e_h, e_d, max_N*max_N*sizeof(datatype), cudaMemcpyDeviceToHost);
  float tolerance = 1e-8;
  for (int i = 0; i < max_N*max_N; i++)
  {
    ASSERT(abs((float)c_h[i] - (float)b[i]*2) < tolerance, "failed at %d, %f, %f\n", i, (float)c[i], (float)c_h[i]);
    ASSERT(abs((float)d_h[i] - (float)b[i]*2) < tolerance, "failed at %d, %f, %f\n", i, (float)c[i], (float)d_h[i]);
    ASSERT(abs((float)e_h[i] - (float)b[i]*2) < tolerance, "failed at %d, %f, %f\n", i, (float)c[i], (float)e_h[i]);
  }
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  cudaFree(d_d);

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
  return 0;
}
