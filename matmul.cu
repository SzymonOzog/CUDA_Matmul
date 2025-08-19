#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include <mma.h>
#include <random>
#include <vector>

#include "kernel_classes.cuh"
#include "utils.cuh"

#define TIMINGS 1
#define START 11



int main()
{
    half* a_d;
    half* b_d;

    long max_N = std::pow<long, long>(2, START+TIMINGS-1);
    half* cublas_ref;
    cudaMalloc((void**) &cublas_ref, max_N*max_N*sizeof(half));
    cudaMemset(cublas_ref, 0, max_N*max_N*sizeof(half));

    cudaMalloc((void**) &a_d, max_N*max_N*sizeof(half));
    cudaMalloc((void**) &b_d, max_N*max_N*sizeof(half));

    half* a = new half[max_N * max_N];
    half* b = new half[max_N * max_N];

    half* cublas_ref_h = new half[max_N*max_N];

    std::random_device rd;
    std::mt19937 e2(rd());
    std::normal_distribution<> dist(0, 1);

    std::vector<BaseKernel*> kernels = {
        //Disabled cause slow
        // new NaiveKernel(max_N),
        // new TiledKernel(max_N),
        // new TensorCoresKernel(max_N),
        new TensorCoresRegKernel(max_N),
        new TensorCoresSmemKernel(max_N),
        new TensorCoresAsyncKernel(max_N),
        new TensorCoresAsyncBTKernel(max_N),
        new TensorCoresAsyncBT_DBKernel(max_N),
        new TensorCoresAsyncBT_DB_IdxKernel(max_N),
        new TensorCoresAsyncBT_DB_FBKernel(max_N),
        new TensorCoresAsyncBT_DB_FB_IdxKernel(max_N),
        new TensorCoresAsyncBT_DB_FB_RegKernel(max_N)
    };

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

        cublasHandle_t handle;
        cublasCreate(&handle);

        half alpha = 1.f;
        half beta = 0.f;
        double cublas_time = measure_performance([&](){ cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N,N,N, &alpha, b_d, N, a_d, N, &beta, cublas_ref, N); });

        cudaMemcpy(cublas_ref_h, cublas_ref, max_N*max_N*sizeof(half), cudaMemcpyDeviceToHost);

        
        long ops = 2*std::pow(N, 3) - std::pow(N, 2);
        std::cout<<"n = "<<N<<" cublas time "<< cublas_time<<" cublas flops "<<(double)ops/(cublas_time*1e6)<<std::endl;
        for (BaseKernel* kernel : kernels)
        {
            double run_time = kernel->run(a_d, b_d, cublas_ref_h, N);
            std::cout<<kernel->kernel_name<<" time: "<<run_time<< " gflops: " <<(double)ops/(run_time*1e6) <<
                " mean difference:" << kernel->mean_diff/kernel->runs << " max diff: " << kernel->max_diff<<std::endl;

        }
    }
    for (auto it = kernels.rbegin(); it != kernels.rend(); it++)
    {
        delete *it;
    }
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(cublas_ref);

    delete [] a;
    delete [] b;
    delete [] cublas_ref_h;

    return 0;
}
