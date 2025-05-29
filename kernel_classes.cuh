#pragma once

#include "utils.cuh"

using layout = nvcuda::wmma::row_major;

class BaseKernel 
{
public:
    BaseKernel(int max_N)
    {
        gpuErrchk(cudaMalloc((void**) &output, max_N*max_N*sizeof(half)));
        gpuErrchk(cudaMemset(output, 0, max_N*max_N*sizeof(half)));
        dimGrid = dim3(1,1,1);
        dimBlock = dim3(1,1,1);
        max_diff = 0.f;
        mean_diff= 0.f;
        runs = 0;
    }
    ~BaseKernel()
    {
        cudaFree(output);
    }

    virtual double run(half* a, half* b, half* cublas_ref, int N) = 0;

    void test_output(half* compare, int N, float tolerance = 2)
    {
        runs++;
        half* d_h = new half[N*N];
        cudaMemcpy(d_h, output, N*N*sizeof(half), cudaMemcpyDeviceToHost);
        for (int j = 0; j < N*N; j++)
        {
            float relative_difference = abs((float)compare[j] - (float)d_h[j]);
            ASSERT(relative_difference < tolerance, "failed at output %s, index %d, %f, %f, rdiff; %f\n", kernel_name.c_str(), j, (float)d_h[j], (float)compare[j], relative_difference);
            max_diff = std::max(relative_difference, max_diff);
            mean_diff += relative_difference;
        } 
        mean_diff/=N*N;
        delete[] d_h;
    }
    std::string kernel_name = "UNDEFINED";
    half* output;

    dim3 dimGrid;
    dim3 dimBlock;

    float max_diff;
    float mean_diff;
    
    int runs;
};

class NaiveKernel : public BaseKernel
{
public:
    NaiveKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "Naive";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TiledKernel : public BaseKernel
{
public:
    TiledKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "Tiled";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresKernel : public BaseKernel
{
public:
    TensorCoresKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCores";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresRegKernel : public BaseKernel
{
public:
    TensorCoresRegKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCoresReg";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresSmemKernel : public BaseKernel
{
public:
    TensorCoresSmemKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCoresSmem";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresShuffleKernel : public BaseKernel
{
public:
    TensorCoresShuffleKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCoresShuffle";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresPrefetchKernel : public BaseKernel {
public:
    TensorCoresPrefetchKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCoresPrefetch";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresAsyncKernel : public BaseKernel {
public:
    TensorCoresAsyncKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCoresAsync";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};

class TensorCoresSwizzleKernel : public BaseKernel {
public:
    TensorCoresSwizzleKernel(int max_N) : BaseKernel(max_N) 
    {
        kernel_name = "TensorCoresSwizzle";
    }
    virtual double run(half* a, half* b, half* cublas_ref, int N) override;
};
