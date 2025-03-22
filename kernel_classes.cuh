#pragma once

#include "utils.cuh"

class BaseKernel 
{
public:
    BaseKernel(int max_N)
    {
        gpuErrchk(cudaMalloc((void**) &output, max_N*max_N*sizeof(half)));
        gpuErrchk(cudaMemset(output, 0, max_N*max_N*sizeof(half)));
        dimGrid = dim3(1,1,1);
        dimBlock = dim3(1,1,1);
    }
    ~BaseKernel()
    {
        cudaFree(output);
    }

    virtual double run(half* a, half* b, half* cublas_ref, int N) = 0;

    void test_output(half* compare, int N, float tolerance = 1e-3)
    {
        half* d_h = new half[N*N];
        cudaMemcpy(d_h, output, N*N*sizeof(half), cudaMemcpyDeviceToHost);
        for (int j = 0; j < N*N; j++)
        {
            float relative_difference = abs(1 - ((float)compare[j] / (float)d_h[j]));
            ASSERT(relative_difference < tolerance, "failed at output %s, index %d, %f, %f, rdiff; %f\n", kernel_name, j, (float)d_h[j], (float)compare[j], relative_difference);
        } 
    }
    std::string kernel_name = "UNDEFINED";
    half* output;

    dim3 dimGrid;
    dim3 dimBlock;
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
