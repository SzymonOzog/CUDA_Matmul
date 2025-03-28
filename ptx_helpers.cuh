#pragma once

template<int R_, int C_>
struct mma_tile
{
    static constexpr int R = R_;
    static constexpr int C = C_;
    static constexpr int len = R*C/(32*2);

    half2 x[len] = {{0.f, 0.f}};
};

static __device__ __forceinline__ void mma(mma_tile<16, 16>& a, mma_tile<16, 16> b, mma_tile<16, 16>& acc)
{
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a.x);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b.x);
    uint32_t* ACC = reinterpret_cast<uint32_t*>(acc.x);
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(ACC[0]), "+r"(ACC[1])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));

    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
            : "+r"(ACC[2]), "+r"(ACC[3])
            : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[2]), "r"(B[3]));
}
