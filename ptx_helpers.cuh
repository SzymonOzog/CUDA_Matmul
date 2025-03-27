#pragma once

template<int R_, int C_>
struct mma_tile
{
    static constexpr int R = R_;
    static constexpr int C = C_;
    static constexpr int len = R*C/(32*2);

    half2 x[len] = {{0.f, 0.f}};
};

static __device__ __forceinline__ void mma(mma_tile<16, 16> a, mma_tile<16, 16> b, mma_tile<16, 16> acc)
{
    asm("mma.sync.aligned.m16n8.k16.row.col.f16.f16.f16.f16 {%0, %1} {%2, %3, %4, %5}, {%6, %7}, {%8, %9};", 
            : "+r"(acc.x[0]), "+r"(acc.x[1]),
            "+r"(a.x[0]), "+r"(a.x[1]), "+r"(a.x[2]), "+r"(a.x[3]), 
            "+r"(b.x[0]), "+r"(b.x[1]), 
            "+r"(acc.x[0]), "+r"(acc.x[1]))

    asm("mma.sync.aligned.m16n8.k16.row.col.f16.f16.f16.f16 {%0, %1} {%2, %3, %4, %5}, {%6, %7}, {%8, %9};", 
            : "+r"(acc.x[2]), "+r"(acc.x[3]),
            "+r"(a.x[0]), "+r"(a.x[1]), "+r"(a.x[2]), "+r"(a.x[3]), 
            "+r"(b.x[2]), "+r"(b.x[3]), 
            "+r"(acc.x[2]), "+r"(acc.x[3]))
}
