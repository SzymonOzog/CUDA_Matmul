#pragma once

template<int R_, int C_>
struct mma_tile
{
    static constexpr int R = R_;
    static constexpr int C = C_;
    static constexpr int len = R*C/(32*2);

    half2 x[len] = {{0.f, 0.f}};
};

static __device__ __forceinline__ void load_tile_a(mma_tile<16, 16>& a_tile, const half* mat, const int stride, const int lane_id)
{
    a_tile.x[0] = reinterpret_cast<const half2*>(&mat[((lane_id>>2))*stride])[lane_id%4];
    a_tile.x[1] = reinterpret_cast<const half2*>(&mat[((lane_id>>2) + 8)*stride])[lane_id%4];
    a_tile.x[2] = reinterpret_cast<const half2*>(&mat[((lane_id>>2))*stride + 8])[lane_id%4];
    a_tile.x[3] = reinterpret_cast<const half2*>(&mat[((lane_id>>2) + 8)*stride + 8])[lane_id%4];
}

static __device__ __forceinline__ void load_tile_a_shared(mma_tile<16, 16>& a_tile, const half* mat, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(a_tile.x);
    int row = ((lane_id/8)%2) * 8 + lane_id%8;
    int col = ((lane_id/16))*8;
    const half* addr = mat + row * stride + col;// + lane_id%8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

static __device__ __forceinline__ void load_tile_b(mma_tile<16, 16>& b_tile, const half* mat, const int stride, const int lane_id)
{
    for (int j = 0; j<4; j++)
    {
        half2 tmp;
        tmp.x = mat[((j%2)*8 + (lane_id%4)*2)*stride + (lane_id>>2) + (j/2)*8];
        tmp.y = mat[((j%2)*8 + (lane_id%4)*2 + 1)*stride + (lane_id>>2) + (j/2)*8];
        b_tile.x[j] = tmp;
    }
}

static __device__ __forceinline__ void load_tile_b_shared(mma_tile<16, 16>& b_tile, const half* mat, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(b_tile.x);
    int row = ((lane_id/8)%2) * 8 + lane_id%8;
    int col = ((lane_id/16))*8;
    const half* addr = mat + row * stride + col;// + lane_id%8;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

static __device__ __forceinline__ void store_matrix(mma_tile<16, 16>& acc, const half* mat, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(acc.x);
    int row = ((lane_id/8)%2) * 8 + lane_id%8;
    int col = ((lane_id/16))*8;
    const half* addr = mat + row * stride + col;// + lane_id%8;
    asm volatile("stmatrix.sync.aligned.m8n8.x4.b16 [%4], {%0, %1, %2, %3};"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

static __device__ __forceinline__ void mma(mma_tile<16, 16>& a, mma_tile<16, 16>& b, mma_tile<16, 16>& acc)
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
