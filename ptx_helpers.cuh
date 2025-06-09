#pragma once

#define S_MASK 0b111000

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

template<int R_, int C_>
struct mma_tile
{
    static constexpr int R = R_;
    static constexpr int C = C_;
    static constexpr int len = R*C/(32*2);

    half2 x[len] = {{0.f, 0.f}};
};

static __device__ __forceinline__ void print_tile(mma_tile<16, 16>& tile, int row, int col, half* addr)
{
    half* A_h = reinterpret_cast<half*>(tile.x);
    printf("loading tile %d, %d, ptr %p for thread %d, vals %f,%f,%f,%f,%f,%f,%f,%f,\n",
            row, col, addr,
            threadIdx.x,
            (float)A_h[0],
            (float)A_h[1],
            (float)A_h[2],
            (float)A_h[3],
            (float)A_h[4],
            (float)A_h[5],
            (float)A_h[6],
            (float)A_h[7]
            );
}

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
    int row = lane_id%16;
    int col = (lane_id/16)*8;
    const half* addr = mat + row * stride + col;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

template<int S_BITS>
static __device__ __forceinline__ void load_tile_a_shared_swizzle(mma_tile<16, 16>& a_tile, const half* mat, const int off_base, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(a_tile.x);
    int row = lane_id%16;
    int col = (lane_id/16)*8;
    int off = off_base + row * stride + col;
    off = off^((off&(S_MASK<<S_BITS))>>S_BITS);
    const half* addr = mat + off;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

static __device__ __forceinline__ void load_tile_a_direct(mma_tile<16, 16>& a_tile, uint32_t addr)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(a_tile.x);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "r"(addr));
}

template<int S_BITS>
static __device__ __forceinline__ void load_tile_b_shared_swizzle_pre_transposed(mma_tile<16, 16>& a_tile, const half* mat, const int off_base, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(a_tile.x);
    int row = (lane_id/16)*8 + lane_id%8;
    int col = ((lane_id/8)%2)*8;
    int off = off_base + row * stride + col;
    off = off^((off&(S_MASK<<S_BITS))>>S_BITS);
    const half* addr = mat + off;
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
    int row = lane_id%16;
    int col = (lane_id/16)*8;
    const half* addr = mat + row * stride + col;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

static __device__ __forceinline__ void load_tile_b_direct(mma_tile<16, 16>& b_tile, uint32_t addr)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(b_tile.x);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "r"(addr));
}

    template<int S_BITS>
static __device__ __forceinline__ void load_tile_b_shared_swizzle(mma_tile<16, 16>& b_tile, const half* mat, const int off_base, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(b_tile.x);
    int row = lane_id%16;
    int col = (lane_id/16)*8;
    int off = off_base + row * stride + col;
    off = off^((off&(S_MASK<<S_BITS))>>S_BITS);
    const half* addr = mat + off;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16  {%0, %1, %2, %3}, [%4];"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3]) : "l"(addr));
}

static __device__ __forceinline__ void store_matrix(mma_tile<16, 16>& acc, const half* mat, const int stride, const int lane_id)
{
    uint32_t* A = reinterpret_cast<uint32_t*>(acc.x);
    int row = ((lane_id/8)%2) * 8 + lane_id%8;
    int col = ((lane_id/16))*8;
    const half* addr = mat + row * stride + col;
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
