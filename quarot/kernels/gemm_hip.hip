#include <gemm.h>

#include <hip/hip_runtime.h>

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

using I32x8       = int32_t __attribute__((ext_vector_type(8)));
using I32x2       = int32_t __attribute__((ext_vector_type(2)));

static constexpr uint32_t kWmmaK = 32;
static constexpr uint32_t kLaneK = kWmmaK / 2u;

namespace {

#if !defined(QUAROT_BPRE_GFX12) &&                                               \
    (defined(__gfx12__) || defined(__gfx1200__) || defined(__gfx1201__) ||       \
     defined(__gfx12_generic__))
#define QUAROT_BPRE_GFX12 1
#endif

void check_hip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
    }
}

size_t b_prepacked_size_bytes(uint32_t N, uint32_t K)
{
    const uint32_t n_tiles16 = (N + 15u) / 16u;
    const uint32_t k_groups  = K / kWmmaK;
    return static_cast<size_t>(n_tiles16) * k_groups * 2u * 16u * 8u;
}

__device__ __forceinline__ I32x2 unpack_u64_to_i32x2(uint64_t packed)
{
    return {static_cast<int32_t>(static_cast<uint32_t>(packed)),
            static_cast<int32_t>(static_cast<uint32_t>(packed >> 32))};
}

__device__ __forceinline__ I32x2 load_row_i4x16_as_i32x2(const Int4Storage* base)
{
    uint32_t packed0 = 0;
    uint32_t packed1 = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        packed0 |= static_cast<uint32_t>(base[i]) << (8 * i);
        packed1 |= static_cast<uint32_t>(base[4 + i]) << (8 * i);
    }
    return {static_cast<int32_t>(packed0), static_cast<int32_t>(packed1)};
}

__device__ __forceinline__ I32x2 load_pre_i4x16_as_i32x2(const Int4Storage* pre,
                                                         uint32_t tile16,
                                                         uint32_t k_group,
                                                         uint32_t row_blk,
                                                         uint32_t lane16,
                                                         uint32_t k_groups)
{
    const size_t offset =
        (((static_cast<size_t>(tile16) * k_groups + k_group) * 2u + row_blk) * 16u +
         lane16) *
        8u;
    return unpack_u64_to_i32x2(*reinterpret_cast<const uint64_t*>(pre + offset));
}

__device__ __forceinline__ uint64_t load_u64_or_tail(const Int4Storage* base,
                                                     uint32_t available_bytes)
{
    if(available_bytes >= 8u)
    {
        return *reinterpret_cast<const uint64_t*>(base);
    }

    uint64_t packed = 0;
#pragma unroll
    for(uint32_t i = 0; i < 8u; ++i)
    {
        if(i < available_bytes)
        {
            packed |= static_cast<uint64_t>(base[i]) << (8u * i);
        }
    }
    return packed;
}

__global__ void prepack_b_kernel(const Int4Storage* __restrict__ B,
                                 Int4Storage* __restrict__ BPre,
                                 uint32_t N,
                                 uint32_t K)
{
    const uint32_t k_groups = K / kWmmaK;
    const uint32_t n_tiles  = (N + 15u) / 16u;
    const uint32_t units    = n_tiles * k_groups * 2u * 16u;
    const uint32_t idx      = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= units)
    {
        return;
    }

    const uint32_t lane16  = idx & 15u;
    const uint32_t tmp0    = idx >> 4u;
    const uint32_t row_blk = tmp0 & 1u;
    const uint32_t tmp1    = tmp0 >> 1u;
    const uint32_t kg      = tmp1 % k_groups;
    const uint32_t tile    = tmp1 / k_groups;
    const uint32_t n       = tile * 16u + lane16;
    const uint32_t k_bytes = K / 2u;
    const uint32_t byte_k  = kg * 16u + row_blk * 8u;

    uint64_t value = 0;
    if(n < N)
    {
        value = *reinterpret_cast<const uint64_t*>(B + static_cast<size_t>(n) * k_bytes + byte_k);
    }
    *reinterpret_cast<uint64_t*>(BPre + static_cast<size_t>(idx) * 8u) = value;
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave>
__global__ void bpre_kernel(const Int4Storage* __restrict__ A,
                            const Int4Storage* __restrict__ BPre,
                            int32_t* __restrict__ C,
                            uint32_t M,
                            uint32_t N,
                            uint32_t K)
{
#if defined(QUAROT_BPRE_GFX12)
    constexpr uint32_t kWavesPerBlock = BlockMWaves * BlockNWaves;
    const uint32_t wave_id = threadIdx.x >> 5u;
    if(wave_id >= kWavesPerBlock)
    {
        return;
    }

    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t lane16  = lane & 15u;
    const uint32_t row_blk = lane >> 4u;
    const uint32_t wave_m  = wave_id / BlockNWaves;
    const uint32_t wave_n  = wave_id - wave_m * BlockNWaves;

    const uint32_t m_tile_base =
        (blockIdx.y * BlockMWaves * MTilesPerWave + wave_m * MTilesPerWave) * 16u;
    const uint32_t n_tile_base =
        (blockIdx.x * BlockNWaves * NTilesPerWave + wave_n * NTilesPerWave) * 16u;

    const uint32_t k_bytes  = K / 2u;
    const uint32_t k_groups = K / kWmmaK;

    I32x8 acc[MTilesPerWave][NTilesPerWave];
    const I32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
        {
            acc[mt][nt] = zero;
        }
    }

    for(uint32_t k = 0; k < K; k += kWmmaK)
    {
        const uint32_t k_group = k / kWmmaK;
        const uint32_t byte_k  = (k + row_blk * kLaneK) / 2u;

        I32x2 b_frags[NTilesPerWave];
#pragma unroll
        for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
        {
            const uint32_t b_col = n_tile_base + nt * 16u + lane16;
            I32x2 b_frag         = {0, 0};
            if(b_col < N)
            {
                const uint32_t b_tile16 = (n_tile_base + nt * 16u) / 16u;
                b_frag =
                    load_pre_i4x16_as_i32x2(BPre, b_tile16, k_group, row_blk, lane16,
                                            k_groups);
            }
            b_frags[nt] = b_frag;
        }

#pragma unroll
        for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
        {
            const uint32_t a_row = m_tile_base + mt * 16u + lane16;
            I32x2 a_frag         = {0, 0};
            if(a_row < M)
            {
                a_frag =
                    load_row_i4x16_as_i32x2(A + static_cast<size_t>(a_row) * k_bytes +
                                             byte_k);
            }

#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                acc[mt][nt] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                    true, a_frag, true, b_frags[nt], acc[mt][nt], false);
            }
        }
    }

#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t row = m_tile_base + mt * 16u + row_blk * 8u + i;
#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                const uint32_t col = n_tile_base + nt * 16u + lane16;
                if(row < M && col < N)
                {
                    C[static_cast<size_t>(row) * N + col] = acc[mt][nt][i];
                }
            }
        }
    }
#else
    (void)A;
    (void)BPre;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
#endif
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave,
          uint32_t LdsK>
__global__ void lds_swizzle_kernel(const Int4Storage* __restrict__ A,
                                   const Int4Storage* __restrict__ B,
                                   int32_t* __restrict__ C,
                                   uint32_t M,
                                   uint32_t N,
                                   uint32_t K)
{
#if defined(QUAROT_BPRE_GFX12)
    constexpr uint32_t kWavesPerBlock  = BlockMWaves * BlockNWaves;
    constexpr uint32_t kThreadsPerBlock = 32u * kWavesPerBlock;
    constexpr uint32_t kLdsKBytes      = LdsK / 2u;
    constexpr uint32_t kLdsKGroups     = LdsK / kWmmaK;
    constexpr uint32_t kBlockMTiles16  = BlockMWaves * MTilesPerWave;
    constexpr uint32_t kBlockNTiles16  = BlockNWaves * NTilesPerWave;
    constexpr uint32_t kAUnits         = kBlockMTiles16 * kLdsKGroups * 2u * 16u;
    constexpr uint32_t kBUnits         = kBlockNTiles16 * kLdsKGroups * 2u * 16u;

    const uint32_t wave_id = threadIdx.x >> 5u;
    if(wave_id >= kWavesPerBlock)
    {
        return;
    }

    const uint32_t lane    = threadIdx.x & 31u;
    const uint32_t lane16  = lane & 15u;
    const uint32_t row_blk = lane >> 4u;
    const uint32_t wave_m  = wave_id / BlockNWaves;
    const uint32_t wave_n  = wave_id - wave_m * BlockNWaves;

    __shared__ uint64_t s_A_u64[kAUnits];
    __shared__ uint64_t s_B_u64[kBUnits];
    Int4Storage* s_A = reinterpret_cast<Int4Storage*>(s_A_u64);
    Int4Storage* s_B = reinterpret_cast<Int4Storage*>(s_B_u64);

    const uint32_t m_tile_base =
        (blockIdx.y * BlockMWaves * MTilesPerWave + wave_m * MTilesPerWave) * 16u;
    const uint32_t n_tile_base =
        (blockIdx.x * BlockNWaves * NTilesPerWave + wave_n * NTilesPerWave) * 16u;
    const uint32_t block_m_base = blockIdx.y * kBlockMTiles16 * 16u;
    const uint32_t block_n_base = blockIdx.x * kBlockNTiles16 * 16u;
    const uint32_t k_bytes      = K / 2u;

    I32x8 acc[MTilesPerWave][NTilesPerWave];
    const I32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
        {
            acc[mt][nt] = zero;
        }
    }

    for(uint32_t k_base = 0; k_base < K; k_base += LdsK)
    {
        const uint32_t a_units_row = kBlockMTiles16 * 16u * (kLdsKBytes / 8u);
        for(uint32_t idx = threadIdx.x; idx < a_units_row; idx += kThreadsPerBlock)
        {
            const uint32_t u64_per_row  = kLdsKBytes / 8u;
            const uint32_t row_in_block = idx / u64_per_row;
            const uint32_t u64_in_k     = idx - row_in_block * u64_per_row;
            const uint32_t tile16       = row_in_block / 16u;
            const uint32_t lane_in_tile = row_in_block - tile16 * 16u;
            const uint32_t kg_local     = u64_in_k / 2u;
            const uint32_t rb           = u64_in_k & 1u;
            const uint32_t row          = block_m_base + row_in_block;
            const uint32_t global_byte  = k_base / 2u + u64_in_k * 8u;
            uint64_t value              = 0;
            if(row < M && global_byte < k_bytes)
            {
                const uint32_t available =
                    ((k_bytes - global_byte) < 8u) ? (k_bytes - global_byte) : 8u;
                value = load_u64_or_tail(A + static_cast<size_t>(row) * k_bytes + global_byte,
                                         available);
            }
            const uint32_t swiz =
                (((tile16 * kLdsKGroups + kg_local) * 2u + rb) * 16u + lane_in_tile);
            s_A_u64[swiz] = value;
        }

        const uint32_t b_units_row = kBlockNTiles16 * 16u * (kLdsKBytes / 8u);
        for(uint32_t idx = threadIdx.x; idx < b_units_row; idx += kThreadsPerBlock)
        {
            const uint32_t u64_per_col  = kLdsKBytes / 8u;
            const uint32_t col_in_block = idx / u64_per_col;
            const uint32_t u64_in_k     = idx - col_in_block * u64_per_col;
            const uint32_t tile16       = col_in_block / 16u;
            const uint32_t lane_in_tile = col_in_block - tile16 * 16u;
            const uint32_t kg_local     = u64_in_k / 2u;
            const uint32_t rb           = u64_in_k & 1u;
            const uint32_t col          = block_n_base + col_in_block;
            const uint32_t global_byte  = k_base / 2u + u64_in_k * 8u;
            uint64_t value              = 0;
            if(col < N && global_byte < k_bytes)
            {
                const uint32_t available =
                    ((k_bytes - global_byte) < 8u) ? (k_bytes - global_byte) : 8u;
                value = load_u64_or_tail(B + static_cast<size_t>(col) * k_bytes + global_byte,
                                         available);
            }
            const uint32_t swiz =
                (((tile16 * kLdsKGroups + kg_local) * 2u + rb) * 16u + lane_in_tile);
            s_B_u64[swiz] = value;
        }
        __syncthreads();

        const uint32_t tile_k = (K - k_base) < LdsK ? (K - k_base) : LdsK;
        for(uint32_t q = 0; q < tile_k; q += kWmmaK)
        {
            const uint32_t kg_local = q / kWmmaK;
            I32x2 b_frags[NTilesPerWave];
#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                const uint32_t b_tile16 = wave_n * NTilesPerWave + nt;
                b_frags[nt] =
                    load_pre_i4x16_as_i32x2(s_B, b_tile16, kg_local, row_blk, lane16,
                                            kLdsKGroups);
            }

#pragma unroll
            for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
            {
                const uint32_t a_tile16 = wave_m * MTilesPerWave + mt;
                const I32x2 a_frag =
                    load_pre_i4x16_as_i32x2(s_A, a_tile16, kg_local, row_blk, lane16,
                                            kLdsKGroups);
#pragma unroll
                for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
                {
                    acc[mt][nt] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(
                        true, a_frag, true, b_frags[nt], acc[mt][nt], false);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for(uint32_t mt = 0; mt < MTilesPerWave; ++mt)
    {
#pragma unroll
        for(uint32_t i = 0; i < 8u; ++i)
        {
            const uint32_t row = m_tile_base + mt * 16u + row_blk * 8u + i;
#pragma unroll
            for(uint32_t nt = 0; nt < NTilesPerWave; ++nt)
            {
                const uint32_t col = n_tile_base + nt * 16u + lane16;
                if(row < M && col < N)
                {
                    C[static_cast<size_t>(row) * N + col] = acc[mt][nt][i];
                }
            }
        }
    }
#else
    (void)A;
    (void)B;
    (void)C;
    (void)M;
    (void)N;
    (void)K;
#endif
}

enum class DispatchVariant
{
    Kv1024LdsSwizzle,
    N4096LdsSwizzle,
    N11008BPre,
    N14336BPre,
    FallbackBPre,
};

DispatchVariant select_variant(uint32_t N)
{
    switch(N)
    {
    case 1024: return DispatchVariant::Kv1024LdsSwizzle;
    case 4096: return DispatchVariant::N4096LdsSwizzle;
    case 11008: return DispatchVariant::N11008BPre;
    case 14336: return DispatchVariant::N14336BPre;
    default: return DispatchVariant::FallbackBPre;
    }
}

const char* variant_name(DispatchVariant variant)
{
    switch(variant)
    {
    case DispatchVariant::Kv1024LdsSwizzle: return "N=1024 LDS swizzle, block4x4 mt2 nt2 LDS64";
    case DispatchVariant::N4096LdsSwizzle: return "N=4096 LDS swizzle, block4x4 mt2 nt2 LDS64";
    case DispatchVariant::N11008BPre: return "N=11008 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::N14336BPre: return "N=14336 cached B-prepack, block2x2 mt2 nt4";
    case DispatchVariant::FallbackBPre: return "fallback cached B-prepack, block4x4 mt1 nt2";
    }
    return "unknown";
}

bool variant_uses_bpre(DispatchVariant variant)
{
    switch(variant)
    {
    case DispatchVariant::Kv1024LdsSwizzle:
    case DispatchVariant::N4096LdsSwizzle: return false;
    case DispatchVariant::N11008BPre:
    case DispatchVariant::N14336BPre:
    case DispatchVariant::FallbackBPre: return true;
    }
    return true;
}

struct BPrepackKey
{
    const Int4Storage* b;
    uint32_t n;
    uint32_t k;
    int device;

    bool operator==(const BPrepackKey& other) const
    {
        return b == other.b && n == other.n && k == other.k && device == other.device;
    }
};

struct BPrepackKeyHash
{
    size_t operator()(const BPrepackKey& key) const
    {
        size_t h = reinterpret_cast<size_t>(key.b);
        h ^= static_cast<size_t>(key.n) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(key.k) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(key.device) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

struct BPrepackEntry
{
    Int4Storage* ptr = nullptr;
    size_t bytes     = 0;
};

std::mutex g_bpre_mutex;
std::unordered_map<BPrepackKey, BPrepackEntry, BPrepackKeyHash> g_bpre_cache;

void launch_prepack_b(const Int4Storage* B, Int4Storage* BPre, uint32_t N, uint32_t K)
{
    const uint32_t units = static_cast<uint32_t>(b_prepacked_size_bytes(N, K) / 8u);
    const uint32_t block = 256;
    const uint32_t grid  = (units + block - 1u) / block;
   hipLaunchKernelGGL(( prepack_b_kernel), dim3(grid), dim3(block), 0, 0, B, BPre, N, K);
    check_hip(hipGetLastError(), "prepack_b_kernel launch");
}

const Int4Storage* get_or_create_bpre(const Int4Storage* B, uint32_t N, uint32_t K)
{
    int device = 0;
    check_hip(hipGetDevice(&device), "hipGetDevice");

    const BPrepackKey key{B, N, K, device};
    std::lock_guard<std::mutex> lock(g_bpre_mutex);

    auto it = g_bpre_cache.find(key);
    if(it != g_bpre_cache.end())
    {
        return it->second.ptr;
    }

    BPrepackEntry entry;
    entry.bytes = b_prepacked_size_bytes(N, K);
    check_hip(hipMalloc(reinterpret_cast<void**>(&entry.ptr), entry.bytes), "hipMalloc B prepack");
    launch_prepack_b(B, entry.ptr, N, K);

    const auto inserted = g_bpre_cache.emplace(key, entry);
    return inserted.first->second.ptr;
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave>
void launch_bpre(const Int4Storage* A,
                 const Int4Storage* BPre,
                 int32_t* C,
                 uint32_t M,
                 uint32_t N,
                 uint32_t K)
{
    constexpr uint32_t kThreadsPerBlock = 32u * BlockMWaves * BlockNWaves;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + 16u * BlockNWaves * NTilesPerWave - 1u) /
                        (16u * BlockNWaves * NTilesPerWave),
                    (M + 16u * BlockMWaves * MTilesPerWave - 1u) /
                        (16u * BlockMWaves * MTilesPerWave));

   hipLaunchKernelGGL(( bpre_kernel<BlockMWaves, BlockNWaves, MTilesPerWave, NTilesPerWave>)
        , dim3(grid), dim3(block), 0, 0, A, BPre, C, M, N, K);
}

template <uint32_t BlockMWaves,
          uint32_t BlockNWaves,
          uint32_t MTilesPerWave,
          uint32_t NTilesPerWave,
          uint32_t LdsK>
void launch_lds_swizzle(const Int4Storage* A,
                        const Int4Storage* B,
                        int32_t* C,
                        uint32_t M,
                        uint32_t N,
                        uint32_t K)
{
    constexpr uint32_t kThreadsPerBlock = 32u * BlockMWaves * BlockNWaves;
    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + 16u * BlockNWaves * NTilesPerWave - 1u) /
                        (16u * BlockNWaves * NTilesPerWave),
                    (M + 16u * BlockMWaves * MTilesPerWave - 1u) /
                        (16u * BlockMWaves * MTilesPerWave));

   hipLaunchKernelGGL(( lds_swizzle_kernel<BlockMWaves, BlockNWaves, MTilesPerWave, NTilesPerWave, LdsK>)
        , dim3(grid), dim3(block), 0, 0, A, B, C, M, N, K);
}

void dispatch_matmul(const Int4Storage* A,
                     const Int4Storage* BRow,
                     const Int4Storage* BPre,
                     uint32_t M,
                     uint32_t N,
                     uint32_t K,
                     int32_t* C)
{
    if((M % 16) != 0 || (N % 16) != 0 || (K % kWmmaK) != 0)
    {
        throw std::runtime_error("M and N must be multiples of 16; K must be a multiple of 32");
    }

    switch(select_variant(N))
    {
    case DispatchVariant::Kv1024LdsSwizzle:
        launch_lds_swizzle<4, 4, 2, 2, 64>(A, BRow, C, M, N, K);
        break;
    case DispatchVariant::N4096LdsSwizzle:
        launch_lds_swizzle<4, 4, 2, 2, 64>(A, BRow, C, M, N, K);
        break;
    case DispatchVariant::N11008BPre:
        launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        break;
    case DispatchVariant::N14336BPre:
        launch_bpre<2, 2, 2, 4>(A, BPre, C, M, N, K);
        break;
    case DispatchVariant::FallbackBPre:
        launch_bpre<4, 4, 1, 2>(A, BPre, C, M, N, K);
        break;
    }
    check_hip(hipGetLastError(), "dispatch_matmul launch");
}

} // namespace

void matmul_host(const Int4Storage* A,
                 const Int4Storage* B,
                 uint32_t M,
                 uint32_t N,
                 uint32_t K,
                 int32_t* C)
{
#if !defined(QUAROT_BPRE_GFX12)
    (void)A;
    (void)B;
    (void)M;
    (void)N;
    (void)K;
    (void)C;
    throw std::runtime_error("GEMM_int4_for_hip requires gfx12/gfx1201 WMMA int4 support");
#else
    if(A == nullptr || B == nullptr || C == nullptr)
    {
        throw std::invalid_argument("matmul_host received a null device pointer");
    }
    if((M % 16) != 0 || (N % 16) != 0 || (K % kWmmaK) != 0)
    {
        throw std::invalid_argument("M and N must be multiples of 16; K must be a multiple of 32");
    }

    const DispatchVariant variant = select_variant(N);
    const Int4Storage* BPre      = variant_uses_bpre(variant) ? get_or_create_bpre(B, N, K)
                                                               : nullptr;
    dispatch_matmul(A, B, BPre, M, N, K, C);
#endif
}

void clear_prepacked_weight_cache()
{
    std::lock_guard<std::mutex> lock(g_bpre_mutex);
    for(auto& item : g_bpre_cache)
    {
        if(item.second.ptr != nullptr)
        {
            check_hip(hipFree(item.second.ptr), "hipFree cached prepacked B");
        }
    }
    g_bpre_cache.clear();
}

size_t prepacked_weight_cache_bytes()
{
    std::lock_guard<std::mutex> lock(g_bpre_mutex);
    size_t total = 0;
    for(const auto& item : g_bpre_cache)
    {
        total += item.second.bytes;
    }
    return total;
}

const char* selected_kernel_name(uint32_t N)
{
    return variant_name(select_variant(N));
}