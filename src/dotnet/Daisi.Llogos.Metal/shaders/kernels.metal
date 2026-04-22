// Daisi.Llogos.Metal — Metal Shading Language compute kernels.
// Ported from Daisi.Llogos.Vulkan/shaders/*.comp. See comments there for
// the full explanation of each quantization block layout.
//
// Conventions:
//   buffer(0) — output (float*)
//   buffer(1) — input  (float*)  — activation vector(s)
//   buffer(2) — weight (uchar*)  — quantized tensor
//   buffer(3) — params (struct)  — small push-constant-style uniform
//
// Threadgroup sizes vary:
//   * matmul kernels: 64 threads (2 SIMD groups) per output row.
//   * reductions / composite ops: 256 threads per workgroup.
//   * pointwise ops: any power-of-two, typically 256.

#include <metal_stdlib>
using namespace metal;

constant uint TG_SIZE = 64u;
constant uint SIMD_WIDTH = 32u;

// Cross-SIMD-group reduce-add within a threadgroup of up to 64 threads
// (at most 2 SIMD groups of 32). Each thread contributes its partial sum;
// thread 0 in the threadgroup receives the total.
inline float tg_reduce_add_64(float v, uint tid, threadgroup float* scratch)
{
    float s = simd_sum(v);
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_id   = tid / SIMD_WIDTH;
    if (simd_lane == 0) scratch[simd_id] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (tid == 0) total = scratch[0] + scratch[1];
    return total;
}

// 256-thread reduce-add: each thread contributes its partial sum; the
// total lands in thread 0. Uses SIMD-group `simd_sum` then a 2nd pass across
// the 8 SIMD-group partial sums.
inline float tg_reduce_add_256(float v, uint tid, threadgroup float* scratch /*[8]*/)
{
    float s = simd_sum(v);
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_id   = tid / SIMD_WIDTH;
    if (simd_lane == 0) scratch[simd_id] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (tid == 0) {
        total = scratch[0] + scratch[1] + scratch[2] + scratch[3]
              + scratch[4] + scratch[5] + scratch[6] + scratch[7];
    }
    return total;
}

// 256-thread reduce-max
inline float tg_reduce_max_256(float v, uint tid, threadgroup float* scratch /*[8]*/)
{
    float s = simd_max(v);
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_id   = tid / SIMD_WIDTH;
    if (simd_lane == 0) scratch[simd_id] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = -INFINITY;
    if (tid == 0) {
        total = max(max(max(scratch[0], scratch[1]), max(scratch[2], scratch[3])),
                    max(max(scratch[4], scratch[5]), max(scratch[6], scratch[7])));
    }
    return total;
}

// Broadcast value from thread 0 to every thread via shared scratch[0].
inline float tg_broadcast(float v, uint tid, threadgroup float* scratch /*[>=1]*/)
{
    if (tid == 0) scratch[0] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

struct MatMulParams {
    uint M;
    uint K;
    uint N;
};

// ── F32 matmul ────────────────────────────────────────────────────────────
kernel void matmul_f32(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const float*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;
    float acc = 0.0f;
    for (uint k = tid; k < p.K; k += TG_SIZE) {
        acc += weight_data[row * p.K + k] * input_data[k];
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── F16 matmul ────────────────────────────────────────────────────────────
kernel void matmul_f16(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const half*         weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;
    float acc = 0.0f;
    for (uint k = tid; k < p.K; k += TG_SIZE) {
        acc += float(weight_data[row * p.K + k]) * input_data[k];
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── I2_S (BitNet ternary) matmul ─────────────────────────────────────────
// 4 ternary quants per byte in a 128-element interleaved group. Within a
// 32-byte stripe, byte gp's 4 2-bit codes go to element offsets
// {0,1,2,3}*32 + gp. Codes: 0b00=-1, 0b01=0, 0b10=+1, 0b11=0.
// Per-tensor F32 scale stored at offset (N*K/4) at the end of the weight blob.
kernel void matmul_i2s_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NR0 = 2;
    constexpr int NSG = 2;
    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    // Read per-tensor scale (F32) appended after the packed weight region.
    const uint scaleOff = p.N * p.K / 4u;
    const float scale = as_type<float>(uint(weight_data[scaleOff    ])       |
                                       (uint(weight_data[scaleOff + 1]) <<  8) |
                                       (uint(weight_data[scaleOff + 2]) << 16) |
                                       (uint(weight_data[scaleOff + 3]) << 24));

    const uint packedBytesPerRow = p.K / 4u;
    const uint groups = p.K / 128u;
    float sumf[NR0] = {0.0f};

    for (uint g = tiisg; g < groups; g += 32u) {
        device const float* yg = input_data + g * 128u;
        for (int row = 0; row < NR0; ++row) {
            if (first_row + row >= p.N) break;
            device const uchar* bp = weight_data + (first_row + row) * packedBytesPerRow + g * 32u;
            float acc = 0.0f;
            #pragma clang loop unroll(full)
            for (short gp = 0; gp < 32; ++gp) {
                uchar b = bp[gp];
                // Code → float: 0→-1, 1→0, 2→+1, 3→0
                // Using branch-free math: v = (c==0 ? -1 : 0) + (c==2 ? +1 : 0)
                uchar c0 = (b >> 6) & 3u;
                uchar c1 = (b >> 4) & 3u;
                uchar c2 = (b >> 2) & 3u;
                uchar c3 =  b       & 3u;
                float v0 = float(c0 == 2u) - float(c0 == 0u);
                float v1 = float(c1 == 2u) - float(c1 == 0u);
                float v2 = float(c2 == 2u) - float(c2 == 0u);
                float v3 = float(c3 == 2u) - float(c3 == 0u);
                acc += yg[ 0 + gp] * v0;
                acc += yg[32 + gp] * v1;
                acc += yg[64 + gp] * v2;
                acc += yg[96 + gp] * v3;
            }
            sumf[row] += acc;
        }
    }

    for (int row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]) * scale;
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── BF16 matmul ──────────────────────────────────────────────────────────
// bfloat16: top 16 bits of a float32. Convert via left-shift + reinterpret.
// No dedicated bfloat type needed; treated as ushort weight blob.
kernel void matmul_bf16(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const ushort*       weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;
    float acc = 0.0f;
    device const ushort* w = weight_data + row * p.K;
    for (uint k = tid; k < p.K; k += TG_SIZE) {
        float wv = as_type<float>(uint(w[k]) << 16);
        acc += wv * input_data[k];
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// Faster BF16 mv: 4 lanes per output row via simd_sum.
kernel void matmul_bf16_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const ushort*       weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NR0 = 2;
    constexpr int NSG = 2;
    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    float sumf[NR0] = {0.0f};
    // Each lane strides K by 32. For K=4096 each lane covers 128 weights.
    for (uint k = tiisg; k < p.K; k += 32u) {
        float y = input_data[k];
        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            if (first_row + row >= p.N) break;
            ushort bf = weight_data[(first_row + row) * p.K + k];
            float wv = as_type<float>(uint(bf) << 16);
            sumf[row] += wv * y;
        }
    }
    for (int row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Q8_0 matmul ───────────────────────────────────────────────────────────
kernel void matmul_q8_0(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float acc = 0.0f;

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint blockOff = (row * blocksPerRow + b) * 34u;
        ushort scaleBits = uint(weight_data[blockOff])
                        | (uint(weight_data[blockOff + 1]) << 8);
        float scale = float(as_type<half>(scaleBits));

        float blockSum = 0.0f;
        uint aBase = b * 32u;
        for (uint i = 0; i < 32u; ++i) {
            int q = (int)((char)weight_data[blockOff + 2u + i]);
            blockSum += float(q) * input_data[aBase + i];
        }
        acc += scale * blockSum;
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q4_0 matmul, 20-byte aligned blocks (uint32 loads) ────────────────────
// Block layout after repack: [scale(fp16, 2b) | pad(2b) | qs(16b)] = 20 bytes.
// Reading the 16 nibble bytes as 4 uint32s cuts weight-load instructions by
// ~4× compared to the per-byte path. Also lets us fuse nibble extraction and
// signed-quant conversion into vector math.
kernel void matmul_q4_0_aligned(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],  // 5 uints per block
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float acc = 0.0f;

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint wordBase = (row * blocksPerRow + b) * 5u;          // 5 uints/block
        uint aBase    = b * 32u;

        // Word 0: low 16 bits = fp16 scale, high 16 bits = padding.
        uint w0 = weight_u32[wordBase];
        float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

        // Words 1..4: 16 bytes of packed nibbles.
        // Nibble layout within byte i: lo = element[i], hi = element[i+16].
        float blockSum = 0.0f;
        for (uint wi = 0; wi < 4u; ++wi) {
            uint packed = weight_u32[wordBase + 1u + wi];
            // Extract four bytes, each with lo/hi nibble.
            float4 lo = float4(
                float(int((packed >>  0) & 0xFu) - 8),
                float(int((packed >>  8) & 0xFu) - 8),
                float(int((packed >> 16) & 0xFu) - 8),
                float(int((packed >> 24) & 0xFu) - 8));
            float4 hi = float4(
                float(int((packed >>  4) & 0xFu) - 8),
                float(int((packed >> 12) & 0xFu) - 8),
                float(int((packed >> 20) & 0xFu) - 8),
                float(int((packed >> 28) & 0xFu) - 8));
            uint loBase = aBase + wi * 4u;
            uint hiBase = aBase + wi * 4u + 16u;
            float4 aLo = float4(input_data[loBase + 0], input_data[loBase + 1],
                                 input_data[loBase + 2], input_data[loBase + 3]);
            float4 aHi = float4(input_data[hiBase + 0], input_data[hiBase + 1],
                                 input_data[hiBase + 2], input_data[hiBase + 3]);
            blockSum += dot(lo, aLo) + dot(hi, aHi);
        }
        acc += scale * blockSum;
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Fused FFN: gate + up + SwiGLU, Q4_0 aligned, 2-SIMD-group × 4-row ─────
// Same parallelisation pattern as matmul_q4_0_aligned_2x4row — 64 threads,
// 8 output rows per TG, no threadgroup scratch.
kernel void matmul_q4_0_aligned_swiglu_2x4row(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uint*         gate_u32    [[buffer(2)]],
    device const uint*         up_u32      [[buffer(3)]],
    constant     MatMulParams& p           [[buffer(4)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float gateAcc[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float upAcc  [ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = input_vec4[aVecBase + i];

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;

            // gate weight
            {
                uint w0 = gate_u32[wordBase];
                float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));
                float blockSum = 0.0f;
                for (uint wi = 0; wi < 4u; ++wi) {
                    uint packed = gate_u32[wordBase + 1u + wi];
                    float4 lo = float4(
                        float(int((packed >>  0) & 0xFu) - 8),
                        float(int((packed >>  8) & 0xFu) - 8),
                        float(int((packed >> 16) & 0xFu) - 8),
                        float(int((packed >> 24) & 0xFu) - 8));
                    float4 hi = float4(
                        float(int((packed >>  4) & 0xFu) - 8),
                        float(int((packed >> 12) & 0xFu) - 8),
                        float(int((packed >> 20) & 0xFu) - 8),
                        float(int((packed >> 28) & 0xFu) - 8));
                    blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
                }
                gateAcc[r] += scale * blockSum;
            }
            // up weight
            {
                uint w0 = up_u32[wordBase];
                float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));
                float blockSum = 0.0f;
                for (uint wi = 0; wi < 4u; ++wi) {
                    uint packed = up_u32[wordBase + 1u + wi];
                    float4 lo = float4(
                        float(int((packed >>  0) & 0xFu) - 8),
                        float(int((packed >>  8) & 0xFu) - 8),
                        float(int((packed >> 16) & 0xFu) - 8),
                        float(int((packed >> 24) & 0xFu) - 8));
                    float4 hi = float4(
                        float(int((packed >>  4) & 0xFu) - 8),
                        float(int((packed >> 12) & 0xFu) - 8),
                        float(int((packed >> 20) & 0xFu) - 8),
                        float(int((packed >> 28) & 0xFu) - 8));
                    blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
                }
                upAcc[r] += scale * blockSum;
            }
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float g = simd_sum(gateAcc[r]);
        float u = simd_sum(upAcc[r]);
        if (lane == 0 && rowBase + r < p.N) {
            output_data[rowBase + r] = u * (g / (1.0f + exp(-g)));
        }
    }
}

// ── Fused FFN: gate matmul + up matmul + SwiGLU, Q4_0 aligned, 4-row ──────
// out[r] = silu(gate_w[r] · a) * (up_w[r] · a). Saves two entire matmul
// dispatches per FFN layer (plus the intermediate _gate/_up writes + reads).
kernel void matmul_q4_0_aligned_swiglu_4row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         gate_u32    [[buffer(2)]],
    device const uint*         up_u32      [[buffer(3)]],
    constant     MatMulParams& p           [[buffer(4)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    threadgroup float scratch[2];
    uint rowBase = tgid * ROWS;
    if (rowBase >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float gateAcc[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float upAcc  [ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint aBase = b * 32u;
        // Load 32 activation values once (reused 8 times: 4 rows × 2 weights).
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) {
            aVec[i] = float4(input_data[aBase + i*4 + 0], input_data[aBase + i*4 + 1],
                             input_data[aBase + i*4 + 2], input_data[aBase + i*4 + 3]);
        }

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;

            // ── gate weight ─────────────────────────────────────────────
            {
                uint w0 = gate_u32[wordBase];
                float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));
                float blockSum = 0.0f;
                for (uint wi = 0; wi < 4u; ++wi) {
                    uint packed = gate_u32[wordBase + 1u + wi];
                    float4 lo = float4(
                        float(int((packed >>  0) & 0xFu) - 8),
                        float(int((packed >>  8) & 0xFu) - 8),
                        float(int((packed >> 16) & 0xFu) - 8),
                        float(int((packed >> 24) & 0xFu) - 8));
                    float4 hi = float4(
                        float(int((packed >>  4) & 0xFu) - 8),
                        float(int((packed >> 12) & 0xFu) - 8),
                        float(int((packed >> 20) & 0xFu) - 8),
                        float(int((packed >> 28) & 0xFu) - 8));
                    blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
                }
                gateAcc[r] += scale * blockSum;
            }
            // ── up weight ───────────────────────────────────────────────
            {
                uint w0 = up_u32[wordBase];
                float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));
                float blockSum = 0.0f;
                for (uint wi = 0; wi < 4u; ++wi) {
                    uint packed = up_u32[wordBase + 1u + wi];
                    float4 lo = float4(
                        float(int((packed >>  0) & 0xFu) - 8),
                        float(int((packed >>  8) & 0xFu) - 8),
                        float(int((packed >> 16) & 0xFu) - 8),
                        float(int((packed >> 24) & 0xFu) - 8));
                    float4 hi = float4(
                        float(int((packed >>  4) & 0xFu) - 8),
                        float(int((packed >> 12) & 0xFu) - 8),
                        float(int((packed >> 20) & 0xFu) - 8),
                        float(int((packed >> 28) & 0xFu) - 8));
                    blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
                }
                upAcc[r] += scale * blockSum;
            }
        }
    }

    // Reduce across threads; thread 0 applies silu(gate)*up and writes out.
    for (uint r = 0; r < ROWS; ++r) {
        float g = tg_reduce_add_64(gateAcc[r], tid, scratch);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float u = tg_reduce_add_64(upAcc[r],   tid, scratch);
        if (tid == 0 && rowBase + r < p.N) {
            output_data[rowBase + r] = u * (g / (1.0f + exp(-g)));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Q4_0 aligned, 8 rows per TG (more activation reuse) ───────────────────
kernel void matmul_q4_0_aligned_8row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    constexpr uint ROWS = 8u;
    threadgroup float scratch[2];
    uint rowBase = tgid * ROWS;
    if (rowBase >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint aBase = b * 32u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) {
            aVec[i] = float4(input_data[aBase + i*4 + 0], input_data[aBase + i*4 + 1],
                             input_data[aBase + i*4 + 2], input_data[aBase + i*4 + 3]);
        }
        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = tg_reduce_add_64(accs[r], tid, scratch);
        if (tid == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Q4_0 aligned, 1 row per 32-thread SIMD group ──────────────────────────
// No threadgroup memory, no barriers — reduction via `simd_sum` only.
// For K=4096 each lane processes 4 blocks. Trades activation reuse for
// smaller TGs (higher per-core occupancy).
kernel void matmul_q4_0_aligned_simd(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    uint row = tgid;
    if (row >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float acc = 0.0f;

    for (uint b = tid; b < blocksPerRow; b += 32u) {
        uint wordBase = (row * blocksPerRow + b) * 5u;
        uint aBase    = b * 32u;

        uint w0 = weight_u32[wordBase];
        float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

        float blockSum = 0.0f;
        for (uint wi = 0; wi < 4u; ++wi) {
            uint packed = weight_u32[wordBase + 1u + wi];
            float4 lo = float4(
                float(int((packed >>  0) & 0xFu) - 8),
                float(int((packed >>  8) & 0xFu) - 8),
                float(int((packed >> 16) & 0xFu) - 8),
                float(int((packed >> 24) & 0xFu) - 8));
            float4 hi = float4(
                float(int((packed >>  4) & 0xFu) - 8),
                float(int((packed >> 12) & 0xFu) - 8),
                float(int((packed >> 20) & 0xFu) - 8),
                float(int((packed >> 28) & 0xFu) - 8));
            uint loBase = aBase + wi * 4u;
            uint hiBase = aBase + wi * 4u + 16u;
            float4 aLo = float4(input_data[loBase + 0], input_data[loBase + 1],
                                 input_data[loBase + 2], input_data[loBase + 3]);
            float4 aHi = float4(input_data[hiBase + 0], input_data[hiBase + 1],
                                 input_data[hiBase + 2], input_data[hiBase + 3]);
            blockSum += dot(lo, aLo) + dot(hi, aHi);
        }
        acc += scale * blockSum;
    }

    float total = simd_sum(acc);
    if (tid == 0) output_data[row] = total;
}

// ── Q4_0 aligned 2x4row with FULL activation preload to threadgroup mem ──
// Loads all K activation floats into tgmem once at kernel start (single
// barrier), then compute reads from tgmem only. For K=4096 the activation
// is 16 KB which fits in tgmem (up to 32 KB on Apple Silicon).
kernel void matmul_q4_0_aligned_2x4row_preload(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    // Up to K=4096 floats = 1024 float4. Declare fixed-size to let the
    // compiler allocate the right amount of tgmem.
    threadgroup float4 tgAct[1024];

    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    uint kVec4Count   = p.K / 4u;  // number of float4s in K floats

    // Cooperative load: 64 threads load K/4 float4s. Thread t loads slots
    // [t, t+64, t+128, ...]. For K=4096 → 1024 float4 / 64 threads = 16 per thread.
    for (uint s = tid; s < kVec4Count; s += 64u) {
        tgAct[s] = input_vec4[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint actBase = b * 8u;  // 8 float4s per block

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, tgAct[actBase + wi])
                         +  dot(hi, tgAct[actBase + wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
    }
}

// ── Q4_0 aligned 2x4row with threadgroup-memory activation tiling ─────────
// Both SIMD groups in a 64-thread TG currently load the same activation
// chunk independently (one copy per SIMD group). This kernel loads each
// activation into threadgroup memory exactly once, cutting activation
// bandwidth in half. Activation is processed in tiles of 32 blocks (1024
// floats / 4 KB of tgmem).
kernel void matmul_q4_0_aligned_2x4row_tiled(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    threadgroup float4 tgAct[256];  // 32 blocks × 8 float4 per block

    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    // Process 32 blocks per tile. For K = 4096 (128 blocks) that's 4 tiles.
    for (uint iterBase = 0; iterBase < blocksPerRow; iterBase += 32u) {
        // Cooperative load of 256 float4s = 1024 floats into tgmem.
        // 64 threads × 4 float4 each = 256 float4. Thread t loads slots [t*4 .. t*4+3].
        uint loadStart = tid * 4u;
        for (uint i = 0; i < 4u; ++i) {
            uint slot = loadStart + i;
            if (slot >= 256u) break;
            uint blockIdx = iterBase + slot / 8u;
            uint within   = slot % 8u;
            if (blockIdx < blocksPerRow) {
                tgAct[slot] = input_vec4[blockIdx * 8u + within];
            } else {
                tgAct[slot] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: thread `lane` in each SIMD group handles block (iterBase + lane).
        uint b = iterBase + lane;
        if (b < blocksPerRow) {
            uint actBase = lane * 8u;  // 8 float4s (32 floats) per block in tgAct

            for (uint r = 0; r < ROWS; ++r) {
                uint row = rowBase + r;
                if (row >= p.N) break;
                uint wordBase = (row * blocksPerRow + b) * 5u;
                uint w0 = weight_u32[wordBase];
                float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

                float blockSum = 0.0f;
                for (uint wi = 0; wi < 4u; ++wi) {
                    uint packed = weight_u32[wordBase + 1u + wi];
                    float4 lo = float4(
                        float(int((packed >>  0) & 0xFu) - 8),
                        float(int((packed >>  8) & 0xFu) - 8),
                        float(int((packed >> 16) & 0xFu) - 8),
                        float(int((packed >> 24) & 0xFu) - 8));
                    float4 hi = float4(
                        float(int((packed >>  4) & 0xFu) - 8),
                        float(int((packed >> 12) & 0xFu) - 8),
                        float(int((packed >> 20) & 0xFu) - 8),
                        float(int((packed >> 28) & 0xFu) - 8));
                    blockSum += dot(lo, tgAct[actBase + wi])
                             +  dot(hi, tgAct[actBase + wi + 4u]);
                }
                accs[r] += scale * blockSum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
    }
}

// ── Q4_0 aligned, 16 rows per TG, 4 SIMD groups × 4 rows each ─────────────
kernel void matmul_q4_0_aligned_4x4row(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * 16u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = input_vec4[aVecBase + i];

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
    }
}

// ── Q4_0 aligned, 8 rows per TG, 2 SIMD groups × 4 rows each ──────────────
// Mirrors llama.cpp's `kernel_mul_mv_q4_0_f32_impl` pattern: 64-thread TG
// with each SIMD group handling its own 4 rows independently. No cross-SIMD
// reduction, no threadgroup scratch. More rows per TG = fewer launches.
kernel void matmul_q4_0_aligned_2x4row(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = input_vec4[aVecBase + i];

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
    }
}

// ── Q4_0 aligned, 4 rows per TG, 64 threads (2 SIMD groups) ───────────────
// Two SIMD groups split the K dimension — each thread does 2 blocks instead
// of 4. Cross-SIMD reduction uses threadgroup scratch.
kernel void matmul_q4_0_aligned_simd2_4row(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    threadgroup float scratch[2];
    uint rowBase = tgid * ROWS;
    if (rowBase >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = tid; b < blocksPerRow; b += 64u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = input_vec4[aVecBase + i];

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = tg_reduce_add_64(accs[r], tid, scratch);
        if (tid == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Q4_0 aligned, 4 rows per SIMD group (32 threads, no scratch) ──────────
// Each SIMD group (32 threads) handles 4 rows. simd_sum reduction, no
// threadgroup memory or barriers. Activation loaded once per block iteration
// and reused across all 4 rows.
kernel void matmul_q4_0_aligned_simd_4row(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],  // float4-typed view
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * ROWS;
    if (rowBase >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = tid; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) {
            aVec[i] = input_vec4[aVecBase + i];
        }
        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (tid == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
    }
}

// ── Q4_0 aligned, 4 rows per TG (activation reuse) ────────────────────────
kernel void matmul_q4_0_aligned_4row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    threadgroup float scratch[2];
    uint rowBase = tgid * ROWS;
    if (rowBase >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint aBase = b * 32u;

        // Load 32 activation values once (shared across the 4 rows).
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) {
            aVec[i] = float4(input_data[aBase + i*4 + 0], input_data[aBase + i*4 + 1],
                             input_data[aBase + i*4 + 2], input_data[aBase + i*4 + 3]);
        }

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = tg_reduce_add_64(accs[r], tid, scratch);
        if (tid == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// ── Q8_0 matrix-vector (llama.cpp mul_mv_q8_0 pattern) ────────────────────
// 4 workers cooperate per 32-weight block (NQ=8 int8 weights per worker).
// Q8_0 blocks are 34 bytes: 2-byte half scale then 32 int8 weights.
kernel void matmul_q8_0_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // NSG=1: each simdgroup covers ALL of K for its assigned rows. With
    // NSG>1 and per-sg row partitioning, different sgs cover disjoint K
    // stripes for their own rows, which misses half the dot product.
    // (The same pattern works in llama.cpp only because they use different
    // sg→row mapping for higher NSG or reduce across sgs.)
    constexpr int  NR0 = 2;
    constexpr int  NSG = 1;
    constexpr int  NQ  = 8;
    constexpr uint BYTES_PER_BLOCK = 34u;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short ix = tiisg / 4;       // 0..7 — worker within block-group
    const short il = tiisg % 4;       // 0..3 — which 8-weight quarter

    const uint nb = p.K / 32u;
    const uint ib0 = ix;

    device const uchar* wx[NR0];
    #pragma clang loop unroll(full)
    for (int row = 0; row < NR0; ++row) {
        wx[row] = weight_data + (first_row + row) * nb * BYTES_PER_BLOCK;
    }

    float sumf[NR0] = {0.0f};
    float yl[NQ];
    device const float* yb = input_data + ib0 * 32 + il * NQ;

    for (uint ib = ib0; ib < nb; ib += NQ) {
        #pragma clang loop unroll(full)
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];

        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            if (first_row + row >= p.N) break;
            device const uchar* blk = wx[row] + ib * BYTES_PER_BLOCK;
            ushort dBits = uint(blk[0]) | (uint(blk[1]) << 8);
            float d = float(as_type<half>(dBits));
            device const char* qs = (device const char*)(blk + 2) + il * NQ;
            float sumq = 0.0f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < NQ; ++i) {
                sumq += float(qs[i]) * yl[i];
            }
            sumf[row] += d * sumq;
        }
        yb += NQ * 32;
    }

    for (int row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Q5_0 matmul (simple, 1 row per TG) — fallback/reference ───────────────
// Direct dequant: byte i of qs holds weight[i] (low nibble) + weight[i+16]
// (high nibble). qh bit i is the high bit for weight i, qh bit (i+16) for
// weight i+16. Value = (low4 | (high_bit<<4)) - 16, then × d.
kernel void matmul_q5_0(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    if (tgid >= p.N) return;
    uint row = tgid;
    uint nb = p.K / 32u;
    float acc = 0.0f;
    device const uchar* rowBase = weight_data + row * nb * 22u;
    for (uint b = tid; b < nb; b += TG_SIZE) {
        device const uchar* blk = rowBase + b * 22u;
        ushort dBits = uint(blk[0]) | (uint(blk[1]) << 8);
        float d = float(as_type<half>(dBits));
        uint qh = uint(blk[2]) | (uint(blk[3]) << 8) | (uint(blk[4]) << 16) | (uint(blk[5]) << 24);
        device const float* y = input_data + b * 32u;
        for (uint i = 0; i < 16u; ++i) {
            uchar byte = blk[6 + i];
            int low  = int(byte & 0xFu) | (int((qh >> i)          & 1u) << 4);
            int high = int(byte >> 4)   | (int((qh >> (i + 16u)) & 1u) << 4);
            acc += y[i     ] * float(low  - 16) * d;
            acc += y[i + 16] * float(high - 16) * d;
        }
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q5_0 matrix-vector (llama.cpp mul_vec_q_n pattern) ────────────────────
// Q5_0 blocks are 22 bytes per 32 weights: 2b d, 4b qh (high bit per
// weight), 16b qs (low 4 bits per weight). Value = (low4 | (high_bit << 4))
// - 16, then × d. Uses yl pre-scaling trick like Q4_0 mv.
kernel void matmul_q5_0_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int  NR0 = 4;
    constexpr int  NSG = 2;
    constexpr int  NQ  = 16;
    constexpr uint BYTES_PER_BLOCK = 22u;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short ix = tiisg / 2;
    const short il = (tiisg % 2) * 8;

    const uint nb = p.K / 32u;

    device const uchar* wx[NR0];
    #pragma clang loop unroll(full)
    for (int row = 0; row < NR0; ++row) {
        wx[row] = weight_data + (first_row + row) * nb * BYTES_PER_BLOCK;
    }

    float sumf[NR0] = {0.0f};
    float yl[16];
    device const float* yb = input_data + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1] / 256.0f;
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.0f;
            yl[i + 9] = yb[i + 17] / 4096.0f;
        }
        const float sumyAll = sumy0 + sumy1;

        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            if (first_row + row >= p.N) break;
            device const uchar* blk = wx[row] + ib * BYTES_PER_BLOCK;
            ushort dBits = uint(blk[0]) | (uint(blk[1]) << 8);
            float d = float(as_type<half>(dBits));
            // qh: 32-bit mask of high bits (one per weight).
            uint qh = uint(blk[2]) | (uint(blk[3]) << 8) |
                      (uint(blk[4]) << 16) | (uint(blk[5]) << 24);
            // qs as uint16* starting at byte 6, offset by il/2.
            device const ushort* qs = (device const ushort*)(blk + 6) + il / 2;

            float4 acc = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i += 2) {
                ushort q = qs[i / 2];
                // Pull the high bit into the nibble position and OR with the
                // stored low nibble. Bit (i+il) of qh maps to weight i+il
                // (low half, bytes 0..15). Bit (i+il+16) maps to weights in
                // the high half (nibble-to-weights 16..31).
                acc[0] += yl[i + 0] * float((q & 0x000F) | (((qh >> (i + 0 + il     )) << 4 ) & 0x0010));
                acc[1] += yl[i + 1] * float((q & 0x0F00) | (((qh >> (i + 1 + il     )) << 12) & 0x1000));
                acc[2] += yl[i + 8] * float((q & 0x00F0) | (((qh >> (i + 0 + il + 16)) << 8 ) & 0x0100));
                acc[3] += yl[i + 9] * float((q & 0xF000) | (((qh >> (i + 1 + il + 16)) << 16) & 0x10000));
            }
            // Q5_0 dequant: value = 5bit - 16, so sum(y*w) = d*(sum(y*bits) - 16*sumy)
            sumf[row] += d * (sumyAll * -16.0f + acc[0] + acc[1] + acc[2] + acc[3]);
        }

        yb += NQ * 32;
    }

    for (int row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Q4_1 matrix-vector (llama.cpp mul_vec_q_n pattern) ────────────────────
// Same 16-workers-per-super-block structure as matmul_q4_0_aligned_mv, but
// the dequant rule is  w = d*nibble + m  (not  d*(nibble - 8)), so the
// final combination is  d*∑acc + m*sumy.
kernel void matmul_q4_1_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int  NR0 = 4;
    constexpr int  NSG = 2;
    constexpr int  NQ  = 16;
    constexpr uint BYTES_PER_BLOCK = 20u;  // 2 d + 2 m + 16 qs

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short ix = tiisg / 2;
    const short il = (tiisg % 2) * 8;

    const uint nb = p.K / 32u;

    device const uchar* wx[NR0];
    #pragma clang loop unroll(full)
    for (int row = 0; row < NR0; ++row) {
        wx[row] = weight_data + (first_row + row) * nb * BYTES_PER_BLOCK;
    }

    float sumf[NR0] = {0.0f};
    float yl[16];
    device const float* yb = input_data + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1] / 256.0f;
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.0f;
            yl[i + 9] = yb[i + 17] / 4096.0f;
        }
        const float sumyAll = sumy0 + sumy1;

        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            if (first_row + row >= p.N) break;
            device const uchar* blk = wx[row] + ib * BYTES_PER_BLOCK;
            ushort dBits = uint(blk[0]) | (uint(blk[1]) << 8);
            ushort mBits = uint(blk[2]) | (uint(blk[3]) << 8);
            float d = float(as_type<half>(dBits));
            float m = float(as_type<half>(mBits));
            // qs as uint16* — nibble bytes start at byte 4, offset by il.
            device const ushort* qs = (device const ushort*)(blk + 4) + il / 2;
            float4 acc = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i += 2) {
                ushort q = qs[i / 2];
                acc[0] += yl[i + 0] * float(q & 0x000Fu);
                acc[1] += yl[i + 1] * float(q & 0x0F00u);
                acc[2] += yl[i + 8] * float(q & 0x00F0u);
                acc[3] += yl[i + 9] * float(q & 0xF000u);
            }
            sumf[row] += d * (acc[0] + acc[1] + acc[2] + acc[3]) + m * sumyAll;
        }

        yb += NQ * 32;
    }

    for (int row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Q4_0 aligned matrix-vector (llama.cpp mul_vec_q_n pattern) ────────────
// 16 workers per super-block (stride-16 ix interleaves) reduce at the
// simdgroup level via simd_sum. Yl pre-scaling trick: instead of
// right-shifting the high nibbles, pre-divide y values so (qs & 0xF000) etc.
// multiplied by pre-scaled yl gives the correct weight × y. Saves per-weight
// shift ops in the inner loop.
kernel void matmul_q4_0_aligned_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NR0 = 4;
    constexpr int NSG = 2;
    constexpr int NQ  = 16;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short ix = tiisg / 2;         // 0..15
    const short il = (tiisg % 2) * 8;    // 0 or 8

    const uint nb = p.K / 32u;
    const uint u32_per_row = nb * 5u;

    // Per-row weight pointers (aligned Q4_0 = 5 uint32 per 32-weight block).
    device const uint* wx[NR0];
    #pragma clang loop unroll(full)
    for (int row = 0; row < NR0; ++row) {
        wx[row] = weight_u32 + (first_row + row) * u32_per_row;
    }

    float sumf[NR0] = {0.0f};
    float yl[16];
    device const float* yb = input_data + ix * 32 + il;

    for (uint ib = ix; ib < nb; ib += NQ) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1] / 256.0f;
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] / 16.0f;
            yl[i + 9] = yb[i + 17] / 4096.0f;
        }
        const float sumyAll = sumy0 + sumy1;

        #pragma clang loop unroll(full)
        for (int row = 0; row < NR0; ++row) {
            if (first_row + row >= p.N) break;
            device const uint* blk = wx[row] + ib * 5u;
            float d = float(as_type<half>(ushort(blk[0] & 0xFFFFu)));
            // qs as uint16* — skip 2 uint16s (scale + pad), then offset by il/2.
            device const ushort* qs = (device const ushort*)blk + 2 + il / 2;
            float4 acc = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i += 2) {
                ushort q = qs[i / 2];
                acc[0] += yl[i + 0] * float(q & 0x000Fu);
                acc[1] += yl[i + 1] * float(q & 0x0F00u);
                acc[2] += yl[i + 8] * float(q & 0x00F0u);
                acc[3] += yl[i + 9] * float(q & 0xF000u);
            }
            sumf[row] += d * (sumyAll * -8.0f + acc[0] + acc[1] + acc[2] + acc[3]);
        }

        yb += NQ * 32;
    }

    for (int row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Batched matmul for prefill — Q4_0 aligned × F32 activations ───────────
// ───────────────────────────────────────────────────────────────────────────
//
// Output [M, N] = A [M, K] × B^T where B is stored as [N, K] in aligned Q4_0
// (20 bytes per 32-weight block). Intended for prefill (M > 1).
//
// Tile layout (matches llama.cpp's kernel_mul_mm pattern):
//   BM = BN = BK = 32, with 4 simdgroups arranged 2×2 (128 threads / TG).
//   Each simdgroup owns a 16×16 output tile = 2×2 array of simdgroup_float8x8
//   accumulators, using Apple7's simdgroup matrix multiply instruction.
//   Grid: (N/32, ⌈M/32⌉, 1).
//
// Threadgroup memory: sA[32×32] + sB[32×32] + sC[32×32] in float = 12 KiB.
//
// Weight load phase dequantizes 32 Q4_0 blocks per TG iteration (one per
// output row in the BN-slice) and writes them transposed into sB as
// sB[k*BN + n].
//
// Activation load phase streams 32×32 floats from A[bm0..bm0+31, kt..kt+31]
// into sA, zero-padding rows beyond M.
//
// Store phase writes accumulators through sC into output[bm0..bm0+31,
// bn0..bn0+31], bounds-checking the M dimension.
#include <metal_simdgroup_matrix>

// Mixed-precision variant (llama.cpp style): half operands × half operands
// → FLOAT accumulator. Apple GPUs execute half-operand simdgroup matrix
// multiply at 2× float throughput while accumulation stays numerically
// precise in float. Output is F32 direct (no conversion needed).
kernel void matmul_mm_q4_0_aligned_h(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint3 tg_id    [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  sg_idx   [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BM = 64u;
    constexpr uint BN = 32u;
    constexpr uint BK = 32u;

    const uint bm0 = tg_id.y * BM;
    const uint bn0 = tg_id.x * BN;
    const uint M = p.M;
    const uint K = p.K;
    const uint N = p.N;
    const uint blocks_per_row_u32 = (K / 32u) * 5u;

    const uint sg_row = sg_idx >> 1;
    const uint sg_col = sg_idx & 1u;
    const uint sg_m_base = sg_row * 32u;
    const uint sg_n_base = sg_col * 16u;

    // Float accumulators for precise reduction across K. Half operands drive
    // the 2× simdgroup-matrix throughput on Apple GPUs.
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup half sA[BM * BK];   // 64×32 = 4096 bytes
    threadgroup half sB[BK * BN];   // 32×32 = 2048 bytes

    for (uint kt = 0; kt < K; kt += BK) {
        // ── Load A[64×32] → sA as half ───────────────────────────────────
        {
            const uint row_lo = tid >> 2;
            const uint col    = (tid & 3u) * 8u;
            const uint k = kt + col;
            for (uint part = 0u; part < 2u; ++part) {
                const uint row = row_lo + part * 32u;
                const uint m = bm0 + row;
                half v0, v1, v2, v3, v4, v5, v6, v7;
                if (m < M) {
                    device const float* src = input_data + m * K + k;
                    v0 = half(src[0]); v1 = half(src[1]); v2 = half(src[2]); v3 = half(src[3]);
                    v4 = half(src[4]); v5 = half(src[5]); v6 = half(src[6]); v7 = half(src[7]);
                } else {
                    v0 = v1 = v2 = v3 = v4 = v5 = v6 = v7 = 0.0h;
                }
                threadgroup half* dst = &sA[row * BK + col];
                dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
                dst[4] = v4; dst[5] = v5; dst[6] = v6; dst[7] = v7;
            }
        }

        // ── Load+dequant B[32×32] as half → sB[k*BN + n] ─────────────────
        // Vectorized: treat 2 packed u32s as uchar4+uchar4, extract 8 nibbles
        // via shift/mask on vectors, and convert to half4 in one expression.
        // Collapses ~40 scalar ops per thread to ~10 vector ops.
        {
            const uint n_local = tid >> 2;
            const uint sub     = tid & 3u;
            const uint n_global = bn0 + n_local;
            const uint block_idx = kt / 32u;
            device const uint* blk = weight_u32 + n_global * blocks_per_row_u32 + block_idx * 5u;
            half scale = as_type<half>(ushort(blk[0] & 0xFFFFu));
            const uint widx0 = 1u + ((sub & 1u) * 2u);
            uchar4 b0 = as_type<uchar4>(blk[widx0]);
            uchar4 b1 = as_type<uchar4>(blk[widx0 + 1u]);
            uchar shift = (sub < 2u) ? uchar(0) : uchar(4);
            // Extract 4 nibbles (either low or high) from each 4-byte group.
            int4 v0 = int4((b0 >> shift) & uchar4(0xFu)) - int4(8);
            int4 v1 = int4((b1 >> shift) & uchar4(0xFu)) - int4(8);
            half4 w_a = half4(float4(v0)) * scale;
            half4 w_b = half4(float4(v1)) * scale;
            const uint k_base = sub * 8u;
            sB[(k_base + 0u) * BN + n_local] = w_a.x;
            sB[(k_base + 1u) * BN + n_local] = w_a.y;
            sB[(k_base + 2u) * BN + n_local] = w_a.z;
            sB[(k_base + 3u) * BN + n_local] = w_a.w;
            sB[(k_base + 4u) * BN + n_local] = w_b.x;
            sB[(k_base + 5u) * BN + n_local] = w_b.y;
            sB[(k_base + 6u) * BN + n_local] = w_b.z;
            sB[(k_base + 7u) * BN + n_local] = w_b.w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Following llama.cpp's kernel_mul_mm: simdgroup_barrier between loads
        // and multiply lets the scheduler pipeline the loads, and the outer
        // threadgroup_barrier alone (at the top of next iter's load phase) is
        // sufficient to synchronize writes-after-reads.
        simdgroup_half8x8 mA0, mA1, mA2, mA3, mB0, mB1;
        #pragma clang loop unroll(full)
        for (uint k_sub = 0u; k_sub < BK; k_sub += 8u) {
            simdgroup_load(mA0, &sA[(sg_m_base + 0u ) * BK + k_sub], BK);
            simdgroup_load(mA1, &sA[(sg_m_base + 8u ) * BK + k_sub], BK);
            simdgroup_load(mA2, &sA[(sg_m_base + 16u) * BK + k_sub], BK);
            simdgroup_load(mA3, &sA[(sg_m_base + 24u) * BK + k_sub], BK);
            simdgroup_load(mB0, &sB[k_sub * BN + sg_n_base + 0u], BN);
            simdgroup_load(mB1, &sB[k_sub * BN + sg_n_base + 8u], BN);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(acc[0][0], mA0, mB0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], mA0, mB1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], mA1, mB0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], mA1, mB1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[2][0], mA2, mB0, acc[2][0]);
            simdgroup_multiply_accumulate(acc[2][1], mA2, mB1, acc[2][1]);
            simdgroup_multiply_accumulate(acc[3][0], mA3, mB0, acc[3][0]);
            simdgroup_multiply_accumulate(acc[3][1], mA3, mB1, acc[3][1]);
        }
    }

    // ── Store float accumulators → output ────────────────────────────────
    // Fully in-bounds TGs write directly from simdgroup_float8x8 to device
    // memory — skips threadgroup staging and the write-out loop entirely.
    // Tail TGs (M not divisible by BM) fall back to sC staging.
    const bool inBounds = (bm0 + BM <= M) && (bn0 + BN <= N);
    if (inBounds) {
        device float* outBase = output_data + (bm0 + sg_m_base) * N + (bn0 + sg_n_base);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) {
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_store(acc[i][j], outBase + (i * 8u) * N + j * 8u, N, 0, false);
            }
        }
        return;
    }
    threadgroup float sC[BM * BN];
    const uint sg_base = sg_m_base * BN + sg_n_base;
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 2u; ++j)
            simdgroup_store(acc[i][j], &sC[sg_base + (i * 8u) * BN + j * 8u], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0u; i < 16u; ++i) {
        const uint local = tid + i * 128u;
        const uint row = local / BN;
        const uint col = local % BN;
        const uint m = bm0 + row;
        const uint n = bn0 + col;
        if (m < M && n < N) {
            output_data[m * N + n] = sC[row * BN + col];
        }
    }
}

// Tile sizes: BM=64 × BN=32 × BK=32. 128 threads (4 simdgroups) arranged 2×2,
// each simdgroup produces a 32×16 output region = 4×2 = 8 accumulator tiles.
// This doubles arithmetic intensity vs the 32×32 tile because weight loads
// are reused across 64 activation rows (vs 32 previously).
kernel void matmul_mm_q4_0_aligned(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint3 tg_id    [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  sg_idx   [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BM = 64u;
    constexpr uint BN = 32u;
    constexpr uint BK = 32u;

    const uint bm0 = tg_id.y * BM;
    const uint bn0 = tg_id.x * BN;
    const uint M = p.M;
    const uint K = p.K;
    const uint N = p.N;
    const uint blocks_per_row_u32 = (K / 32u) * 5u;

    // 2×2 simdgroup arrangement. Each sg at (sg_row, sg_col) owns 32×16.
    const uint sg_row = sg_idx >> 1;
    const uint sg_col = sg_idx & 1u;
    const uint sg_m_base = sg_row * 32u;   // 0 or 32
    const uint sg_n_base = sg_col * 16u;   // 0 or 16

    // 4×2 accumulator tiles per simdgroup = 8 float8x8 matrices.
    simdgroup_float8x8 acc[4][2];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup float sA[BM * BK];   // 64×32 = 8192 bytes
    threadgroup float sB[BK * BN];   // 32×32 = 4096 bytes

    for (uint kt = 0; kt < K; kt += BK) {
        // ── Load A[BM=64, BK=32] → sA ────────────────────────────────────
        // 128 threads × 16 floats = 2048 values. Each thread writes two
        // 8-wide strips: rows tid/4 and tid/4+32, col (tid%4)*8.
        {
            const uint row_lo = tid >> 2;        // 0..31
            const uint col    = (tid & 3u) * 8u; // 0,8,16,24
            const uint k = kt + col;
            for (uint part = 0u; part < 2u; ++part) {
                const uint row = row_lo + part * 32u;   // 0..31 or 32..63
                const uint m = bm0 + row;
                float v0, v1, v2, v3, v4, v5, v6, v7;
                if (m < M) {
                    device const float* src = input_data + m * K + k;
                    v0 = src[0]; v1 = src[1]; v2 = src[2]; v3 = src[3];
                    v4 = src[4]; v5 = src[5]; v6 = src[6]; v7 = src[7];
                } else {
                    v0 = v1 = v2 = v3 = v4 = v5 = v6 = v7 = 0.0f;
                }
                threadgroup float* dst = &sA[row * BK + col];
                dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
                dst[4] = v4; dst[5] = v5; dst[6] = v6; dst[7] = v7;
            }
        }

        // ── Load+dequant B[BN rows × BK cols] transposed → sB[k*BN + n] ──
        // Vectorized dequant via uchar4/int4: ~10 vector ops replace ~40
        // scalar ops, at the cost of 2 scalar stores of half4 into sB.
        {
            const uint n_local = tid >> 2;       // 0..31
            const uint sub     = tid & 3u;       // 0..3
            const uint n_global = bn0 + n_local;
            const uint block_idx = kt / 32u;
            device const uint* blk = weight_u32 + n_global * blocks_per_row_u32 + block_idx * 5u;
            float scale = float(as_type<half>(ushort(blk[0] & 0xFFFFu)));
            const uint widx0 = 1u + ((sub & 1u) * 2u);
            uchar4 b0 = as_type<uchar4>(blk[widx0]);
            uchar4 b1 = as_type<uchar4>(blk[widx0 + 1u]);
            uchar shift = (sub < 2u) ? uchar(0) : uchar(4);
            int4 v0 = int4((b0 >> shift) & uchar4(0xFu)) - int4(8);
            int4 v1 = int4((b1 >> shift) & uchar4(0xFu)) - int4(8);
            float4 w_a = float4(v0) * scale;
            float4 w_b = float4(v1) * scale;
            const uint k_base = sub * 8u;
            sB[(k_base + 0u) * BN + n_local] = w_a.x;
            sB[(k_base + 1u) * BN + n_local] = w_a.y;
            sB[(k_base + 2u) * BN + n_local] = w_a.z;
            sB[(k_base + 3u) * BN + n_local] = w_a.w;
            sB[(k_base + 4u) * BN + n_local] = w_b.x;
            sB[(k_base + 5u) * BN + n_local] = w_b.y;
            sB[(k_base + 6u) * BN + n_local] = w_b.z;
            sB[(k_base + 7u) * BN + n_local] = w_b.w;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Matmul over the 32-wide K-chunk, 4× 8-wide sub-steps ──────────
        simdgroup_float8x8 mA0, mA1, mA2, mA3, mB0, mB1;
        #pragma clang loop unroll(full)
        for (uint k_sub = 0u; k_sub < BK; k_sub += 8u) {
            simdgroup_load(mA0, &sA[(sg_m_base + 0u ) * BK + k_sub], BK);
            simdgroup_load(mA1, &sA[(sg_m_base + 8u ) * BK + k_sub], BK);
            simdgroup_load(mA2, &sA[(sg_m_base + 16u) * BK + k_sub], BK);
            simdgroup_load(mA3, &sA[(sg_m_base + 24u) * BK + k_sub], BK);
            simdgroup_load(mB0, &sB[k_sub * BN + sg_n_base + 0u], BN);
            simdgroup_load(mB1, &sB[k_sub * BN + sg_n_base + 8u], BN);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(acc[0][0], mA0, mB0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], mA0, mB1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], mA1, mB0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], mA1, mB1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[2][0], mA2, mB0, acc[2][0]);
            simdgroup_multiply_accumulate(acc[2][1], mA2, mB1, acc[2][1]);
            simdgroup_multiply_accumulate(acc[3][0], mA3, mB0, acc[3][0]);
            simdgroup_multiply_accumulate(acc[3][1], mA3, mB1, acc[3][1]);
        }
    }

    // ── Store accumulators → output ──────────────────────────────────────
    // In-bounds fast path: direct simdgroup_store to device memory (skips sC).
    const bool inBounds = (bm0 + BM <= M) && (bn0 + BN <= N);
    if (inBounds) {
        device float* outBase = output_data + (bm0 + sg_m_base) * N + (bn0 + sg_n_base);
        #pragma clang loop unroll(full)
        for (uint i = 0u; i < 4u; ++i) {
            #pragma clang loop unroll(full)
            for (uint j = 0u; j < 2u; ++j) {
                simdgroup_store(acc[i][j], outBase + (i * 8u) * N + j * 8u, N, 0, false);
            }
        }
        return;
    }

    threadgroup float sC[BM * BN];    // 64×32 = 8192 bytes
    const uint sg_base = sg_m_base * BN + sg_n_base;
    for (uint i = 0u; i < 4u; ++i) {
        for (uint j = 0u; j < 2u; ++j) {
            simdgroup_store(acc[i][j], &sC[sg_base + (i * 8u) * BN + j * 8u], BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0u; i < 16u; ++i) {
        const uint local = tid + i * 128u;
        const uint row = local / BN;
        const uint col = local % BN;
        const uint m = bm0 + row;
        const uint n = bn0 + col;
        if (m < M && n < N) {
            output_data[m * N + n] = sC[row * BN + col];
        }
    }
}

// ── Batched matmul for prefill — Q8_0 × F32 activations ──────────────────
// Same tile layout as matmul_mm_q4_0_aligned: BM=64, BN=32, BK=32, 4 sgs (2×2),
// each sg produces a 32×16 = 4×2 output tile. Q8_0 blocks are 34 bytes:
// 2-byte half scale then 32 int8 weights, not 4-byte aligned.
kernel void matmul_mm_q8_0(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint3 tg_id    [[threadgroup_position_in_grid]],
    uint  tid      [[thread_index_in_threadgroup]],
    uint  sg_idx   [[simdgroup_index_in_threadgroup]])
{
    constexpr uint BM = 64u;
    constexpr uint BN = 32u;
    constexpr uint BK = 32u;

    const uint bm0 = tg_id.y * BM;
    const uint bn0 = tg_id.x * BN;
    const uint M = p.M;
    const uint K = p.K;
    const uint N = p.N;
    const uint blocks_per_row = K / 32u;

    const uint sg_row = sg_idx >> 1;
    const uint sg_col = sg_idx & 1u;
    const uint sg_m_base = sg_row * 32u;
    const uint sg_n_base = sg_col * 16u;

    simdgroup_float8x8 acc[4][2];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 2u; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup float sA[BM * BK];
    threadgroup float sB[BK * BN];

    for (uint kt = 0; kt < K; kt += BK) {
        // ── Load A[64×32] ────────────────────────────────────────────────
        {
            const uint row_lo = tid >> 2;
            const uint col    = (tid & 3u) * 8u;
            const uint k = kt + col;
            for (uint part = 0u; part < 2u; ++part) {
                const uint row = row_lo + part * 32u;
                const uint m = bm0 + row;
                float v0, v1, v2, v3, v4, v5, v6, v7;
                if (m < M) {
                    device const float* src = input_data + m * K + k;
                    v0 = src[0]; v1 = src[1]; v2 = src[2]; v3 = src[3];
                    v4 = src[4]; v5 = src[5]; v6 = src[6]; v7 = src[7];
                } else {
                    v0 = v1 = v2 = v3 = v4 = v5 = v6 = v7 = 0.0f;
                }
                threadgroup float* dst = &sA[row * BK + col];
                dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
                dst[4] = v4; dst[5] = v5; dst[6] = v6; dst[7] = v7;
            }
        }

        // ── Load+dequant Q8_0 B[32×32] → sB[k*BN + n] ─────────────────────
        // 128 threads: n_local = tid>>2, sub = tid&3. Each thread dequants
        // 8 consecutive int8 weights for its row at cols [8*sub..8*sub+7].
        {
            const uint n_local = tid >> 2;
            const uint sub     = tid & 3u;
            const uint n_global = bn0 + n_local;
            const uint block_idx = kt / 32u;
            const uint block_off = (n_global * blocks_per_row + block_idx) * 34u;
            device const uchar* blk = weight_data + block_off;
            ushort scaleBits = (ushort(blk[1]) << 8) | ushort(blk[0]);
            float scale = float(as_type<half>(scaleBits));

            const uint qBase = 2u + sub * 8u;
            const uint k_base = sub * 8u;
            for (uint i = 0u; i < 8u; ++i) {
                int q = int((int8_t)blk[qBase + i]);
                sB[(k_base + i) * BN + n_local] = float(q) * scale;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mA0, mA1, mA2, mA3, mB0, mB1;
        for (uint k_sub = 0u; k_sub < BK; k_sub += 8u) {
            simdgroup_load(mA0, &sA[(sg_m_base + 0u ) * BK + k_sub], BK);
            simdgroup_load(mA1, &sA[(sg_m_base + 8u ) * BK + k_sub], BK);
            simdgroup_load(mA2, &sA[(sg_m_base + 16u) * BK + k_sub], BK);
            simdgroup_load(mA3, &sA[(sg_m_base + 24u) * BK + k_sub], BK);
            simdgroup_load(mB0, &sB[k_sub * BN + sg_n_base + 0u], BN);
            simdgroup_load(mB1, &sB[k_sub * BN + sg_n_base + 8u], BN);
            simdgroup_multiply_accumulate(acc[0][0], mA0, mB0, acc[0][0]);
            simdgroup_multiply_accumulate(acc[0][1], mA0, mB1, acc[0][1]);
            simdgroup_multiply_accumulate(acc[1][0], mA1, mB0, acc[1][0]);
            simdgroup_multiply_accumulate(acc[1][1], mA1, mB1, acc[1][1]);
            simdgroup_multiply_accumulate(acc[2][0], mA2, mB0, acc[2][0]);
            simdgroup_multiply_accumulate(acc[2][1], mA2, mB1, acc[2][1]);
            simdgroup_multiply_accumulate(acc[3][0], mA3, mB0, acc[3][0]);
            simdgroup_multiply_accumulate(acc[3][1], mA3, mB1, acc[3][1]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float sC[BM * BN];
    const uint sg_base = sg_m_base * BN + sg_n_base;
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 2u; ++j)
            simdgroup_store(acc[i][j], &sC[sg_base + (i * 8u) * BN + j * 8u], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0u; i < 16u; ++i) {
        const uint local = tid + i * 128u;
        const uint row = local / BN;
        const uint col = local % BN;
        const uint m = bm0 + row;
        const uint n = bn0 + col;
        if (m < M && n < N) {
            output_data[m * N + n] = sC[row * BN + col];
        }
    }
}

// ── Q4_0 matmul (1 row per threadgroup) ───────────────────────────────────
kernel void matmul_q4_0(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float acc = 0.0f;

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint blockOff = (row * blocksPerRow + b) * 18u;
        uint aBase    = b * 32u;

        ushort scaleBits = uint(weight_data[blockOff])
                        | (uint(weight_data[blockOff + 1]) << 8);
        float scale = float(as_type<half>(scaleBits));

        float blockSum = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            uint packed = weight_data[blockOff + 2u + i];
            int lo = int(packed & 0xFu) - 8;
            int hi = int(packed >> 4)   - 8;
            blockSum += float(lo) * input_data[aBase + i]
                     +  float(hi) * input_data[aBase + i + 16u];
        }
        acc += scale * blockSum;
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q4_1 matmul ───────────────────────────────────────────────────────────
// Block layout: 2b FP16 scale + 2b FP16 min + 16b packed nibbles = 20 bytes/block.
// value = scale * nibble + min.  (Unlike Q4_0, the nibble is unsigned 0..15 and we add a per-block min.)
// Q4_1 2-SIMD-group 4-rows-each (llama.cpp pattern). 8 rows per TG.
kernel void matmul_q4_1_2x4row(
    device       float*        output_data [[buffer(0)]],
    device const float4*       input_vec4  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = input_vec4[aVecBase + i];
        // Precompute activation sum for the min-term.
        float aSum = 0.0f;
        for (uint i = 0; i < 8u; ++i)
            aSum += aVec[i].x + aVec[i].y + aVec[i].z + aVec[i].w;

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint blockOff = (row * blocksPerRow + b) * 20u;
            ushort dBits = uint(weight_data[blockOff])     | (uint(weight_data[blockOff + 1]) << 8);
            ushort mBits = uint(weight_data[blockOff + 2]) | (uint(weight_data[blockOff + 3]) << 8);
            float scale = float(as_type<half>(dBits));
            float minV  = float(as_type<half>(mBits));

            float blockSum = 0.0f;
            // 16 nibble bytes, each containing lo (→element i) and hi (→element i+16).
            // Read 4 bytes per iteration to cut load count.
            for (uint i = 0; i < 4u; ++i) {
                uint b0 = weight_data[blockOff + 4u + i*4 + 0];
                uint b1 = weight_data[blockOff + 4u + i*4 + 1];
                uint b2 = weight_data[blockOff + 4u + i*4 + 2];
                uint b3 = weight_data[blockOff + 4u + i*4 + 3];
                float4 lo = float4(float(b0 & 0xFu), float(b1 & 0xFu),
                                   float(b2 & 0xFu), float(b3 & 0xFu));
                float4 hi = float4(float(b0 >> 4),   float(b1 >> 4),
                                   float(b2 >> 4),  float(b3 >> 4));
                blockSum += dot(lo, aVec[i]) + dot(hi, aVec[i + 4u]);
            }
            accs[r] += scale * blockSum + minV * aSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
    }
}

kernel void matmul_q4_1(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float acc = 0.0f;

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint blockOff = (row * blocksPerRow + b) * 20u;
        uint aBase    = b * 32u;

        ushort dBits = uint(weight_data[blockOff])       | (uint(weight_data[blockOff + 1]) << 8);
        ushort mBits = uint(weight_data[blockOff + 2])   | (uint(weight_data[blockOff + 3]) << 8);
        float scale = float(as_type<half>(dBits));
        float minV  = float(as_type<half>(mBits));

        float blockSum = 0.0f;
        float aSum = 0.0f;
        for (uint i = 0; i < 16u; ++i) {
            uint packed = weight_data[blockOff + 4u + i];
            float lo = float(packed & 0xFu);
            float hi = float(packed >> 4);
            float a0 = input_data[aBase + i];
            float a1 = input_data[aBase + i + 16u];
            blockSum += lo * a0 + hi * a1;
            aSum     += a0 + a1;
        }
        acc += scale * blockSum + minV * aSum;
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q5_K matmul ───────────────────────────────────────────────────────────
// Super-block layout (176 bytes, 256 elements):
//   [0..1]   d     (half)        super-block scale
//   [2..3]   dmin  (half)        super-block min
//   [4..15]  scales (12 bytes)   packed 6-bit scales[0..7] and mins[0..7]
//   [16..47] qh     (32 bytes)   high bit for each of 256 elements
//   [48..175] qs    (128 bytes)  low 4 bits for each element
//
// Layout replicates CPU dequant order: 4 chunks j=0..3, each pulling 64 elements
// from qs[j*32..j*32+31]. For each l in 0..31:
//   y[j*64+l    ] = d*scales[2j  ] * ((qs[j*32+l] & 0xF) + (qh[l] & (1<<(2j  )) ? 16 : 0)) - dmin*mins[2j  ]
//   y[j*64+l+32] = d*scales[2j+1] * ((qs[j*32+l] >> 4 ) + (qh[l] & (1<<(2j+1)) ? 16 : 0)) - dmin*mins[2j+1]
// Q5_K 16-row-per-TG variant: 64 threads = 2 SIMD groups, each produces 8
// rows with 4 threads cooperating per row. Same parallelisation pattern as
// matmul_q6_k_16row.
kernel void matmul_q5_k_16row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;

    {
        float acc = 0.0f;

        for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
            uint base = (rowBase * superBlocksPerRow + sb) * 176u;
            uint aBase = sb * 256u;

            ushort dBits = uint(weight_data[base])     | (uint(weight_data[base + 1]) << 8);
            ushort mBits = uint(weight_data[base + 2]) | (uint(weight_data[base + 3]) << 8);
            float d = float(as_type<half>(dBits));
            float dmin = float(as_type<half>(mBits));

            float scales[8];
            float mins[8];
            for (uint j = 0; j < 4u; ++j) {
                scales[j] = float(weight_data[base + 4 + j] & 63u);
                mins[j]   = float(weight_data[base + 4 + j + 4] & 63u);
            }
            for (uint j = 4; j < 8u; ++j) {
                uint lo = weight_data[base + 4 + j + 4] & 0xFu;
                uint hi = (weight_data[base + 4 + j - 4] >> 6) & 0x3u;
                scales[j] = float(lo | (hi << 4));
                lo = weight_data[base + 4 + j + 4] >> 4;
                hi = (weight_data[base + 4 + j] >> 6) & 0x3u;
                mins[j] = float(lo | (hi << 4));
            }

            uint qhOff = base + 16u;
            uint qsOff = base + 48u;
            uint u1mask = 1u;
            uint u2mask = 2u;
            for (uint j = 0; j < 4u; ++j) {
                float d1 = d * scales[2 * j];
                float m1 = dmin * mins[2 * j];
                float d2 = d * scales[2 * j + 1];
                float m2 = dmin * mins[2 * j + 1];
                for (uint l = 0; l < 32u; ++l) {
                    uint qs = weight_data[qsOff + j * 32u + l];
                    uint qh = weight_data[qhOff + l];
                    float lo = float((qs & 0xFu) + ((qh & u1mask) != 0u ? 16u : 0u));
                    float hi = float((qs >> 4) + ((qh & u2mask) != 0u ? 16u : 0u));
                    float w0 = d1 * lo - m1;
                    float w1 = d2 * hi - m2;
                    acc += w0 * input_data[aBase + j * 64u + l];
                    acc += w1 * input_data[aBase + j * 64u + l + 32u];
                }
                u1mask <<= 2;
                u2mask <<= 2;
            }
        }

        acc += simd_shuffle_xor(acc, 1u);
        acc += simd_shuffle_xor(acc, 2u);
        if (threadInRow == 0) output_data[rowBase] = acc;
    }
}

// ── Batched Q5_K (prefill): one dispatch covers all M activation rows ──
// Same per-row math as matmul_q5_k_16row, wrapped in an outer M loop so one
// dispatch handles all batched rows. This eliminates the M× per-row
// dispatches AND the 2M× CopyTensorRegion/CopyTensorSlice overhead from the
// C# fallback path. Dequant is repeated per M iteration (no amortization);
// the win is pure dispatch-count reduction.
kernel void matmul_mm_q5_k_16row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    uint M = p.M;

    for (uint m = 0u; m < M; ++m) {
        float acc = 0.0f;
        uint aRowBase = m * p.K;

        for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
            uint base  = (rowBase * superBlocksPerRow + sb) * 176u;
            uint aBase = aRowBase + sb * 256u;

            ushort dBits = uint(weight_data[base])     | (uint(weight_data[base + 1]) << 8);
            ushort mBits = uint(weight_data[base + 2]) | (uint(weight_data[base + 3]) << 8);
            float d = float(as_type<half>(dBits));
            float dmin = float(as_type<half>(mBits));

            float scales[8];
            float mins[8];
            for (uint j = 0; j < 4u; ++j) {
                scales[j] = float(weight_data[base + 4 + j] & 63u);
                mins[j]   = float(weight_data[base + 4 + j + 4] & 63u);
            }
            for (uint j = 4; j < 8u; ++j) {
                uint lo = weight_data[base + 4 + j + 4] & 0xFu;
                uint hi = (weight_data[base + 4 + j - 4] >> 6) & 0x3u;
                scales[j] = float(lo | (hi << 4));
                lo = weight_data[base + 4 + j + 4] >> 4;
                hi = (weight_data[base + 4 + j] >> 6) & 0x3u;
                mins[j] = float(lo | (hi << 4));
            }

            uint qhOff = base + 16u;
            uint qsOff = base + 48u;
            uint u1mask = 1u;
            uint u2mask = 2u;
            for (uint j = 0; j < 4u; ++j) {
                float d1 = d * scales[2 * j];
                float m1 = dmin * mins[2 * j];
                float d2 = d * scales[2 * j + 1];
                float m2 = dmin * mins[2 * j + 1];
                for (uint l = 0; l < 32u; ++l) {
                    uint qs = weight_data[qsOff + j * 32u + l];
                    uint qh = weight_data[qhOff + l];
                    float lo = float((qs & 0xFu) + ((qh & u1mask) != 0u ? 16u : 0u));
                    float hi = float((qs >> 4) + ((qh & u2mask) != 0u ? 16u : 0u));
                    float w0 = d1 * lo - m1;
                    float w1 = d2 * hi - m2;
                    acc += w0 * input_data[aBase + j * 64u + l];
                    acc += w1 * input_data[aBase + j * 64u + l + 32u];
                }
                u1mask <<= 2;
                u2mask <<= 2;
            }
        }

        acc += simd_shuffle_xor(acc, 1u);
        acc += simd_shuffle_xor(acc, 2u);
        if (threadInRow == 0) output_data[m * p.N + rowBase] = acc;
    }
}

// ── Unused stub (retained to minimize diff surface) ────────────────────────
#if 0
constant constexpr uint Q5K_MBLOCK = 16u;

kernel void matmul_mm_q5_k_16row_mblock(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    uint M = p.M;

    for (uint m0 = 0u; m0 < M; m0 += Q5K_MBLOCK) {
        uint mLen = min(Q5K_MBLOCK, M - m0);

        float acc[Q5K_MBLOCK];
        for (uint i = 0u; i < Q5K_MBLOCK; ++i) acc[i] = 0.0f;

        for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
            uint base  = (rowBase * superBlocksPerRow + sb) * 176u;
            uint aBase = sb * 256u;

            ushort dBits = uint(weight_data[base])     | (uint(weight_data[base + 1]) << 8);
            ushort mBits = uint(weight_data[base + 2]) | (uint(weight_data[base + 3]) << 8);
            float d = float(as_type<half>(dBits));
            float dmin = float(as_type<half>(mBits));

            float scales[8];
            float mins[8];
            for (uint j = 0; j < 4u; ++j) {
                scales[j] = float(weight_data[base + 4 + j] & 63u);
                mins[j]   = float(weight_data[base + 4 + j + 4] & 63u);
            }
            for (uint j = 4; j < 8u; ++j) {
                uint lo = weight_data[base + 4 + j + 4] & 0xFu;
                uint hi = (weight_data[base + 4 + j - 4] >> 6) & 0x3u;
                scales[j] = float(lo | (hi << 4));
                lo = weight_data[base + 4 + j + 4] >> 4;
                hi = (weight_data[base + 4 + j] >> 6) & 0x3u;
                mins[j] = float(lo | (hi << 4));
            }

            uint qhOff = base + 16u;
            uint qsOff = base + 48u;
            uint u1mask = 1u;
            uint u2mask = 2u;
            for (uint j = 0; j < 4u; ++j) {
                float d1 = d * scales[2 * j];
                float m1 = dmin * mins[2 * j];
                float d2 = d * scales[2 * j + 1];
                float m2 = dmin * mins[2 * j + 1];
                for (uint l = 0; l < 32u; ++l) {
                    uint qs = weight_data[qsOff + j * 32u + l];
                    uint qh = weight_data[qhOff + l];
                    float lo = float((qs & 0xFu) + ((qh & u1mask) != 0u ? 16u : 0u));
                    float hi = float((qs >> 4) + ((qh & u2mask) != 0u ? 16u : 0u));
                    float w0 = d1 * lo - m1;
                    float w1 = d2 * hi - m2;
                    uint inLo = aBase + j * 64u + l;
                    uint inHi = inLo + 32u;
                    for (uint mi = 0u; mi < Q5K_MBLOCK; ++mi) {
                        if (mi >= mLen) break;
                        uint aRow = (m0 + mi) * p.K;
                        acc[mi] += w0 * input_data[aRow + inLo];
                        acc[mi] += w1 * input_data[aRow + inHi];
                    }
                }
                u1mask <<= 2;
                u2mask <<= 2;
            }
        }

        for (uint mi = 0u; mi < Q5K_MBLOCK; ++mi) {
            if (mi >= mLen) break;
            float v = acc[mi];
            v += simd_shuffle_xor(v, 1u);
            v += simd_shuffle_xor(v, 2u);
            if (threadInRow == 0) output_data[(m0 + mi) * p.N + rowBase] = v;
        }
    }
}
#endif

// ── Q2_K matrix-vector (llama.cpp mul_mv_q2_K pattern) ────────────────────
// Q2_K blocks are 84 bytes per 256 weights: 16b scales+mins (packed 4+4 per
// byte × 16 sub-blocks), 64b 2-bit quants, 2b d, 2b dmin. Sub-blocks hold 16
// weights each. 32 lanes cooperate with stride-4 ix × 8 it interleaving.
kernel void matmul_q2_k_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int  NR0 = 2;
    constexpr int  NSG = 2;
    constexpr uint Q2K_BYTES   = 84u;
    constexpr uint SCALES_OFF  = 0u;
    constexpr uint QS_OFF      = 16u;
    constexpr uint D_OFF       = 80u;
    constexpr uint DMIN_OFF    = 82u;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;
    const short is = (8 * ir) / 16;

    const uint sbPerRow = p.K / 256u;

    float yl[32];
    float sumf[NR0] = {0.0f};

    device const float* y4 = input_data + ix * 256 + 128 * iq + 8 * ir;

    for (uint ib = ix; ib < sbPerRow; ib += 4u) {
        float4 sumy = float4(0.0f);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yl[i + 16] = y4[i + 64]; sumy[2] += yl[i + 16];
            yl[i + 24] = y4[i + 96]; sumy[3] += yl[i + 24];
        }

        const uint sbOff = ib * Q2K_BYTES;
        for (short row = 0; row < NR0 && first_row + row < p.N; ++row) {
            const uint rowOff = (first_row + row) * sbPerRow * Q2K_BYTES + sbOff;
            device const uchar*  sc = weight_data + rowOff + SCALES_OFF + 8 * iq + is;
            device const ushort* qs = (device const ushort*)(weight_data + rowOff + QS_OFF) + 16 * iq + 4 * ir;

            ushort dBits  = uint(weight_data[rowOff + D_OFF])    | (uint(weight_data[rowOff + D_OFF + 1])    << 8);
            ushort dmBits = uint(weight_data[rowOff + DMIN_OFF]) | (uint(weight_data[rowOff + DMIN_OFF + 1]) << 8);
            float dall = float(as_type<half>(dBits));
            float dmin = float(as_type<half>(dmBits)) / 16.0f;

            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 8; i += 2) {
                ushort q = qs[i / 2];
                acc1[0] += yl[i +  0] * float(q & 0x0003);
                acc2[0] += yl[i +  1] * float(q & 0x0300);
                acc1[1] += yl[i +  8] * float(q & 0x000c);
                acc2[1] += yl[i +  9] * float(q & 0x0c00);
                acc1[2] += yl[i + 16] * float(q & 0x0030);
                acc2[2] += yl[i + 17] * float(q & 0x3000);
                acc1[3] += yl[i + 24] * float(q & 0x00c0);
                acc2[3] += yl[i + 25] * float(q & 0xc000);
            }
            sumf[row] += dall * ((acc1[0] + acc2[0] / 256.0f) * (sc[0] & 0xF)                +
                                 (acc1[1] + acc2[1] / 256.0f) * (sc[2] & 0xF) /  4.0f +
                                 (acc1[2] + acc2[2] / 256.0f) * (sc[4] & 0xF) / 16.0f +
                                 (acc1[3] + acc2[3] / 256.0f) * (sc[6] & 0xF) / 64.0f)
                        - dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                  sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));
        }

        y4 += 4 * 256;
    }

    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Q4_K matrix-vector (llama.cpp mul_mv_q4_K pattern) ────────────────────
// Q4_K blocks are 144 bytes per 256 weights: 2b d, 2b dmin, 12b packed
// scales+mins, 128b of 4-bit nibbles. Similar structure to Q5_K but without
// the qh (high-bit) array. 32 lanes cooperate with stride-4 ix interleaving
// (8 lanes per super-block).
kernel void matmul_q4_k_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int  NR0 = 2;
    constexpr int  NSG = 2;
    constexpr uint Q4K_BYTES  = 144u;
    constexpr uint D_OFF      = 0u;
    constexpr uint DMIN_OFF   = 2u;
    constexpr uint SCALES_OFF = 4u;
    constexpr uint QS_OFF     = 16u;
    constexpr ushort kmask1   = 0x3f3f;
    constexpr ushort kmask2   = 0x0f0f;
    constexpr ushort kmask3   = 0xc0c0;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const uint sbPerRow = p.K / 256u;

    float yl[16], yh[16];
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;
    float sumf[NR0] = {0.0f};

    device const float* y4 = input_data + ix * 256 + 64 * iq + 8 * ir;

    for (uint ib = ix; ib < sbPerRow; ib += 4u) {
        float4 sumy = float4(0.0f);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        const uint sbOff = ib * Q4K_BYTES;
        for (short row = 0; row < NR0 && first_row + row < p.N; ++row) {
            const uint rowOff = (first_row + row) * sbPerRow * Q4K_BYTES + sbOff;
            device const ushort* sc = (device const ushort*)(weight_data + rowOff + SCALES_OFF) + iq;
            device const ushort* q1 = (device const ushort*)(weight_data + rowOff + QS_OFF) + 16 * iq + 4 * ir;
            device const half*   dh = (device const half*)(weight_data + rowOff + D_OFF);

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * float(q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * float(q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * float(q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * float(q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * float(q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * float(q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * float(q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * float(q2[i] & 0xF000);
            }
            float d    = float(dh[0]);
            float dmin = float(dh[1]);
            sumf[row] += d * ((acc1[0] + acc1[1] / 256.0f) * sc8[0] +
                              (acc1[2] + acc1[3] / 256.0f) * sc8[1] / 16.0f +
                              (acc2[0] + acc2[1] / 256.0f) * sc8[4] +
                              (acc2[2] + acc2[3] / 256.0f) * sc8[5] / 16.0f)
                        - dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }

        y4 += 4 * 256;
    }

    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// ── Q5_K matrix-vector (llama.cpp mul_mv_q5_K pattern) ────────────────────
// 8 lanes cooperate per super-block (stride-4 ix interleaves 4 super-blocks
// concurrently per sg). Uses the d/dmin bias split: per-weight value is
// (d*scale*q5 − dmin*min), the kernel computes sum_y * (d*scale*q5) and
// sum_y * (dmin*min) separately.
//
// Thread layout (per sg):
//   tid = tiisg/4 ∈ [0,8);  ix = tiisg%4 ∈ [0,4)
//   iq  = tid/4 ∈ {0,1}     — which 128-weight half
//   ir  = tid%4 ∈ [0,4)     — which 8-value chunk
kernel void matmul_q5_k_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint   tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    // NR0=2 per sg tested faster than llama.cpp's NR0=1 for our shapes —
    // probably because our TGs are less fully utilized at smaller NR0.
    constexpr int NR0 = 2;
    constexpr int NSG = 2;
    constexpr uint Q5K_BYTES   = 176u;
    constexpr uint D_OFF       = 0u;
    constexpr uint DMIN_OFF    = 2u;
    constexpr uint SCALES_OFF  = 4u;
    constexpr uint QH_OFF      = 16u;
    constexpr uint QS_OFF      = 48u;
    constexpr ushort kmask1    = 0x3f3f;
    constexpr ushort kmask2    = 0x0f0f;
    constexpr ushort kmask3    = 0xc0c0;

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short tid = tiisg / 4;
    const short ix  = tiisg % 4;
    const short iq  = tid / 4;
    const short ir  = tid % 4;
    const short l0  = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = uchar(1u << (2 * iq));
    const uchar hm2 = uchar(hm1 << 1);
    const uchar hm3 = uchar(hm1 << 4);
    const uchar hm4 = uchar(hm2 << 4);

    const uint sbPerRow = p.K / 256u;

    float yl[16], yh[16];
    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    // Outer M loop: when M>1 (prefill) we iterate across input rows inside
    // the kernel, eliminating the M× per-row dispatch + 2M× copy overhead
    // from the C# fallback. When M=1 (decode) the loop executes once.
    for (uint m = 0u; m < p.M; ++m) {
        float sumf[NR0] = {0.0f};
        const uint inputRowBase = m * p.K;
        device const float* y1 = input_data + inputRowBase + ix * 256 + y_offset;

        for (uint i = ix; i < sbPerRow; i += 4u) {
            const uint sbOff = i * Q5K_BYTES;
            device const float* y2 = y1 + 128;
            float4 sumy = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short l = 0; l < 8; ++l) {
                yl[l + 0] = y1[l +  0]; sumy[0] += yl[l + 0];
                yl[l + 8] = y1[l + 32]; sumy[1] += yl[l + 8];
                yh[l + 0] = y2[l +  0]; sumy[2] += yh[l + 0];
                yh[l + 8] = y2[l + 32]; sumy[3] += yh[l + 8];
            }

            for (short row = 0; row < NR0 && first_row + row < p.N; ++row) {
                const uint rowOff = (first_row + row) * sbPerRow * Q5K_BYTES + sbOff;
                device const uchar*  q1 = weight_data + rowOff + QS_OFF + q_offset;
                device const uchar*  q2 = q1 + 64;
                device const uchar*  qh = weight_data + rowOff + QH_OFF + l0;
                device const ushort* a  = (device const ushort*)(weight_data + rowOff + SCALES_OFF) + iq;

                ushort dBits  = uint(weight_data[rowOff + D_OFF])    | (uint(weight_data[rowOff + D_OFF + 1])    << 8);
                ushort dmBits = uint(weight_data[rowOff + DMIN_OFF]) | (uint(weight_data[rowOff + DMIN_OFF + 1]) << 8);
                float d_row    = float(as_type<half>(dBits));
                float dmin_row = float(as_type<half>(dmBits));

                sc16[0] = a[0] & kmask1;
                sc16[1] = a[2] & kmask1;
                sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
                sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

                float4 acc1 = float4(0.0f);
                float4 acc2 = float4(0.0f);
                #pragma clang loop unroll(full)
                for (short l = 0; l < 8; ++l) {
                    uchar h = qh[l];
                    acc1[0] += yl[l + 0] * float(q1[l] & 0x0F);
                    acc1[1] += yl[l + 8] * float(q1[l] & 0xF0);
                    acc1[2] += yh[l + 0] * float(q2[l] & 0x0F);
                    acc1[3] += yh[l + 8] * float(q2[l] & 0xF0);
                    acc2[0] += (h & hm1) ? yl[l + 0] : 0.0f;
                    acc2[1] += (h & hm2) ? yl[l + 8] : 0.0f;
                    acc2[2] += (h & hm3) ? yh[l + 0] : 0.0f;
                    acc2[3] += (h & hm4) ? yh[l + 8] : 0.0f;
                }
                sumf[row] += d_row * (sc8[0] * (acc1[0]          + 16.0f * acc2[0]) +
                                      sc8[1] * (acc1[1] / 16.0f  + 16.0f * acc2[1]) +
                                      sc8[4] * (acc1[2]          + 16.0f * acc2[2]) +
                                      sc8[5] * (acc1[3] / 16.0f  + 16.0f * acc2[3]))
                            - dmin_row * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                                          sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }

            y1 += 4 * 256;
        }

        for (short row = 0; row < NR0; ++row) {
            float s = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < p.N) {
                output_data[m * p.N + first_row + row] = s;
            }
        }
    }
}

// Q5_K with a small TG (16 threads = 16 super-blocks when K=4096).
// All threads stay active; reduction is a single `simd_sum` within one
// SIMD group, no threadgroup memory or barrier needed. Much better packing
// per core than the 64-thread version when superBlocksPerRow ≤ 16.
kernel void matmul_q5_k_tg16(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    uint row = tgid;
    if (row >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;

    for (uint sb = tid; sb < superBlocksPerRow; sb += 16u) {
        uint base = (row * superBlocksPerRow + sb) * 176u;
        uint aBase = sb * 256u;

        ushort dBits = uint(weight_data[base])     | (uint(weight_data[base + 1]) << 8);
        ushort mBits = uint(weight_data[base + 2]) | (uint(weight_data[base + 3]) << 8);
        float d = float(as_type<half>(dBits));
        float dmin = float(as_type<half>(mBits));

        float scales[8];
        float mins[8];
        for (uint j = 0; j < 4u; ++j) {
            scales[j] = float(weight_data[base + 4 + j] & 63u);
            mins[j]   = float(weight_data[base + 4 + j + 4] & 63u);
        }
        for (uint j = 4; j < 8u; ++j) {
            uint lo = weight_data[base + 4 + j + 4] & 0xFu;
            uint hi = (weight_data[base + 4 + j - 4] >> 6) & 0x3u;
            scales[j] = float(lo | (hi << 4));
            lo = weight_data[base + 4 + j + 4] >> 4;
            hi = (weight_data[base + 4 + j] >> 6) & 0x3u;
            mins[j] = float(lo | (hi << 4));
        }

        uint qhOff = base + 16u;
        uint qsOff = base + 48u;
        uint u1mask = 1u;
        uint u2mask = 2u;
        for (uint j = 0; j < 4u; ++j) {
            float d1 = d * scales[2 * j];
            float m1 = dmin * mins[2 * j];
            float d2 = d * scales[2 * j + 1];
            float m2 = dmin * mins[2 * j + 1];
            for (uint l = 0; l < 32u; ++l) {
                uint qs = weight_data[qsOff + j * 32u + l];
                uint qh = weight_data[qhOff + l];
                float lo = float((qs & 0xFu) + ((qh & u1mask) != 0u ? 16u : 0u));
                float hi = float((qs >> 4) + ((qh & u2mask) != 0u ? 16u : 0u));
                float w0 = d1 * lo - m1;
                float w1 = d2 * hi - m2;
                acc += w0 * input_data[aBase + j * 64u + l];
                acc += w1 * input_data[aBase + j * 64u + l + 32u];
            }
            u1mask <<= 2;
            u2mask <<= 2;
        }
    }
    // All threads in one SIMD group — no threadgroup scratch needed.
    float total = simd_sum(acc);
    if (tid == 0) output_data[row] = total;
}

kernel void matmul_q5_k(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;

    // One super-block = 176 bytes, covers 256 input floats.
    for (uint sb = tid; sb < superBlocksPerRow; sb += TG_SIZE) {
        uint base = (row * superBlocksPerRow + sb) * 176u;
        uint aBase = sb * 256u;

        ushort dBits = uint(weight_data[base])     | (uint(weight_data[base + 1]) << 8);
        ushort mBits = uint(weight_data[base + 2]) | (uint(weight_data[base + 3]) << 8);
        float d = float(as_type<half>(dBits));
        float dmin = float(as_type<half>(mBits));

        // Unpack 6-bit scales/mins.
        float scales[8];
        float mins[8];
        for (uint j = 0; j < 4u; ++j) {
            scales[j] = float(weight_data[base + 4 + j] & 63u);
            mins[j]   = float(weight_data[base + 4 + j + 4] & 63u);
        }
        for (uint j = 4; j < 8u; ++j) {
            uint lo = weight_data[base + 4 + j + 4] & 0xFu;
            uint hi = (weight_data[base + 4 + j - 4] >> 6) & 0x3u;
            scales[j] = float(lo | (hi << 4));
            lo = weight_data[base + 4 + j + 4] >> 4;
            hi = (weight_data[base + 4 + j] >> 6) & 0x3u;
            mins[j] = float(lo | (hi << 4));
        }

        uint qhOff = base + 16u;
        uint qsOff = base + 48u;

        // 4 chunks × 64 elements = 256 elements.
        uint u1mask = 1u;
        uint u2mask = 2u;
        for (uint j = 0; j < 4u; ++j) {
            float d1 = d * scales[2 * j];
            float m1 = dmin * mins[2 * j];
            float d2 = d * scales[2 * j + 1];
            float m2 = dmin * mins[2 * j + 1];
            for (uint l = 0; l < 32u; ++l) {
                uint qs = weight_data[qsOff + j * 32u + l];
                uint qh = weight_data[qhOff + l];
                float lo = float((qs & 0xFu) + ((qh & u1mask) != 0u ? 16u : 0u));
                float hi = float((qs >> 4) + ((qh & u2mask) != 0u ? 16u : 0u));
                float w0 = d1 * lo - m1;
                float w1 = d2 * hi - m2;
                acc += w0 * input_data[aBase + j * 64u + l];
                acc += w1 * input_data[aBase + j * 64u + l + 32u];
            }
            u1mask <<= 2;
            u2mask <<= 2;
        }
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q6_K matmul ───────────────────────────────────────────────────────────
// Super-block layout (210 bytes, 256 elements):
//   [0..127]   ql       (128 bytes) — 2 nibbles per byte, each element's low 4 bits
//   [128..191] qh       (64 bytes)  — 4 elements share one byte, each element's high 2 bits
//   [192..207] scales   (16 bytes)  — int8 scales
//   [208..209] d        (half)      — super-block scale
//
// CPU layout uses two 128-element halves; for l in 0..31 we produce 4 values at
// l, l+32, l+64, l+96 per half. Each value = d * scale * (((ql&0xF) | (qh&3)<<4) - 32).
// Q6_K 2-rows-per-SIMD-group, 4 threads per row.
// For K=4096 (16 super-blocks): 4 threads × 4 SBs each = 16 SBs/row,
// 2 rows per 8-thread slice, 4 slices per SIMD group = 8 rows per SIMD.
// Dispatched with TG=64 (2 SIMD groups) and gridX = N/16.
kernel void matmul_q6_k_16row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    // Each SIMD group (32 threads) produces 8 rows. 4 threads per row.
    uint lane      = tid % 32u;
    uint threadInRow = lane % 4u;           // 0..3
    uint localRow    = lane / 4u;            // 0..7 (which of 8 rows within SIMD)
    uint rowBase   = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;
    device const uchar* raw = weight_data;

    for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
        uint base = (rowBase * superBlocksPerRow + sb) * 210u;
        uint aBase = sb * 256u;

        ushort dBits = uint(raw[base + 208]) | (uint(raw[base + 209]) << 8);
        float d = float(as_type<half>(dBits));

        float scales[16];
        for (uint s = 0; s < 16u; ++s) scales[s] = d * float((char)raw[base + 192 + s]);

        for (uint half_ = 0u; half_ < 2u; ++half_) {
            uint qlOff = base + half_ * 64u;
            uint qhOff = base + 128u + half_ * 32u;
            uint scIdx = half_ * 8u;
            uint aOff  = aBase + half_ * 128u;

            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            for (uint l = 0; l < 16u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sA0 * float(q1) * input_data[aOff + l];
                acc += sA1 * float(q2) * input_data[aOff + l + 32u];
                acc += sA2 * float(q3) * input_data[aOff + l + 64u];
                acc += sA3 * float(q4) * input_data[aOff + l + 96u];
            }
            for (uint l = 16; l < 32u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sB0 * float(q1) * input_data[aOff + l];
                acc += sB1 * float(q2) * input_data[aOff + l + 32u];
                acc += sB2 * float(q3) * input_data[aOff + l + 64u];
                acc += sB3 * float(q4) * input_data[aOff + l + 96u];
            }
        }
    }

    // Reduce across the 4 threads that worked on this row.
    // simd_shuffle_xor with offset 1, then 2 → sum within group of 4.
    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    if (threadInRow == 0) output_data[rowBase] = acc;
}

// ── Q6_K matrix-vector (llama.cpp mul_mv_q6_K pattern) ────────────────────
// 16 lanes cooperate on ONE super-block instead of having 32 lanes scatter
// across 32 different super-blocks (which wrecked memory coalescing on our
// previous Q6_K kernels). Each simdgroup (32 lanes) produces nr0 output
// rows; lane layout is:
//   tid = tiisg/2 ∈ [0,16)   — which piece of the super-block this lane does
//   ix  = tiisg%2  ∈ {0,1}   — interleaves lanes across super-blocks
// Per lane, 4 weights × 4 sums = 16 MACs per super-block per row. Memory
// reads from ql/qh/scales are coalesced within each ix-group of 16 lanes.
//
// Dispatch: gridX = ⌈N / (NSG × nr0)⌉, tgX = 32 × NSG threads. Caller picks
// NSG=1 (32 threads/TG) to maximize parallel TGs for huge-N cases like
// lm_head.
kernel void matmul_q6_k_mv(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr int NR0 = 2;   // output rows per simdgroup
    constexpr int NSG = 2;   // simdgroups per TG

    // Block layout constants (Q6_K = 210 bytes per 256 weights).
    constexpr uint Q6K_BYTES = 210u;
    constexpr uint QL_OFF    = 0u;     // [0..127] low 4-bit nibbles × 2
    constexpr uint QH_OFF    = 128u;   // [128..191] high 2-bit per weight
    constexpr uint SC_OFF    = 192u;   // [192..207] 16 int8 scales
    constexpr uint D_OFF     = 208u;   // [208..209] half super-scale

    const uint first_row = (tgid * NSG + sgitg) * NR0;
    if (first_row >= p.N) return;

    const short tid = tiisg / 2;       // 0..15
    const short ix  = tiisg % 2;       // 0 or 1
    const short ip  = tid / 8;         // 0 or 1 — which half of super-block
    const short il  = tid % 8;         // 0..7
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;
    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    float yl[16];
    float sumf[NR0] = { 0.0f };

    const uint sbPerRow = p.K / 256u;
    device const uchar* weightBase0 = weight_data + first_row * sbPerRow * Q6K_BYTES;

    // Outer super-block loop, stride 2 (ix interleaves).
    for (uint i = ix; i < sbPerRow; i += 2u) {
        const uint sbByteOff = i * Q6K_BYTES;
        device const float* y = input_data + i * 256u + y_offset;
        #pragma clang loop unroll(full)
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        device const uchar* rowBase = weightBase0;
        for (short row = 0; row < NR0; ++row) {
            // Per-row pointers into this super-block.
            device const uchar* q1 = rowBase + sbByteOff + QL_OFF + q_offset_l;
            device const uchar* q2 = q1 + 32;
            device const uchar* qh = rowBase + sbByteOff + QH_OFF + q_offset_h;
            device const char*  sc = (device const char*)(rowBase + sbByteOff + SC_OFF + is);
            ushort dBits = uint(rowBase[sbByteOff + D_OFF]) | (uint(rowBase[sbByteOff + D_OFF + 1]) << 8);
            float d = float(as_type<half>(dBits));

            float4 sums = float4(0.0f);
            #pragma clang loop unroll(full)
            for (short l = 0; l < 4; ++l) {
                int w0 = int((int8_t)((q1[l] & 0xFu) | ((qh[l] & 0x03u) << 4))) - 32;
                int w1 = int((int8_t)((q2[l] & 0xFu) | ((qh[l] & 0x0Cu) << 2))) - 32;
                int w2 = int((int8_t)((q1[l]  >> 4) | ((qh[l] & 0x30u) << 0))) - 32;
                int w3 = int((int8_t)((q2[l]  >> 4) | ((qh[l] & 0xC0u) >> 2))) - 32;
                sums[0] += yl[4*l + 0] * float(w0);
                sums[1] += yl[4*l + 1] * float(w1);
                sums[2] += yl[4*l + 2] * float(w2);
                sums[3] += yl[4*l + 3] * float(w3);
            }
            sumf[row] += d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) +
                              sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            rowBase += sbPerRow * Q6K_BYTES;  // advance to next row's data
        }
    }

    // Reduce across the 32 lanes of this simdgroup, one row at a time.
    for (short row = 0; row < NR0; ++row) {
        float s = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < p.N) {
            output_data[first_row + row] = s;
        }
    }
}

// Q6_K 32-rows-per-TG: 128 threads = 4 SIMD groups, 8 rows per SIMD group
// (same scheme as 16row but doubled). Better TG utilization for large-N
// matmuls like lm_head.
kernel void matmul_q6_k_32row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 32u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;
    device const uchar* raw = weight_data;

    for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
        uint base = (rowBase * superBlocksPerRow + sb) * 210u;
        uint aBase = sb * 256u;

        ushort dBits = uint(raw[base + 208]) | (uint(raw[base + 209]) << 8);
        float d = float(as_type<half>(dBits));

        float scales[16];
        for (uint s = 0; s < 16u; ++s) scales[s] = d * float((char)raw[base + 192 + s]);

        for (uint half_ = 0u; half_ < 2u; ++half_) {
            uint qlOff = base + half_ * 64u;
            uint qhOff = base + 128u + half_ * 32u;
            uint scIdx = half_ * 8u;
            uint aOff  = aBase + half_ * 128u;

            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            for (uint l = 0; l < 16u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sA0 * float(q1) * input_data[aOff + l];
                acc += sA1 * float(q2) * input_data[aOff + l + 32u];
                acc += sA2 * float(q3) * input_data[aOff + l + 64u];
                acc += sA3 * float(q4) * input_data[aOff + l + 96u];
            }
            for (uint l = 16; l < 32u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sB0 * float(q1) * input_data[aOff + l];
                acc += sB1 * float(q2) * input_data[aOff + l + 32u];
                acc += sB2 * float(q3) * input_data[aOff + l + 64u];
                acc += sB3 * float(q4) * input_data[aOff + l + 96u];
            }
        }
    }

    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    if (threadInRow == 0) output_data[rowBase] = acc;
}

// Q6_K with 16 threads per TG — one thread per super-block when K=4096.
// All threads active, reduction via a single `simd_sum` (no threadgroup
// scratch or barrier). Much better SIMD packing than the 64-thread version.
kernel void matmul_q6_k_tg16(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    uint row = tgid;
    if (row >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;
    device const uchar* raw = weight_data;

    for (uint sb = tid; sb < superBlocksPerRow; sb += 16u) {
        uint base = (row * superBlocksPerRow + sb) * 210u;
        uint aBase = sb * 256u;

        ushort dBits = uint(raw[base + 208]) | (uint(raw[base + 209]) << 8);
        float d = float(as_type<half>(dBits));

        uint qlBase = base;
        uint qhBase = base + 128u;
        uint scBase = base + 192u;

        float scales[16];
        for (uint s = 0; s < 16u; ++s) {
            scales[s] = d * float((char)raw[scBase + s]);
        }

        for (uint half_ = 0u; half_ < 2u; ++half_) {
            uint qlOff = qlBase + half_ * 64u;
            uint qhOff = qhBase + half_ * 32u;
            uint scIdx = half_ * 8u;
            uint aOff  = aBase + half_ * 128u;

            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            for (uint l = 0; l < 16u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;

                acc += sA0 * float(q1) * input_data[aOff + l];
                acc += sA1 * float(q2) * input_data[aOff + l + 32u];
                acc += sA2 * float(q3) * input_data[aOff + l + 64u];
                acc += sA3 * float(q4) * input_data[aOff + l + 96u];
            }
            for (uint l = 16; l < 32u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;

                acc += sB0 * float(q1) * input_data[aOff + l];
                acc += sB1 * float(q2) * input_data[aOff + l + 32u];
                acc += sB2 * float(q3) * input_data[aOff + l + 64u];
                acc += sB3 * float(q4) * input_data[aOff + l + 96u];
            }
        }
    }
    float total = simd_sum(acc);
    if (tid == 0) output_data[row] = total;
}

kernel void matmul_q6_k(
    device       float*        output_data [[buffer(0)]],
    device const uchar*        weight_data [[buffer(2)]],
    device const float*        input_data  [[buffer(1)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;
    device const uchar* raw = weight_data;

    for (uint sb = tid; sb < superBlocksPerRow; sb += TG_SIZE) {
        uint base = (row * superBlocksPerRow + sb) * 210u;
        uint aBase = sb * 256u;

        ushort dBits = uint(raw[base + 208]) | (uint(raw[base + 209]) << 8);
        float d = float(as_type<half>(dBits));

        uint qlBase = base;
        uint qhBase = base + 128u;
        uint scBase = base + 192u;

        // Hoist all 16 scales out of the inner loop — the original kernel
        // re-loaded them 16× per l-iteration.
        float scales[16];
        for (uint s = 0; s < 16u; ++s) {
            scales[s] = d * float((char)raw[scBase + s]);
        }

        for (uint half_ = 0u; half_ < 2u; ++half_) {
            uint qlOff = qlBase + half_ * 64u;
            uint qhOff = qhBase + half_ * 32u;
            uint scIdx = half_ * 8u;
            uint aOff  = aBase + half_ * 128u;

            // Per-half the 4 scale groups are constant across 16 l-iterations.
            // groupA: l in [0,16) → scIdx+0,+2,+4,+6
            // groupB: l in [16,32) → scIdx+1,+3,+5,+7
            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            // First 16 elements per quadrant (scales = A group).
            for (uint l = 0; l < 16u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;

                acc += sA0 * float(q1) * input_data[aOff + l];
                acc += sA1 * float(q2) * input_data[aOff + l + 32u];
                acc += sA2 * float(q3) * input_data[aOff + l + 64u];
                acc += sA3 * float(q4) * input_data[aOff + l + 96u];
            }
            // Second 16 elements per quadrant (scales = B group).
            for (uint l = 16; l < 32u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;

                acc += sB0 * float(q1) * input_data[aOff + l];
                acc += sB1 * float(q2) * input_data[aOff + l + 32u];
                acc += sB2 * float(q3) * input_data[aOff + l + 64u];
                acc += sB3 * float(q4) * input_data[aOff + l + 96u];
            }
        }
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q6_K aligned matmul (224-byte super-blocks, uint32 loads) ─────────────
// After repack each 256-element super-block is 56 uint32 words:
//   word 0       : low 16b fp16 d, high 16b pad
//   words 1..4   : 16 int8 scales (4 per word, little-endian)
//   words 5..36  : 128 ql bytes (4 per word)
//   words 37..52 : 64  qh bytes (4 per word)
//   words 53..55 : pad
kernel void matmul_q6_k_aligned(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint sbPerRow = p.K / 256u;
    float acc = 0.0f;

    for (uint sb = tid; sb < sbPerRow; sb += TG_SIZE) {
        uint wordBase = (row * sbPerRow + sb) * 56u;

        // d in low 16 bits of word 0.
        uint w0 = weight_u32[wordBase];
        float d = float(as_type<half>(ushort(w0 & 0xFFFFu)));

        // Unpack 16 int8 scales from 4 little-endian uint32 words into regs.
        float scales[16];
        for (uint w = 0; w < 4u; ++w) {
            uint packed = weight_u32[wordBase + 1u + w];
            for (uint bi = 0; bi < 4u; ++bi) {
                int s = int((packed >> (bi * 8u)) & 0xFFu);
                if (s >= 128) s -= 256;  // sign-extend int8
                scales[w * 4u + bi] = d * float(s);
            }
        }

        uint qlBase = wordBase + 5u;   // 32 words of ql (128 bytes)
        uint qhBase = wordBase + 37u;  // 16 words of qh (64 bytes)
        uint aBase  = sb * 256u;

        // Process two halves of 128 elements. Per half: 16 l-iterations × 4 quants.
        for (uint h = 0u; h < 2u; ++h) {
            uint qlWordBase = qlBase + h * 16u;   // 16 words per half (64 bytes)
            uint qhWordBase = qhBase + h * 8u;    // 8 words per half (32 bytes)
            uint aOff       = aBase + h * 128u;
            uint scIdx      = h * 8u;

            // Hoist the two scale groups used per-l into registers.
            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            // Inner body parameterised on (l, scales): compute 4 quants and FMA.
            for (uint l = 0; l < 32u; ++l) {
                // ql[l]      lives at byte (l) of ql half → word (l/4), byte (l%4).
                // ql[l+32]   lives at byte (l+32) → word (l/4 + 8), byte (l%4).
                uint ql0Word = weight_u32[qlWordBase + (l >> 2)];
                uint ql1Word = weight_u32[qlWordBase + (l >> 2) + 8u];
                uint qhWord  = weight_u32[qhWordBase + (l >> 2)];
                uint shift = (l & 3u) * 8u;
                uint ql0 = (ql0Word >> shift) & 0xFFu;
                uint ql1 = (ql1Word >> shift) & 0xFFu;
                uint qh0 = (qhWord  >> shift) & 0xFFu;
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;

                float s0, s1, s2, s3;
                if (l < 16u) { s0 = sA0; s1 = sA1; s2 = sA2; s3 = sA3; }
                else         { s0 = sB0; s1 = sB1; s2 = sB2; s3 = sB3; }

                acc += s0 * float(q1) * input_data[aOff + l];
                acc += s1 * float(q2) * input_data[aOff + l + 32u];
                acc += s2 * float(q3) * input_data[aOff + l + 64u];
                acc += s3 * float(q4) * input_data[aOff + l + 96u];
            }
        }
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = total;
}

// ── Q4_0 matmul — 4 rows per threadgroup (activation reuse) ───────────────
kernel void matmul_q4_0_4row(
    device       float*        output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    threadgroup float scratch[2];

    uint rowBase = tgid * ROWS;
    if (rowBase >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint aBase = b * 32u;

        float aBuf[32];
        for (uint i = 0; i < 32u; ++i) aBuf[i] = input_data[aBase + i];

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;

            uint blockOff = (row * blocksPerRow + b) * 18u;
            ushort scaleBits = uint(weight_data[blockOff])
                            | (uint(weight_data[blockOff + 1]) << 8);
            float scale = float(as_type<half>(scaleBits));

            float blockSum = 0.0f;
            for (uint i = 0; i < 16u; ++i) {
                uint packed = weight_data[blockOff + 2u + i];
                int lo = int(packed & 0xFu) - 8;
                int hi = int(packed >> 4)   - 8;
                blockSum += float(lo) * aBuf[i] + float(hi) * aBuf[i + 16u];
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = tg_reduce_add_64(accs[r], tid, scratch);
        if (tid == 0 && rowBase + r < p.N) output_data[rowBase + r] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ── Copy kernel ───────────────────────────────────────────────────────────
struct UintParams { uint n; uint extra0; uint extra1; uint extra2; };

kernel void copy_f32(
    device       float*       dst [[buffer(0)]],
    device const float*       src [[buffer(1)]],
    constant     UintParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = src[gid];
}

// Copy `n` floats from src[extra0 ..] to dst[0 ..].
kernel void copy_f32_region(
    device       float*       dst [[buffer(0)]],
    device const float*       src [[buffer(1)]],
    constant     UintParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = src[p.extra0 + gid];
}

// Copy `n` floats from src[extra0 ..] to dst[extra1 ..].
kernel void copy_f32_slice(
    device       float*       dst [[buffer(0)]],
    device const float*       src [[buffer(1)]],
    constant     UintParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[p.extra1 + gid] = src[p.extra0 + gid];
}

kernel void zero_f32(
    device       float*       dst [[buffer(0)]],
    constant     UintParams&  p   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = 0.0f;
}

kernel void fill_f32(
    device       float*       dst   [[buffer(0)]],
    constant     UintParams&  p     [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = as_type<float>(p.extra0);
}

// ── Element-wise add/mul ──────────────────────────────────────────────────
struct ElementParams { uint n; };

kernel void element_add(
    device       float*          out [[buffer(0)]],
    device const float*          a   [[buffer(1)]],
    device const float*          b   [[buffer(2)]],
    constant     ElementParams&  p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) out[gid] = a[gid] + b[gid];
}

// Broadcast bias add: out[m, i] = a[m, i] + bias[i]  for m ∈ [0, M), i ∈ [0, rowDim)
// Total threads = M × rowDim. p.n = total elements; p.extra0 = rowDim.
struct BroadcastAddParams { uint n; uint rowDim; };
kernel void element_add_broadcast_row(
    device       float*               out  [[buffer(0)]],
    device const float*               a    [[buffer(1)]],
    device const float*               bias [[buffer(2)]],
    constant     BroadcastAddParams&  p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    uint col = gid % p.rowDim;
    out[gid] = a[gid] + bias[col];
}

kernel void element_mul(
    device       float*          out [[buffer(0)]],
    device const float*          a   [[buffer(1)]],
    device const float*          b   [[buffer(2)]],
    constant     ElementParams&  p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) out[gid] = a[gid] * b[gid];
}

// ── SiLU ──────────────────────────────────────────────────────────────────
kernel void silu(
    device       float*          out [[buffer(0)]],
    device const float*          in  [[buffer(1)]],
    constant     ElementParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = in[gid];
    out[gid] = x / (1.0f + exp(-x));
}

kernel void silu_in_place(
    device       float*          data [[buffer(0)]],
    constant     ElementParams&  p    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = data[gid];
    data[gid] = x / (1.0f + exp(-x));
}

// out[i] = data[i] * silu(gate[i])
struct SiLUGateParams { uint n; };
kernel void silu_gate(
    device       float*          out  [[buffer(0)]],
    device const float*          data [[buffer(1)]],
    device const float*          gate [[buffer(2)]],
    constant     SiLUGateParams& p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float g = gate[gid];
    out[gid] = data[gid] * (g / (1.0f + exp(-g)));
}

// out[i] = silu(fused[i]) * fused[N + i]
struct SplitSwiGLUParams { uint n; };
kernel void split_swiglu(
    device       float*                out    [[buffer(0)]],
    device const float*                fused  [[buffer(1)]],
    constant     SplitSwiGLUParams&    p      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float g = fused[gid];
    float u = fused[p.n + gid];
    out[gid] = (g / (1.0f + exp(-g))) * u;
}

// data[i] = max(0,x)^2
kernel void squared_relu(
    device       float*          data [[buffer(0)]],
    constant     ElementParams&  p    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = max(0.0f, data[gid]);
    data[gid] = x * x;
}

// ── RmsNorm ───────────────────────────────────────────────────────────────
// out[i] = input[i] * rsqrt(mean(input²)+eps) * weight[i]
// One threadgroup of 256, uses SIMD reductions.
struct RmsNormParams { uint n; float eps; };

kernel void rmsnorm(
    device       float*           out    [[buffer(0)]],
    device const float*           in_    [[buffer(1)]],
    device const float*           weight [[buffer(2)]],
    constant     RmsNormParams&   p      [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    // Per-row when dispatched with gridX=M: each TG handles one row of p.n
    // elements. gridX=1 single-row callers still work (tgid == 0 → off == 0).
    uint off = tgid * p.n;
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = in_[off + i];
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    float inv_rms = 0.0f;
    if (tid == 0) {
        inv_rms = rsqrt(total / float(p.n) + p.eps);
        scratch[0] = inv_rms;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u) {
        out[off + i] = in_[off + i] * inv_rms * weight[i];
    }
}

// Per-head RmsNorm (in-place): one head per threadgroup.
struct PerHeadRmsNormParams { uint numHeads; uint headDim; float eps; };

kernel void per_head_rmsnorm(
    device       float*                data   [[buffer(0)]],
    device const float*                weight [[buffer(1)]],
    constant     PerHeadRmsNormParams& p      [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.numHeads) return;
    threadgroup float scratch[8];
    uint off = tgid * p.headDim;

    float local_sum = 0.0f;
    for (uint i = tid; i < p.headDim; i += 256u) {
        float v = data[off + i];
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.headDim) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.headDim; i += 256u) {
        data[off + i] = data[off + i] * inv_rms * weight[i];
    }
}

// ── Softmax (in-place on a 1D float vector) ───────────────────────────────
kernel void softmax(
    device       float*          out [[buffer(0)]],
    device const float*          in_ [[buffer(1)]],
    constant     ElementParams&  p   [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    float local_max = -INFINITY;
    for (uint i = tid; i < p.n; i += 256u) local_max = max(local_max, in_[i]);
    float max_val = tg_reduce_max_256(local_max, tid, scratch);
    if (tid == 0) scratch[0] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = scratch[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float e = exp(in_[i] - max_val);
        out[i] = e;
        local_sum += e;
    }
    float sum_val = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / scratch[0];
    for (uint i = tid; i < p.n; i += 256u) out[i] *= inv;
}

// ── RoPE (interleaved pairs (2i, 2i+1)) ────────────────────────────────────
// Apply in place to q and k. freqFactors ∈ {null, [ropeDim/2]}.
struct RoPEParams {
    uint qTotal;
    uint kTotal;
    uint headDim;
    uint ropeDim;
    int  positionOffset;
    float ropeTheta;
    uint useFreqFactors; // 0 / 1
    uint neox;           // 0 = interleaved, 1 = neox (pair (i, i+ropeDim/2))
};

kernel void rope(
    device       float*       q_data       [[buffer(0)]],
    device       float*       k_data       [[buffer(1)]],
    device const float*       freq_factors [[buffer(2)]], // may be ignored
    constant     RoPEParams&  p            [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint maxPairs = max(p.qTotal, p.kTotal) / 2u;
    if (gid >= maxPairs) return;

    // For each (head, pairIndex), compute the rotation once.
    // Q path
    if (gid * 2u < p.qTotal) {
        uint headIdx = (gid * 2u) / p.headDim;
        uint dimIdx = (gid * 2u) % p.headDim;
        if (dimIdx < p.ropeDim) {
            uint pairIdx = (p.neox != 0u) ? (dimIdx) : (dimIdx / 2u); // neox pair index is just dim
            // interleaved: pair index is dimIdx/2; dim index in the freq table likewise.
            float base_idx;
            if (p.neox != 0u) {
                // In NEOX mode we don't actually enter here (see separate NEOX dispatch).
                base_idx = float(dimIdx);
            } else {
                base_idx = float(dimIdx);
            }
            float freq = pow(p.ropeTheta, -base_idx / float(p.ropeDim));
            if (p.useFreqFactors != 0u) freq = freq / freq_factors[dimIdx / 2u];
            float angle = float(p.positionOffset) * freq;
            float c = cos(angle), s = sin(angle);
            float x0 = q_data[gid * 2u];
            float x1 = q_data[gid * 2u + 1u];
            q_data[gid * 2u]      = x0 * c - x1 * s;
            q_data[gid * 2u + 1u] = x0 * s + x1 * c;
        }
    }
    // K path
    if (gid * 2u < p.kTotal) {
        uint headIdx = (gid * 2u) / p.headDim;
        uint dimIdx = (gid * 2u) % p.headDim;
        if (dimIdx < p.ropeDim) {
            float base_idx = float(dimIdx);
            float freq = pow(p.ropeTheta, -base_idx / float(p.ropeDim));
            if (p.useFreqFactors != 0u) freq = freq / freq_factors[dimIdx / 2u];
            float angle = float(p.positionOffset) * freq;
            float c = cos(angle), s = sin(angle);
            float x0 = k_data[gid * 2u];
            float x1 = k_data[gid * 2u + 1u];
            k_data[gid * 2u]      = x0 * c - x1 * s;
            k_data[gid * 2u + 1u] = x0 * s + x1 * c;
        }
    }
}

// NEOX RoPE: pairs are (i, i + ropeDim/2) inside each head.
kernel void rope_neox(
    device       float*       q_data       [[buffer(0)]],
    device       float*       k_data       [[buffer(1)]],
    device const float*       freq_factors [[buffer(2)]],
    constant     RoPEParams&  p            [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint halfDim = p.ropeDim / 2u;
    uint qHeads = p.qTotal / p.headDim;
    uint kHeads = p.kTotal / p.headDim;

    // Q
    if (gid < qHeads * halfDim) {
        uint head = gid / halfDim;
        uint i = gid % halfDim;
        float base_idx = float(i * 2u);
        float freq = pow(p.ropeTheta, -base_idx / float(p.ropeDim));
        if (p.useFreqFactors != 0u) freq = freq / freq_factors[i];
        float angle = float(p.positionOffset) * freq;
        float c = cos(angle), s = sin(angle);
        uint base = head * p.headDim;
        float x0 = q_data[base + i];
        float x1 = q_data[base + i + halfDim];
        q_data[base + i]           = x0 * c - x1 * s;
        q_data[base + i + halfDim] = x0 * s + x1 * c;
    }
    // K
    if (gid < kHeads * halfDim) {
        uint head = gid / halfDim;
        uint i = gid % halfDim;
        float base_idx = float(i * 2u);
        float freq = pow(p.ropeTheta, -base_idx / float(p.ropeDim));
        if (p.useFreqFactors != 0u) freq = freq / freq_factors[i];
        float angle = float(p.positionOffset) * freq;
        float c = cos(angle), s = sin(angle);
        uint base = head * p.headDim;
        float x0 = k_data[base + i];
        float x1 = k_data[base + i + halfDim];
        k_data[base + i]           = x0 * c - x1 * s;
        k_data[base + i + halfDim] = x0 * s + x1 * c;
    }
}

// ── Batched RoPE (prefill): each token gets positionOffset + tokenIdx ──
struct BatchedRoPEParams {
    uint qTotal;        // M × numHeads × headDim
    uint kTotal;        // M × numKvHeads × headDim
    uint headDim;
    uint ropeDim;
    int  positionOffset;
    float ropeTheta;
    uint numHeads;
    uint numKvHeads;
};

kernel void batched_rope(
    device       float*              q_data [[buffer(0)]],
    device       float*              k_data [[buffer(1)]],
    constant     BatchedRoPEParams&  p      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint maxPairs = max(p.qTotal, p.kTotal) / 2u;
    if (gid >= maxPairs) return;

    uint qHeadPairs = p.headDim / 2u;
    uint qPairsPerToken = p.numHeads   * qHeadPairs;
    uint kPairsPerToken = p.numKvHeads * qHeadPairs;

    // Q path
    if (gid * 2u < p.qTotal) {
        uint tokenIdx = gid / qPairsPerToken;
        uint pairInRow = gid % qPairsPerToken;
        uint pairInHead = pairInRow % qHeadPairs;
        uint dimIdx = pairInHead * 2u;
        if (dimIdx < p.ropeDim) {
            float base_idx = float(dimIdx);
            float freq = pow(p.ropeTheta, -base_idx / float(p.ropeDim));
            float angle = float(p.positionOffset + int(tokenIdx)) * freq;
            float c = cos(angle), s = sin(angle);
            float x0 = q_data[gid * 2u];
            float x1 = q_data[gid * 2u + 1u];
            q_data[gid * 2u]      = x0 * c - x1 * s;
            q_data[gid * 2u + 1u] = x0 * s + x1 * c;
        }
    }
    // K path
    if (gid * 2u < p.kTotal) {
        uint tokenIdx = gid / kPairsPerToken;
        uint pairInRow = gid % kPairsPerToken;
        uint pairInHead = pairInRow % qHeadPairs;
        uint dimIdx = pairInHead * 2u;
        if (dimIdx < p.ropeDim) {
            float base_idx = float(dimIdx);
            float freq = pow(p.ropeTheta, -base_idx / float(p.ropeDim));
            float angle = float(p.positionOffset + int(tokenIdx)) * freq;
            float c = cos(angle), s = sin(angle);
            float x0 = k_data[gid * 2u];
            float x1 = k_data[gid * 2u + 1u];
            k_data[gid * 2u]      = x0 * c - x1 * s;
            k_data[gid * 2u + 1u] = x0 * s + x1 * c;
        }
    }
}

// ── SplitQKV ──────────────────────────────────────────────────────────────
struct SplitQKVParams { uint innerSize; };
kernel void split_qkv(
    device       float*              q    [[buffer(0)]],
    device       float*              k    [[buffer(1)]],
    device       float*              v    [[buffer(2)]],
    device const float*              qkv  [[buffer(3)]],
    constant     SplitQKVParams&     p    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.innerSize) return;
    q[gid] = qkv[gid];
    k[gid] = qkv[p.innerSize + gid];
    v[gid] = qkv[p.innerSize * 2u + gid];
}

// ── De-interleave Q: [q0, g0, q1, g1, ...] → q:[q0,q1,...], g:[g0,g1,...] ─
struct DeInterleaveParams { uint numHeads; uint headDim; };
kernel void de_interleave_q(
    device       float*                 qAttn [[buffer(0)]],
    device       float*                 qGate [[buffer(1)]],
    device const float*                 qFull [[buffer(2)]],
    constant     DeInterleaveParams&    p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.numHeads * p.headDim;
    if (gid >= total) return;
    uint head = gid / p.headDim;
    uint dim  = gid % p.headDim;
    qAttn[gid] = qFull[head * 2u * p.headDim + dim];
    qGate[gid] = qFull[head * 2u * p.headDim + p.headDim + dim];
}

// ── KV cache write (F32 cache; we don't enable FP16 cache on Metal yet) ───
struct KvWriteParams {
    uint nKvHeads;
    uint keyLength;
    uint valueLength;
    uint maxSeqLen;
    uint position;
};
kernel void kv_cache_write_f32(
    device       float*          kCache [[buffer(0)]],
    device       float*          vCache [[buffer(1)]],
    device const float*          k      [[buffer(2)]],
    device const float*          v      [[buffer(3)]],
    constant     KvWriteParams&  p      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint kMax = p.nKvHeads * p.keyLength;
    uint vMax = p.nKvHeads * p.valueLength;
    uint maxE = max(kMax, vMax);
    if (gid >= maxE) return;
    if (gid < kMax) {
        uint h = gid / p.keyLength;
        uint d = gid % p.keyLength;
        uint off = (h * p.maxSeqLen + p.position) * p.keyLength + d;
        kCache[off] = k[gid];
    }
    if (gid < vMax) {
        uint h = gid / p.valueLength;
        uint d = gid % p.valueLength;
        uint off = (h * p.maxSeqLen + p.position) * p.valueLength + d;
        vCache[off] = v[gid];
    }
}

// ── Gated attention: per-head tiled with online softmax ───────────────────
struct GatedAttnParams {
    uint numHeads;
    uint numKvHeads;
    uint keyLength;
    uint valueLength;
    uint maxSeqLen;
    uint seqLen;
    float scale;
};
kernel void gated_attention(
    device       float*             out    [[buffer(0)]],
    device const float*             qAttn  [[buffer(1)]],
    device const float*             qGate  [[buffer(2)]],
    device const float*             kCache [[buffer(3)]],
    device const float*             vCache [[buffer(4)]],
    constant     GatedAttnParams&   p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.numHeads) return;
    uint h = tgid;
    threadgroup float tile_scores[256];
    threadgroup float scratch[8];

    uint headsPerGroup = p.numHeads / p.numKvHeads;
    uint kvHead = h / headsPerGroup;
    uint qOff = h * p.keyLength;
    uint kvKBase = kvHead * p.maxSeqLen * p.keyLength;
    uint kvVBase = kvHead * p.maxSeqLen * p.valueLength;
    uint outOff = h * p.valueLength;

    // Zero output once
    for (uint d = tid; d < p.valueLength; d += 256u) out[outOff + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const uint TILE = 256u;

    for (uint tileStart = 0u; tileStart < p.seqLen; tileStart += TILE) {
        uint tileEnd = min(tileStart + TILE, p.seqLen);
        uint tileLen = tileEnd - tileStart;

        // Score for this thread's position within the tile
        float my_score = -INFINITY;
        if (tid < tileLen) {
            uint pos = tileStart + tid;
            float dot = 0.0f;
            for (uint d = 0; d < p.keyLength; ++d)
                dot += qAttn[qOff + d] * kCache[kvKBase + pos * p.keyLength + d];
            my_score = dot * p.scale;
        }
        tile_scores[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_max = tg_reduce_max_256(my_score, tid, scratch);
        if (tid == 0) scratch[0] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = scratch[0];

        float my_exp = (tid < tileLen) ? exp(tile_scores[tid] - tile_max) : 0.0f;
        tile_scores[tid] = my_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_sum = tg_reduce_add_256(my_exp, tid, scratch);
        if (tid == 0) scratch[0] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = scratch[0];

        float new_max = max(running_max, tile_max);
        float corr_old = exp(running_max - new_max);
        float corr_new = exp(tile_max    - new_max);

        for (uint d = tid; d < p.valueLength; d += 256u) {
            float tile_val = 0.0f;
            for (uint t = 0; t < tileLen; ++t)
                tile_val += tile_scores[t] * vCache[kvVBase + (tileStart + t) * p.valueLength + d];
            out[outOff + d] = out[outOff + d] * corr_old + tile_val * corr_new;
        }
        running_sum = running_sum * corr_old + tile_sum * corr_new;
        running_max = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint d = tid; d < p.valueLength; d += 256u) {
        float gate_val = 1.0f;
        if (d < p.keyLength) {
            float gv = qGate[h * p.keyLength + d];
            gate_val = 1.0f / (1.0f + exp(-gv));
        }
        out[outOff + d] = out[outOff + d] * inv_sum * gate_val;
    }
}

// ── Batched gated attention (prefill) ─────────────────────────────────────
// Grid: gridX = M × numHeads. tgid encodes (tokenIdx, head).
// seqLen for tokenIdx = startPosition + tokenIdx + 1 (causal).
struct BatchedGatedAttnParams {
    uint numHeads;
    uint numKvHeads;
    uint keyLength;
    uint valueLength;
    uint maxSeqLen;
    uint startPosition;
    uint M;
    float scale;
};
kernel void batched_gated_attention(
    device       float*                     out    [[buffer(0)]],
    device const float*                     qAttn  [[buffer(1)]],
    device const float*                     qGate  [[buffer(2)]],
    device const float*                     kCache [[buffer(3)]],
    device const float*                     vCache [[buffer(4)]],
    constant     BatchedGatedAttnParams&    p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    uint totalWork = p.M * p.numHeads;
    if (tgid >= totalWork) return;
    uint tokenIdx = tgid / p.numHeads;
    uint h        = tgid % p.numHeads;
    uint seqLen   = p.startPosition + tokenIdx + 1u;

    threadgroup float tile_scores[256];
    threadgroup float scratch[8];

    uint headsPerGroup = p.numHeads / p.numKvHeads;
    uint kvHead = h / headsPerGroup;
    uint qBase = tokenIdx * p.numHeads * p.keyLength + h * p.keyLength;
    uint gBase = qBase;   // qGate layout matches qAttn
    uint outBase = tokenIdx * p.numHeads * p.valueLength + h * p.valueLength;
    uint kvKBase = kvHead * p.maxSeqLen * p.keyLength;
    uint kvVBase = kvHead * p.maxSeqLen * p.valueLength;

    for (uint d = tid; d < p.valueLength; d += 256u) out[outBase + d] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const uint TILE = 256u;

    for (uint tileStart = 0u; tileStart < seqLen; tileStart += TILE) {
        uint tileEnd = min(tileStart + TILE, seqLen);
        uint tileLen = tileEnd - tileStart;

        float my_score = -INFINITY;
        if (tid < tileLen) {
            uint pos = tileStart + tid;
            float dot = 0.0f;
            for (uint d = 0; d < p.keyLength; ++d)
                dot += qAttn[qBase + d] * kCache[kvKBase + pos * p.keyLength + d];
            my_score = dot * p.scale;
        }
        tile_scores[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_max = tg_reduce_max_256(my_score, tid, scratch);
        if (tid == 0) scratch[0] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = scratch[0];

        float my_exp = (tid < tileLen) ? exp(tile_scores[tid] - tile_max) : 0.0f;
        tile_scores[tid] = my_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_sum = tg_reduce_add_256(my_exp, tid, scratch);
        if (tid == 0) scratch[0] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = scratch[0];

        float new_max = max(running_max, tile_max);
        float corr_old = exp(running_max - new_max);
        float corr_new = exp(tile_max    - new_max);

        for (uint d = tid; d < p.valueLength; d += 256u) {
            float tile_val = 0.0f;
            for (uint t = 0; t < tileLen; ++t)
                tile_val += tile_scores[t] * vCache[kvVBase + (tileStart + t) * p.valueLength + d];
            out[outBase + d] = out[outBase + d] * corr_old + tile_val * corr_new;
        }
        running_sum = running_sum * corr_old + tile_sum * corr_new;
        running_max = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint d = tid; d < p.valueLength; d += 256u) {
        float gate_val = 1.0f;
        if (d < p.keyLength) {
            float gv = qGate[gBase + d];
            gate_val = 1.0f / (1.0f + exp(-gv));
        }
        out[outBase + d] = out[outBase + d] * inv_sum * gate_val;
    }
}

// ── Batched KV cache write (prefill) ──────────────────────────────────────
// Writes M tokens' (k,v) into the cache starting at startPosition.
// For each token t, k/v row t goes to cache slot (startPosition + t).
struct BatchedKvWriteParams {
    uint nKvHeads;
    uint keyLength;
    uint valueLength;
    uint maxSeqLen;
    uint startPosition;
    uint M;
};
kernel void batched_kv_cache_write(
    device       float*                     kCache [[buffer(0)]],
    device       float*                     vCache [[buffer(1)]],
    device const float*                     k      [[buffer(2)]],
    device const float*                     v      [[buffer(3)]],
    constant     BatchedKvWriteParams&      p      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint kRowElems = p.nKvHeads * p.keyLength;
    uint vRowElems = p.nKvHeads * p.valueLength;
    uint maxElems  = max(kRowElems, vRowElems) * p.M;
    if (gid >= maxElems) return;

    // K path
    uint kTotal = kRowElems * p.M;
    if (gid < kTotal) {
        uint t       = gid / kRowElems;
        uint inRow   = gid % kRowElems;
        uint h       = inRow / p.keyLength;
        uint d       = inRow % p.keyLength;
        uint dstSlot = p.startPosition + t;
        uint dstOff  = h * p.maxSeqLen * p.keyLength + dstSlot * p.keyLength + d;
        kCache[dstOff] = k[gid];
    }
    // V path
    uint vTotal = vRowElems * p.M;
    if (gid < vTotal) {
        uint t       = gid / vRowElems;
        uint inRow   = gid % vRowElems;
        uint h       = inRow / p.valueLength;
        uint d       = inRow % p.valueLength;
        uint dstSlot = p.startPosition + t;
        uint dstOff  = h * p.maxSeqLen * p.valueLength + dstSlot * p.valueLength + d;
        vCache[dstOff] = v[gid];
    }
}

// ── Batched embedding lookup (prefill) ────────────────────────────────────
// For each of M token IDs, write [hiddenDim] floats of the embedding row to
// output[t*hiddenDim..]. Handles only the common F32/F16/Q8_0/Q4_0 aligned
// cases here — other types fall back to the C# loop in MetalBackend.
struct BatchedEmbedParams {
    uint hiddenDim;
    uint tableType;   // 0=f32 1=q8_0 2=f16 5=q4_0 8=q4_0 aligned
    uint M;
};
kernel void batched_embedding_lookup(
    device       float*                     out       [[buffer(0)]],
    device const uchar*                     table     [[buffer(1)]],
    device const int*                       tokenIds  [[buffer(2)]],
    constant     BatchedEmbedParams&        p         [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.M) return;
    uint tokenId = uint(tokenIds[tgid]);
    uint outBase = tgid * p.hiddenDim;

    if (p.tableType == 0u) {
        device const float* t = (device const float*)table;
        for (uint i = tid; i < p.hiddenDim; i += 256u)
            out[outBase + i] = t[tokenId * p.hiddenDim + i];
    }
    else if (p.tableType == 2u) {
        device const half* t = (device const half*)table;
        for (uint i = tid; i < p.hiddenDim; i += 256u)
            out[outBase + i] = float(t[tokenId * p.hiddenDim + i]);
    }
    else if (p.tableType == 1u) {
        // Q8_0: 34 bytes per 32-weight block — 2 byte scale + 32 int8 weights
        uint blocksPerRow = p.hiddenDim / 32u;
        device const uchar* rowBase = table + tokenId * blocksPerRow * 34u;
        for (uint i = tid; i < p.hiddenDim; i += 256u) {
            uint blk = i / 32u;
            uint inb = i % 32u;
            device const uchar* blkPtr = rowBase + blk * 34u;
            ushort scaleBits = (ushort(blkPtr[1]) << 8) | ushort(blkPtr[0]);
            float scale = float(as_type<half>(scaleBits));
            int w = int((int8_t)blkPtr[2u + inb]);
            out[outBase + i] = float(w) * scale;
        }
    }
    else if (p.tableType == 8u) {
        // Q4_0 aligned: 20 bytes per block (u32[0]=scale, u32[1..4]=nibbles)
        uint blocksPerRow = p.hiddenDim / 32u;
        device const uint* rowBase = (device const uint*)(table + tokenId * blocksPerRow * 20u);
        for (uint i = tid; i < p.hiddenDim; i += 256u) {
            uint blk = i / 32u;
            uint inb = i % 32u;
            device const uint* blkPtr = rowBase + blk * 5u;
            float scale = float(as_type<half>(ushort(blkPtr[0] & 0xFFFFu)));
            // Nibble layout: inb < 16 → low nibble of byte[inb] in u32[1+inb/4]
            //                inb >= 16 → high nibble of byte[inb-16] in u32[1+(inb-16)/4]
            uint b = inb & 0xFu;
            uint u = blkPtr[1u + (b >> 2)];
            uint shift = (b & 3u) * 8u;
            uint nib = (inb < 16u) ? ((u >> shift) & 0xFu)
                                   : ((u >> (shift + 4u)) & 0xFu);
            out[outBase + i] = float(int(nib) - 8) * scale;
        }
    }
    else if (p.tableType == 5u) {
        // Q4_0 packed (18 bytes per block: 2 byte scale + 16 nibble bytes)
        uint blocksPerRow = p.hiddenDim / 32u;
        device const uchar* rowBase = table + tokenId * blocksPerRow * 18u;
        for (uint i = tid; i < p.hiddenDim; i += 256u) {
            uint blk = i / 32u;
            uint inb = i % 32u;
            device const uchar* blkPtr = rowBase + blk * 18u;
            ushort scaleBits = (ushort(blkPtr[1]) << 8) | ushort(blkPtr[0]);
            float scale = float(as_type<half>(scaleBits));
            uint b = inb & 0xFu;
            uchar byte = blkPtr[2u + b];
            uint nib = (inb < 16u) ? (byte & 0xFu) : (uint(byte) >> 4);
            out[outBase + i] = float(int(nib) - 8) * scale;
        }
    }
}

// ── DeltaNet step ─────────────────────────────────────────────────────────
struct DeltaNetParams {
    uint groupCount;
    uint headDim;
    float scale;
    float normEps;
};
kernel void deltanet_step(
    device       float*              out    [[buffer(0)]],
    device const float*              q      [[buffer(1)]],
    device const float*              k      [[buffer(2)]],
    device const float*              v      [[buffer(3)]],
    device       float*              state  [[buffer(4)]],
    device const float*              decay  [[buffer(5)]],
    device const float*              beta_  [[buffer(6)]],
    device const float*              normW  [[buffer(7)]],
    constant     DeltaNetParams&     p      [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.groupCount) return;
    uint g = tgid;
    threadgroup float scratch[8];

    uint baseOff  = g * p.headDim;
    uint stateOff = g * p.headDim * p.headDim;
    float d = decay[g];
    float b = beta_[g];

    // Step 1+2+3: For each column j, compute sk_j, error_j, then update state column j.
    for (uint j = tid; j < p.headDim; j += 256u) {
        float sk = 0.0f;
        for (uint i = 0; i < p.headDim; ++i)
            sk += state[stateOff + i * p.headDim + j] * k[baseOff + i];
        float err_j = (v[baseOff + j] - d * sk) * b;
        for (uint i = 0; i < p.headDim; ++i) {
            uint idx = stateOff + i * p.headDim + j;
            state[idx] = d * state[idx] + k[baseOff + i] * err_j;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: o[j] = Sᵀ·q*scale
    for (uint j = tid; j < p.headDim; j += 256u) {
        float sum = 0.0f;
        for (uint i = 0; i < p.headDim; ++i)
            sum += state[stateOff + i * p.headDim + j] * q[baseOff + i];
        out[baseOff + j] = sum * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Per-head RMSNorm with weight
    float local_sum = 0.0f;
    for (uint i = tid; i < p.headDim; i += 256u) {
        float vv = out[baseOff + i];
        local_sum += vv * vv;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.headDim) + p.normEps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.headDim; i += 256u)
        out[baseOff + i] = out[baseOff + i] * inv_rms * normW[i];
}

// ── Batched fused DeltaNet prefill ────────────────────────────────────────
// Replaces the entire per-token DeltaNet inner loop for prefill. Fuses
// (per token, inline): split QKV, L2-normalize Q/K, compute decay/beta,
// DeltaNet state update, per-head RmsNorm, SiLU gate, and output write.
//
// Key architectural wins vs the per-token path:
//   1. ONE dispatch per DeltaNet layer instead of ~14 × M dispatches.
//   2. Zero CopyTensorSlice overhead — operates directly on batched buffers.
//   3. Inner M loop inside a single kernel launch.
//
// Grid: gridX = numVHeads TGs, 256 threads per TG. Each TG handles one
// V head across all M tokens sequentially (state is order-dependent).
//
// Per-head DeltaNet state is kept in GLOBAL memory (state[numVHeads ×
// headDim × headDim]). For Qwen3.5-9B headDim=128 → 64 KiB per head, too
// large for threadgroup memory. Global residency is fine because only this
// TG accesses this head's state region, and the unified memory on Apple
// Silicon makes global reads cheap (essentially TG-memory speed for well-
// aligned access).
//
// Threadgroup memory: sQ, sK, sV, sOut (headDim each) + scratch[8]. Stays
// under 4 KiB for headDim up to 128.
constant constexpr uint DN_MAX_HEAD_DIM = 128u;

struct BatchedDeltaNetParams {
    uint  M;
    uint  qkvOutDim;    // 2*keyDim + valueDim
    uint  keyDim;       // numKHeads * headDim
    uint  valueDim;     // numVHeads * headDim
    uint  numKHeads;
    uint  numVHeads;
    uint  headDim;
    uint  repeatFactor; // numVHeads / numKHeads
    float scale;
    float normEps;
};

kernel void batched_deltanet_fused(
    device       float*                       out       [[buffer(0)]],
    device const float*                       qkv       [[buffer(1)]],
    device const float*                       alpha     [[buffer(2)]],
    device const float*                       beta_     [[buffer(3)]],
    device const float*                       gate_     [[buffer(4)]],
    device       float*                       state     [[buffer(5)]],
    device const float*                       ssmA      [[buffer(6)]],
    device const float*                       dtBias    [[buffer(7)]],
    device const float*                       normW     [[buffer(8)]],
    constant     BatchedDeltaNetParams&       p         [[buffer(9)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    const uint g = tgid;
    if (g >= p.numVHeads) return;
    // RepeatTile replicates [0, keyDim) as data[gid] = data[gid % keyDim], so
    // V head g reads Q/K from K-head (g mod numKHeads), NOT (g / repeatFactor).
    const uint gKV = g % p.numKHeads;
    const uint hd  = p.headDim;
    const uint stateSize = hd * hd;
    const uint stateOff  = g * stateSize;

    threadgroup float sQ[DN_MAX_HEAD_DIM];
    threadgroup float sK[DN_MAX_HEAD_DIM];
    threadgroup float sV[DN_MAX_HEAD_DIM];
    threadgroup float sOut[DN_MAX_HEAD_DIM];
    threadgroup float scratch[8];

    // Per-head state stays in global memory (too large for TG mem at
    // headDim=128). Each TG exclusively owns its head's state slab so no
    // cross-TG synchronization is needed.
    device float* sState = state + stateOff;

    const float ssmA_g   = ssmA[g];
    const float dtBias_g = dtBias[g];

    for (uint t = 0u; t < p.M; ++t) {
        const uint qkvBase = t * p.qkvOutDim;
        const uint qSrc = qkvBase + gKV * hd;
        const uint kSrc = qkvBase + p.keyDim + gKV * hd;
        const uint vSrc = qkvBase + 2u * p.keyDim + g * hd;

        // Load Q, K, V for this head, this token.
        for (uint i = tid; i < hd; i += 256u) {
            sQ[i] = qkv[qSrc + i];
            sK[i] = qkv[kSrc + i];
            sV[i] = qkv[vSrc + i];
        }
        // Flush both device (qkv reads) and threadgroup (sQ/sK/sV writes).
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // L2 normalize Q per head: q / sqrt(sum_sq + 1e-12).
        float qsum = 0.0f;
        for (uint i = tid; i < hd; i += 256u) qsum += sQ[i] * sQ[i];
        float qtotal = tg_reduce_add_256(qsum, tid, scratch);
        if (tid == 0) scratch[0] = rsqrt(qtotal + 1e-12f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float qinv = scratch[0];
        for (uint i = tid; i < hd; i += 256u) sQ[i] = sQ[i] * qinv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // L2 normalize K per head.
        float ksum = 0.0f;
        for (uint i = tid; i < hd; i += 256u) ksum += sK[i] * sK[i];
        float ktotal = tg_reduce_add_256(ksum, tid, scratch);
        if (tid == 0) scratch[0] = rsqrt(ktotal + 1e-12f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kinv = scratch[0];
        for (uint i = tid; i < hd; i += 256u) sK[i] = sK[i] * kinv;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute decay & beta for this head & token.
        float alpha_tg = alpha[t * p.numVHeads + g];
        float beta_tg  = beta_[t * p.numVHeads + g];
        float sp = log(1.0f + exp(alpha_tg + dtBias_g));
        float d  = exp(ssmA_g * sp);
        float bt = 1.0f / (1.0f + exp(-beta_tg));

        // DeltaNet state update: per-column work (each thread owns columns
        // j, j+256, ...). sk_j = Σᵢ S[i,j]·k[i]; err_j = (v[j] − d·sk_j)·β;
        // S[i,j] ← d·S[i,j] + k[i]·err_j   (all i)
        // State is in device memory; each thread owns disjoint columns so
        // no cross-thread race within this phase, but we need mem_device to
        // ensure writes are visible to subsequent reads.
        for (uint j = tid; j < hd; j += 256u) {
            float sk = 0.0f;
            for (uint i = 0u; i < hd; ++i)
                sk += sState[i * hd + j] * sK[i];
            float err_j = (sV[j] - d * sk) * bt;
            for (uint i = 0u; i < hd; ++i) {
                uint idx = i * hd + j;
                sState[idx] = d * sState[idx] + sK[i] * err_j;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        // Output: o[j] = Σᵢ S[i,j]·q[i] · scale
        for (uint j = tid; j < hd; j += 256u) {
            float sum = 0.0f;
            for (uint i = 0u; i < hd; ++i)
                sum += sState[i * hd + j] * sQ[i];
            sOut[j] = sum * p.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Per-head RmsNorm with weight.
        float osum = 0.0f;
        for (uint i = tid; i < hd; i += 256u) osum += sOut[i] * sOut[i];
        float ototal = tg_reduce_add_256(osum, tid, scratch);
        if (tid == 0) scratch[0] = rsqrt(ototal / float(hd) + p.normEps);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv_rms = scratch[0];
        for (uint i = tid; i < hd; i += 256u) sOut[i] = sOut[i] * inv_rms * normW[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SiLU gate and write to batched output: out[t,g,i] = sOut[i] * silu(gate[t,g,i])
        const uint gateOff = t * p.valueDim + g * hd;
        const uint outOff  = t * p.valueDim + g * hd;
        for (uint i = tid; i < hd; i += 256u) {
            float gv = gate_[gateOff + i];
            float silu_g = gv / (1.0f + exp(-gv));
            out[outOff + i] = sOut[i] * silu_g;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // State is updated in place in device memory; no final write-back needed.
}

// ── Compute decay/beta ────────────────────────────────────────────────────
struct DecayBetaParams { uint groupCount; };
kernel void compute_decay_beta(
    device       float*              decay   [[buffer(0)]],
    device       float*              beta_   [[buffer(1)]],
    device const float*              alphaP  [[buffer(2)]],
    device const float*              betaP   [[buffer(3)]],
    device const float*              ssmA    [[buffer(4)]],
    device const float*              dtBias  [[buffer(5)]],
    constant     DecayBetaParams&    p       [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.groupCount) return;
    float sp = log(1.0f + exp(alphaP[gid] + dtBias[gid]));
    decay[gid] = exp(ssmA[gid] * sp);
    beta_[gid] = 1.0f / (1.0f + exp(-betaP[gid]));
}

// ── Depthwise causal conv1d (per-channel, kernel size K) ──────────────────
struct Conv1dParams { uint channels; uint kernelSize; };
kernel void causal_conv1d(
    device       float*          qkv    [[buffer(0)]],
    device       float*          buf    [[buffer(1)]],
    device const float*          weight [[buffer(2)]],
    constant     Conv1dParams&   p      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.channels) return;
    uint slots = p.kernelSize - 1u;
    float cur = qkv[gid];
    float acc = weight[gid * p.kernelSize + slots] * cur;
    for (uint k = 0; k < slots; ++k)
        acc += weight[gid * p.kernelSize + k] * buf[k * p.channels + gid];
    qkv[gid] = acc;

    // Shift buf: for k = 0..slots-2: buf[k] = buf[k+1]; buf[slots-1] = cur.
    for (uint k = 0; k < slots - 1u; ++k)
        buf[k * p.channels + gid] = buf[(k + 1u) * p.channels + gid];
    if (slots > 0u) buf[(slots - 1u) * p.channels + gid] = cur;
}

// Fused Conv1d + SiLU: the ForwardPass always runs SiLU immediately after
// the conv1d on the same buffer. Doing both in one kernel saves a dispatch
// per DeltaNet layer and keeps `acc` in register between ops.
kernel void causal_conv1d_silu(
    device       float*          qkv    [[buffer(0)]],
    device       float*          buf    [[buffer(1)]],
    device const float*          weight [[buffer(2)]],
    constant     Conv1dParams&   p      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.channels) return;
    uint slots = p.kernelSize - 1u;
    float cur = qkv[gid];
    float acc = weight[gid * p.kernelSize + slots] * cur;
    for (uint k = 0; k < slots; ++k)
        acc += weight[gid * p.kernelSize + k] * buf[k * p.channels + gid];
    // Apply SiLU in-register before writing out.
    qkv[gid] = acc / (1.0f + exp(-acc));

    for (uint k = 0; k < slots - 1u; ++k)
        buf[k * p.channels + gid] = buf[(k + 1u) * p.channels + gid];
    if (slots > 0u) buf[(slots - 1u) * p.channels + gid] = cur;
}

// ── Batched causal conv1d + SiLU (prefill) ───────────────────────────────
// Loops M tokens internally. Per thread (= per channel), keeps the
// conv-history slots in registers and walks through the M tokens,
// reading/writing qkv at offsets t * channels. Eliminates the per-token
// dispatch + CopyTensorSlice overhead. kernelSize limited to 8 (plenty for
// typical DeltaNet / Mamba conv kernels of size 4).
struct BatchedConv1dParams { uint channels; uint kernelSize; uint M; };

kernel void batched_causal_conv1d_silu(
    device       float*                     qkv    [[buffer(0)]],   // [M × channels]
    device       float*                     buf    [[buffer(1)]],   // [(K-1) × channels]
    device const float*                     weight [[buffer(2)]],   // [channels × K]
    constant     BatchedConv1dParams&       p      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.channels) return;
    const uint K = p.kernelSize;
    const uint slots = K - 1u;

    // Load conv history slots (up to kernelSize-1 = 7) into registers.
    float history[7];
    for (uint k = 0; k < slots; ++k) {
        history[k] = buf[k * p.channels + gid];
    }
    // Load weights for this channel.
    float wt[8];
    for (uint k = 0; k < K; ++k) {
        wt[k] = weight[gid * K + k];
    }

    // Walk through all M tokens sequentially for this channel.
    for (uint t = 0; t < p.M; ++t) {
        const uint off = t * p.channels + gid;
        float cur = qkv[off];
        float acc = wt[slots] * cur;
        for (uint k = 0; k < slots; ++k) {
            acc += wt[k] * history[k];
        }
        // SiLU in place
        qkv[off] = acc / (1.0f + exp(-acc));

        // Shift history, append cur
        for (uint k = 0; k + 1u < slots; ++k) {
            history[k] = history[k + 1u];
        }
        if (slots > 0u) history[slots - 1u] = cur;
    }

    // Write back final conv history.
    for (uint k = 0; k < slots; ++k) {
        buf[k * p.channels + gid] = history[k];
    }
}

// ── Embedding lookup ──────────────────────────────────────────────────────
// tableType: 0=F32, 1=Q8_0, 2=F16, 3=Q4_K, 5=Q4_0, 6=Q4_1
struct EmbedParams { uint hiddenDim; uint tokenId; uint tableType; };
kernel void embedding_lookup(
    device       float*         out   [[buffer(0)]],
    device const uchar*         table [[buffer(1)]],
    constant     EmbedParams&   p     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.hiddenDim) return;

    if (p.tableType == 0u) {
        // F32
        uint byteOff = (p.tokenId * p.hiddenDim + gid) * 4u;
        uint bits = uint(table[byteOff])
                 | (uint(table[byteOff + 1u]) << 8)
                 | (uint(table[byteOff + 2u]) << 16)
                 | (uint(table[byteOff + 3u]) << 24);
        out[gid] = as_type<float>(bits);
    } else if (p.tableType == 1u) {
        // Q8_0 (unaligned 34-byte blocks)
        uint blocksPerRow = p.hiddenDim / 32u;
        uint blk = gid / 32u;
        uint off = gid % 32u;
        uint blockByteOff = (p.tokenId * blocksPerRow + blk) * 34u;
        ushort scaleBits = uint(table[blockByteOff]) | (uint(table[blockByteOff + 1u]) << 8);
        float scale = float(as_type<half>(scaleBits));
        int q = (int)((char)table[blockByteOff + 2u + off]);
        out[gid] = scale * float(q);
    } else if (p.tableType == 2u) {
        uint byteOff = (p.tokenId * p.hiddenDim + gid) * 2u;
        ushort bits = uint(table[byteOff]) | (uint(table[byteOff + 1u]) << 8);
        out[gid] = float(as_type<half>(bits));
    } else if (p.tableType == 5u) {
        // Q4_0 (18-byte blocks, unaligned layout)
        uint blocksPerRow = p.hiddenDim / 32u;
        uint blk = gid / 32u;
        uint elemInBlock = gid % 32u;
        uint blockByteOff = (p.tokenId * blocksPerRow + blk) * 18u;
        ushort scaleBits = uint(table[blockByteOff]) | (uint(table[blockByteOff + 1u]) << 8);
        float scale = float(as_type<half>(scaleBits));
        uint byteIdx = (elemInBlock < 16u) ? elemInBlock : (elemInBlock - 16u);
        uint packed = table[blockByteOff + 2u + byteIdx];
        uint nibble = (elemInBlock < 16u) ? (packed & 0xFu) : (packed >> 4);
        out[gid] = scale * float(int(nibble) - 8);
    } else if (p.tableType == 8u) {
        // Q4_0 aligned (20-byte blocks: scale[2] + pad[2] + qs[16])
        uint blocksPerRow = p.hiddenDim / 32u;
        uint blk = gid / 32u;
        uint elemInBlock = gid % 32u;
        uint blockByteOff = (p.tokenId * blocksPerRow + blk) * 20u;
        ushort scaleBits = uint(table[blockByteOff]) | (uint(table[blockByteOff + 1u]) << 8);
        float scale = float(as_type<half>(scaleBits));
        uint byteIdx = (elemInBlock < 16u) ? elemInBlock : (elemInBlock - 16u);
        // Nibble bytes start at offset 4 in aligned layout.
        uint packed = table[blockByteOff + 4u + byteIdx];
        uint nibble = (elemInBlock < 16u) ? (packed & 0xFu) : (packed >> 4);
        out[gid] = scale * float(int(nibble) - 8);
    }
    // (K-quants etc. can fall back to host-side dequant for embedding rows.)
}

// ── ArgMax over first `count` elements ────────────────────────────────────
struct ArgMaxParams { uint count; };
kernel void argmax(
    device const float*          in_  [[buffer(0)]],
    device       int*            out  [[buffer(1)]],
    constant     ArgMaxParams&   p    [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float vScratch[8];
    threadgroup uint  iScratch[8];
    uint n = p.count;

    float localMax = -INFINITY;
    uint  localIdx = 0u;
    for (uint i = tid; i < n; i += 256u) {
        float v = in_[i];
        if (v > localMax) { localMax = v; localIdx = i; }
    }

    // Intra-SIMD reduction picking max via pairwise compare.
    // Apple's simd_max only returns the value, not the owning lane, so we
    // do a two-stage shared-memory reduction.
    uint simd_lane = tid % SIMD_WIDTH;
    uint simd_id   = tid / SIMD_WIDTH;

    // SIMD-local reduction via shuffle down.
    for (uint off = SIMD_WIDTH / 2u; off > 0u; off >>= 1u) {
        float oVal = simd_shuffle_down(localMax, off);
        uint  oIdx = simd_shuffle_down(localIdx, off);
        bool newer = (oVal > localMax);
        localMax = newer ? oVal : localMax;
        localIdx = newer ? oIdx : localIdx;
    }
    if (simd_lane == 0) { vScratch[simd_id] = localMax; iScratch[simd_id] = localIdx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float bestV = vScratch[0]; uint bestI = iScratch[0];
        for (uint i = 1u; i < 8u; ++i) {
            if (vScratch[i] > bestV) { bestV = vScratch[i]; bestI = iScratch[i]; }
        }
        out[0] = int(bestI);
    }
}

// ── SplitUnequalQKV: [Q:keyDim, K:keyDim, V:valueDim] → q (valueDim buf), k (valueDim buf), v (valueDim)
// Q and K destination tensors are valueDim-sized; we zero-pad the trailing (valueDim - keyDim) entries.
struct SplitUnequalQKVParams { uint keyDim; uint valueDim; };
kernel void split_unequal_qkv(
    device       float*                     q     [[buffer(0)]],
    device       float*                     k     [[buffer(1)]],
    device       float*                     v     [[buffer(2)]],
    device const float*                     qkv   [[buffer(3)]],
    constant     SplitUnequalQKVParams&     p     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    // q and k are sized valueDim, so we cover up to valueDim indices.
    if (gid < p.valueDim) {
        q[gid] = (gid < p.keyDim) ? qkv[gid] : 0.0f;
        k[gid] = (gid < p.keyDim) ? qkv[p.keyDim + gid] : 0.0f;
        v[gid] = qkv[p.keyDim * 2u + gid];
    }
}

// ── RepeatTile: replicate first srcSize = numHeads*headDim entries to factor*srcSize ──
// Dispatched with dstSize threads. Each thread copies one element from src[gid % srcSize] → dst[gid].
// Operates in-place: writer of index r*srcSize+i reads src[i] (within [0..srcSize)). Each index
// in [srcSize..) is written before the next r-iteration reads it. We split across factor separate
// dispatches to avoid in-place hazards; the caller issues one dispatch covering all factor copies.
// Implementation detail: we just write dst[gid] = src[gid % srcSize]; the first srcSize writes are
// idempotent (src[i] → dst[i] where they already equal), then subsequent ones read from the stable
// prefix. This is safe because Metal compute within one dispatch does not guarantee ordering, so
// we require the source region and destination region to not overlap — which they DO overlap for
// the first block. Fix: threads only copy for r >= 1; r == 0 is a no-op (src already in place).
struct RepeatTileParams { uint srcSize; uint factor; };
kernel void repeat_tile_f32(
    device       float*               data [[buffer(0)]],
    constant     RepeatTileParams&    p    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.srcSize * p.factor;
    if (gid >= total) return;
    uint srcIdx = gid % p.srcSize;
    // Block 0 is the source; leave it alone. Writes to blocks 1..factor-1 read from block 0.
    if (gid >= p.srcSize) data[gid] = data[srcIdx];
}

// ── L2 normalize groups in place ─────────────────────────────────────────
// One threadgroup per group. Each threadgroup reduces its groupDim elements, then normalizes.
struct L2NormGroupsParams { uint groupDim; };
kernel void l2norm_groups_f32(
    device       float*                 data [[buffer(0)]],
    constant     L2NormGroupsParams&    p    [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    uint off = tgid * p.groupDim;
    float local_sum = 0.0f;
    for (uint i = tid; i < p.groupDim; i += 256u) {
        float v = data[off + i];
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total + 1e-12f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = scratch[0];
    for (uint i = tid; i < p.groupDim; i += 256u) {
        data[off + i] = data[off + i] * inv;
    }
}

// ── Fused: hidden[i] = a[i] + b[i]; residual[i] = hidden[i]; out = RmsNorm(hidden, weight) ─
struct AddRmsNormResidualParams { uint n; float eps; };
kernel void add_rmsnorm_residual(
    device       float*                          out      [[buffer(0)]],
    device       float*                          hidden   [[buffer(1)]],
    device       float*                          residual [[buffer(2)]],
    device const float*                          b_       [[buffer(3)]],
    device const float*                          weight   [[buffer(4)]],
    constant     AddRmsNormResidualParams&       p        [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    uint off = tgid * p.n;
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = hidden[off + i] + b_[off + i];
        hidden[off + i] = v;
        residual[off + i] = v;
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[off + i] = hidden[off + i] * inv_rms * weight[i];
}

// hidden[i] = input[i]; residual[i] = input[i]; out = RmsNorm(input, weight)
kernel void rmsnorm_residual(
    device       float*                          out      [[buffer(0)]],
    device       float*                          residual [[buffer(1)]],
    device const float*                          input    [[buffer(2)]],
    device const float*                          weight   [[buffer(3)]],
    constant     AddRmsNormResidualParams&       p        [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    uint off = tgid * p.n;
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = input[off + i];
        residual[off + i] = v;
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[off + i] = input[off + i] * inv_rms * weight[i];
}

// hidden[i] = a[i] + b[i]; out = RmsNorm(hidden, weight)
kernel void add_rmsnorm(
    device       float*                          out     [[buffer(0)]],
    device       float*                          hidden  [[buffer(1)]],
    device const float*                          a_      [[buffer(2)]],
    device const float*                          b_      [[buffer(3)]],
    device const float*                          weight  [[buffer(4)]],
    constant     AddRmsNormResidualParams&       p       [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    uint off = tgid * p.n;
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = a_[off + i] + b_[off + i];
        hidden[off + i] = v;
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[off + i] = hidden[off + i] * inv_rms * weight[i];
}

// ══════════════════════════════════════════════════════════════════════════
// FP16 ACTIVATION VARIANTS
// ══════════════════════════════════════════════════════════════════════════
// These kernels take half-precision activation inputs/outputs. Weights and
// norm tensors remain F32. Math is done in F32 inside the kernel for
// precision; only the memory interface is F16. Halves activation memory
// bandwidth.

// ─── pointwise ops ──────────────────────────────────────────────────────

kernel void copy_f16(
    device       half*        dst [[buffer(0)]],
    device const half*        src [[buffer(1)]],
    constant     UintParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = src[gid];
}

kernel void copy_f16_region(
    device       half*        dst [[buffer(0)]],
    device const half*        src [[buffer(1)]],
    constant     UintParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = src[p.extra0 + gid];
}

kernel void copy_f16_slice(
    device       half*        dst [[buffer(0)]],
    device const half*        src [[buffer(1)]],
    constant     UintParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[p.extra1 + gid] = src[p.extra0 + gid];
}

kernel void zero_f16(
    device       half*        dst [[buffer(0)]],
    constant     UintParams&  p   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = 0.0h;
}

kernel void fill_f16(
    device       half*        dst [[buffer(0)]],
    constant     UintParams&  p   [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) dst[gid] = half(as_type<float>(p.extra0));
}

kernel void element_add_f16(
    device       half*           out [[buffer(0)]],
    device const half*           a   [[buffer(1)]],
    device const half*           b   [[buffer(2)]],
    constant     ElementParams&  p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) out[gid] = a[gid] + b[gid];
}

kernel void element_mul_f16(
    device       half*           out [[buffer(0)]],
    device const half*           a   [[buffer(1)]],
    device const half*           b   [[buffer(2)]],
    constant     ElementParams&  p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < p.n) out[gid] = a[gid] * b[gid];
}

kernel void silu_f16(
    device       half*           out [[buffer(0)]],
    device const half*           in  [[buffer(1)]],
    constant     ElementParams&  p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = float(in[gid]);
    out[gid] = half(x / (1.0f + exp(-x)));
}

kernel void silu_in_place_f16(
    device       half*           data [[buffer(0)]],
    constant     ElementParams&  p    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = float(data[gid]);
    data[gid] = half(x / (1.0f + exp(-x)));
}

kernel void silu_gate_f16(
    device       half*            out  [[buffer(0)]],
    device const half*            data [[buffer(1)]],
    device const half*            gate [[buffer(2)]],
    constant     SiLUGateParams&  p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float g = float(gate[gid]);
    out[gid] = half(float(data[gid]) * (g / (1.0f + exp(-g))));
}

kernel void split_swiglu_f16(
    device       half*                out    [[buffer(0)]],
    device const half*                fused  [[buffer(1)]],
    constant     SplitSwiGLUParams&   p      [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float g = float(fused[gid]);
    float u = float(fused[p.n + gid]);
    out[gid] = half((g / (1.0f + exp(-g))) * u);
}

kernel void squared_relu_f16(
    device       half*           data [[buffer(0)]],
    constant     ElementParams&  p    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.n) return;
    float x = max(0.0f, float(data[gid]));
    data[gid] = half(x * x);
}

// ─── norm / softmax ─────────────────────────────────────────────────────

kernel void rmsnorm_f16(
    device       half*            out    [[buffer(0)]],
    device const half*            in_    [[buffer(1)]],
    device const float*           weight [[buffer(2)]],
    constant     RmsNormParams&   p      [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = float(in_[i]);
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[i] = half(float(in_[i]) * inv_rms * weight[i]);
}

kernel void per_head_rmsnorm_f16(
    device       half*                 data   [[buffer(0)]],
    device const float*                weight [[buffer(1)]],
    constant     PerHeadRmsNormParams& p      [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.numHeads) return;
    threadgroup float scratch[8];
    uint off = tgid * p.headDim;

    float local_sum = 0.0f;
    for (uint i = tid; i < p.headDim; i += 256u) {
        float v = float(data[off + i]);
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.headDim) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.headDim; i += 256u) {
        data[off + i] = half(float(data[off + i]) * inv_rms * weight[i]);
    }
}

kernel void softmax_f16(
    device       half*           out [[buffer(0)]],
    device const half*           in_ [[buffer(1)]],
    constant     ElementParams&  p   [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    float local_max = -INFINITY;
    for (uint i = tid; i < p.n; i += 256u) local_max = max(local_max, float(in_[i]));
    float max_val = tg_reduce_max_256(local_max, tid, scratch);
    if (tid == 0) scratch[0] = max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    max_val = scratch[0];

    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float e = exp(float(in_[i]) - max_val);
        out[i] = half(e);
        local_sum += e;
    }
    float sum_val = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / scratch[0];
    for (uint i = tid; i < p.n; i += 256u) out[i] = half(float(out[i]) * inv);
}

// ─── fused norms / residuals ────────────────────────────────────────────

kernel void add_rmsnorm_residual_f16(
    device       half*                           out      [[buffer(0)]],
    device       half*                           hidden   [[buffer(1)]],
    device       half*                           residual [[buffer(2)]],
    device const half*                           b_       [[buffer(3)]],
    device const float*                          weight   [[buffer(4)]],
    constant     AddRmsNormResidualParams&       p        [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = float(hidden[i]) + float(b_[i]);
        hidden[i] = half(v);
        residual[i] = half(v);
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[i] = half(float(hidden[i]) * inv_rms * weight[i]);
}

kernel void rmsnorm_residual_f16(
    device       half*                           out      [[buffer(0)]],
    device       half*                           residual [[buffer(1)]],
    device const half*                           input    [[buffer(2)]],
    device const float*                          weight   [[buffer(3)]],
    constant     AddRmsNormResidualParams&       p        [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = float(input[i]);
        residual[i] = half(v);
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[i] = half(float(input[i]) * inv_rms * weight[i]);
}

kernel void add_rmsnorm_f16(
    device       half*                           out     [[buffer(0)]],
    device       half*                           hidden  [[buffer(1)]],
    device const half*                           a_      [[buffer(2)]],
    device const half*                           b_      [[buffer(3)]],
    device const float*                          weight  [[buffer(4)]],
    constant     AddRmsNormResidualParams&       p       [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    float local_sum = 0.0f;
    for (uint i = tid; i < p.n; i += 256u) {
        float v = float(a_[i]) + float(b_[i]);
        hidden[i] = half(v);
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.n) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.n; i += 256u)
        out[i] = half(float(hidden[i]) * inv_rms * weight[i]);
}

// ─── RoPE ────────────────────────────────────────────────────────────────

kernel void rope_f16(
    device       half*        q_data       [[buffer(0)]],
    device       half*        k_data       [[buffer(1)]],
    device const float*       freq_factors [[buffer(2)]],
    constant     RoPEParams&  p            [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint maxPairs = max(p.qTotal, p.kTotal) / 2u;
    if (gid >= maxPairs) return;

    if (gid * 2u < p.qTotal) {
        uint dimIdx = (gid * 2u) % p.headDim;
        if (dimIdx < p.ropeDim) {
            float freq = pow(p.ropeTheta, -float(dimIdx) / float(p.ropeDim));
            if (p.useFreqFactors != 0u) freq = freq / freq_factors[dimIdx / 2u];
            float angle = float(p.positionOffset) * freq;
            float c = cos(angle), s = sin(angle);
            float x0 = float(q_data[gid * 2u]);
            float x1 = float(q_data[gid * 2u + 1u]);
            q_data[gid * 2u]      = half(x0 * c - x1 * s);
            q_data[gid * 2u + 1u] = half(x0 * s + x1 * c);
        }
    }
    if (gid * 2u < p.kTotal) {
        uint dimIdx = (gid * 2u) % p.headDim;
        if (dimIdx < p.ropeDim) {
            float freq = pow(p.ropeTheta, -float(dimIdx) / float(p.ropeDim));
            if (p.useFreqFactors != 0u) freq = freq / freq_factors[dimIdx / 2u];
            float angle = float(p.positionOffset) * freq;
            float c = cos(angle), s = sin(angle);
            float x0 = float(k_data[gid * 2u]);
            float x1 = float(k_data[gid * 2u + 1u]);
            k_data[gid * 2u]      = half(x0 * c - x1 * s);
            k_data[gid * 2u + 1u] = half(x0 * s + x1 * c);
        }
    }
}

// ─── SplitQKV family (F16) ──────────────────────────────────────────────

kernel void split_qkv_f16(
    device       half*               q    [[buffer(0)]],
    device       half*               k    [[buffer(1)]],
    device       half*               v    [[buffer(2)]],
    device const half*               qkv  [[buffer(3)]],
    constant     SplitQKVParams&     p    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.innerSize) return;
    q[gid] = qkv[gid];
    k[gid] = qkv[p.innerSize + gid];
    v[gid] = qkv[p.innerSize * 2u + gid];
}

kernel void de_interleave_q_f16(
    device       half*                 qAttn [[buffer(0)]],
    device       half*                 qGate [[buffer(1)]],
    device const half*                 qFull [[buffer(2)]],
    constant     DeInterleaveParams&   p     [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.numHeads * p.headDim;
    if (gid >= total) return;
    uint head = gid / p.headDim;
    uint dim  = gid % p.headDim;
    qAttn[gid] = qFull[head * 2u * p.headDim + dim];
    qGate[gid] = qFull[head * 2u * p.headDim + p.headDim + dim];
}

kernel void split_unequal_qkv_f16(
    device       half*                   q    [[buffer(0)]],
    device       half*                   k    [[buffer(1)]],
    device       half*                   v    [[buffer(2)]],
    device const half*                   qkv  [[buffer(3)]],
    constant     SplitUnequalQKVParams&  p    [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.keyDim + p.keyDim + p.valueDim;
    if (gid >= total) return;
    if (gid < p.keyDim) {
        q[gid] = qkv[gid];
    } else if (gid < p.keyDim * 2u) {
        k[gid - p.keyDim] = qkv[gid];
    } else {
        v[gid - p.keyDim * 2u] = qkv[gid];
    }
}

kernel void repeat_tile_f16(
    device       half*              tensor [[buffer(0)]],
    constant     RepeatTileParams&  p      [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    uint total = p.srcSize * p.factor;
    if (gid >= total) return;
    if (gid < p.srcSize) return;
    uint srcIdx = gid % p.srcSize;
    tensor[gid] = tensor[srcIdx];
}

kernel void l2norm_groups_f16(
    device       half*                data [[buffer(0)]],
    constant     L2NormGroupsParams&  p    [[buffer(1)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[8];
    uint off = tgid * p.groupDim;
    float local_sum = 0.0f;
    for (uint i = tid; i < p.groupDim; i += 256u) {
        float v = float(data[off + i]);
        local_sum += v * v;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = scratch[0];
    for (uint i = tid; i < p.groupDim; i += 256u)
        data[off + i] = half(float(data[off + i]) * inv);
}

// ─── KV cache write (F16 cache) ─────────────────────────────────────────

kernel void kv_cache_write_f16(
    device       half*           kCache [[buffer(0)]],
    device       half*           vCache [[buffer(1)]],
    device const half*           k      [[buffer(2)]],
    device const half*           v      [[buffer(3)]],
    constant     KvWriteParams&  p      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    uint kMax = p.nKvHeads * p.keyLength;
    uint vMax = p.nKvHeads * p.valueLength;
    uint maxE = max(kMax, vMax);
    if (gid >= maxE) return;
    if (gid < kMax) {
        uint h = gid / p.keyLength;
        uint d = gid % p.keyLength;
        uint off = (h * p.maxSeqLen + p.position) * p.keyLength + d;
        kCache[off] = k[gid];
    }
    if (gid < vMax) {
        uint h = gid / p.valueLength;
        uint d = gid % p.valueLength;
        uint off = (h * p.maxSeqLen + p.position) * p.valueLength + d;
        vCache[off] = v[gid];
    }
}

// ─── Gated attention (F16 activations, F16 KV cache) ────────────────────

kernel void gated_attention_f16(
    device       half*              out    [[buffer(0)]],
    device const half*              qAttn  [[buffer(1)]],
    device const half*              qGate  [[buffer(2)]],
    device const half*              kCache [[buffer(3)]],
    device const half*              vCache [[buffer(4)]],
    constant     GatedAttnParams&   p      [[buffer(5)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.numHeads) return;
    uint h = tgid;
    threadgroup float tile_scores[256];
    threadgroup float scratch[8];

    uint headsPerGroup = p.numHeads / p.numKvHeads;
    uint kvHead = h / headsPerGroup;
    uint qOff = h * p.keyLength;
    uint kvKBase = kvHead * p.maxSeqLen * p.keyLength;
    uint kvVBase = kvHead * p.maxSeqLen * p.valueLength;
    uint outOff = h * p.valueLength;

    for (uint d = tid; d < p.valueLength; d += 256u) out[outOff + d] = 0.0h;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const uint TILE = 256u;

    for (uint tileStart = 0u; tileStart < p.seqLen; tileStart += TILE) {
        uint tileEnd = min(tileStart + TILE, p.seqLen);
        uint tileLen = tileEnd - tileStart;

        float my_score = -INFINITY;
        if (tid < tileLen) {
            uint pos = tileStart + tid;
            float dot = 0.0f;
            for (uint d = 0; d < p.keyLength; ++d)
                dot += float(qAttn[qOff + d]) * float(kCache[kvKBase + pos * p.keyLength + d]);
            my_score = dot * p.scale;
        }
        tile_scores[tid] = my_score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_max = tg_reduce_max_256(my_score, tid, scratch);
        if (tid == 0) scratch[0] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = scratch[0];

        float my_exp = (tid < tileLen) ? exp(tile_scores[tid] - tile_max) : 0.0f;
        tile_scores[tid] = my_exp;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_sum = tg_reduce_add_256(my_exp, tid, scratch);
        if (tid == 0) scratch[0] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_sum = scratch[0];

        float new_max = max(running_max, tile_max);
        float corr_old = exp(running_max - new_max);
        float corr_new = exp(tile_max    - new_max);

        for (uint d = tid; d < p.valueLength; d += 256u) {
            float tile_val = 0.0f;
            for (uint t = 0; t < tileLen; ++t)
                tile_val += tile_scores[t] * float(vCache[kvVBase + (tileStart + t) * p.valueLength + d]);
            out[outOff + d] = half(float(out[outOff + d]) * corr_old + tile_val * corr_new);
        }
        running_sum = running_sum * corr_old + tile_sum * corr_new;
        running_max = new_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (uint d = tid; d < p.valueLength; d += 256u) {
        float gate_val = 1.0f;
        if (d < p.keyLength) {
            float gv = float(qGate[h * p.keyLength + d]);
            gate_val = 1.0f / (1.0f + exp(-gv));
        }
        out[outOff + d] = half(float(out[outOff + d]) * inv_sum * gate_val);
    }
}

// ─── DeltaNet (F16 activations, F16 state — state is large, ~500KB) ─────

kernel void deltanet_step_f16(
    device       half*               out    [[buffer(0)]],
    device const half*               q      [[buffer(1)]],
    device const half*               k      [[buffer(2)]],
    device const half*               v      [[buffer(3)]],
    device       half*               state  [[buffer(4)]],
    device const half*               decay  [[buffer(5)]],
    device const half*               beta_  [[buffer(6)]],
    device const float*              normW  [[buffer(7)]],
    constant     DeltaNetParams&     p      [[buffer(8)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    if (tgid >= p.groupCount) return;
    uint g = tgid;
    threadgroup float scratch[8];

    uint baseOff  = g * p.headDim;
    uint stateOff = g * p.headDim * p.headDim;
    float d = float(decay[g]);
    float b = float(beta_[g]);

    for (uint j = tid; j < p.headDim; j += 256u) {
        float sk = 0.0f;
        for (uint i = 0; i < p.headDim; ++i)
            sk += float(state[stateOff + i * p.headDim + j]) * float(k[baseOff + i]);
        float err_j = (float(v[baseOff + j]) - d * sk) * b;
        for (uint i = 0; i < p.headDim; ++i) {
            uint idx = stateOff + i * p.headDim + j;
            state[idx] = half(d * float(state[idx]) + float(k[baseOff + i]) * err_j);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < p.headDim; j += 256u) {
        float sum = 0.0f;
        for (uint i = 0; i < p.headDim; ++i)
            sum += float(state[stateOff + i * p.headDim + j]) * float(q[baseOff + i]);
        out[baseOff + j] = half(sum * p.scale);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint i = tid; i < p.headDim; i += 256u) {
        float vv = float(out[baseOff + i]);
        local_sum += vv * vv;
    }
    float total = tg_reduce_add_256(local_sum, tid, scratch);
    if (tid == 0) scratch[0] = rsqrt(total / float(p.headDim) + p.normEps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = scratch[0];
    for (uint i = tid; i < p.headDim; i += 256u)
        out[baseOff + i] = half(float(out[baseOff + i]) * inv_rms * normW[i]);
}

kernel void compute_decay_beta_f16(
    device       half*               decay   [[buffer(0)]],
    device       half*               beta_   [[buffer(1)]],
    device const half*               alphaP  [[buffer(2)]],
    device const half*               betaP   [[buffer(3)]],
    device const float*              ssmA    [[buffer(4)]],   // remains F32 (loaded weight)
    device const float*              dtBias  [[buffer(5)]],   // remains F32 (loaded weight)
    constant     DecayBetaParams&    p       [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.groupCount) return;
    float sp = log(1.0f + exp(float(alphaP[gid]) + dtBias[gid]));
    decay[gid] = half(exp(ssmA[gid] * sp));
    beta_[gid] = half(1.0f / (1.0f + exp(-float(betaP[gid]))));
}

kernel void causal_conv1d_silu_f16(
    device       half*            qkv    [[buffer(0)]],
    device       half*            buf    [[buffer(1)]],
    device const float*           weight [[buffer(2)]],  // static weight, F32
    constant     Conv1dParams&    p      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.channels) return;
    uint slots = p.kernelSize - 1u;
    float cur = float(qkv[gid]);
    float acc = weight[gid * p.kernelSize + slots] * cur;
    for (uint k = 0; k < slots; ++k)
        acc += weight[gid * p.kernelSize + k] * float(buf[k * p.channels + gid]);
    qkv[gid] = half(acc / (1.0f + exp(-acc)));

    for (uint k = 0; k < slots - 1u; ++k)
        buf[k * p.channels + gid] = buf[(k + 1u) * p.channels + gid];
    if (slots > 0u) buf[(slots - 1u) * p.channels + gid] = half(cur);
}

// ─── Embedding lookup (output F16) ──────────────────────────────────────

kernel void embedding_lookup_f16(
    device       half*          out   [[buffer(0)]],
    device const uchar*         table [[buffer(1)]],
    constant     EmbedParams&   p     [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= p.hiddenDim) return;
    float value = 0.0f;
    if (p.tableType == 0u) {
        uint byteOff = (p.tokenId * p.hiddenDim + gid) * 4u;
        uint bits = uint(table[byteOff])
                 | (uint(table[byteOff + 1u]) << 8)
                 | (uint(table[byteOff + 2u]) << 16)
                 | (uint(table[byteOff + 3u]) << 24);
        value = as_type<float>(bits);
    } else if (p.tableType == 1u) {
        uint blocksPerRow = p.hiddenDim / 32u;
        uint blk = gid / 32u;
        uint off = gid % 32u;
        uint blockByteOff = (p.tokenId * blocksPerRow + blk) * 34u;
        ushort scaleBits = uint(table[blockByteOff]) | (uint(table[blockByteOff + 1u]) << 8);
        float scale = float(as_type<half>(scaleBits));
        int q = (int)((char)table[blockByteOff + 2u + off]);
        value = scale * float(q);
    } else if (p.tableType == 2u) {
        uint byteOff = (p.tokenId * p.hiddenDim + gid) * 2u;
        ushort bits = uint(table[byteOff]) | (uint(table[byteOff + 1u]) << 8);
        value = float(as_type<half>(bits));
    } else if (p.tableType == 5u) {
        uint blocksPerRow = p.hiddenDim / 32u;
        uint blk = gid / 32u;
        uint elemInBlock = gid % 32u;
        uint blockByteOff = (p.tokenId * blocksPerRow + blk) * 18u;
        ushort scaleBits = uint(table[blockByteOff]) | (uint(table[blockByteOff + 1u]) << 8);
        float scale = float(as_type<half>(scaleBits));
        uint byteIdx = (elemInBlock < 16u) ? elemInBlock : (elemInBlock - 16u);
        uint packed = table[blockByteOff + 2u + byteIdx];
        uint nibble = (elemInBlock < 16u) ? (packed & 0xFu) : (packed >> 4);
        value = scale * float(int(nibble) - 8);
    } else if (p.tableType == 8u) {
        uint blocksPerRow = p.hiddenDim / 32u;
        uint blk = gid / 32u;
        uint elemInBlock = gid % 32u;
        uint blockByteOff = (p.tokenId * blocksPerRow + blk) * 20u;
        ushort scaleBits = uint(table[blockByteOff]) | (uint(table[blockByteOff + 1u]) << 8);
        float scale = float(as_type<half>(scaleBits));
        uint byteIdx = (elemInBlock < 16u) ? elemInBlock : (elemInBlock - 16u);
        uint packed = table[blockByteOff + 4u + byteIdx];
        uint nibble = (elemInBlock < 16u) ? (packed & 0xFu) : (packed >> 4);
        value = scale * float(int(nibble) - 8);
    }
    out[gid] = half(value);
}

// ─── matmul F16 variants (half activations in/out) ──────────────────────

// F32 input × F32 weight → F16 output variant kept for boundary cases.
kernel void matmul_f32_out_f16(
    device       half*         output_data [[buffer(0)]],
    device const float*        input_data  [[buffer(1)]],
    device const float*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;
    float acc = 0.0f;
    for (uint k = tid; k < p.K; k += TG_SIZE) {
        acc += weight_data[row * p.K + k] * input_data[k];
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = half(total);
}

kernel void matmul_q4_0_aligned_2x4row_f16(
    device       half*         output_data [[buffer(0)]],
    device const half4*        input_vec4  [[buffer(1)]],
    device const uint*         weight_u32  [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = float4(input_vec4[aVecBase + i]);

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint wordBase = (row * blocksPerRow + b) * 5u;
            uint w0 = weight_u32[wordBase];
            float scale = float(as_type<half>(ushort(w0 & 0xFFFFu)));

            float blockSum = 0.0f;
            for (uint wi = 0; wi < 4u; ++wi) {
                uint packed = weight_u32[wordBase + 1u + wi];
                float4 lo = float4(
                    float(int((packed >>  0) & 0xFu) - 8),
                    float(int((packed >>  8) & 0xFu) - 8),
                    float(int((packed >> 16) & 0xFu) - 8),
                    float(int((packed >> 24) & 0xFu) - 8));
                float4 hi = float4(
                    float(int((packed >>  4) & 0xFu) - 8),
                    float(int((packed >> 12) & 0xFu) - 8),
                    float(int((packed >> 20) & 0xFu) - 8),
                    float(int((packed >> 28) & 0xFu) - 8));
                blockSum += dot(lo, aVec[wi]) + dot(hi, aVec[wi + 4u]);
            }
            accs[r] += scale * blockSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = half(total);
    }
}

kernel void matmul_q4_1_2x4row_f16(
    device       half*         output_data [[buffer(0)]],
    device const half4*        input_vec4  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    constexpr uint ROWS = 4u;
    uint rowBase = tgid * 8u + sgitg * ROWS;
    if (rowBase >= p.N) return;

    uint lane = tid % 32u;
    uint blocksPerRow = p.K / 32u;
    float accs[ROWS] = { 0.0f, 0.0f, 0.0f, 0.0f };

    for (uint b = lane; b < blocksPerRow; b += 32u) {
        uint aVecBase = b * 8u;
        float4 aVec[8];
        for (uint i = 0; i < 8u; ++i) aVec[i] = float4(input_vec4[aVecBase + i]);
        float aSum = 0.0f;
        for (uint i = 0; i < 8u; ++i)
            aSum += aVec[i].x + aVec[i].y + aVec[i].z + aVec[i].w;

        for (uint r = 0; r < ROWS; ++r) {
            uint row = rowBase + r;
            if (row >= p.N) break;
            uint blockOff = (row * blocksPerRow + b) * 20u;
            ushort dBits = uint(weight_data[blockOff])     | (uint(weight_data[blockOff + 1]) << 8);
            ushort mBits = uint(weight_data[blockOff + 2]) | (uint(weight_data[blockOff + 3]) << 8);
            float scale = float(as_type<half>(dBits));
            float minV  = float(as_type<half>(mBits));

            float blockSum = 0.0f;
            for (uint i = 0; i < 4u; ++i) {
                uint b0 = weight_data[blockOff + 4u + i*4 + 0];
                uint b1 = weight_data[blockOff + 4u + i*4 + 1];
                uint b2 = weight_data[blockOff + 4u + i*4 + 2];
                uint b3 = weight_data[blockOff + 4u + i*4 + 3];
                float4 lo = float4(float(b0 & 0xFu), float(b1 & 0xFu),
                                   float(b2 & 0xFu), float(b3 & 0xFu));
                float4 hi = float4(float(b0 >> 4),   float(b1 >> 4),
                                   float(b2 >> 4),  float(b3 >> 4));
                blockSum += dot(lo, aVec[i]) + dot(hi, aVec[i + 4u]);
            }
            accs[r] += scale * blockSum + minV * aSum;
        }
    }

    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(accs[r]);
        if (lane == 0 && rowBase + r < p.N) output_data[rowBase + r] = half(total);
    }
}

kernel void matmul_q8_0_f16(
    device       half*         output_data [[buffer(0)]],
    device const half*         input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_position_in_threadgroup]])
{
    threadgroup float scratch[2];
    uint row = tgid;
    if (row >= p.N) return;

    uint blocksPerRow = p.K / 32u;
    float acc = 0.0f;

    for (uint b = tid; b < blocksPerRow; b += TG_SIZE) {
        uint blockOff = (row * blocksPerRow + b) * 34u;
        ushort scaleBits = uint(weight_data[blockOff]) | (uint(weight_data[blockOff + 1]) << 8);
        float scale = float(as_type<half>(scaleBits));

        float blockSum = 0.0f;
        uint aBase = b * 32u;
        for (uint i = 0; i < 32u; ++i) {
            int q = (int)((char)weight_data[blockOff + 2u + i]);
            blockSum += float(q) * float(input_data[aBase + i]);
        }
        acc += scale * blockSum;
    }
    float total = tg_reduce_add_64(acc, tid, scratch);
    if (tid == 0) output_data[row] = half(total);
}

kernel void matmul_q5_k_16row_f16(
    device       half*         output_data [[buffer(0)]],
    device const half*         input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;

    for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
        uint base = (rowBase * superBlocksPerRow + sb) * 176u;
        uint aBase = sb * 256u;

        ushort dBits = uint(weight_data[base])     | (uint(weight_data[base + 1]) << 8);
        ushort mBits = uint(weight_data[base + 2]) | (uint(weight_data[base + 3]) << 8);
        float d = float(as_type<half>(dBits));
        float dmin = float(as_type<half>(mBits));

        float scales[8];
        float mins[8];
        for (uint j = 0; j < 4u; ++j) {
            scales[j] = float(weight_data[base + 4 + j] & 63u);
            mins[j]   = float(weight_data[base + 4 + j + 4] & 63u);
        }
        for (uint j = 4; j < 8u; ++j) {
            uint lo = weight_data[base + 4 + j + 4] & 0xFu;
            uint hi = (weight_data[base + 4 + j - 4] >> 6) & 0x3u;
            scales[j] = float(lo | (hi << 4));
            lo = weight_data[base + 4 + j + 4] >> 4;
            hi = (weight_data[base + 4 + j] >> 6) & 0x3u;
            mins[j] = float(lo | (hi << 4));
        }

        uint qhOff = base + 16u;
        uint qsOff = base + 48u;
        uint u1mask = 1u;
        uint u2mask = 2u;
        for (uint j = 0; j < 4u; ++j) {
            float d1 = d * scales[2 * j];
            float m1 = dmin * mins[2 * j];
            float d2 = d * scales[2 * j + 1];
            float m2 = dmin * mins[2 * j + 1];
            for (uint l = 0; l < 32u; ++l) {
                uint qs = weight_data[qsOff + j * 32u + l];
                uint qh = weight_data[qhOff + l];
                float lo = float((qs & 0xFu) + ((qh & u1mask) != 0u ? 16u : 0u));
                float hi = float((qs >> 4) + ((qh & u2mask) != 0u ? 16u : 0u));
                float w0 = d1 * lo - m1;
                float w1 = d2 * hi - m2;
                acc += w0 * float(input_data[aBase + j * 64u + l]);
                acc += w1 * float(input_data[aBase + j * 64u + l + 32u]);
            }
            u1mask <<= 2;
            u2mask <<= 2;
        }
    }

    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    if (threadInRow == 0) output_data[rowBase] = half(acc);
}

kernel void matmul_q6_k_16row_f16(
    device       half*         output_data [[buffer(0)]],
    device const half*         input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;
    device const uchar* raw = weight_data;

    for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
        uint base = (rowBase * superBlocksPerRow + sb) * 210u;
        uint aBase = sb * 256u;

        ushort dBits = uint(raw[base + 208]) | (uint(raw[base + 209]) << 8);
        float d = float(as_type<half>(dBits));

        float scales[16];
        for (uint s = 0; s < 16u; ++s) scales[s] = d * float((char)raw[base + 192 + s]);

        for (uint half_ = 0u; half_ < 2u; ++half_) {
            uint qlOff = base + half_ * 64u;
            uint qhOff = base + 128u + half_ * 32u;
            uint scIdx = half_ * 8u;
            uint aOff  = aBase + half_ * 128u;

            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            for (uint l = 0; l < 16u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sA0 * float(q1) * float(input_data[aOff + l]);
                acc += sA1 * float(q2) * float(input_data[aOff + l + 32u]);
                acc += sA2 * float(q3) * float(input_data[aOff + l + 64u]);
                acc += sA3 * float(q4) * float(input_data[aOff + l + 96u]);
            }
            for (uint l = 16; l < 32u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sB0 * float(q1) * float(input_data[aOff + l]);
                acc += sB1 * float(q2) * float(input_data[aOff + l + 32u]);
                acc += sB2 * float(q3) * float(input_data[aOff + l + 64u]);
                acc += sB3 * float(q4) * float(input_data[aOff + l + 96u]);
            }
        }
    }

    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    if (threadInRow == 0) output_data[rowBase] = half(acc);
}

// ─── lm_head: Q6_K weight × F16 activation → F32 logits ─────────────────
kernel void matmul_q6_k_16row_hact_fout(
    device       float*        output_data [[buffer(0)]],
    device const half*         input_data  [[buffer(1)]],
    device const uchar*        weight_data [[buffer(2)]],
    constant     MatMulParams& p           [[buffer(3)]],
    uint  tgid  [[threadgroup_position_in_grid]],
    uint  tid   [[thread_position_in_threadgroup]],
    uint  sgitg [[simdgroup_index_in_threadgroup]])
{
    uint lane        = tid % 32u;
    uint threadInRow = lane % 4u;
    uint localRow    = lane / 4u;
    uint rowBase     = tgid * 16u + sgitg * 8u + localRow;
    if (rowBase >= p.N) return;

    uint superBlocksPerRow = p.K / 256u;
    float acc = 0.0f;
    device const uchar* raw = weight_data;

    for (uint sb = threadInRow; sb < superBlocksPerRow; sb += 4u) {
        uint base = (rowBase * superBlocksPerRow + sb) * 210u;
        uint aBase = sb * 256u;

        ushort dBits = uint(raw[base + 208]) | (uint(raw[base + 209]) << 8);
        float d = float(as_type<half>(dBits));

        float scales[16];
        for (uint s = 0; s < 16u; ++s) scales[s] = d * float((char)raw[base + 192 + s]);

        for (uint half_ = 0u; half_ < 2u; ++half_) {
            uint qlOff = base + half_ * 64u;
            uint qhOff = base + 128u + half_ * 32u;
            uint scIdx = half_ * 8u;
            uint aOff  = aBase + half_ * 128u;

            float sA0 = scales[scIdx + 0u], sA1 = scales[scIdx + 2u];
            float sA2 = scales[scIdx + 4u], sA3 = scales[scIdx + 6u];
            float sB0 = scales[scIdx + 1u], sB1 = scales[scIdx + 3u];
            float sB2 = scales[scIdx + 5u], sB3 = scales[scIdx + 7u];

            for (uint l = 0; l < 16u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sA0 * float(q1) * float(input_data[aOff + l]);
                acc += sA1 * float(q2) * float(input_data[aOff + l + 32u]);
                acc += sA2 * float(q3) * float(input_data[aOff + l + 64u]);
                acc += sA3 * float(q4) * float(input_data[aOff + l + 96u]);
            }
            for (uint l = 16; l < 32u; ++l) {
                uint ql0 = raw[qlOff + l];
                uint ql1 = raw[qlOff + l + 32u];
                uint qh0 = raw[qhOff + l];
                int q1 = int((ql0 & 0xFu) | (((qh0 >> 0) & 3u) << 4)) - 32;
                int q2 = int((ql1 & 0xFu) | (((qh0 >> 2) & 3u) << 4)) - 32;
                int q3 = int((ql0 >> 4)   | (((qh0 >> 4) & 3u) << 4)) - 32;
                int q4 = int((ql1 >> 4)   | (((qh0 >> 6) & 3u) << 4)) - 32;
                acc += sB0 * float(q1) * float(input_data[aOff + l]);
                acc += sB1 * float(q2) * float(input_data[aOff + l + 32u]);
                acc += sB2 * float(q3) * float(input_data[aOff + l + 64u]);
                acc += sB3 * float(q4) * float(input_data[aOff + l + 96u]);
            }
        }
    }

    acc += simd_shuffle_xor(acc, 1u);
    acc += simd_shuffle_xor(acc, 2u);
    if (threadInRow == 0) output_data[rowBase] = acc;
}
