using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Metal;

/// <summary>
/// Apple Metal compute backend. Uses MSL kernels dispatched via the
/// Objective-C runtime. Buffers live in unified (shared) memory so CPU
/// fallbacks can touch the same bytes after we wait for GPU completion.
/// </summary>
public sealed class MetalBackend : IComputeBackend, IDisposable
{
    private readonly MetalDevice _dev;
    private readonly Dictionary<string, IntPtr> _pipelines = new();
    private bool _disposed;

    // ── Pending command buffer / encoder for batching ─────────────────────
    // Dispatches accumulate on one command buffer / one compute encoder until
    // we hit a CPU-side read (CopyTensor, AsFloatSpan, argmax readback…) or
    // the caller explicitly flushes (token boundary). This collapses ~500
    // commit+wait round trips per token into ~1.
    private IntPtr _pendingCmdBuf;
    private IntPtr _pendingEncoder;
    private IntPtr _lastBoundPipeline;


    // Set via env var DAISI_METAL_BATCH=0 to force commit+wait per dispatch (debug).
    private readonly bool _batchEnabled = Environment.GetEnvironmentVariable("DAISI_METAL_BATCH") != "0";
    // DAISI_METAL_FP16=1 enables fp16-backed activation tensors. Halves
    // activation memory traffic at the cost of some precision. Logits stay F32.
    private readonly bool _fp16Activations = Environment.GetEnvironmentVariable("DAISI_METAL_FP16") == "1";
    private readonly bool _traceDispatch = Environment.GetEnvironmentVariable("DAISI_METAL_TRACE") == "1";
    // DAISI_METAL_GPUPROF=1 enables per-dispatch GPU timing via GPUStartTime /
    // GPUEndTime / wall-clock before commit + after wait. Reveals the three
    // latency components: submit (CPU→GPU scheduling), execute (GPU kernel
    // runtime), drain (GPU done → CPU unblocked).
    private readonly bool _gpuProf = Environment.GetEnvironmentVariable("DAISI_METAL_GPUPROF") == "1";
    private readonly List<(string fn, double submit, double exec, double drain)> _gpuProfRows = new();

    // DAISI_METAL_BARRIER=1 inserts `memoryBarrierWithScope(.buffers)` between
    // every dispatch on a batched encoder. Diagnostic — in theory redundant
    // because hazard-tracked shared-storage buffers already get auto-barriers.
    private readonly bool _forceBarriers = Environment.GetEnvironmentVariable("DAISI_METAL_BARRIER") == "1";
    private readonly bool _flushAfterKv = Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_KV") == "1";
    private readonly bool _flushAfterAttn = Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_ATTN") == "1";
    private readonly bool _flushAfterDeltaNet = Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_DN") == "1";
    private readonly bool _flushAfterMatMul = Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_MM") == "1";
    private readonly bool _halfMatMul = Environment.GetEnvironmentVariable("DAISI_METAL_MM_HALF") == "1";
    // Experimental: keep encoder alive across dispatches. On M1 Pro this
    // regressed perf (~15%) — presumably Metal's auto-hazard tracking costs
    // more than the saved encoder setup. Left off by default.
    private readonly bool _reuseEncoder = Environment.GetEnvironmentVariable("DAISI_METAL_REUSE_ENCODER") == "1";
    private readonly bool _flushAfterCopy = Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_COPY") == "1";
    private readonly int _maxBatchSize = int.TryParse(Environment.GetEnvironmentVariable("DAISI_METAL_BATCH_SIZE"), out var n) && n > 0 ? n : int.MaxValue;
    private int _encodedInBatch;
    private int _totalDispatches;
    private int _totalFlushes;
    private long _traceDispatchCount;
    private long _traceDispatchNanos;
    private long _traceFlushCount;
    private long _traceFlushNanos;
    private readonly Dictionary<string, (long count, long nanos)> _traceByKernel = new();

    // ── Param structs (must match MSL layouts in kernels.metal) ──────────

    [StructLayout(LayoutKind.Sequential)]
    private struct MatMulParams { public uint M; public uint K; public uint N; }

    [StructLayout(LayoutKind.Sequential)]
    private struct UintParams { public uint N; public uint Extra0; public uint Extra1; public uint Extra2; }

    [StructLayout(LayoutKind.Sequential)]
    private struct ElementParams { public uint N; }

    [StructLayout(LayoutKind.Sequential)]
    private struct RmsNormParams { public uint N; public float Eps; }

    [StructLayout(LayoutKind.Sequential)]
    private struct PerHeadRmsNormParams { public uint NumHeads; public uint HeadDim; public float Eps; }

    [StructLayout(LayoutKind.Sequential)]
    private struct SplitQKVParams { public uint InnerSize; }

    [StructLayout(LayoutKind.Sequential)]
    private struct DeInterleaveParams { public uint NumHeads; public uint HeadDim; }

    [StructLayout(LayoutKind.Sequential)]
    private struct KvWriteParams
    {
        public uint NKvHeads;
        public uint KeyLength;
        public uint ValueLength;
        public uint MaxSeqLen;
        public uint Position;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct GatedAttnParams
    {
        public uint NumHeads;
        public uint NumKvHeads;
        public uint KeyLength;
        public uint ValueLength;
        public uint MaxSeqLen;
        public uint SeqLen;
        public float Scale;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DeltaNetParams
    {
        public uint GroupCount;
        public uint HeadDim;
        public float Scale;
        public float NormEps;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct DecayBetaParams { public uint GroupCount; }

    [StructLayout(LayoutKind.Sequential)]
    private struct Conv1dParams { public uint Channels; public uint KernelSize; }

    [StructLayout(LayoutKind.Sequential)]
    private struct EmbedParams { public uint HiddenDim; public uint TokenId; public uint TableType; }

    [StructLayout(LayoutKind.Sequential)]
    private struct ArgMaxParams { public uint Count; }

    [StructLayout(LayoutKind.Sequential)]
    private struct RoPEParams
    {
        public uint QTotal;
        public uint KTotal;
        public uint HeadDim;
        public uint RopeDim;
        public int PositionOffset;
        public float RopeTheta;
        public uint UseFreqFactors;
        public uint Neox;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct AddRmsNormResidualParams { public uint N; public float Eps; }

    [StructLayout(LayoutKind.Sequential)]
    private struct BatchedRoPEParams
    {
        public uint QTotal;
        public uint KTotal;
        public uint HeadDim;
        public uint RopeDim;
        public int PositionOffset;
        public float RopeTheta;
        public uint NumHeads;
        public uint NumKvHeads;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BatchedGatedAttnParams
    {
        public uint NumHeads;
        public uint NumKvHeads;
        public uint KeyLength;
        public uint ValueLength;
        public uint MaxSeqLen;
        public uint StartPosition;
        public uint M;
        public float Scale;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BatchedKvWriteParams
    {
        public uint NKvHeads;
        public uint KeyLength;
        public uint ValueLength;
        public uint MaxSeqLen;
        public uint StartPosition;
        public uint M;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BatchedEmbedParams { public uint HiddenDim; public uint TableType; public uint M; }

    [StructLayout(LayoutKind.Sequential)]
    private struct BatchedConv1dParams
    {
        public uint Channels;
        public uint KernelSize;
        public uint M;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BatchedDeltaNetParams
    {
        public uint M;
        public uint QkvOutDim;
        public uint KeyDim;
        public uint ValueDim;
        public uint NumKHeads;
        public uint NumVHeads;
        public uint HeadDim;
        public uint RepeatFactor;
        public float Scale;
        public float NormEps;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SplitUnequalQkvParams { public uint KeyDim; public uint ValueDim; }

    [StructLayout(LayoutKind.Sequential)]
    private struct RepeatTileParams { public uint SrcSize; public uint Factor; }

    [StructLayout(LayoutKind.Sequential)]
    private struct L2NormGroupsParams { public uint GroupDim; }

    public MetalBackend()
    {
        _dev = new MetalDevice();
        if (_traceDispatch)
            Console.Error.WriteLine($"[metal] device: {_dev.DeviceName}, appleSilicon={_dev.IsAppleSilicon}");
        _dev.LoadLibrary(LoadShaderSource());

        // Pre-warm the pipelines we'll actually use every layer.
        foreach (var fn in new[]
        {
            "matmul_f32", "matmul_f16", "matmul_q8_0", "matmul_q4_0", "matmul_q4_0_4row",
            "matmul_q4_1", "matmul_q5_k", "matmul_q5_k_tg16", "matmul_q6_k", "matmul_q6_k_tg16",
            "matmul_q6_k_16row", "matmul_q6_k_32row", "matmul_q6_k_mv", "matmul_q5_k_16row", "matmul_q5_k_mv", "matmul_q4_0_aligned_mv", "matmul_q4_1_2x4row", "matmul_q4_1_mv", "matmul_q8_0_mv",
            // Quants added for broader model coverage:
            "matmul_q4_k_mv", "matmul_q2_k_mv", "matmul_q5_0_mv", "matmul_q5_0", "matmul_bf16", "matmul_bf16_mv", "matmul_i2s_mv",
            "element_add_broadcast_row",
            "matmul_q4_0_aligned", "matmul_q4_0_aligned_4row", "matmul_q4_0_aligned_swiglu_4row",
            "matmul_q4_0_aligned_swiglu_2x4row",
            "matmul_q4_0_aligned_simd", "matmul_q4_0_aligned_simd_4row", "matmul_q4_0_aligned_simd2_4row",
            "matmul_q4_0_aligned_2x4row", "matmul_q4_0_aligned_2x4row_tiled",
            "matmul_q4_0_aligned_2x4row_preload", "matmul_q4_0_aligned_4x4row",
            "matmul_mm_q4_0_aligned", "matmul_mm_q4_0_aligned_h", "matmul_mm_q8_0", "matmul_mm_q5_k_16row",
            // batched prefill kernels
            "batched_rope", "batched_gated_attention", "batched_kv_cache_write", "batched_embedding_lookup",
            "batched_causal_conv1d_silu", "batched_deltanet_fused",
            // fp16 activation variants
            "copy_f16", "copy_f16_region", "copy_f16_slice", "zero_f16", "fill_f16",
            "element_add_f16", "element_mul_f16",
            "silu_f16", "silu_in_place_f16", "silu_gate_f16", "split_swiglu_f16", "squared_relu_f16",
            "rmsnorm_f16", "per_head_rmsnorm_f16", "softmax_f16",
            "add_rmsnorm_residual_f16", "rmsnorm_residual_f16", "add_rmsnorm_f16",
            "rope_f16",
            "split_qkv_f16", "de_interleave_q_f16", "split_unequal_qkv_f16",
            "repeat_tile_f16", "l2norm_groups_f16",
            "kv_cache_write_f16",
            "gated_attention_f16",
            "deltanet_step_f16", "compute_decay_beta_f16", "causal_conv1d_silu_f16",
            "embedding_lookup_f16",
            "matmul_f32_out_f16",
            "matmul_q4_0_aligned_2x4row_f16", "matmul_q4_1_2x4row_f16",
            "matmul_q5_k_16row_f16", "matmul_q6_k_16row_f16", "matmul_q6_k_16row_hact_fout",
            "matmul_q8_0_f16",
            "copy_f32", "copy_f32_region", "copy_f32_slice", "zero_f32", "fill_f32",
            "element_add", "element_mul",
            "silu", "silu_in_place", "silu_gate", "split_swiglu", "squared_relu",
            "rmsnorm", "per_head_rmsnorm", "softmax",
            "rope", "rope_neox",
            "split_qkv", "de_interleave_q",
            "kv_cache_write_f32",
            "gated_attention", "deltanet_step", "compute_decay_beta", "causal_conv1d", "causal_conv1d_silu",
            "embedding_lookup", "argmax",
            "add_rmsnorm_residual", "rmsnorm_residual", "add_rmsnorm",
            "split_unequal_qkv", "repeat_tile_f32", "l2norm_groups_f32",
        })
        {
            GetPipeline(fn);
        }
    }

    public string Name => $"Metal ({_dev.DeviceName})";

    /// <summary>
    /// Metal supports batched prefill ops. When true, ForwardPass calls
    /// BatchedEmbeddingLookup / BatchedRoPE / BatchedGatedAttention /
    /// BatchedKvCacheWrite and the batched MatMul path — which routes to the
    /// simdgroup_matrix tiled kernel for Q4_0. That's where the prefill win
    /// comes from.
    /// </summary>
    public bool SupportsBatchedOps => Environment.GetEnvironmentVariable("DAISI_METAL_NO_BATCH") != "1";

    // ── Tensor creation ──────────────────────────────────────────────────

    public ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions)
    {
        // In fp16 mode, back F32 tensors with half-precision storage. Logits
        // (which go into the sampler with full F32 precision) are the only
        // F32 tensor that stays F32.
        bool f16 = _fp16Activations
                   && type == GgmlType.F32
                   && name != "logits";
        return new MetalTensor(_dev, name, type, dimensions, f16Backed: f16);
    }

    public ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data)
    {
        // Repack Q4_0 (18-byte blocks) into 20-byte aligned blocks so the GPU
        // kernel can use uint32 loads (4× bandwidth / instruction throughput
        // vs per-byte reads). Layout after repack:
        //   [0..1]  FP16 scale
        //   [2..3]  zero padding (alignment)
        //   [4..19] 16 packed-nibble bytes (same as source [2..17])
        if (type == GgmlType.Q4_0 && dimensions.Length >= 2)
        {
            int blockCount = data.Length / 18;
            var aligned = new byte[blockCount * 20];
            for (int i = 0; i < blockCount; i++)
            {
                int src = i * 18;
                int dst = i * 20;
                aligned[dst]     = data[src];
                aligned[dst + 1] = data[src + 1];
                // aligned[dst+2, dst+3] left zero
                data.Slice(src + 2, 16).CopyTo(aligned.AsSpan(dst + 4, 16));
            }
            return new MetalTensor(_dev, name, type, dimensions, aligned, isAlignedQ4_0: true);
        }
        return new MetalTensor(_dev, name, type, dimensions, data);
    }

    public ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions) =>
        new MetalTensor(_dev, name, type, dimensions);

    // ── Pipeline + dispatch helpers ──────────────────────────────────────

    private IntPtr GetPipeline(string fn)
    {
        if (_pipelines.TryGetValue(fn, out var pso)) return pso;
        pso = _dev.NewComputePipeline(fn);
        _pipelines[fn] = pso;
        return pso;
    }

    /// <summary>
    /// Encode one compute dispatch onto the pending command buffer. No commit
    /// or wait — callers that need host-side synchronization must call <see
    /// cref="Flush"/> first. All dispatches share one encoder until flushed.
    /// </summary>
    private unsafe void Dispatch(
        string fn,
        ReadOnlySpan<IntPtr> buffers,
        void* paramsPtr, int paramsLen, int paramsIndex,
        uint gridX, uint tgX,
        bool useThreadgroups = true)
        => Dispatch2D(fn, buffers, paramsPtr, paramsLen, paramsIndex, gridX, 1u, tgX, useThreadgroups);

    /// <summary>
    /// Dispatch with a 2-D threadgroup grid (tg_id.x ∈ [0,gridX), tg_id.y ∈ [0,gridY)).
    /// Used for batched matmul kernels that tile in both M and N.
    /// </summary>
    private unsafe void Dispatch2D(
        string fn,
        ReadOnlySpan<IntPtr> buffers,
        void* paramsPtr, int paramsLen, int paramsIndex,
        uint gridX, uint gridY, uint tgX,
        bool useThreadgroups = true)
    {
        // When _reuseEncoder is true (default), keep one compute encoder open
        // across many dispatches; Metal's hazard tracking on shared-storage
        // MTLBuffers inserts the needed fences automatically. Falls back to
        // encoder-per-dispatch if DAISI_METAL_REUSE_ENCODER=0.
        if (_pendingCmdBuf == IntPtr.Zero)
        {
            _pendingCmdBuf = ObjC.Send(_dev.CommandQueue, Sel.commandBuffer);
        }
        if (!_reuseEncoder && _pendingEncoder != IntPtr.Zero)
        {
            ObjC.SendVoid(_pendingEncoder, Sel.endEncoding);
            _pendingEncoder = IntPtr.Zero;
        }
        if (_pendingEncoder == IntPtr.Zero)
        {
            _pendingEncoder = ObjC.Send(_pendingCmdBuf, Sel.computeCommandEncoder);
            _lastBoundPipeline = IntPtr.Zero;
        }

        IntPtr pso = GetPipeline(fn);
        if (pso != _lastBoundPipeline)
        {
            ObjC.SendVoid(_pendingEncoder, Sel.setComputePipelineState, pso);
            _lastBoundPipeline = pso;
        }

        for (int i = 0; i < buffers.Length; i++)
        {
            ObjC.SetBuffer(_pendingEncoder, Sel.setBuffer_offset_atIndex, buffers[i], 0, (nuint)i);
        }
        if (paramsLen > 0)
        {
            ObjC.SetBytes(_pendingEncoder, Sel.setBytes_length_atIndex, paramsPtr, (nuint)paramsLen, (nuint)paramsIndex);
        }

        var tg = new MTLSize((nuint)tgX, 1, 1);
        if (useThreadgroups)
        {
            var grid = new MTLSize((nuint)gridX, (nuint)gridY, 1);
            ObjC.SendVoid(_pendingEncoder, Sel.dispatchThreadgroups_threadsPerThreadgroup, grid, tg);
        }
        else
        {
            var grid = new MTLSize((nuint)gridX, (nuint)gridY, 1);
            ObjC.SendVoid(_pendingEncoder, Sel.dispatchThreads_threadsPerThreadgroup, grid, tg);
        }
        _encodedInBatch++;
        _totalDispatches++;

        if (_gpuProf)
        {
            // All times in same clock (CACurrentMediaTime == GPUStartTime clock).
            double tCommit = ObjC.CACurrentMediaTime();
            if (_pendingEncoder != IntPtr.Zero)
            {
                ObjC.SendVoid(_pendingEncoder, Sel.endEncoding);
                _pendingEncoder = IntPtr.Zero;
                _lastBoundPipeline = IntPtr.Zero;
            }
            ObjC.SendVoid(_pendingCmdBuf, Sel.commit);
            ObjC.SendVoid(_pendingCmdBuf, Sel.waitUntilCompleted);
            double gpuStart = ObjC.SendDouble(_pendingCmdBuf, Sel.GPUStartTime);
            double gpuEnd = ObjC.SendDouble(_pendingCmdBuf, Sel.GPUEndTime);
            double tDone = ObjC.CACurrentMediaTime();
            double submitMs = (gpuStart - tCommit) * 1000.0;
            double execMs = (gpuEnd - gpuStart) * 1000.0;
            double drainMs = (tDone - gpuEnd) * 1000.0;
            if (submitMs < 0) submitMs = 0;
            if (drainMs < 0) drainMs = 0;
            _gpuProfRows.Add((fn, submitMs, execMs, drainMs));
            _pendingCmdBuf = IntPtr.Zero;
            _encodedInBatch = 0;
            return;
        }

        if (_traceDispatch)
        {
            System.Threading.Interlocked.Increment(ref _traceDispatchCount);
            var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            Flush();
            long ns = (long)(System.Diagnostics.Stopwatch.GetElapsedTime(t0).Ticks * 100);
            if (!_traceByKernel.TryGetValue(fn, out var acc)) acc = (0L, 0L);
            _traceByKernel[fn] = (acc.count + 1, acc.nanos + ns);
            return;
        }

        if (!_batchEnabled)
        {
            Flush();
        }
        else if (_encodedInBatch >= _maxBatchSize)
        {
            Flush();
        }
    }

    /// <summary>
    /// End encoding, commit the command buffer, and block until the GPU is
    /// done. Call this before any host-side read/write of MetalTensor bytes.
    /// </summary>
    internal void Flush()
    {
        _encodedInBatch = 0;
        if (_pendingEncoder != IntPtr.Zero)
        {
            ObjC.SendVoid(_pendingEncoder, Sel.endEncoding);
            _pendingEncoder = IntPtr.Zero;
            _lastBoundPipeline = IntPtr.Zero;
        }
        if (_pendingCmdBuf != IntPtr.Zero)
        {
            _totalFlushes++;
            if (_totalFlushes < 10 && Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_TRACE") == "1")
                Console.Error.WriteLine($"[metal-flush-trace] flush #{_totalFlushes} dispatches-in-batch={_encodedInBatch}\n{new System.Diagnostics.StackTrace(1, false)}");
            var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            ObjC.SendVoid(_pendingCmdBuf, Sel.commit);
            ObjC.SendVoid(_pendingCmdBuf, Sel.waitUntilCompleted);
            if (_traceDispatch)
            {
                System.Threading.Interlocked.Increment(ref _traceFlushCount);
                System.Threading.Interlocked.Add(ref _traceFlushNanos,
                    (long)(System.Diagnostics.Stopwatch.GetElapsedTime(t0).Ticks * 100));
            }
            IntPtr err = ObjC.Send(_pendingCmdBuf, Sel.error);
            if (err != IntPtr.Zero)
            {
                string msg = ObjC.NSStringToManaged(ObjC.Send(err, Sel.localizedDescription)) ?? "metal dispatch error";
                _pendingCmdBuf = IntPtr.Zero;
                throw new InvalidOperationException($"Metal command buffer failed: {msg}");
            }
            _pendingCmdBuf = IntPtr.Zero;
        }
    }

    public void DumpTrace()
    {
        if (_traceDispatch)
        {
            Console.Error.WriteLine($"metal trace: dispatches={_traceDispatchCount}, encodeTime={_traceDispatchNanos / 1e6:F1}ms, flushes={_traceFlushCount}, gpuTime={_traceFlushNanos / 1e6:F1}ms");
            Console.Error.WriteLine("  per-kernel breakdown (count, total_ms, avg_ms):");
            foreach (var (fn, acc) in _traceByKernel.OrderByDescending(kv => kv.Value.nanos))
            {
                Console.Error.WriteLine($"    {fn,-28} {acc.count,6}  {acc.nanos / 1e6,8:F1} ms  {acc.nanos / 1e6 / acc.count,6:F3} ms/op");
            }
            _traceDispatchCount = 0; _traceDispatchNanos = 0; _traceFlushCount = 0; _traceFlushNanos = 0;
            _traceByKernel.Clear();
        }

        if (_gpuProf)
        {
            var agg = new Dictionary<string, (int count, double submitMs, double execMs, double drainMs)>();
            foreach (var (fn, sub, exec, drain) in _gpuProfRows)
            {
                agg.TryGetValue(fn, out var a);
                agg[fn] = (a.count + 1, a.submitMs + sub, a.execMs + exec, a.drainMs + drain);
            }
            double gSub = _gpuProfRows.Sum(r => r.submit);
            double gExec = _gpuProfRows.Sum(r => r.exec);
            double gDrain = _gpuProfRows.Sum(r => r.drain);
            Console.Error.WriteLine($"metal gpuprof: {_gpuProfRows.Count} dispatches, totals: submit={gSub:F0}ms exec={gExec:F0}ms drain={gDrain:F0}ms");
            Console.Error.WriteLine($"  {"kernel",-28} {"n",5} {"subAvg",8} {"execAvg",8} {"drainAvg",9} {"total",8}");
            foreach (var (fn, a) in agg.OrderByDescending(kv => kv.Value.submitMs + kv.Value.execMs + kv.Value.drainMs))
            {
                double total = a.submitMs + a.execMs + a.drainMs;
                Console.Error.WriteLine($"  {fn,-28} {a.count,5} {a.submitMs / a.count,5:F2} ms {a.execMs / a.count,5:F3} ms {a.drainMs / a.count,6:F2} ms {total,5:F0} ms");
            }
            _gpuProfRows.Clear();
        }
    }

    public void BeginCommands() { /* encoder is created lazily on first Dispatch */ }
    public void FlushCommands() => Flush();
    public void Synchronize() => Flush();

    private static IntPtr H(ITensor t) => ((MetalTensor)t).Buffer.Handle;

    // ── MatMul ───────────────────────────────────────────────────────────

    public unsafe void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N)
    {
        if (M == 1)
        {
            MatMulRow((MetalTensor)output, (MetalTensor)a, (MetalTensor)b, K, N);
            if (_flushAfterMatMul) Flush();
            return;
        }

        // ── Batched prefill path ────────────────────────────────────────
        // For compatible shapes, use the simdgroup_matrix tiled kernel that
        // computes the full M×N output in a single dispatch. This is the
        // critical prefill optimization — replaces M× per-row dispatches +
        // copies with one tiled GEMM using Apple's 8×8 matrix MAC unit.
        var mtB = (MetalTensor)b;
        var mtA = (MetalTensor)a;
        var mtO = (MetalTensor)output;
        if (!mtA.IsF16Backed && !mtO.IsF16Backed
            && (N & 31) == 0 && (K & 31) == 0)
        {
            string? mmFn = null;
            if (mtB.Type == GgmlType.Q4_0 && mtB.IsAlignedQ4_0) mmFn = "matmul_mm_q4_0_aligned";
            else if (mtB.Type == GgmlType.Q8_0) mmFn = "matmul_mm_q8_0";

            if (mmFn != null)
            {
                // Opt-in half-precision variant — Apple GPUs execute half
                // simdgroup matrix-multiply at 2× the float rate. Enabled via
                // DAISI_METAL_MM_HALF=1 to A/B against the float-accumulator
                // baseline while we validate accuracy in long generations.
                if (_halfMatMul && mmFn == "matmul_mm_q4_0_aligned") mmFn = "matmul_mm_q4_0_aligned_h";

                var p = new MatMulParams { M = (uint)M, K = (uint)K, N = (uint)N };
                var bufs = stackalloc IntPtr[3] { mtO.Buffer.Handle, mtA.Buffer.Handle, mtB.Buffer.Handle };
                var span = new ReadOnlySpan<IntPtr>(bufs, 3);
                // Tile: BM=64, BN=32. Grid covers ⌈M/64⌉ × ⌈N/32⌉ threadgroups.
                uint gridX = (uint)(N / 32);
                uint gridY = (uint)((M + 63) / 64);
                Dispatch2D(mmFn, span, &p, sizeof(MatMulParams), 3,
                           gridX, gridY, tgX: 128);
                if (_flushAfterMatMul) Flush();
                return;
            }
        }

        // Q5_K: the mv kernel DOES have an internal M loop, but tested slower
        // than the per-row fallback — serial M work in one TG loses more
        // parallelism than it saves in dispatch overhead for these shapes.
        // (The M-loop branch stays in the kernel for M=1 correctness.)



        // Fallback: fully-on-GPU per-row dispatches. For each of the M rows,
        // copy a[i,:] → scratch row, run matmul, copy output row → o[i,:].
        // All queued onto the batched command buffer — no per-row Flush.
        bool aF16 = mtA.IsF16Backed;
        bool oF16 = mtO.IsF16Backed;
        using var aRow = new MetalTensor(_dev, "_mm_a_row", GgmlType.F32, new long[] { K }, f16Backed: aF16);
        using var oRow = new MetalTensor(_dev, "_mm_o_row", GgmlType.F32, new long[] { N }, f16Backed: oF16);
        for (int i = 0; i < M; i++)
        {
            CopyTensorRegion(aRow, a, i * K, K);
            MatMulRow(oRow, aRow, (MetalTensor)b, K, N);
            CopyTensorSlice(output, i * N, oRow, 0, N);
        }
    }

    private unsafe void MatMulRow(MetalTensor output, MetalTensor a, MetalTensor b, int K, int N)
    {
        if (Environment.GetEnvironmentVariable("DAISI_METAL_ALL_MM_CPU") == "1")
        {
            CpuFallbackMatMul(output, a, b, 1, K, N);
            return;
        }
        var p = new MatMulParams { M = 1, K = (uint)K, N = (uint)N };
        var bufs = stackalloc IntPtr[3] { output.Buffer.Handle, a.Buffer.Handle, b.Buffer.Handle };
        var span = new ReadOnlySpan<IntPtr>(bufs, 3);

        // fp16 activation routing. The key case: lm_head has F16 input but F32
        // logits output — dispatch `matmul_q6_k_16row_hact_fout`.
        bool aF16 = a.IsF16Backed;
        bool oF16 = output.IsF16Backed;

        string fn;
        uint grid;
        switch (b.Type)
        {
            case GgmlType.F32:
                if (oF16)
                {
                    fn = "matmul_f32_out_f16"; grid = (uint)N;
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                fn = "matmul_f32"; grid = (uint)N; break;
            case GgmlType.F16:
                fn = "matmul_f16"; grid = (uint)N; break;
            case GgmlType.Q8_0:
                if (aF16 && oF16)
                {
                    fn = "matmul_q8_0_f16"; grid = (uint)N;
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                // llama.cpp-style mv kernel: 4 workers cooperate per block,
                // NR0=2 rows × NSG=1 sg = 2 rows per TG.
                if ((N & 1) == 0 && Environment.GetEnvironmentVariable("DAISI_METAL_Q8_MV_OFF") != "1")
                {
                    fn = "matmul_q8_0_mv"; grid = (uint)(N / 2);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 32);
                    return;
                }
                fn = "matmul_q8_0"; grid = (uint)N; break;
            case GgmlType.Q4_0:
                if (b.IsAlignedQ4_0)
                {
                    if (aF16 && oF16 && (N & 7) == 0)
                    {
                        fn = "matmul_q4_0_aligned_2x4row_f16"; grid = (uint)(N / 8);
                        Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                        return;
                    }
                    // llama.cpp-style mv kernel: 16 workers cooperate per
                    // super-block with yl pre-scaling. NR0=4 rows per sg ×
                    // NSG=2 sgs = 8 rows per TG (same as 2x4row).
                    if ((N & 7) == 0)
                    {
                        fn = "matmul_q4_0_aligned_mv"; grid = (uint)(N / 8);
                        Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                        return;
                    }
                    if ((N & 3) == 0)
                    {
                        fn = "matmul_q4_0_aligned_simd_4row"; grid = (uint)(N / 4);
                        Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 32);
                        return;
                    }
                    fn = "matmul_q4_0_aligned_simd"; grid = (uint)N;
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 32);
                    return;
                }
                fn = "matmul_q4_0"; grid = (uint)N; break;
            case GgmlType.Q4_1:
                if (aF16 && oF16 && (N & 7) == 0)
                {
                    fn = "matmul_q4_1_2x4row_f16"; grid = (uint)(N / 8);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                // llama.cpp-style mv kernel: 16 workers cooperate per
                // super-block (same layout as Q4_0 mv but with +min term).
                if ((N & 7) == 0)
                {
                    fn = "matmul_q4_1_mv"; grid = (uint)(N / 8);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                fn = "matmul_q4_1"; grid = (uint)N; break;
            case GgmlType.Q5_K:
                if (Environment.GetEnvironmentVariable("DAISI_METAL_Q5K_CPU") == "1")
                {
                    CpuFallbackMatMul(output, a, b, 1, K, N);
                    return;
                }
                if (aF16 && oF16 && K / 256 <= 16 && (N & 15) == 0)
                {
                    fn = "matmul_q5_k_16row_f16"; grid = (uint)(N / 16);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                // llama.cpp-style mv kernel: 8 lanes cooperate per super-block
                // via stride-4 ix interleaving. NR0=2 rows per sg × NSG=2 sgs.
                if ((N & 3) == 0)
                {
                    fn = "matmul_q5_k_mv"; grid = (uint)(N / 4);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                if (K / 256 <= 16 && (N & 15) == 0)
                {
                    fn = "matmul_q5_k_16row"; grid = (uint)(N / 16);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                if (K / 256 <= 16)
                {
                    fn = "matmul_q5_k_tg16"; grid = (uint)N;
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 16);
                    return;
                }
                fn = "matmul_q5_k"; grid = (uint)N; break;
            case GgmlType.Q6_K:
                // lm_head boundary: F16 input, F32 logits output.
                if (aF16 && !oF16 && K / 256 <= 16 && (N & 15) == 0)
                {
                    fn = "matmul_q6_k_16row_hact_fout"; grid = (uint)(N / 16);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                if (aF16 && oF16 && K / 256 <= 16 && (N & 15) == 0)
                {
                    fn = "matmul_q6_k_16row_f16"; grid = (uint)(N / 16);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                // New mv-pattern kernel: 16 lanes cooperate per super-block
                // (coalesced reads) vs old 4-lane-per-row layout. NR0=2 rows
                // per sg × NSG=2 sgs = 4 rows per TG.
                if ((N & 3) == 0 && Environment.GetEnvironmentVariable("DAISI_METAL_Q6K_MV_OFF") != "1")
                {
                    fn = "matmul_q6_k_mv"; grid = (uint)(N / 4);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                if (K / 256 <= 16 && (N & 31) == 0)
                {
                    fn = "matmul_q6_k_32row"; grid = (uint)(N / 32);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 128);
                    return;
                }
                if (K / 256 <= 16 && (N & 15) == 0)
                {
                    fn = "matmul_q6_k_16row"; grid = (uint)(N / 16);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                if (K / 256 <= 16)
                {
                    fn = "matmul_q6_k_tg16"; grid = (uint)N;
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 16);
                    return;
                }
                fn = "matmul_q6_k"; grid = (uint)N; break;
            case GgmlType.Q4_K:
                // llama.cpp-style mv kernel: 8 lanes cooperate per super-block
                // via stride-4 ix. Same dispatch pattern as Q5_K.
                if ((N & 3) == 0 && Environment.GetEnvironmentVariable("DAISI_METAL_Q4K_CPU") != "1")
                {
                    fn = "matmul_q4_k_mv"; grid = (uint)(N / 4);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                CpuFallbackMatMul(output, a, b, 1, K, N);
                return;
            case GgmlType.Q2_K:
                // llama.cpp-style mv kernel: 8 lanes cooperate per super-block
                // via stride-4 ix × 8 it. Sub-blocks are 16 weights each.
                if ((N & 3) == 0)
                {
                    fn = "matmul_q2_k_mv"; grid = (uint)(N / 4);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                CpuFallbackMatMul(output, a, b, 1, K, N);
                return;
            case GgmlType.BF16:
                if ((N & 3) == 0)
                {
                    fn = "matmul_bf16_mv"; grid = (uint)(N / 4);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                fn = "matmul_bf16"; grid = (uint)N; break;
            case GgmlType.Q5_0:
                if (Environment.GetEnvironmentVariable("DAISI_METAL_Q5_0_CPU") == "1")
                {
                    CpuFallbackMatMul(output, a, b, 1, K, N);
                    return;
                }
                if ((N & 7) == 0 && Environment.GetEnvironmentVariable("DAISI_METAL_Q5_0_SIMPLE") != "1")
                {
                    fn = "matmul_q5_0_mv"; grid = (uint)(N / 8);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                // Fallback / diagnostic kernel.
                fn = "matmul_q5_0"; grid = (uint)N; break;
            case GgmlType.I2_S:
                // I2_S (BitNet ternary): 4 ternary codes per byte. Scale is
                // a single F32 at end of weight blob. K must be a multiple of
                // 128 (group size); N in 2-row blocks per sg.
                if ((N & 3) == 0 && (K & 127) == 0)
                {
                    fn = "matmul_i2s_mv"; grid = (uint)(N / 4);
                    Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
                    return;
                }
                CpuFallbackMatMul(output, a, b, 1, K, N);
                return;
            default:
                CpuFallbackMatMul(output, a, b, 1, K, N);
                return;
        }
        Dispatch(fn, span, &p, sizeof(MatMulParams), 3, grid, tgX: 64);
    }

    private void CpuFallbackMatMul(MetalTensor output, MetalTensor a, MetalTensor b, int M, int K, int N)
    {
        if (Environment.GetEnvironmentVariable("DAISI_METAL_FALLBACK_TRACE") == "1")
            Console.Error.WriteLine($"[metal-cpu-mm] {b.Name} type={b.Type} M={M} K={K} N={N}");
        Flush();
        var o = output.AsFloatSpan();
        var aSpan = a.AsFloatSpan();
        var bBytes = b.RawBytes();
        switch (b.Type)
        {
            case GgmlType.F32: Cpu.MatMul.Multiply(o, aSpan, MemoryMarshal.Cast<byte, float>(bBytes), M, K, N); return;
            case GgmlType.Q8_0: Cpu.MatMul.MultiplyQ8_0(o, aSpan, bBytes, M, K, N); return;
            case GgmlType.Q4_0: Cpu.MatMul.MultiplyQ4_0(o, aSpan, bBytes, M, K, N); return;
            case GgmlType.Q4_K: Cpu.MatMul.MultiplyQ4_K(o, aSpan, bBytes, M, K, N); return;
            case GgmlType.BF16: Cpu.MatMul.MultiplyBF16(o, aSpan, bBytes, M, K, N); return;
            case GgmlType.F16: Cpu.MatMul.MultiplyF16(o, aSpan, MemoryMarshal.Cast<byte, Half>(bBytes), M, K, N); return;
            case GgmlType.I2_S: Cpu.I2SDequant.Multiply(o, aSpan, bBytes, M, K, N); return;
            case GgmlType.TQ1_0: Cpu.TernaryMatMul.MultiplyTQ1_0(o, aSpan, bBytes, M, K, N); return;
            default: GenericDequantMatMul(o, aSpan, b, M, K, N); return;
        }
    }

    private static void GenericDequantMatMul(Span<float> output, ReadOnlySpan<float> a, MetalTensor b, int M, int K, int N)
    {
        var dequant = new float[b.ElementCount];
        b.DequantizeTo(dequant);
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                float dot = 0;
                for (int k = 0; k < K; k++) dot += a[i * K + k] * dequant[j * K + k];
                output[i * N + j] = dot;
            }
    }

    // ── Elementwise ──────────────────────────────────────────────────────

    // Kernel suffix for fp16-backed tensors. Empty string = F32 path.
    private string Sx(ITensor t) => ((MetalTensor)t).IsF16Backed ? "_f16" : "";

    public unsafe void ElementAdd(ITensor output, ITensor a, ITensor b)
    {
        var outElems = ((MetalTensor)output).ElementCount;
        var bElems   = ((MetalTensor)b).ElementCount;
        // Broadcast path: `b` is a bias (row-sized) and `output`/`a` is the
        // batched [M × rowDim] activation. Repeats the bias row M times.
        // Needed for Qwen2 attention biases during ForwardBatchedPrefill.
        if (outElems > bElems && outElems % bElems == 0 && !((MetalTensor)output).IsF16Backed)
        {
            var pBc = new UintParams { N = (uint)outElems, Extra0 = (uint)bElems };
            var bufsBc = stackalloc IntPtr[3] { H(output), H(a), H(b) };
            Dispatch("element_add_broadcast_row", new ReadOnlySpan<IntPtr>(bufsBc, 3),
                     &pBc, sizeof(UintParams), 3, (uint)outElems, 256, useThreadgroups: false);
            return;
        }
        var p = new ElementParams { N = (uint)outElems };
        var bufs = stackalloc IntPtr[3] { H(output), H(a), H(b) };
        Dispatch("element_add" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(ElementParams), 3, p.N, 256, useThreadgroups: false);
    }

    public unsafe void ElementMul(ITensor output, ITensor a, ITensor b)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)output).ElementCount };
        var bufs = stackalloc IntPtr[3] { H(output), H(a), H(b) };
        Dispatch("element_mul" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(ElementParams), 3, p.N, 256, useThreadgroups: false);
    }

    public unsafe void SiLU(ITensor output, ITensor input)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)output).ElementCount };
        var bufs = stackalloc IntPtr[2] { H(output), H(input) };
        Dispatch("silu" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(ElementParams), 2, p.N, 256, useThreadgroups: false);
    }

    public unsafe void SiLUInPlace(ITensor data)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)data).ElementCount };
        var bufs = stackalloc IntPtr[1] { H(data) };
        Dispatch("silu_in_place" + Sx(data), new ReadOnlySpan<IntPtr>(bufs, 1), &p, sizeof(ElementParams), 1, p.N, 256, useThreadgroups: false);
    }

    public unsafe void SiLUGate(ITensor output, ITensor data, ITensor gate)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)output).ElementCount };
        var bufs = stackalloc IntPtr[3] { H(output), H(data), H(gate) };
        Dispatch("silu_gate" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(ElementParams), 3, p.N, 256, useThreadgroups: false);
    }

    // Fused SwiGLU: out[i] = silu(gate[i]) * up[i]. Default is 2 dispatches;
    // silu_gate already computes `data * silu(gate)` which is the same thing.
    public unsafe void SwiGLU(ITensor output, ITensor gate, ITensor up)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)output).ElementCount };
        var bufs = stackalloc IntPtr[3] { H(output), H(up), H(gate) };
        Dispatch("silu_gate" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(ElementParams), 3, p.N, 256, useThreadgroups: false);
    }

    /// <summary>
    /// Fused gate+up+SwiGLU for Q4_0-aligned weights. Replaces 3 separate
    /// kernels (gate matmul, up matmul, silu_gate) with one, and eliminates
    /// the intermediate gate/up activation buffers.
    /// </summary>
    public unsafe void MatMulSwiGLU(ITensor output, ITensor a, ITensor gateWeights, ITensor upWeights, int M, int K, int N)
    {
        var gw = (MetalTensor)gateWeights;
        var uw = (MetalTensor)upWeights;
        bool aF16 = ((MetalTensor)a).IsF16Backed;
        // F32 fused path only when activations are F32. For F16 we don't have
        // a fused F16 variant yet — fall through to separate matmul + silu_gate
        // (both have F16 variants).
        if (!aF16 && M == 1 && gw.Type == GgmlType.Q4_0 && uw.Type == GgmlType.Q4_0
            && gw.IsAlignedQ4_0 && uw.IsAlignedQ4_0 && (N & 7) == 0)
        {
            var p = new MatMulParams { M = 1, K = (uint)K, N = (uint)N };
            var bufs = stackalloc IntPtr[4] { H(output), H(a), gw.Buffer.Handle, uw.Buffer.Handle };
            Dispatch("matmul_q4_0_aligned_swiglu_2x4row",
                new ReadOnlySpan<IntPtr>(bufs, 4), &p, sizeof(MatMulParams), 4,
                gridX: (uint)(N / 8), tgX: 64);
            return;
        }
        // Fallback: default (separate ops).
        MatMul(output, a, gateWeights, M, K, N);
        var temp = CreateTensor("_swiGLU_temp", output.Type, output.Dimensions);
        MatMul(temp, a, upWeights, M, K, N);
        SwiGLU(output, output, temp);
        temp.Dispose();
    }

    public unsafe void SplitSwiGLU(ITensor output, ITensor fusedInput, int N)
    {
        var p = new ElementParams { N = (uint)N };
        var bufs = stackalloc IntPtr[2] { H(output), H(fusedInput) };
        Dispatch("split_swiglu" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(ElementParams), 2, p.N, 256, useThreadgroups: false);
    }

    public unsafe void SquaredReLU(ITensor data)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)data).ElementCount };
        var bufs = stackalloc IntPtr[1] { H(data) };
        Dispatch("squared_relu" + Sx(data), new ReadOnlySpan<IntPtr>(bufs, 1), &p, sizeof(ElementParams), 1, p.N, 256, useThreadgroups: false);
    }

    // ── Copy/Fill/Zero ───────────────────────────────────────────────────

    public unsafe void CopyTensor(ITensor dst, ITensor src)
    {
        var s = (MetalTensor)src;
        var d = (MetalTensor)dst;
        if (s.Type == GgmlType.F32 && d.Type == GgmlType.F32)
        {
            var p = new UintParams { N = (uint)s.ElementCount };
            var bufs = stackalloc IntPtr[2] { d.Buffer.Handle, s.Buffer.Handle };
            string fn = s.IsF16Backed ? "copy_f16" : "copy_f32";
            Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(UintParams), 2, p.N, 256, useThreadgroups: false);
            return;
        }
        Flush();
        s.RawBytes().CopyTo(d.Buffer.AsByteSpan().Slice(0, (int)s.ByteSize));
    }

    public unsafe void CopyTensorBytes(ITensor dst, ITensor src, long byteCount)
    {
        // If dst & src are F16-backed, copy halves. Otherwise F32.
        var s = (MetalTensor)src;
        if (s.IsF16Backed && (byteCount & 1) == 0)
        {
            var p = new UintParams { N = (uint)(byteCount / 2) };
            var bufs = stackalloc IntPtr[2] { H(dst), H(src) };
            Dispatch("copy_f16", new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(UintParams), 2, p.N, 256, useThreadgroups: false);
            return;
        }
        if ((byteCount & 3) == 0)
        {
            var p = new UintParams { N = (uint)(byteCount / 4) };
            var bufs = stackalloc IntPtr[2] { H(dst), H(src) };
            Dispatch("copy_f32", new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(UintParams), 2, p.N, 256, useThreadgroups: false);
            return;
        }
        Flush();
        ((MetalTensor)src).Buffer.AsByteSpan().Slice(0, (int)byteCount).CopyTo(((MetalTensor)dst).Buffer.AsByteSpan());
    }

    public unsafe void CopyTensorRegion(ITensor dst, ITensor src, int srcOffset, int count)
    {
        var p = new UintParams { N = (uint)count, Extra0 = (uint)srcOffset };
        var bufs = stackalloc IntPtr[2] { H(dst), H(src) };
        string fn = ((MetalTensor)src).IsF16Backed ? "copy_f16_region" : "copy_f32_region";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(UintParams), 2, p.N, 256, useThreadgroups: false);
        if (_flushAfterCopy) Flush();
    }

    public unsafe void CopyTensorSlice(ITensor dst, int dstOffset, ITensor src, int srcOffset, int count)
    {
        var p = new UintParams { N = (uint)count, Extra0 = (uint)srcOffset, Extra1 = (uint)dstOffset };
        var bufs = stackalloc IntPtr[2] { H(dst), H(src) };
        string fn = ((MetalTensor)src).IsF16Backed ? "copy_f16_slice" : "copy_f32_slice";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(UintParams), 2, p.N, 256, useThreadgroups: false);
        if (_flushAfterCopy) Flush();
    }

    public unsafe void ZeroTensor(ITensor tensor)
    {
        var t = (MetalTensor)tensor;
        // F16-backed: 2 bytes per element. F32: 4 bytes per element.
        uint n = t.IsF16Backed ? (uint)t.ElementCount : (uint)(t.ByteSize / 4);
        var p = new UintParams { N = n };
        var bufs = stackalloc IntPtr[1] { t.Buffer.Handle };
        string fn = t.IsF16Backed ? "zero_f16" : "zero_f32";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 1), &p, sizeof(UintParams), 1, p.N, 256, useThreadgroups: false);
    }

    public unsafe void FillTensor(ITensor tensor, float value)
    {
        var t = (MetalTensor)tensor;
        var p = new UintParams { N = (uint)t.ElementCount, Extra0 = BitConverter.SingleToUInt32Bits(value) };
        var bufs = stackalloc IntPtr[1] { t.Buffer.Handle };
        string fn = t.IsF16Backed ? "fill_f16" : "fill_f32";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 1), &p, sizeof(UintParams), 1, p.N, 256, useThreadgroups: false);
    }

    // ── RmsNorm / Softmax ────────────────────────────────────────────────

    public unsafe void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
    {
        // The rmsnorm kernel is per-row (p.n = row width, gridX = M). Derive M
        // from the ratio of input elements to weight elements — the weight has
        // one parameter per feature, so weight.ElementCount = rowDim.
        uint rowDim = (uint)((MetalTensor)weight).ElementCount;
        uint M = (uint)((MetalTensor)input).ElementCount / rowDim;
        var p = new RmsNormParams { N = rowDim, Eps = eps };
        var bufs = stackalloc IntPtr[3] { H(output), H(input), H(weight) };
        Dispatch("rmsnorm" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(RmsNormParams), 3, gridX: M, tgX: 256);
    }

    public unsafe void Softmax(ITensor output, ITensor input)
    {
        var p = new ElementParams { N = (uint)((MetalTensor)input).ElementCount };
        var bufs = stackalloc IntPtr[2] { H(output), H(input) };
        Dispatch("softmax" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(ElementParams), 2, gridX: 1, tgX: 256);
    }

    public unsafe void PerHeadRmsNorm(ITensor data, ITensor weight, int numHeads, int headDim, float eps)
    {
        var p = new PerHeadRmsNormParams { NumHeads = (uint)numHeads, HeadDim = (uint)headDim, Eps = eps };
        var bufs = stackalloc IntPtr[2] { H(data), H(weight) };
        Dispatch("per_head_rmsnorm" + Sx(data), new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(PerHeadRmsNormParams), 2, gridX: (uint)numHeads, tgX: 256);
    }

    // ── RoPE ─────────────────────────────────────────────────────────────

    public unsafe void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        RoPEDispatch(q, k, headDim, ropeDim, positionOffset, ropeTheta, freqFactors: null, neox: false);
    }

    public unsafe void RoPEWithFreqFactors(ITensor q, ITensor k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ITensor? freqFactors)
    {
        RoPEDispatch(q, k, headDim, ropeDim, positionOffset, ropeTheta, freqFactors, neox: false);
    }

    public unsafe void RoPENeox(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        RoPEDispatch(q, k, headDim, ropeDim, positionOffset, ropeTheta, freqFactors: null, neox: true);
    }

    public unsafe void RoPENeoxWithFreqFactors(ITensor q, ITensor k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ITensor? freqFactors)
    {
        RoPEDispatch(q, k, headDim, ropeDim, positionOffset, ropeTheta, freqFactors, neox: true);
    }

    private unsafe void RoPEDispatch(ITensor q, ITensor k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ITensor? freqFactors, bool neox)
    {
        uint qTotal = (uint)((MetalTensor)q).ElementCount;
        uint kTotal = (uint)((MetalTensor)k).ElementCount;
        if (ropeDim == 0) ropeDim = headDim;

        var p = new RoPEParams
        {
            QTotal = qTotal,
            KTotal = kTotal,
            HeadDim = (uint)headDim,
            RopeDim = (uint)ropeDim,
            PositionOffset = positionOffset,
            RopeTheta = ropeTheta,
            UseFreqFactors = freqFactors != null ? 1u : 0u,
            Neox = neox ? 1u : 0u,
        };

        // If freqFactors is null we still need to bind *some* buffer. Bind q's
        // buffer harmlessly — the kernel only reads it when UseFreqFactors != 0.
        IntPtr ff = freqFactors != null ? H(freqFactors) : H(q);
        var bufs = stackalloc IntPtr[3] { H(q), H(k), ff };
        string fn = neox ? "rope_neox" : "rope";

        string sx = Sx(q);
        if (!neox)
        {
            uint pairs = Math.Max(qTotal, kTotal) / 2u;
            Dispatch(fn + sx, new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(RoPEParams), 3, pairs, 256, useThreadgroups: false);
        }
        else
        {
            uint halfDim = (uint)ropeDim / 2u;
            uint qHeads = qTotal / (uint)headDim;
            uint kHeads = kTotal / (uint)headDim;
            uint total = Math.Max(qHeads, kHeads) * halfDim;
            Dispatch(fn + sx, new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(RoPEParams), 3, total, 256, useThreadgroups: false);
        }
    }

    // ── SplitQKV / DeInterleaveQ ─────────────────────────────────────────

    public unsafe void SplitQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int innerSize)
    {
        var p = new SplitQKVParams { InnerSize = (uint)innerSize };
        var bufs = stackalloc IntPtr[4] { H(q), H(k), H(v), H(qkv) };
        Dispatch("split_qkv" + Sx(q), new ReadOnlySpan<IntPtr>(bufs, 4), &p, sizeof(SplitQKVParams), 4, (uint)innerSize, 256, useThreadgroups: false);
    }

    public unsafe void DeInterleaveQ(ITensor qAttn, ITensor qGate, ITensor qFull, int numHeads, int headDim)
    {
        var p = new DeInterleaveParams { NumHeads = (uint)numHeads, HeadDim = (uint)headDim };
        var bufs = stackalloc IntPtr[3] { H(qAttn), H(qGate), H(qFull) };
        Dispatch("de_interleave_q" + Sx(qAttn), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(DeInterleaveParams), 3,
            (uint)(numHeads * headDim), 256, useThreadgroups: false);
    }

    // ── KV cache write ───────────────────────────────────────────────────

    public unsafe void KvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int position)
    {
        var p = new KvWriteParams
        {
            NKvHeads = (uint)nKvHeads,
            KeyLength = (uint)keyLength,
            ValueLength = (uint)valueLength,
            MaxSeqLen = (uint)maxSeqLen,
            Position = (uint)position,
        };
        var bufs = stackalloc IntPtr[4] { H(kCache), H(vCache), H(k), H(v) };
        uint maxE = (uint)Math.Max(nKvHeads * keyLength, nKvHeads * valueLength);
        string fn = ((MetalTensor)kCache).IsF16Backed ? "kv_cache_write_f16" : "kv_cache_write_f32";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 4), &p, sizeof(KvWriteParams), 4,
            maxE, 256, useThreadgroups: false);
        if (_flushAfterKv) Flush();
    }

    // ── Gated attention ──────────────────────────────────────────────────

    public unsafe void GatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale)
    {
        var p = new GatedAttnParams
        {
            NumHeads = (uint)numHeads,
            NumKvHeads = (uint)numKvHeads,
            KeyLength = (uint)keyLength,
            ValueLength = (uint)valueLength,
            MaxSeqLen = (uint)maxSeqLen,
            SeqLen = (uint)seqLen,
            Scale = scale,
        };
        var bufs = stackalloc IntPtr[5] { H(output), H(qAttn), H(qGate), H(kCache), H(vCache) };
        Dispatch("gated_attention" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 5), &p, sizeof(GatedAttnParams), 5,
            (uint)numHeads, 256);
        if (_flushAfterAttn) Flush();
    }

    private void GatedAttentionCpu(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale)
    {
        Flush();
        var q = ((MetalTensor)qAttn).AsFloatSpan();
        var g = ((MetalTensor)qGate).AsFloatSpan();
        var kC = ((MetalTensor)kCache).AsFloatSpan();
        var vC = ((MetalTensor)vCache).AsFloatSpan();
        var o = ((MetalTensor)output).AsFloatSpan();

        int groupSize = numHeads / numKvHeads;
        var scores = new float[seqLen];
        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / groupSize;
            int qOff = h * keyLength;
            int kBase = kvHead * maxSeqLen * keyLength;
            for (int t = 0; t < seqLen; t++)
            {
                float dot = 0;
                int kOff = kBase + t * keyLength;
                for (int d = 0; d < keyLength; d++) dot += q[qOff + d] * kC[kOff + d];
                scores[t] = dot * scale;
            }
            float max = float.NegativeInfinity;
            for (int t = 0; t < seqLen; t++) if (scores[t] > max) max = scores[t];
            float sum = 0;
            for (int t = 0; t < seqLen; t++) { scores[t] = MathF.Exp(scores[t] - max); sum += scores[t]; }
            float inv = 1.0f / sum;
            for (int t = 0; t < seqLen; t++) scores[t] *= inv;

            int vBase = kvHead * maxSeqLen * valueLength;
            int oOff = h * valueLength;
            for (int d = 0; d < valueLength; d++)
            {
                float acc = 0;
                for (int t = 0; t < seqLen; t++) acc += scores[t] * vC[vBase + t * valueLength + d];
                o[oOff + d] = acc;
            }
            int gOff = h * keyLength;
            for (int d = 0; d < valueLength; d++)
            {
                float gv = g[gOff + d];
                o[oOff + d] *= 1.0f / (1.0f + MathF.Exp(-gv));
            }
        }
    }

    // ── DeltaNet ─────────────────────────────────────────────────────────

    public unsafe void DeltaNetStep(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps)
    {
        var p = new DeltaNetParams
        {
            GroupCount = (uint)groupCount,
            HeadDim = (uint)headDim,
            Scale = scale,
            NormEps = normEps,
        };
        var bufs = stackalloc IntPtr[8] { H(output), H(q), H(k), H(v), H(state), H(decay), H(beta), H(normWeight) };
        Dispatch("deltanet_step" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 8), &p, sizeof(DeltaNetParams), 8,
            (uint)groupCount, 256);
        if (_flushAfterDeltaNet) Flush();
    }

    private void DeltaNetStepCpu(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps)
    {
        Flush();
        var qS = ((MetalTensor)q).AsFloatSpan();
        var kS = ((MetalTensor)k).AsFloatSpan();
        var vS = ((MetalTensor)v).AsFloatSpan();
        var oS = ((MetalTensor)output).AsFloatSpan();
        var st = ((MetalTensor)state).AsFloatSpan();
        var dc = ((MetalTensor)decay).AsFloatSpan();
        var bt = ((MetalTensor)beta).AsFloatSpan();
        var nw = ((MetalTensor)normWeight).AsFloatSpan();

        for (int g = 0; g < groupCount; g++)
        {
            int stOff = g * headDim * headDim;
            int vecOff = g * headDim;
            float d = dc[g]; float beta1 = bt[g];
            var sk = new float[headDim];
            for (int i = 0; i < headDim; i++)
            {
                float s = 0;
                for (int j = 0; j < headDim; j++) s += st[stOff + j * headDim + i] * kS[vecOff + j];
                sk[i] = s;
            }
            var err = new float[headDim];
            for (int i = 0; i < headDim; i++) err[i] = (vS[vecOff + i] - d * sk[i]) * beta1;
            var outLocal = new float[headDim];
            for (int i = 0; i < headDim; i++)
                for (int j = 0; j < headDim; j++)
                {
                    int idx = stOff + i * headDim + j;
                    st[idx] = d * st[idx] + kS[vecOff + i] * err[j];
                }
            for (int j = 0; j < headDim; j++)
            {
                float s = 0;
                for (int i = 0; i < headDim; i++) s += st[stOff + i * headDim + j] * qS[vecOff + i];
                outLocal[j] = s * scale;
            }
            double ss = 0;
            for (int j = 0; j < headDim; j++) ss += (double)outLocal[j] * outLocal[j];
            float inv = (float)(1.0 / Math.Sqrt(ss / headDim + normEps));
            for (int j = 0; j < headDim; j++) oS[vecOff + j] = outLocal[j] * inv * nw[j];
        }
    }

    public unsafe void ComputeDecayBeta(ITensor decay, ITensor beta, ITensor alphaProj, ITensor betaProj,
        ITensor ssmA, ITensor dtBias, int groupCount)
    {
        var p = new DecayBetaParams { GroupCount = (uint)groupCount };
        var bufs = stackalloc IntPtr[6] { H(decay), H(beta), H(alphaProj), H(betaProj), H(ssmA), H(dtBias) };
        Dispatch("compute_decay_beta" + Sx(decay), new ReadOnlySpan<IntPtr>(bufs, 6), &p, sizeof(DecayBetaParams), 6,
            (uint)groupCount, 256, useThreadgroups: false);
    }

    public unsafe void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        var p = new Conv1dParams { Channels = (uint)channels, KernelSize = (uint)kernelSize };
        var bufs = stackalloc IntPtr[3] { H(qkv), H(convBuffer), H(convWeight) };
        Dispatch("causal_conv1d", new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(Conv1dParams), 3,
            (uint)channels, 256, useThreadgroups: false);
    }

    public unsafe void CausalConv1dSiLU(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        var p = new Conv1dParams { Channels = (uint)channels, KernelSize = (uint)kernelSize };
        var bufs = stackalloc IntPtr[3] { H(qkv), H(convBuffer), H(convWeight) };
        Dispatch("causal_conv1d_silu" + Sx(qkv), new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(Conv1dParams), 3,
            (uint)channels, 256, useThreadgroups: false);
    }

    public unsafe void BatchedCausalConv1dSiLU(ITensor qkv, ITensor convBuffer, ITensor convWeight,
        int channels, int kernelSize, int M)
    {
        var p = new BatchedConv1dParams { Channels = (uint)channels, KernelSize = (uint)kernelSize, M = (uint)M };
        var bufs = stackalloc IntPtr[3] { H(qkv), H(convBuffer), H(convWeight) };
        Dispatch("batched_causal_conv1d_silu", new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(BatchedConv1dParams), 3,
            (uint)channels, 256, useThreadgroups: false);
    }

    public unsafe void BatchedDeltaNetFused(
        ITensor output, ITensor qkv, ITensor alpha, ITensor beta,
        ITensor gate, ITensor state, ITensor ssmA, ITensor dtBias, ITensor ssmNorm,
        int M, int qkvOutDim, int keyDim, int valueDim,
        int numKHeads, int numVHeads, int headDim,
        float scale, float normEps)
    {
        var p = new BatchedDeltaNetParams
        {
            M = (uint)M,
            QkvOutDim = (uint)qkvOutDim,
            KeyDim = (uint)keyDim,
            ValueDim = (uint)valueDim,
            NumKHeads = (uint)numKHeads,
            NumVHeads = (uint)numVHeads,
            HeadDim = (uint)headDim,
            RepeatFactor = (uint)(numVHeads / numKHeads),
            Scale = scale,
            NormEps = normEps,
        };
        var bufs = stackalloc IntPtr[9]
        {
            H(output), H(qkv), H(alpha), H(beta), H(gate),
            H(state), H(ssmA), H(dtBias), H(ssmNorm),
        };
        Dispatch("batched_deltanet_fused", new ReadOnlySpan<IntPtr>(bufs, 9), &p, sizeof(BatchedDeltaNetParams), 9,
            gridX: (uint)numVHeads, tgX: 256);
    }

    /// <summary>
    /// Fused DeltaNet is only safe when per-head state fits in threadgroup
    /// memory (headDim² floats). Reported conditionally based on model
    /// headDim — but we can't see headDim here, so the FP decides per-layer.
    /// This flag just says the backend has the kernel available.
    /// </summary>
    public bool SupportsFusedDeltaNetPrefill => true;

    // ── L2-norm groups (GPU) ─────────────────────────────────────────────

    public unsafe void L2NormGroups(ITensor data, int numGroups, int groupDim)
    {
        var p = new L2NormGroupsParams { GroupDim = (uint)groupDim };
        var bufs = stackalloc IntPtr[1] { H(data) };
        string fn = ((MetalTensor)data).IsF16Backed ? "l2norm_groups_f16" : "l2norm_groups_f32";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 1), &p, sizeof(L2NormGroupsParams), 1,
            (uint)numGroups, 256);
    }

    public unsafe void SplitUnequalQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int keyDim, int valueDim)
    {
        var p = new SplitUnequalQkvParams { KeyDim = (uint)keyDim, ValueDim = (uint)valueDim };
        var bufs = stackalloc IntPtr[4] { H(q), H(k), H(v), H(qkv) };
        Dispatch("split_unequal_qkv" + Sx(q), new ReadOnlySpan<IntPtr>(bufs, 4), &p, sizeof(SplitUnequalQkvParams), 4,
            (uint)valueDim, 256, useThreadgroups: false);
    }

    public unsafe void RepeatTile(ITensor tensor, int numHeads, int headDim, int factor)
    {
        int srcSize = numHeads * headDim;
        int total = srcSize * factor;
        var p = new RepeatTileParams { SrcSize = (uint)srcSize, Factor = (uint)factor };
        var bufs = stackalloc IntPtr[1] { H(tensor) };
        string fn = ((MetalTensor)tensor).IsF16Backed ? "repeat_tile_f16" : "repeat_tile_f32";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 1), &p, sizeof(RepeatTileParams), 1,
            (uint)total, 256, useThreadgroups: false);
    }

    // ── Embedding lookup ─────────────────────────────────────────────────

    public unsafe void EmbeddingLookup(ITensor output, ITensor table, int tokenId)
    {
        var t = (MetalTensor)table;
        uint hiddenDim = (uint)t.Dimensions[0];

        uint tableType = t.Type switch
        {
            GgmlType.F32 => 0u,
            GgmlType.Q8_0 => 1u,
            GgmlType.F16 => 2u,
            GgmlType.Q4_0 => t.IsAlignedQ4_0 ? 8u : 5u,
            _ => uint.MaxValue,
        };

        if (tableType == uint.MaxValue)
        {
            Flush();
            var outSpan = ((MetalTensor)output).AsFloatSpan();
            int blockSize = GgmlTypeInfo.BlockSize(t.Type);
            int typeSize = GgmlTypeInfo.TypeSize(t.Type);
            int bytesPerRow = (int)(hiddenDim / blockSize) * typeSize;
            var rowBytes = t.RawBytes().Slice(tokenId * bytesPerRow, bytesPerRow);
            using var rowT = new Cpu.CpuTensor("_emb_row", t.Type, new long[] { hiddenDim }, rowBytes);
            rowT.DequantizeTo(outSpan);
            return;
        }

        var p = new EmbedParams { HiddenDim = hiddenDim, TokenId = (uint)tokenId, TableType = tableType };
        var bufs = stackalloc IntPtr[2] { H(output), H(table) };
        string fn = ((MetalTensor)output).IsF16Backed ? "embedding_lookup_f16" : "embedding_lookup";
        Dispatch(fn, new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(EmbedParams), 2,
            hiddenDim, 256, useThreadgroups: false);
    }

    // ── Batched prefill ops ──────────────────────────────────────────────

    /// <summary>
    /// Write M token embeddings into output[M × hiddenDim]. Uploads the token
    /// ID list once and dispatches a single kernel that reads per-token.
    /// </summary>
    public unsafe void BatchedEmbeddingLookup(ITensor output, ITensor table, int[] tokenIds)
    {
        var t = (MetalTensor)table;
        int M = tokenIds.Length;
        uint hiddenDim = (uint)t.Dimensions[0];

        uint tableType = t.Type switch
        {
            GgmlType.F32 => 0u,
            GgmlType.Q8_0 => 1u,
            GgmlType.F16 => 2u,
            GgmlType.Q4_0 => t.IsAlignedQ4_0 ? 8u : 5u,
            _ => uint.MaxValue,
        };

        if (tableType == uint.MaxValue || ((MetalTensor)output).IsF16Backed)
        {
            // Fallback: per-token (handles exotic quant types and F16 output).
            int hDim = (int)hiddenDim;
            using var row = new MetalTensor(_dev, "_emb_scratch", GgmlType.F32, new long[] { hDim },
                f16Backed: ((MetalTensor)output).IsF16Backed);
            for (int i = 0; i < M; i++)
            {
                EmbeddingLookup(row, table, tokenIds[i]);
                CopyTensorSlice(output, i * hDim, row, 0, hDim);
            }
            return;
        }

        // Upload token IDs into a Metal buffer.
        using var idsBuf = new MetalBuffer(_dev, M * sizeof(int));
        MemoryMarshal.Cast<byte, int>(idsBuf.AsByteSpan()).Slice(0, M).Clear();
        var idsSpan = MemoryMarshal.Cast<byte, int>(idsBuf.AsByteSpan());
        for (int i = 0; i < M; i++) idsSpan[i] = tokenIds[i];

        var p = new BatchedEmbedParams { HiddenDim = hiddenDim, TableType = tableType, M = (uint)M };
        var bufs = stackalloc IntPtr[3] { H(output), H(table), idsBuf.Handle };
        Dispatch("batched_embedding_lookup", new ReadOnlySpan<IntPtr>(bufs, 3), &p, sizeof(BatchedEmbedParams), 3,
            gridX: (uint)M, tgX: 256);
        // idsBuf is disposed at end of scope — Metal captures its contents on
        // dispatch but the buffer lifetime must outlive the in-flight batch.
        // Force flush to ensure it executes before disposal.
        Flush();
    }

    public unsafe void BatchedRoPE(ITensor q, ITensor k, int headDim, int ropeDim,
        int startPosition, float ropeTheta, int numHeads, int numKvHeads)
    {
        var p = new BatchedRoPEParams
        {
            QTotal = (uint)((MetalTensor)q).ElementCount,
            KTotal = (uint)((MetalTensor)k).ElementCount,
            HeadDim = (uint)headDim,
            RopeDim = (uint)(ropeDim > 0 ? ropeDim : headDim),
            PositionOffset = startPosition,
            RopeTheta = ropeTheta,
            NumHeads = (uint)numHeads,
            NumKvHeads = (uint)numKvHeads,
        };
        var bufs = stackalloc IntPtr[2] { H(q), H(k) };
        uint maxPairs = (uint)(Math.Max(p.QTotal, p.KTotal) / 2);
        Dispatch("batched_rope", new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(BatchedRoPEParams), 2,
            maxPairs, 256, useThreadgroups: false);
    }

    public unsafe void BatchedKvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int startPosition, int M)
    {
        var p = new BatchedKvWriteParams
        {
            NKvHeads = (uint)nKvHeads,
            KeyLength = (uint)keyLength,
            ValueLength = (uint)valueLength,
            MaxSeqLen = (uint)maxSeqLen,
            StartPosition = (uint)startPosition,
            M = (uint)M,
        };
        var bufs = stackalloc IntPtr[4] { H(kCache), H(vCache), H(k), H(v) };
        uint total = (uint)(Math.Max(nKvHeads * keyLength, nKvHeads * valueLength) * M);
        Dispatch("batched_kv_cache_write", new ReadOnlySpan<IntPtr>(bufs, 4), &p, sizeof(BatchedKvWriteParams), 4,
            total, 256, useThreadgroups: false);
        if (_flushAfterKv) Flush();
    }

    public unsafe void BatchedGatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int startPosition, int M, float scale)
    {
        var p = new BatchedGatedAttnParams
        {
            NumHeads = (uint)numHeads,
            NumKvHeads = (uint)numKvHeads,
            KeyLength = (uint)keyLength,
            ValueLength = (uint)valueLength,
            MaxSeqLen = (uint)maxSeqLen,
            StartPosition = (uint)startPosition,
            M = (uint)M,
            Scale = scale,
        };
        var bufs = stackalloc IntPtr[5] { H(output), H(qAttn), H(qGate), H(kCache), H(vCache) };
        Dispatch("batched_gated_attention", new ReadOnlySpan<IntPtr>(bufs, 5), &p, sizeof(BatchedGatedAttnParams), 5,
            gridX: (uint)(M * numHeads), tgX: 256);
        if (_flushAfterAttn) Flush();
    }

    // ── ArgMax ───────────────────────────────────────────────────────────

    public unsafe int ArgMax(ITensor tensor, int count)
    {
        var p = new ArgMaxParams { Count = (uint)count };
        using var outBuf = new MetalTensor(_dev, "_argmax_out", GgmlType.F32, new long[] { 1 });
        var bufs = stackalloc IntPtr[2] { H(tensor), outBuf.Buffer.Handle };
        Dispatch("argmax", new ReadOnlySpan<IntPtr>(bufs, 2), &p, sizeof(ArgMaxParams), 2, gridX: 1, tgX: 256);
        Flush(); // read-back of GPU output
        var bytes = outBuf.Buffer.AsByteSpan().Slice(0, 4);
        return BitConverter.ToInt32(bytes);
    }

    // ── Fused ops (single-kernel, saves 2 dispatches per layer each) ─────
    // Before batching correctness was fixed, these kernels appeared to race —
    // the root cause was actually upstream (SplitUnequalQKV / RepeatTile CPU
    // fallbacks reading pending GPU writes). With that fixed, the fused path
    // is now safe and materially cuts per-layer dispatch count.

    public unsafe void RmsNormResidual(ITensor output, ITensor residual, ITensor input, ITensor weight, float eps)
    {
        uint rowDim = (uint)((MetalTensor)weight).ElementCount;
        uint M = (uint)((MetalTensor)input).ElementCount / rowDim;
        var p = new AddRmsNormResidualParams { N = rowDim, Eps = eps };
        var bufs = stackalloc IntPtr[4] { H(output), H(residual), H(input), H(weight) };
        Dispatch("rmsnorm_residual" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 4), &p, sizeof(AddRmsNormResidualParams), 4,
            gridX: M, tgX: 256);
    }

    public unsafe void AddRmsNormResidual(ITensor output, ITensor hidden, ITensor residual, ITensor b, ITensor weight, float eps)
    {
        uint rowDim = (uint)((MetalTensor)weight).ElementCount;
        uint M = (uint)((MetalTensor)hidden).ElementCount / rowDim;
        var p = new AddRmsNormResidualParams { N = rowDim, Eps = eps };
        var bufs = stackalloc IntPtr[5] { H(output), H(hidden), H(residual), H(b), H(weight) };
        Dispatch("add_rmsnorm_residual" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 5), &p, sizeof(AddRmsNormResidualParams), 5,
            gridX: M, tgX: 256);
    }

    public unsafe void AddRmsNorm(ITensor output, ITensor hidden, ITensor a, ITensor b, ITensor weight, float eps)
    {
        uint rowDim = (uint)((MetalTensor)weight).ElementCount;
        uint M = (uint)((MetalTensor)hidden).ElementCount / rowDim;
        var p = new AddRmsNormResidualParams { N = rowDim, Eps = eps };
        var bufs = stackalloc IntPtr[5] { H(output), H(hidden), H(a), H(b), H(weight) };
        Dispatch("add_rmsnorm" + Sx(output), new ReadOnlySpan<IntPtr>(bufs, 5), &p, sizeof(AddRmsNormResidualParams), 5,
            gridX: M, tgX: 256);
    }

    // ── Shader source loader ─────────────────────────────────────────────

    private static string LoadShaderSource()
    {
        var asm = typeof(MetalBackend).Assembly;
        foreach (var n in asm.GetManifestResourceNames())
        {
            if (n.EndsWith("kernels.metal", StringComparison.Ordinal))
            {
                using var s = asm.GetManifestResourceStream(n)!;
                using var r = new StreamReader(s, Encoding.UTF8);
                return r.ReadToEnd();
            }
        }
        throw new InvalidOperationException("kernels.metal embedded resource not found.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        if (_traceDispatch || _gpuProf) DumpTrace();
        if (_traceDispatch || Environment.GetEnvironmentVariable("DAISI_METAL_FLUSH_TRACE") == "1" ||
            Environment.GetEnvironmentVariable("DAISI_METAL_FALLBACK_TRACE") == "1")
        {
            Console.Error.WriteLine($"[metal] totalDispatches={_totalDispatches} totalFlushes={_totalFlushes} (ratio={(_totalFlushes > 0 ? _totalDispatches / (double)_totalFlushes : 0):F1} disp/flush)");
        }
        foreach (var pso in _pipelines.Values) ObjC.Release(pso);
        _pipelines.Clear();
        _dev.Dispose();
        _disposed = true;
    }
}
