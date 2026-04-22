using Daisi.Llogos.Gguf;

namespace Daisi.Llogos;

/// <summary>
/// Abstraction for a compute provider that creates tensors and executes tensor operations.
/// Each backend targets specific hardware (CPU/SIMD, CUDA, Vulkan, Metal).
/// The inference engine works exclusively through this interface.
/// </summary>
public interface IComputeBackend : IDisposable
{
    /// <summary>
    /// Human-readable backend name (e.g. "CPU", "CUDA").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Allocate an empty tensor with the given shape and type.
    /// </summary>
    ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions);

    /// <summary>
    /// Allocate a tensor and populate it with raw data from a GGUF file.
    /// The data span must match the expected byte size for the given type and dimensions.
    /// </summary>
    ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data);

    // ── Math Operations ──────────────────────────────────────────────────────

    /// <summary>
    /// Matrix multiplication: output = a × b.
    /// a is [M × K], b is [K × N], output is [M × N]. All stored as 1D row-major.
    /// b may be quantized — the backend fuses dequantization into the multiply.
    /// </summary>
    void MatMul(ITensor output, ITensor a, ITensor b, int M, int K, int N);

    /// <summary>
    /// Repack a weight tensor into a backend-specific layout optimised for
    /// batched (M&gt;1) matmul. Returns an opaque object (or null if the backend
    /// has no such optimisation). The caller passes the returned handle back to
    /// <see cref="MatMulRepacked"/>. Lifetime of the repacked data is owned by
    /// the caller; the backend must not reference the original tensor afterwards.
    /// </summary>
    object? RepackWeightForBatchedMatMul(ITensor weight) => null;

    /// <summary>
    /// Batched matmul using a previously-repacked weight handle. Default
    /// implementation falls back to <see cref="MatMul"/> if the backend does
    /// not implement the fast path.
    /// </summary>
    void MatMulRepacked(ITensor output, ITensor a, object repackedWeight, ITensor originalWeight, int M, int K, int N)
    {
        MatMul(output, a, originalWeight, M, K, N);
    }

    /// <summary>
    /// RMS normalization: output[i] = (input[i] / rms) * weight[i]
    /// where rms = sqrt(mean(input²) + eps).
    /// All tensors are 1D with the same element count.
    /// </summary>
    void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps);

    /// <summary>
    /// Numerically stable softmax over a 1D float tensor.
    /// output[i] = exp(input[i] - max) / sum(exp(input - max)).
    /// </summary>
    void Softmax(ITensor output, ITensor input);

    /// <summary>
    /// SiLU activation: output[i] = input[i] * sigmoid(input[i]).
    /// </summary>
    void SiLU(ITensor output, ITensor input);

    /// <summary>
    /// Rotary position embedding applied in-place to q and k tensors.
    /// Rotates pairs of dimensions (2i, 2i+1) by position-dependent angles.
    /// Only the first <paramref name="ropeDim"/> dimensions of each head are rotated.
    /// </summary>
    /// <param name="q">Query tensor [nHeads × headDim], modified in-place.</param>
    /// <param name="k">Key tensor [nKvHeads × headDim], modified in-place.</param>
    /// <param name="headDim">Dimension of each attention head.</param>
    /// <param name="ropeDim">Number of dimensions to apply rotation to (0 = all).</param>
    /// <param name="positionOffset">Starting position index for the rotation.</param>
    /// <param name="ropeTheta">RoPE frequency base (e.g. 1000000.0).</param>
    void RoPE(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta);

    /// <summary>
    /// Element-wise multiply: output[i] = a[i] * b[i].
    /// </summary>
    void ElementMul(ITensor output, ITensor a, ITensor b);

    /// <summary>
    /// Element-wise add: output[i] = a[i] + b[i].
    /// </summary>
    void ElementAdd(ITensor output, ITensor a, ITensor b);

    /// <summary>
    /// Copy a single row from a (possibly quantized) embedding table into an FP32 output tensor.
    /// Dequantizes the row if the table is quantized.
    /// </summary>
    void EmbeddingLookup(ITensor output, ITensor table, int tokenId);

    // ── Composite Operations (used by forward pass) ─────────────────────────

    /// <summary>
    /// Copy tensor data from src to dst. Both must be same type and size.
    /// </summary>
    void CopyTensor(ITensor dst, ITensor src);

    /// <summary>
    /// In-place SiLU activation: data[i] = data[i] * sigmoid(data[i]).
    /// </summary>
    void SiLUInPlace(ITensor data);

    /// <summary>
    /// L2-normalize groups of elements in-place.
    /// For each group g of size groupDim: normalize data[g*groupDim .. (g+1)*groupDim-1].
    /// </summary>
    void L2NormGroups(ITensor data, int numGroups, int groupDim);

    /// <summary>
    /// Per-head RMSNorm in-place: apply RMSNorm independently to each head-sized slice.
    /// weight has headDim elements, shared across all heads.
    /// </summary>
    void PerHeadRmsNorm(ITensor data, ITensor weight, int numHeads, int headDim, float eps);

    /// <summary>
    /// De-interleave Q projection output from [q_h0, gate_h0, q_h1, gate_h1, ...]
    /// into separate contiguous Q_attn and Q_gate buffers.
    /// </summary>
    void DeInterleaveQ(ITensor qAttn, ITensor qGate, ITensor qFull, int numHeads, int headDim);

    /// <summary>
    /// Write K and V vectors for a single position into the KV cache tensors.
    /// k: [nKvHeads × keyLength], kCache: [nKvHeads × maxSeqLen × keyLength].
    /// </summary>
    void KvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int position);

    /// <summary>
    /// Compute multi-head attention: scores, softmax, weighted V sum, sigmoid gating.
    /// output = sigmoid(qGate) * softmax(qAttn @ kCache^T / scale) @ vCache.
    /// </summary>
    void GatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int seqLen, float scale);

    /// <summary>
    /// Batched RoPE: apply rotary position embedding to M tokens.
    /// q: [M × numHeads × headDim], k: [M × numKvHeads × headDim].
    /// Positions are [startPosition .. startPosition + M - 1].
    /// </summary>
    void BatchedRoPE(ITensor q, ITensor k, int headDim, int ropeDim,
        int startPosition, float ropeTheta, int numHeads, int numKvHeads)
    {
        // Default: not supported, batched prefill requires CUDA backend
        throw new NotSupportedException("BatchedRoPE requires a backend with batched prefill support (CUDA).");
    }

    /// <summary>
    /// Write M K/V pairs at consecutive positions [startPos..startPos+M-1].
    /// k: [M × nKvHeads × keyLength], v: [M × nKvHeads × valueLength].
    /// </summary>
    void BatchedKvCacheWrite(ITensor kCache, ITensor vCache, ITensor k, ITensor v,
        int nKvHeads, int keyLength, int valueLength, int maxSeqLen, int startPosition, int M)
    {
        throw new NotSupportedException("BatchedKvCacheWrite requires a backend with batched prefill support (CUDA).");
    }

    /// <summary>
    /// Batched causal attention for M query tokens.
    /// qAttn: [M × numHeads × keyLength], qGate: [M × numHeads × keyLength].
    /// output: [M × numHeads × valueLength].
    /// Each query m attends to keys [0..startPosition+m] (causal mask).
    /// KV cache must already contain all entries up to startPosition+M-1.
    /// </summary>
    void BatchedGatedAttention(ITensor output, ITensor qAttn, ITensor qGate,
        ITensor kCache, ITensor vCache,
        int numHeads, int numKvHeads, int keyLength, int valueLength,
        int maxSeqLen, int startPosition, int M, float scale)
    {
        throw new NotSupportedException("BatchedGatedAttention requires a backend with batched prefill support (CUDA).");
    }

    /// <summary>
    /// Apply depthwise causal conv1d with shift buffer.
    /// Modifies qkv in-place and updates convBuffer.
    /// </summary>
    void CausalConv1d(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize);

    /// <summary>
    /// Fused CausalConv1d + in-place SiLU on the qkv buffer. ForwardPass
    /// always does these back-to-back in the DeltaNet path — fusing saves
    /// a dispatch and keeps the conv result in register between ops.
    /// </summary>
    void CausalConv1dSiLU(ITensor qkv, ITensor convBuffer, ITensor convWeight, int channels, int kernelSize)
    {
        CausalConv1d(qkv, convBuffer, convWeight, channels, kernelSize);
        SiLUInPlace(qkv);
    }

    /// <summary>
    /// Compute DeltaNet decay and beta from alpha projection, ssmA, dt_bias.
    /// decay[g] = exp(ssmA[g] * softplus(alpha[g] + dtBias[g]))
    /// beta[g] = sigmoid(betaProj[g])
    /// </summary>
    void ComputeDecayBeta(ITensor decay, ITensor beta, ITensor alphaProj, ITensor betaProj,
        ITensor ssmA, ITensor dtBias, int groupCount);

    /// <summary>
    /// Fused DeltaNet state update + output computation:
    /// For each group: sk = S^T*k, error = (v - d*sk)*beta, S = d*S + k*error, o = S^T*q * scale.
    /// Then per-head RMSNorm on output using normWeight.
    /// </summary>
    void DeltaNetStep(ITensor output, ITensor q, ITensor k, ITensor v,
        ITensor state, ITensor decay, ITensor beta,
        ITensor normWeight, int groupCount, int headDim, float scale, float normEps);

    /// <summary>
    /// Element-wise: output[i] = data[i] * silu(gate[i]).
    /// </summary>
    void SiLUGate(ITensor output, ITensor data, ITensor gate);

    /// <summary>
    /// Split a concatenated [Q|K|V] buffer into separate tensors.
    /// qkv has 3*innerSize elements; q, k, v each have innerSize elements.
    /// </summary>
    void SplitQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int innerSize);

    /// <summary>
    /// Zero all elements of a tensor.
    /// </summary>
    void ZeroTensor(ITensor tensor);

    /// <summary>
    /// Copy the first byteCount bytes from src to dst. dst may be larger than src.
    /// Used for growing cache reallocation.
    /// </summary>
    void CopyTensorBytes(ITensor dst, ITensor src, long byteCount);

    /// <summary>
    /// Create a tensor in host-accessible memory (pinned for GPU, regular for CPU).
    /// GPU kernels can access this tensor via mapped device pointers, but at reduced bandwidth.
    /// </summary>
    ITensor CreateHostTensor(string name, GgmlType type, ReadOnlySpan<long> dimensions);

    /// <summary>
    /// Fill all elements of an F32 tensor with a constant value.
    /// </summary>
    void FillTensor(ITensor tensor, float value);

    /// <summary>
    /// In-place Squared ReLU: data[i] = max(0, data[i])².
    /// Used by BitNet b1.58 instead of SiLU.
    /// </summary>
    void SquaredReLU(ITensor data);

    /// <summary>
    /// Begin batching GPU commands. Dispatches will be recorded without
    /// submitting until FlushCommands() is called. No-op for backends
    /// that don't support batching (CPU, CUDA with async streams).
    /// </summary>
    /// <summary>Disable graph capture / command batching. Used when multiple models share one backend.</summary>
    void DisableGraphCapture() { }

    void BeginCommands() { }

    /// <summary>
    /// Submit all batched commands and wait for completion.
    /// Called automatically before any GPU→CPU readback.
    /// </summary>
    void FlushCommands() { }

    // ── Fused Operations (optional, default to separate calls) ──────────────

    /// <summary>
    /// Copy a region of floats from source to destination tensor.
    /// Copies 'count' floats starting at 'srcOffset' in source to start of destination.
    /// </summary>
    void CopyTensorRegion(ITensor dst, ITensor src, int srcOffset, int count)
    {
        CopyTensorBytes(dst, src, (long)count * sizeof(float));
        // Default implementation doesn't handle offset — backends override for efficiency
    }

    /// <summary>
    /// Copy 'count' floats from src at srcOffset to dst at dstOffset.
    /// Used for placing slices into larger batched tensors.
    /// </summary>
    /// <summary>Whether this backend supports batched prefill operations (CopyTensorSlice, etc.).</summary>
    bool SupportsBatchedOps => false;

    /// <summary>
    /// Whether this backend has a fused batched DeltaNet prefill kernel that
    /// does conv1d+SiLU, split/L2norm/decay/step/gate in M-loop-per-TG form.
    /// When true, ForwardBatchedPrefill replaces the per-token DeltaNet loop
    /// with a single dispatch per layer.
    /// </summary>
    bool SupportsFusedDeltaNetPrefill => false;

    /// <summary>
    /// Batched causal conv1d + SiLU: loops M tokens internally. qkv layout
    /// is [M × channels]; convBuf is persistent per-channel history
    /// [(kernelSize-1) × channels]. Writes qkv in-place with SiLU applied.
    /// </summary>
    void BatchedCausalConv1dSiLU(ITensor qkv, ITensor convBuf, ITensor convWeight,
        int channels, int kernelSize, int M)
    {
        throw new NotSupportedException("BatchedCausalConv1dSiLU not implemented by this backend.");
    }

    /// <summary>
    /// Batched fused DeltaNet: for each of numVHeads heads, processes all M
    /// tokens sequentially in a single TG, keeping per-head state in
    /// threadgroup memory. Replaces the per-token loop of split/L2norm/
    /// repeat/decay/step/silugate.
    /// </summary>
    void BatchedDeltaNetFused(
        ITensor output, ITensor qkv, ITensor alpha, ITensor beta,
        ITensor gate, ITensor state, ITensor ssmA, ITensor dtBias, ITensor ssmNorm,
        int M, int qkvOutDim, int keyDim, int valueDim,
        int numKHeads, int numVHeads, int headDim,
        float scale, float normEps)
    {
        throw new NotSupportedException("BatchedDeltaNetFused not implemented by this backend.");
    }

    /// <summary>Wait for all pending operations to complete (for profiling). No-op on CPU.</summary>
    void Synchronize() { }

    /// <summary>
    /// Batched embedding lookup: write M token embeddings to output[M × hiddenDim].
    /// Default: sequential lookups.
    /// </summary>
    void BatchedEmbeddingLookup(ITensor output, ITensor table, int[] tokenIds)
    {
        throw new NotSupportedException("BatchedEmbeddingLookup requires backend support.");
    }


    void CopyTensorSlice(ITensor dst, int dstOffset, ITensor src, int srcOffset, int count)
    {
        throw new NotSupportedException("CopyTensorSlice requires backend support.");
    }

    /// <summary>
    /// Split QKV buffer with unequal sizes: [Q:keyDim, K:keyDim, V:valueDim].
    /// Q and K tensors are valueDim-sized (zero-padded after keyDim elements).
    /// </summary>
    void SplitUnequalQKV(ITensor q, ITensor k, ITensor v, ITensor qkv, int keyDim, int valueDim)
    {
        // Default: download, split on CPU, re-upload (slow for GPU)
        int totalElems = keyDim * 2 + valueDim;
        var buf = new float[totalElems];
        qkv.DequantizeTo(buf.AsSpan(0, totalElems));

        var qBuf = new byte[q.ByteSize];
        Buffer.BlockCopy(buf, 0, qBuf, 0, keyDim * sizeof(float));
        q.CopyFrom(qBuf);

        var kBuf = new byte[k.ByteSize];
        Buffer.BlockCopy(buf, keyDim * sizeof(float), kBuf, 0, keyDim * sizeof(float));
        k.CopyFrom(kBuf);

        var vBuf = new byte[v.ByteSize];
        Buffer.BlockCopy(buf, keyDim * 2 * sizeof(float), vBuf, 0, valueDim * sizeof(float));
        v.CopyFrom(vBuf);
    }

    /// <summary>
    /// Tile Q/K heads: [h0..hN] → [h0..hN, h0..hN, ...] repeated 'factor' times.
    /// Tensor must be large enough for the expanded result (numHeads*factor*headDim).
    /// Only the first numHeads*headDim elements are source data.
    /// </summary>
    void RepeatTile(ITensor tensor, int numHeads, int headDim, int factor)
    {
        // Default: download, tile on CPU, re-upload (slow for GPU)
        int srcSize = numHeads * headDim;
        int dstSize = numHeads * factor * headDim;
        var fullBuf = new float[dstSize];
        tensor.DequantizeTo(fullBuf);
        var src = new float[srcSize];
        Array.Copy(fullBuf, 0, src, 0, srcSize);
        for (int r = 0; r < factor; r++)
            Array.Copy(src, 0, fullBuf, r * srcSize, srcSize);
        var bytes = new byte[dstSize * sizeof(float)];
        Buffer.BlockCopy(fullBuf, 0, bytes, 0, bytes.Length);
        tensor.CopyFrom(bytes);
    }

    /// <summary>
    /// Fused: split gate+up from fused matmul output and compute SwiGLU in one pass.
    /// fusedInput is [gate(N) | up(N)], output[i] = SiLU(gate[i]) * up[i].
    /// </summary>
    void SplitSwiGLU(ITensor output, ITensor fusedInput, int N)
    {
        // Default: download, split+silu*mul on CPU, upload
        var buf = new float[N * 2];
        fusedInput.DequantizeTo(buf);
        var result = new float[N];
        for (int i = 0; i < N; i++)
        {
            float g = buf[i];
            float u = buf[N + i];
            result[i] = (g / (1.0f + MathF.Exp(-g))) * u;
        }
        var bytes = new byte[N * sizeof(float)];
        Buffer.BlockCopy(result, 0, bytes, 0, bytes.Length);
        output.CopyFrom(bytes);
    }

    /// <summary>
    /// Fused: split fused QKV, per-head norms, RoPE, KV cache write — all in one launch.
    /// </summary>
    void PostQkvNormRopeCache(ITensor qOut, ITensor kOut, ITensor vOut,
        ITensor fusedQkv, ITensor kCache, ITensor vCache,
        int qDim, int kDim, int vDim,
        int numHeads, int numKvHeads, int headDim, int ropeDim,
        int position, float ropeTheta, float normEps,
        int maxSeqLen, int seqLen, ITensor? qNormWeight, ITensor? kNormWeight)
    {
        // Default: separate operations
        CopyTensorRegion(qOut, fusedQkv, 0, qDim);
        CopyTensorRegion(kOut, fusedQkv, qDim, kDim);
        CopyTensorRegion(vOut, fusedQkv, qDim + kDim, vDim);
        if (qNormWeight != null)
        {
            PerHeadRmsNorm(qOut, qNormWeight, numHeads, headDim, normEps);
            PerHeadRmsNorm(kOut, kNormWeight!, numKvHeads, headDim, normEps);
        }
        RoPE(qOut, kOut, headDim, ropeDim, position, ropeTheta);
        KvCacheWrite(kCache, vCache, kOut, vOut, numKvHeads, headDim, headDim, maxSeqLen, position);
    }

    /// <summary>
    /// Find the index of the maximum value in a tensor. Returns -1 if tensor is empty.
    /// GPU backends can compute this on-device to avoid downloading the full tensor.
    /// </summary>
    int ArgMax(ITensor tensor) => ArgMax(tensor, (int)tensor.ElementCount);

    /// <summary>
    /// Find the index of the maximum value in the first <paramref name="count"/> elements.
    /// Used for partial vocab logit computation where only a subset of logits are valid.
    /// </summary>
    int ArgMax(ITensor tensor, int count)
    {
        // Default: download and scan on CPU
        var buf = new float[tensor.ElementCount];
        tensor.DequantizeTo(buf);
        int n = Math.Min(count, buf.Length);
        int best = 0;
        float bestVal = buf[0];
        for (int i = 1; i < n; i++)
            if (buf[i] > bestVal) { bestVal = buf[i]; best = i; }
        return best;
    }

    /// <summary>
    /// Fused: residual[i] = input[i], output[i] = RmsNorm(input, weight).
    /// Saves one kernel launch and one memory round-trip vs CopyTensor + RmsNorm.
    /// </summary>
    void RmsNormResidual(ITensor output, ITensor residual, ITensor input, ITensor weight, float eps)
    {
        CopyTensor(residual, input);
        RmsNorm(output, input, weight, eps);
    }

    /// <summary>
    /// Fused: hidden[i] += b[i]; residual[i] = hidden[i]; output[i] = RmsNorm(hidden, weight).
    /// Combines ElementAdd + CopyTensor + RmsNorm across layer boundaries (saves 2 launches).
    /// </summary>
    void AddRmsNormResidual(ITensor output, ITensor hidden, ITensor residual, ITensor b, ITensor weight, float eps)
    {
        ElementAdd(hidden, hidden, b);
        CopyTensor(residual, hidden);
        RmsNorm(output, hidden, weight, eps);
    }

    /// <summary>
    /// Fused SwiGLU: output[i] = SiLU(gate[i]) * up[i].
    /// Saves one kernel launch vs SiLU + ElementMul.
    /// </summary>
    void SwiGLU(ITensor output, ITensor gate, ITensor up)
    {
        SiLU(gate, gate);
        ElementMul(output, gate, up);
    }

    /// <summary>
    /// Fused gate+up matmul with SwiGLU: output[i] = SiLU(dot(a, gate[i])) * dot(a, up[i]).
    /// Eliminates intermediate tensor writes and extra kernel launches vs separate gate/up/SwiGLU.
    /// </summary>
    void MatMulSwiGLU(ITensor output, ITensor a, ITensor gateWeights, ITensor upWeights, int M, int K, int N)
    {
        // Default: separate operations (overridden by CudaBackend with fused kernel)
        MatMul(output, a, gateWeights, M, K, N);      // output = gate projection
        var temp = CreateTensor("_swiGLU_temp", output.Type, output.Dimensions);
        MatMul(temp, a, upWeights, M, K, N);           // temp = up projection
        SwiGLU(output, output, temp);                   // output = SiLU(gate) * up
        temp.Dispose();
    }

    /// <summary>
    /// Fused: hidden[i] = a[i] + b[i], output[i] = RmsNorm(hidden, weight).
    /// Saves kernel launches vs ElementAdd + CopyTensor + RmsNorm.
    /// </summary>
    void AddRmsNorm(ITensor output, ITensor hidden, ITensor a, ITensor b, ITensor weight, float eps)
    {
        ElementAdd(hidden, a, b);
        RmsNorm(output, hidden, weight, eps);
    }

    // ── Gemma 4 operations ──────────────────────────────────────────────────

    /// <summary>
    /// GeLU activation using the tanh approximation (matches PyTorch's
    /// `gelu_pytorch_tanh` and ggml's `ggml_gelu`):
    ///   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    /// </summary>
    void GeluTanh(ITensor output, ITensor input)
    {
        var o = output.AsFloatSpan();
        var i = input.AsFloatSpan();
        const float c = 0.7978845608028654f; // sqrt(2/π)
        for (int k = 0; k < i.Length; k++)
        {
            float x = i[k];
            float inner = c * (x + 0.044715f * x * x * x);
            o[k] = 0.5f * x * (1.0f + MathF.Tanh(inner));
        }
    }

    /// <summary>
    /// In-place GeLU (tanh approximation).
    /// </summary>
    void GeluTanhInPlace(ITensor data) => GeluTanh(data, data);

    /// <summary>
    /// Fused parallel GeGLU: output[i] = GeluTanh(gate[i]) * up[i].
    /// Equivalent to <see cref="SwiGLU"/> but with GeLU instead of SiLU.
    /// </summary>
    void GeGLU(ITensor output, ITensor gate, ITensor up)
    {
        GeluTanh(gate, gate);
        ElementMul(output, gate, up);
    }

    /// <summary>
    /// RMS normalization with NO learned weight: output[i] = input[i] / rms(input).
    /// Used by Gemma 4 to normalize the V projection at runtime.
    /// </summary>
    void RmsNormUnit(ITensor output, ITensor input, float eps)
    {
        var o = output.AsFloatSpan();
        var i = input.AsFloatSpan();
        double sumSq = 0;
        for (int k = 0; k < i.Length; k++) sumSq += (double)i[k] * i[k];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / i.Length + eps));
        for (int k = 0; k < i.Length; k++) o[k] = i[k] * invRms;
    }

    /// <summary>
    /// Per-head RMS normalization with NO learned weight, in-place.
    /// Each contiguous head_dim slice is independently normalized: data[h*hd + i] /= rms(head_h).
    /// Used for Gemma 4's V projection which has shape [head_dim, n_kv_heads] reshaped from [n_kv_heads × head_dim].
    /// </summary>
    void PerHeadRmsNormUnit(ITensor data, int numHeads, int headDim, float eps)
    {
        var span = data.AsFloatSpan();
        for (int h = 0; h < numHeads; h++)
        {
            int off = h * headDim;
            double sumSq = 0;
            for (int i = 0; i < headDim; i++) sumSq += (double)span[off + i] * span[off + i];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / headDim + eps));
            for (int i = 0; i < headDim; i++) span[off + i] *= invRms;
        }
    }

    /// <summary>
    /// Apply tanh-based logit softcap in-place: data[i] = cap * tanh(data[i] / cap).
    /// Used by Gemma 2/3/4 final logits.
    /// </summary>
    void LogitSoftcap(ITensor data, float cap)
    {
        var span = data.AsFloatSpan();
        float invCap = 1.0f / cap;
        for (int i = 0; i < span.Length; i++)
            span[i] = cap * MathF.Tanh(span[i] * invCap);
    }

    /// <summary>
    /// Multiply every element by a scalar in-place. Used to scale token embeddings
    /// by sqrt(hidden_dim) (Gemma) and to apply per-layer output scales.
    /// </summary>
    void ScaleInPlace(ITensor data, float scale)
    {
        var span = data.AsFloatSpan();
        for (int i = 0; i < span.Length; i++) span[i] *= scale;
    }

    /// <summary>
    /// RoPE with explicit per-pair frequency multipliers. Used by Gemma 4 full-attention
    /// layers for proportional/precomputed RoPE.
    /// freqFactors is a [headDim/2] (or [ropeDim/2]) F32 tensor of multipliers applied
    /// to each rotation pair's angle. Pass null to use plain theta-derived frequencies.
    /// </summary>
    void RoPEWithFreqFactors(ITensor q, ITensor k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ITensor? freqFactors)
    {
        // Default: ignore freqFactors and call regular RoPE.
        // CPU backend overrides this to honor the freq table.
        if (freqFactors == null)
        {
            RoPE(q, k, headDim, ropeDim, positionOffset, ropeTheta);
            return;
        }
        throw new NotSupportedException(
            $"{Name} backend does not implement RoPEWithFreqFactors. " +
            "Required for Gemma 4 full-attention layers.");
    }

    /// <summary>
    /// NEOX-style RoPE: pairs are (head[i], head[i+ropeDim/2]) instead of the
    /// "interleaved" (head[2i], head[2i+1]) used by the regular <see cref="RoPE"/>.
    /// Required for Gemma 4 (LLAMA_ROPE_TYPE_NEOX) and any other arch where the
    /// converter does NOT pre-permute Q/K weights into interleaved layout.
    /// </summary>
    void RoPENeox(ITensor q, ITensor k, int headDim, int ropeDim, int positionOffset, float ropeTheta)
    {
        throw new NotSupportedException(
            $"{Name} backend does not implement RoPENeox. Required for Gemma 4 sliding-window layers.");
    }

    /// <summary>
    /// NEOX-style RoPE with explicit per-pair frequency multipliers (proportional RoPE).
    /// Required for Gemma 4 full-attention layers.
    /// </summary>
    void RoPENeoxWithFreqFactors(ITensor q, ITensor k, int headDim, int ropeDim,
        int positionOffset, float ropeTheta, ITensor? freqFactors)
    {
        if (freqFactors == null)
        {
            RoPENeox(q, k, headDim, ropeDim, positionOffset, ropeTheta);
            return;
        }
        throw new NotSupportedException(
            $"{Name} backend does not implement RoPENeoxWithFreqFactors. " +
            "Required for Gemma 4 full-attention layers.");
    }
}
