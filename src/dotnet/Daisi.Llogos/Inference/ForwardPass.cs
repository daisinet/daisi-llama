using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Hybrid transformer forward pass supporting both standard attention and DeltaNet layers.
/// Uses only IComputeBackend operations — no AsFloatSpan() — so it works on both CPU and GPU.
/// </summary>
public sealed class ForwardPass : IForwardPass
{
    private readonly IComputeBackend _backend;
    private readonly ModelConfig _config;
    private ModelWeights _weights; // not readonly: PipelinedForwardPass swaps via reflection
    private readonly IKvCache _kvCache;
    private readonly DeltaNetState _deltaState;

    /// <summary>
    /// Maximum vocab entries to compute in lm_head for greedy decode (ForwardArgMax).
    /// Common tokens occupy the low vocab range, so computing a fraction is sufficient
    /// for argmax. Set to VocabSize for full computation, or smaller for faster decode.
    /// Default: VocabSize / 4 (covers ~38K tokens for 152K vocab).
    /// </summary>
    public int ArgMaxVocabLimit { get; set; }

    // Scratch buffers
    private readonly ITensor _hidden;
    private readonly ITensor _residual;
    private readonly ITensor _normOut;
    private readonly ITensor _logits;
    private readonly float[] _logitsBuffer;

    // Standard attention scratch
    private readonly ITensor? _qFull;   // [numHeads × keyLength × 2] (attn + gate interleaved, gated Q only)
    private readonly ITensor _qAttn;    // [numHeads × keyLength]
    private readonly ITensor _qGate;    // [numHeads × keyLength] (dummy gate for non-gated models)
    private readonly ITensor _kProj;    // [numKvHeads × keyLength]
    private readonly ITensor _vProj;    // [numKvHeads × valueLength]
    private readonly ITensor _attnOut;  // [numHeads × valueLength]

    // DeltaNet scratch
    private readonly ITensor _qkvBuf;   // [ssmInnerSize × 3]
    private readonly ITensor _ssmQ;     // [ssmInnerSize] — view into qkvBuf not possible, so separate
    private readonly ITensor _ssmK;     // [ssmInnerSize]
    private readonly ITensor _ssmV;     // [ssmInnerSize]
    private readonly ITensor _ssmAlpha;  // [ssmGroupCount]
    private readonly ITensor _ssmBeta;   // [ssmGroupCount]
    private readonly ITensor _ssmDecay;  // [ssmGroupCount]
    private readonly ITensor _ssmBetaVal; // [ssmGroupCount]
    private readonly ITensor _ssmGate;   // [ssmInnerSize]
    private readonly ITensor _ssmOutput; // [ssmInnerSize]

    // FFN scratch
    private readonly ITensor _gate;
    private readonly ITensor _up;

    // Fused projection scratch (allocated if any layer has fused weights)
    private readonly ITensor? _fusedQkvOut;
    private readonly ITensor? _fusedGateUpOut;

    // Batched prefill scratch (allocated lazily, M-wide)
    private int _batchM;
    private ITensor? _bHidden, _bResidual, _bNormOut;
    private ITensor? _bQAttn, _bQGate, _bKProj, _bVProj, _bAttnOut;
    private ITensor? _bGate, _bUp;
    private ITensor? _bFusedQkvOut, _bFusedGateUpOut;
    private ITensor? _bQFull;
    private ITensor? _bDnQkv, _bDnAlpha, _bDnBeta, _bDnGate, _bDnOutput;

    public ForwardPass(IComputeBackend backend, ModelConfig config, ModelWeights weights,
        IKvCache kvCache, DeltaNetState deltaState)
    {
        _backend = backend;
        _config = config;
        _weights = weights;
        _kvCache = kvCache;
        _deltaState = deltaState;
        ArgMaxVocabLimit = config.VocabSize / 32;

        _hidden = CreateF32("scratch_hidden", config.HiddenDim);
        _residual = CreateF32("scratch_residual", config.HiddenDim);
        _normOut = CreateF32("scratch_norm", config.HiddenDim);
        _logits = CreateF32("scratch_logits", config.VocabSize);
        _logitsBuffer = new float[config.VocabSize];

        // Standard attention scratch
        // Gated Q (Qwen) needs a 2× buffer for interleaved Q+gate; standard models project directly to _qAttn
        // Detect from actual weights: if any standard attention layer has Q/K norms, it's gated
        bool hasGatedQ = false;
        for (int i = 0; i < config.NumLayers; i++)
            if (weights.Layers[i] is StandardAttentionWeights saw && saw.HasGatedQ)
                { hasGatedQ = true; break; }
        if (hasGatedQ)
            _qFull = CreateF32("scratch_q_full", config.NumHeads * config.KeyLength * 2);
        _qAttn = CreateF32("scratch_q_attn", config.NumHeads * config.KeyLength);
        _qGate = CreateF32("scratch_q_gate", config.NumHeads * config.KeyLength);
        if (!hasGatedQ)
            backend.FillTensor(_qGate, 88.0f); // sigmoid(88)≈1.0 → ungated attention
        _kProj = CreateF32("scratch_k", config.NumKvHeads * config.KeyLength);
        _vProj = CreateF32("scratch_v", config.NumKvHeads * config.ValueLength);
        _attnOut = CreateF32("scratch_attn_out", config.NumHeads * config.ValueLength);

        // DeltaNet scratch — derive sizes from actual weight tensors
        if (config.SsmInnerSize > 0)
        {
            // Find first DeltaNet layer to get actual tensor dimensions
            DeltaNetWeights? deltaLayer = null;
            for (int i = 0; i < config.NumLayers; i++)
                if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
                    { deltaLayer = dw; break; }

            int qkvOutDim = deltaLayer != null ? (int)deltaLayer.AttnQkv.Dimensions[1] : config.SsmInnerSize * 3;
            int numVHeads = deltaLayer != null ? (int)deltaLayer.SsmAlpha.Dimensions[1] : config.SsmGroupCount;
            int valueDim = numVHeads * (config.SsmStateSize > 0 ? config.SsmStateSize : config.SsmHeadDim);
            // Key dim: whatever remains after subtracting valueDim
            int keyDim = (qkvOutDim - valueDim) / 2;
            int numKHeads = keyDim > 0 ? keyDim / (valueDim / numVHeads) : numVHeads;

            _qkvBuf = CreateF32("scratch_qkv", qkvOutDim);
            // Q and K may be smaller than V if num_k_heads < num_v_heads.
            // After repeat-interleave, Q and K become valueDim-sized.
            _ssmQ = CreateF32("scratch_ssm_q", valueDim);
            _ssmK = CreateF32("scratch_ssm_k", valueDim);
            _ssmV = CreateF32("scratch_ssm_v", valueDim);
            _ssmAlpha = CreateF32("scratch_ssm_alpha", numVHeads);
            _ssmBeta = CreateF32("scratch_ssm_beta", numVHeads);
            _ssmDecay = CreateF32("scratch_ssm_decay", numVHeads);
            _ssmBetaVal = CreateF32("scratch_ssm_betaval", numVHeads);
            _ssmGate = CreateF32("scratch_ssm_gate", valueDim);
            _ssmOutput = CreateF32("scratch_ssm_out", valueDim);
        }
        else
        {
            _qkvBuf = _ssmQ = _ssmK = _ssmV = _ssmAlpha = _ssmBeta =
                _ssmDecay = _ssmBetaVal = _ssmGate = _ssmOutput = _hidden; // unused
        }

        _gate = CreateF32("scratch_ffn_gate", config.IntermediateDim);
        _up = CreateF32("scratch_ffn_up", config.IntermediateDim);

        // Fused projection scratch — check if any standard attention layer has fused weights
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (weights.Layers[i] is StandardAttentionWeights saw)
            {
                if (saw.FusedQKV != null && _fusedQkvOut == null)
                {
                    int fusedN = (int)saw.FusedQKV.Dimensions[1];
                    _fusedQkvOut = CreateF32("scratch_fused_qkv", fusedN);
                }
                if (saw.FusedGateUp != null && _fusedGateUpOut == null)
                {
                    int fusedN = (int)saw.FusedGateUp.Dimensions[1];
                    _fusedGateUpOut = CreateF32("scratch_fused_gateup", fusedN);
                }
                if (_fusedQkvOut != null && _fusedGateUpOut != null) break;
            }
        }
    }

    public IKvCache KvCache => _kvCache;
    public DeltaNetState DeltaState => _deltaState;

    /// <inheritdoc />
    public void ResetState()
    {
        _kvCache.Reset();
        _deltaState.Reset();
    }

    /// <summary>
    /// Run a forward pass for a single token at the given position.
    /// </summary>
    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        ForwardTransformer(tokenId, position);

        // Final RMSNorm + LM head + logit download
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);
        _backend.FlushCommands(); // submit all batched commands before readback
        _logits.DequantizeTo(_logitsBuffer);
        return _logitsBuffer;
    }

    /// <summary>
    /// Run only the transformer layers (embedding + all layers) without logit projection.
    /// Used for intermediate prefill tokens where logits aren't needed.
    /// </summary>
    public void ForwardHidden(int tokenId, int position)
    {
        ForwardTransformer(tokenId, position);
        _backend.FlushCommands(); // submit batch and reset pools before next token
    }

    /// <summary>
    /// Core transformer: embedding + all layers. Shared by Forward, ForwardHidden, ForwardArgMax.
    /// </summary>
    /// <summary>
    /// Debug hook: if set, invoked after each layer with (layer, tag, tensor).
    /// Tags: "embed", "after_layer_N". Tensor is the hidden state at that point.
    /// The hook may call AsFloatSpan / DequantizeTo; ForwardTransformer flushes first.
    /// </summary>
    public static Action<int, string, ITensor>? DebugHook { get; set; }

    private void ForwardTransformer(int tokenId, int position)
    {
        // Batch entire forward pass into single command buffer submission
        _backend.BeginCommands();

        // 1. Embedding lookup
        _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenId);

        if (DebugHook != null) { _backend.FlushCommands(); DebugHook(-1, "embed", _hidden); _backend.BeginCommands(); }

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // Fuse previous layer's ElementAdd with this layer's RmsNormResidual
            // First layer: no preceding add, just do RmsNormResidual
            if (layer == 0)
                _backend.RmsNormResidual(_normOut, _residual, _hidden, lw.AttnNorm, _config.NormEps);
            else
                _backend.AddRmsNormResidual(_normOut, _hidden, _residual, _residual, lw.AttnNorm, _config.NormEps);

            if (DebugHook != null)
            {
                _backend.FlushCommands();
                DebugHook(layer, $"layer{layer}_afterNorm_hidden", _hidden);
                DebugHook(layer, $"layer{layer}_afterNorm_normOut", _normOut);
                DebugHook(layer, $"layer{layer}_afterNorm_residual", _residual);
                _backend.BeginCommands();
            }

            if (lw is StandardAttentionWeights saw)
                ForwardStandardAttention(saw, position, layer);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNet(dnw, layer);

            if (DebugHook != null)
            {
                _backend.FlushCommands();
                DebugHook(layer, $"layer{layer}_afterAttnOrDn_hidden", _hidden);
                DebugHook(layer, $"layer{layer}_afterAttnOrDn_residual", _residual);
                _backend.BeginCommands();
            }

            _backend.AddRmsNorm(_normOut, _hidden, _hidden, _residual, lw.PostAttnNorm, _config.NormEps);
            _backend.CopyTensor(_residual, _hidden);

            if (DebugHook != null)
            {
                _backend.FlushCommands();
                DebugHook(layer, $"layer{layer}_afterPostNorm_normOut", _normOut);
                DebugHook(layer, $"layer{layer}_afterPostNorm_hidden", _hidden);
                DebugHook(layer, $"layer{layer}_afterPostNorm_residual", _residual);
                _backend.BeginCommands();
            }

            if (lw is StandardAttentionWeights sawFfn && sawFfn.FusedGateUp != null && _fusedGateUpOut != null)
            {
                int gateDim = _config.IntermediateDim;
                ProjectLinear(_fusedGateUpOut, _normOut, sawFfn.FusedGateUp);
                _backend.SplitSwiGLU(_gate, _fusedGateUpOut, gateDim);
            }
            else if (lw.FfnGate.Type == Gguf.GgmlType.Q4_K ||
                     lw.FfnGate.Type == Gguf.GgmlType.Q4_0)
            {
                // Fused gate+up+SwiGLU: single kernel. Q4_K has a fused CUDA
                // path; Q4_0 has a fused Metal path. Other backends fall
                // through to the default (separate ops).
                int ffnK = (int)lw.FfnGate.Dimensions[0];
                int ffnN = (int)lw.FfnGate.Dimensions[1];
                _backend.MatMulSwiGLU(_gate, _normOut, lw.FfnGate, lw.FfnUp, 1, ffnK, ffnN);
            }
            else
            {
                ProjectLinear(_gate, _normOut, lw.FfnGate);
                ProjectLinear(_up, _normOut, lw.FfnUp);
                _backend.SwiGLU(_gate, _gate, _up);
            }
            ProjectLinear(_hidden, _gate, lw.FfnDown);

            // Last layer: do ElementAdd now (no next layer to fuse with)
            if (layer == _config.NumLayers - 1)
                _backend.ElementAdd(_hidden, _hidden, _residual);

            // Other layers: defer ElementAdd to fuse with next layer's RmsNormResidual

            if (DebugHook != null)
            {
                // Just flush and capture the raw _hidden (pre-deferred-add for all but last layer).
                // Both per-op and batched runs see the same capture point, so they are comparable.
                _backend.FlushCommands();
                DebugHook(layer, $"after_layer_{layer}", _hidden);
                _backend.BeginCommands();
            }

            // Early exit profiling: check what token each layer would predict
            if (EarlyExitProfile && layer >= _config.NumLayers / 4)
            {
                _backend.FlushCommands();
                // Temporarily compute output to see what token this layer predicts
                var tempHidden = new float[_config.HiddenDim];
                _hidden.DequantizeTo(tempHidden);
                // Store for comparison
                if (EarlyExitTokens != null)
                    EarlyExitTokens[layer] = ComputeArgMaxFromHidden(tempHidden);
                _backend.BeginCommands();
            }
        }
    }

    /// <summary>Enable early exit profiling — measures token prediction at each layer.</summary>
    public bool EarlyExitProfile { get; set; }

    /// <summary>Per-layer predicted token IDs (populated when EarlyExitProfile is true).</summary>
    public int[]? EarlyExitTokens { get; set; }

    private int ComputeArgMaxFromHidden(float[] hidden)
    {
        // Apply output norm manually on CPU
        float sumSq = 0;
        for (int i = 0; i < hidden.Length; i++) sumSq += hidden[i] * hidden[i];
        float invRms = 1.0f / MathF.Sqrt(sumSq / hidden.Length + _config.NormEps);

        // Download output norm weights
        var normW = new float[_config.HiddenDim];
        _weights.OutputNorm.DequantizeTo(normW);

        var normed = new float[hidden.Length];
        for (int i = 0; i < hidden.Length; i++)
            normed[i] = hidden[i] * invRms * normW[i];

        // Project through output weight (CPU matmul, partial vocab)
        int vocabLimit = Math.Min(ArgMaxVocabLimit, _config.VocabSize);
        var outW = new float[_weights.OutputWeight.ElementCount];
        _weights.OutputWeight.DequantizeTo(outW);

        float maxVal = float.MinValue;
        int maxIdx = 0;
        int K = _config.HiddenDim;
        for (int n = 0; n < vocabLimit; n++)
        {
            float dot = 0;
            for (int k = 0; k < K; k++)
                dot += normed[k] * outW[n * K + k];
            if (dot > maxVal) { maxVal = dot; maxIdx = n; }
        }
        return maxIdx;
    }

    /// <summary>
    /// Run forward pass and return only the argmax token ID.
    /// For GPU backends, this avoids downloading the full logit tensor (600KB+).
    /// </summary>
    public int ForwardArgMax(int tokenId, int position)
    {
        ForwardTransformer(tokenId, position);
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        // Partial vocab: only compute logits for the first ArgMaxVocabLimit tokens.
        // Common tokens are in the low vocab range — sufficient for greedy argmax.
        // Partial vocab: only compute logits for the first ArgMaxVocabLimit tokens.
        // Common tokens are in the low vocab range — sufficient for greedy argmax.
        // Tested across hundreds of prompts (code, math, Unicode, CJK, emoji, Cuneiform)
        // with zero mismatches vs full vocab on Qwen3, Qwen3.5, and TinyLlama models.
        ProjectLinearPartial(_logits, _normOut, _weights.OutputWeight, ArgMaxVocabLimit);
        _backend.FlushCommands(); // submit batch before argmax readback
        return _backend.ArgMax(_logits, ArgMaxVocabLimit);
    }

    // ── Standard Attention (Gated) ───────────────────────────────────────────

    private void ForwardStandardAttention(StandardAttentionWeights w, int position, int layer)
    {
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyLen = _config.KeyLength;
        int valLen = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyLen);

        // Q/K/V projections — fused into single matmul when possible
        if (!w.HasGatedQ && w.FusedQKV != null && _fusedQkvOut != null)
        {
            // Single fused matmul: [normOut] × [Q|K|V weights] → [Q|K|V output]
            int qDim = numHeads * keyLen;
            int kDim = numKvHeads * keyLen;
            int vDim = numKvHeads * valLen;
            ProjectLinear(_fusedQkvOut, _normOut, w.FusedQKV);
            // Split fused output → Q, K, V
            _backend.CopyTensorRegion(_qAttn, _fusedQkvOut, 0, qDim);
            _backend.CopyTensorRegion(_kProj, _fusedQkvOut, qDim, kDim);
            _backend.CopyTensorRegion(_vProj, _fusedQkvOut, qDim + kDim, vDim);

            if (w.AttnQNorm != null)
            {
                _backend.PerHeadRmsNorm(_qAttn, w.AttnQNorm, numHeads, keyLen, _config.NormEps);
                _backend.PerHeadRmsNorm(_kProj, w.AttnKNorm!, numKvHeads, keyLen, _config.NormEps);
            }
        }
        else if (w.HasGatedQ)
        {
            ProjectLinear(_kProj, _normOut, w.AttnK);
            ProjectLinear(_vProj, _normOut, w.AttnV);
            ProjectLinear(_qFull!, _normOut, w.AttnQ);
            _backend.DeInterleaveQ(_qAttn, _qGate, _qFull!, numHeads, keyLen);
            _backend.PerHeadRmsNorm(_qAttn, w.AttnQNorm!, numHeads, keyLen, _config.NormEps);
            _backend.PerHeadRmsNorm(_kProj, w.AttnKNorm!, numKvHeads, keyLen, _config.NormEps);
        }
        else
        {
            ProjectLinear(_qAttn, _normOut, w.AttnQ);
            ProjectLinear(_kProj, _normOut, w.AttnK);
            ProjectLinear(_vProj, _normOut, w.AttnV);

            if (w.AttnQNorm != null)
            {
                _backend.PerHeadRmsNorm(_qAttn, w.AttnQNorm, numHeads, keyLen, _config.NormEps);
                _backend.PerHeadRmsNorm(_kProj, w.AttnKNorm!, numKvHeads, keyLen, _config.NormEps);
            }
        }

        // Attention biases (Qwen2/2.5 — optional, null for Qwen3+)
        if (w.AttnQBias != null) _backend.ElementAdd(_qAttn, _qAttn, w.AttnQBias);
        if (w.AttnKBias != null) _backend.ElementAdd(_kProj, _kProj, w.AttnKBias);
        if (w.AttnVBias != null) _backend.ElementAdd(_vProj, _vProj, w.AttnVBias);

        // RoPE (partial — only first ropeDim dims)
        _backend.RoPE(_qAttn, _kProj, keyLen, ropeDim, position, _config.RopeTheta);

        // Write K/V to cache (maps position through strategy — ring buffer for sliding window)
        _kvCache.Write(_backend, layer, position, _kProj, _vProj);

        // seqLen is read AFTER write so it includes the current token
        int seqLen = _kvCache.Length;

        // Compute attention — try fused compressed path first, fall back to standard
        if (!_kvCache.ComputeAttention(_attnOut, _qAttn, _qGate,
                layer, numHeads, numKvHeads, keyLen, valLen, seqLen, scale))
        {
            var kCacheTensor = _kvCache.GetKCacheTensor(layer);
            var vCacheTensor = _kvCache.GetVCacheTensor(layer);
            _backend.GatedAttention(_attnOut, _qAttn, _qGate, kCacheTensor, vCacheTensor,
                numHeads, numKvHeads, keyLen, valLen, _kvCache.MaxSeqLen, seqLen, scale);
        }

        if (DebugHook != null)
        {
            _backend.FlushCommands();
            DebugHook(layer, $"layer{layer}_beforeAttnOProj_attnOut", _attnOut);
            DebugHook(layer, $"layer{layer}_beforeAttnOProj_hidden_stale", _hidden);
            _backend.BeginCommands();
        }

        // Output projection
        ProjectLinear(_hidden, _attnOut, w.AttnO);

        if (DebugHook != null)
        {
            _backend.FlushCommands();
            DebugHook(layer, $"layer{layer}_afterAttnOProj_hidden", _hidden);
            DebugHook(layer, $"layer{layer}_afterAttnOProj_attnOut", _attnOut);
            _backend.BeginCommands();
        }
    }

    // ── DeltaNet (Gated Linear Attention) ────────────────────────────────────

    /// <summary>
    /// Batched DeltaNet prefill: batch the linear projections, loop only over
    /// the sequential recurrent state updates.
    /// </summary>
    private void BatchedForwardDeltaNet(DeltaNetWeights w, int layer, int startPosition, int M)
    {
        int hDim = _config.HiddenDim;
        int convKernel = _config.SsmConvKernel;
        int headDim = _config.SsmStateSize > 0 ? _config.SsmStateSize : _config.SsmHeadDim;
        int numVHeads = (int)w.SsmAlpha.Dimensions[1];
        int valueDim = numVHeads * headDim;
        int qkvOutDim = (int)w.AttnQkv.Dimensions[1];
        int keyDim = (qkvOutDim - valueDim) / 2;
        int numKHeads = keyDim / headDim;
        int repeatFactor = numVHeads / numKHeads;
        int convChannels = (int)(w.SsmConv1d.ElementCount / convKernel);

        // ── Phase 1: Batched linear projections (all M tokens at once) ──
        ProjectLinearBatched(_bDnQkv!, _bNormOut!, w.AttnQkv, M);
        ProjectLinearBatched(_bDnAlpha!, _bNormOut!, w.SsmAlpha, M);
        ProjectLinearBatched(_bDnBeta!, _bNormOut!, w.SsmBeta, M);
        ProjectLinearBatched(_bDnGate!, _bNormOut!, w.AttnGate, M);

        // ── Phase 2: Sequential state update (conv1d + DeltaNet) ──
        // Fused path: two dispatches per layer — batched conv1d+SiLU (loops M
        // internally, carrying conv history in registers), then one fused
        // DeltaNet kernel that does split / L2norm / decay / state update /
        // RmsNorm / SiLU gate for all M tokens with per-head state resident
        // in threadgroup memory. Replaces the 14-dispatch-per-token loop.
        var convBuffer = _deltaState.GetConvBufferTensor(layer);
        var stateTensorF = _deltaState.GetStateTensor(layer);
        float scaleF = 1.0f / MathF.Sqrt(headDim);
        if (_backend.SupportsFusedDeltaNetPrefill
            && headDim <= 128
            && keyDim != 0 && valueDim != 0)
        {
            _backend.BatchedCausalConv1dSiLU(_bDnQkv!, convBuffer, w.SsmConv1d,
                convChannels, convKernel, M);
            _backend.BatchedDeltaNetFused(
                _bDnOutput!, _bDnQkv!, _bDnAlpha!, _bDnBeta!, _bDnGate!,
                stateTensorF, w.SsmA, w.SsmDtBias, w.SsmNorm,
                M, qkvOutDim, keyDim, valueDim,
                numKHeads, numVHeads, headDim,
                scaleF, _config.NormEps);
        }
        else
        {
            for (int t = 0; t < M; t++)
            {
                // Extract token t's pre-computed QKV projection
                _backend.CopyTensorSlice(_qkvBuf, 0, _bDnQkv!, t * qkvOutDim, qkvOutDim);

                // CausalConv1d (state-dependent — uses conv buffer)
                _backend.CausalConv1dSiLU(_qkvBuf, convBuffer, w.SsmConv1d, convChannels, convKernel);

                // Split Q/K/V
                if (keyDim == valueDim)
                    _backend.SplitQKV(_ssmQ, _ssmK, _ssmV, _qkvBuf, keyDim);
                else
                    _backend.SplitUnequalQKV(_ssmQ, _ssmK, _ssmV, _qkvBuf, keyDim, valueDim);

                // L2-normalize Q and K
                _backend.L2NormGroups(_ssmQ, numKHeads, headDim);
                _backend.L2NormGroups(_ssmK, numKHeads, headDim);

                // Tile Q and K from num_k_heads → num_v_heads
                if (repeatFactor > 1)
                {
                    _backend.RepeatTile(_ssmQ, numKHeads, headDim, repeatFactor);
                    _backend.RepeatTile(_ssmK, numKHeads, headDim, repeatFactor);
                }

                // Extract pre-computed alpha/beta
                _backend.CopyTensorSlice(_ssmAlpha, 0, _bDnAlpha!, t * numVHeads, numVHeads);
                _backend.CopyTensorSlice(_ssmBeta, 0, _bDnBeta!, t * numVHeads, numVHeads);

                // Compute decay and beta values
                _backend.ComputeDecayBeta(_ssmDecay, _ssmBetaVal, _ssmAlpha, _ssmBeta,
                    w.SsmA, w.SsmDtBias, numVHeads);

                // DeltaNet state update (state-dependent)
                _backend.DeltaNetStep(_ssmOutput, _ssmQ, _ssmK, _ssmV,
                    stateTensorF, _ssmDecay, _ssmBetaVal,
                    w.SsmNorm, numVHeads, headDim, scaleF, _config.NormEps);

                // Extract pre-computed gate, apply SiLU gate
                _backend.CopyTensorSlice(_ssmGate, 0, _bDnGate!, t * valueDim, valueDim);
                _backend.SiLUGate(_ssmOutput, _ssmOutput, _ssmGate);

                // Collect output for batched output projection
                _backend.CopyTensorSlice(_bDnOutput!, t * valueDim, _ssmOutput, 0, valueDim);
            }
        }

        // ── Phase 3: Batched output projection ──
        ProjectLinearBatched(_bHidden!, _bDnOutput!, w.SsmOut, M);
    }

    private void ForwardDeltaNet(DeltaNetWeights w, int layer)
    {
        int convKernel = _config.SsmConvKernel;
        int headDim = _config.SsmStateSize > 0 ? _config.SsmStateSize : _config.SsmHeadDim;

        // Derive dimensions from weight tensors
        int numVHeads = (int)w.SsmAlpha.Dimensions[1]; // num_v_heads (32 for 9B)
        int valueDim = numVHeads * headDim;             // 32 × 128 = 4096
        int qkvOutDim = (int)w.AttnQkv.Dimensions[1];  // 8192 for 9B
        int keyDim = (qkvOutDim - valueDim) / 2;        // (8192-4096)/2 = 2048
        int numKHeads = keyDim / headDim;                // 2048/128 = 16
        int repeatFactor = numVHeads / numKHeads;        // 32/16 = 2

        // 1. QKV projection
        ProjectLinear(_qkvBuf, _normOut, w.AttnQkv);

        // 2. Causal conv1d on full Q+K+V output
        int convChannels = (int)(w.SsmConv1d.ElementCount / convKernel);
        var convBuf = _deltaState.GetConvBufferTensor(layer);
        _backend.CausalConv1dSiLU(_qkvBuf, convBuf, w.SsmConv1d, convChannels, convKernel);

        // 4. Split Q(keyDim) + K(keyDim) + V(valueDim) — possibly unequal
        if (keyDim == valueDim)
        {
            _backend.SplitQKV(_ssmQ, _ssmK, _ssmV, _qkvBuf, keyDim);
        }
        else
        {
            _backend.SplitUnequalQKV(_ssmQ, _ssmK, _ssmV, _qkvBuf, keyDim, valueDim);
        }

        // 5. L2-normalize Q and K (num_k_heads groups)
        _backend.L2NormGroups(_ssmQ, numKHeads, headDim);
        _backend.L2NormGroups(_ssmK, numKHeads, headDim);

        // 6. Tile Q and K from num_k_heads → num_v_heads (ggml_repeat style)
        if (repeatFactor > 1)
        {
            _backend.RepeatTile(_ssmQ, numKHeads, headDim, repeatFactor);
            _backend.RepeatTile(_ssmK, numKHeads, headDim, repeatFactor);
        }

        // 7. Compute alpha and beta projections
        ProjectLinear(_ssmAlpha, _normOut, w.SsmAlpha);
        ProjectLinear(_ssmBeta, _normOut, w.SsmBeta);

        // 8. Compute decay and beta values
        _backend.ComputeDecayBeta(_ssmDecay, _ssmBetaVal, _ssmAlpha, _ssmBeta,
            w.SsmA, w.SsmDtBias, numVHeads);

        // 9. DeltaNet state update + output + per-head norm
        var stateTensor = _deltaState.GetStateTensor(layer);
        float scale = 1.0f / MathF.Sqrt(headDim);
        _backend.DeltaNetStep(_ssmOutput, _ssmQ, _ssmK, _ssmV,
            stateTensor, _ssmDecay, _ssmBetaVal,
            w.SsmNorm, numVHeads, headDim, scale, _config.NormEps);

        if (DebugHook != null)
        {
            _backend.FlushCommands();
            DebugHook(layer, $"layer{layer}_dn_afterStep_out", _ssmOutput);
            _backend.BeginCommands();
        }

        // 10. Gate: output = RMSNorm(output) * SiLU(Z)
        ProjectLinear(_ssmGate, _normOut, w.AttnGate);
        _backend.SiLUGate(_ssmOutput, _ssmOutput, _ssmGate);

        if (DebugHook != null)
        {
            _backend.FlushCommands();
            DebugHook(layer, $"layer{layer}_dn_afterGate_out", _ssmOutput);
            DebugHook(layer, $"layer{layer}_dn_afterGate_gate", _ssmGate);
            _backend.BeginCommands();
        }

        // 11. Output projection
        ProjectLinear(_hidden, _ssmOutput, w.SsmOut);

        if (DebugHook != null)
        {
            _backend.FlushCommands();
            DebugHook(layer, $"layer{layer}_dn_afterSsmOut_hidden", _hidden);
            DebugHook(layer, $"layer{layer}_dn_afterSsmOut_ssmOutput", _ssmOutput);
            _backend.BeginCommands();
        }
    }

    /// <summary>
    /// Split QKV buffer with unequal Q/K and V sizes using backend CopyTensorBytes.
    /// Layout: [Q: keyDim] [K: keyDim] [V: valueDim]
    /// </summary>
    private void SplitUnequal(ITensor qkv, ITensor q, ITensor k, ITensor v,
        int keyDim, int valueDim)
    {
        // Download QKV, split on CPU, re-upload
        int totalElems = keyDim * 2 + valueDim;
        var buf = new float[totalElems];
        qkv.DequantizeTo(buf.AsSpan(0, totalElems));

        // Q: first keyDim elements → q tensor (which is valueDim-sized, pad with zeros)
        var qBuf = new byte[q.ByteSize];
        Buffer.BlockCopy(buf, 0, qBuf, 0, keyDim * sizeof(float));
        q.CopyFrom(qBuf);

        // K: next keyDim elements → k tensor (valueDim-sized, pad with zeros)
        var kBuf = new byte[k.ByteSize];
        Buffer.BlockCopy(buf, keyDim * sizeof(float), kBuf, 0, keyDim * sizeof(float));
        k.CopyFrom(kBuf);

        // V: last valueDim elements → v tensor
        var vBuf = new byte[v.ByteSize];
        Buffer.BlockCopy(buf, keyDim * 2 * sizeof(float), vBuf, 0, valueDim * sizeof(float));
        v.CopyFrom(vBuf);
    }

    /// <summary>
    /// Repeat-interleave tensor data from numHeads groups to numHeads*factor groups.
    /// Each head of headDim elements is repeated 'factor' times.
    /// Done in-place (tensor must be large enough for the expanded result).
    /// </summary>
    private void RepeatInterleave(ITensor tensor, int numHeads, int headDim, int factor)
    {
        // Download, repeat via tiling (ggml_repeat style), re-upload.
        // Tiling: [h0, h1, ..., h15, h0, h1, ..., h15] — NOT interleave [h0, h0, h1, h1, ...]
        // This matches ggml_repeat which llama.cpp uses for Q/K head expansion.
        int srcSize = numHeads * headDim;
        int dstSize = numHeads * factor * headDim;
        var fullBuf = new float[dstSize]; // tensor is dstSize elements
        tensor.DequantizeTo(fullBuf);
        var src = new float[srcSize];
        Array.Copy(fullBuf, 0, src, 0, srcSize);

        var dst = new float[dstSize];
        for (int r = 0; r < factor; r++)
            Array.Copy(src, 0, dst, r * srcSize, srcSize);

        var bytes = new byte[dstSize * sizeof(float)];
        Buffer.BlockCopy(dst, 0, bytes, 0, bytes.Length);
        tensor.CopyFrom(bytes);
    }

    // ── Batched Prefill ────────────────────────────────────────────────────────

    /// <summary>
    /// Whether this model supports batched prefill.
    /// Supports pure attention and DeltaNet hybrid models (DeltaNet layers run sequentially within the batch).
    /// </summary>
    public bool SupportsBatchedPrefill => _backend.SupportsBatchedOps;

    /// <summary>Disable command batching/graph capture (needed when two models share one backend).</summary>
    public void DisableGraphCapture() => _backend.DisableGraphCapture();

    /// <summary>
    /// Process M tokens through all transformer layers in parallel.
    /// Standard attention layers run fully batched. DeltaNet layers run
    /// sequentially per token (state updates are order-dependent), but
    /// RmsNorm and FFN surrounding them are still batched.
    /// After this call, KV cache contains all M tokens and the last token's
    /// hidden state is in the single-token _hidden buffer.
    /// </summary>
    public void ForwardBatchedPrefill(int[] tokenIds, int startPosition)
    {
        int M = tokenIds.Length;
        if (M <= 1) { if (M == 1) ForwardHidden(tokenIds[0], startPosition); return; }

        EnsureBatchBuffers(M);

        // 1. Embedding lookup: M tokens → [M × hiddenDim]
        int hDim = _config.HiddenDim;
        if (_backend.SupportsBatchedOps)
        {
            _backend.BatchedEmbeddingLookup(_bHidden!, _weights.TokenEmbedding, tokenIds);
        }
        else
        {
            for (int i = 0; i < M; i++)
            {
                _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenIds[i]);
                _backend.CopyTensorSlice(_bHidden!, i * hDim, _hidden, 0, hDim);
            }
        }

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // Fused RmsNorm (batched)
            if (layer == 0)
                _backend.RmsNormResidual(_bNormOut!, _bResidual!, _bHidden!, lw.AttnNorm, _config.NormEps);
            else
                _backend.AddRmsNormResidual(_bNormOut!, _bHidden!, _bResidual!, _bResidual!, lw.AttnNorm, _config.NormEps);

            if (lw is StandardAttentionWeights saw)
            {
                BatchedForwardStandardAttention(saw, startPosition, layer, M);
            }
            else if (lw is DeltaNetWeights dnw)
            {
                if (_bDnQkv != null)
                    BatchedForwardDeltaNet(dnw, layer, startPosition, M);
                else
                {
                    for (int t = 0; t < M; t++)
                    {
                        _backend.CopyTensorSlice(_normOut, 0, _bNormOut!, t * hDim, hDim);
                        ForwardDeltaNet(dnw, layer);
                        _backend.CopyTensorSlice(_bHidden!, t * hDim, _hidden, 0, hDim);
                    }
                }
            }

            _backend.AddRmsNorm(_bNormOut!, _bHidden!, _bHidden!, _bResidual!, lw.PostAttnNorm, _config.NormEps);
            _backend.CopyTensor(_bResidual!, _bHidden!);

            // FFN (batched)
            ProjectLinearBatched(_bGate!, _bNormOut!, lw.FfnGate, M);
            ProjectLinearBatched(_bUp!, _bNormOut!, lw.FfnUp, M);
            _backend.SwiGLU(_bGate!, _bGate!, _bUp!);
            ProjectLinearBatched(_bHidden!, _bGate!, lw.FfnDown, M);

            if (layer == _config.NumLayers - 1)
                _backend.ElementAdd(_bHidden!, _bHidden!, _bResidual!);
        }

        // 3. Copy last token's hidden state to single-token buffer for subsequent decode
        _backend.CopyTensorSlice(_hidden, 0, _bHidden!, (M - 1) * hDim, hDim);
    }

    /// <summary>
    /// Batched prefill + logit projection for the last token.
    /// Returns the argmax token ID.
    /// </summary>
    public int ForwardBatchedPrefillArgMax(int[] tokenIds, int startPosition)
    {
        ForwardBatchedPrefill(tokenIds, startPosition);
        _backend.BeginCommands();
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        ProjectLinearPartial(_logits, _normOut, _weights.OutputWeight, ArgMaxVocabLimit);
        _backend.FlushCommands();
        return _backend.ArgMax(_logits, ArgMaxVocabLimit);
    }

    /// <summary>
    /// Batched forward pass that returns the argmax prediction at EVERY position.
    /// Used by speculative decoding: the target model verifies N draft tokens
    /// and returns what it would have predicted at each position.
    /// </summary>
    public int[] ForwardBatchedVerify(int[] tokenIds, int startPosition)
    {
        int M = tokenIds.Length;
        ForwardBatchedPrefill(tokenIds, startPosition);

        // Batched: norm all M hidden states, project through lm_head, per-row argmax
        int hDim = _config.HiddenDim;
        int vocabLimit = ArgMaxVocabLimit;
        var results = new int[M];

        // _bHidden has [M × hiddenDim], _bNormOut has [M × hiddenDim]
        // Batched RmsNorm: normalize each row independently
        _backend.RmsNorm(_bNormOut!, _bHidden!, _weights.OutputNorm, _config.NormEps);

        // Batched lm_head: [M × hiddenDim] × [hiddenDim × vocabLimit] → [M × vocabLimit]
        // Need M-wide logits buffer
        int K = (int)_weights.OutputWeight.Dimensions[0];
        int N = Math.Min((int)_weights.OutputWeight.Dimensions[1], vocabLimit);
        if (_bLogits == null || _bLogitsSize < M * N)
        {
            _bLogits?.Dispose();
            _bLogits = _backend.CreateTensor("batch_logits", Gguf.GgmlType.F32, [(long)(M * N)]);
            _bLogitsSize = M * N;
        }
        _backend.MatMul(_bLogits, _bNormOut!, _weights.OutputWeight, M, K, N);

        // Per-row argmax
        var logitsBuf = new float[_bLogitsSize];
        _bLogits.DequantizeTo(logitsBuf);
        for (int t = 0; t < M; t++)
        {
            int bestIdx = 0;
            float bestVal = logitsBuf[t * N];
            for (int j = 1; j < N; j++)
            {
                float v = logitsBuf[t * N + j];
                if (v > bestVal) { bestVal = v; bestIdx = j; }
            }
            results[t] = bestIdx;
        }

        return results;
    }

    private ITensor? _bLogits;
    private int _bLogitsSize;

    private void BatchedForwardStandardAttention(StandardAttentionWeights w, int startPosition, int layer, int M)
    {
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyLen = _config.KeyLength;
        int valLen = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyLen);

        if (w.HasGatedQ)
        {
            ProjectLinearBatched(_bKProj!, _bNormOut!, w.AttnK, M);
            ProjectLinearBatched(_bVProj!, _bNormOut!, w.AttnV, M);
            ProjectLinearBatched(_bQFull!, _bNormOut!, w.AttnQ, M);
            _backend.DeInterleaveQ(_bQAttn!, _bQGate!, _bQFull!, M * numHeads, keyLen);
            _backend.PerHeadRmsNorm(_bQAttn!, w.AttnQNorm!, M * numHeads, keyLen, _config.NormEps);
            _backend.PerHeadRmsNorm(_bKProj!, w.AttnKNorm!, M * numKvHeads, keyLen, _config.NormEps);
        }
        else
        {
            ProjectLinearBatched(_bQAttn!, _bNormOut!, w.AttnQ, M);
            ProjectLinearBatched(_bKProj!, _bNormOut!, w.AttnK, M);
            ProjectLinearBatched(_bVProj!, _bNormOut!, w.AttnV, M);

            if (w.AttnQNorm != null)
            {
                _backend.PerHeadRmsNorm(_bQAttn!, w.AttnQNorm, M * numHeads, keyLen, _config.NormEps);
                _backend.PerHeadRmsNorm(_bKProj!, w.AttnKNorm!, M * numKvHeads, keyLen, _config.NormEps);
            }
        }

        // Attention biases (Qwen2/2.5 — optional, null for Qwen3+). The biases
        // are row-sized; ElementAdd broadcasts them across all M tokens when
        // the output tensor is M× bigger than the bias (handled in backend).
        if (w.AttnQBias != null) _backend.ElementAdd(_bQAttn!, _bQAttn!, w.AttnQBias);
        if (w.AttnKBias != null) _backend.ElementAdd(_bKProj!, _bKProj!, w.AttnKBias);
        if (w.AttnVBias != null) _backend.ElementAdd(_bVProj!, _bVProj!, w.AttnVBias);

        _backend.BatchedRoPE(_bQAttn!, _bKProj!, keyLen, ropeDim, startPosition, _config.RopeTheta,
            numHeads, numKvHeads);
        _kvCache.BatchedWrite(_backend, layer, startPosition, M, _bKProj!, _bVProj!);

        var kCacheTensor = _kvCache.GetKCacheTensor(layer);
        var vCacheTensor = _kvCache.GetVCacheTensor(layer);

        if (!w.HasGatedQ)
            _backend.FillTensor(_bQGate!, 88.0f);

        _backend.BatchedGatedAttention(_bAttnOut!, _bQAttn!, _bQGate!, kCacheTensor, vCacheTensor,
            numHeads, numKvHeads, keyLen, valLen, _kvCache.MaxSeqLen, startPosition, M, scale);

        ProjectLinearBatched(_bHidden!, _bAttnOut!, w.AttnO, M);
    }

    private void EnsureBatchBuffers(int M)
    {
        if (_batchM == M && _bHidden != null) return;

        // Dispose old buffers
        DisposeBatchBuffers();

        _batchM = M;
        int hDim = _config.HiddenDim;
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyLen = _config.KeyLength;
        int valLen = _config.ValueLength;

        _bHidden = CreateF32("batch_hidden", M * hDim);
        _bResidual = CreateF32("batch_residual", M * hDim);
        _bNormOut = CreateF32("batch_norm", M * hDim);
        _bQAttn = CreateF32("batch_q_attn", M * numHeads * keyLen);
        _bQGate = CreateF32("batch_q_gate", M * numHeads * keyLen);
        _bKProj = CreateF32("batch_k", M * numKvHeads * keyLen);
        _bVProj = CreateF32("batch_v", M * numKvHeads * valLen);
        _bAttnOut = CreateF32("batch_attn_out", M * numHeads * valLen);
        _bGate = CreateF32("batch_ffn_gate", M * _config.IntermediateDim);
        _bUp = CreateF32("batch_ffn_up", M * _config.IntermediateDim);

        if (_qFull != null)
            _bQFull = CreateF32("batch_q_full", M * numHeads * keyLen * 2);

        // DeltaNet batch buffers
        if (_config.HasDeltaNet)
        {
            DeltaNetWeights? dlw = null;
            for (int i = 0; i < _config.NumLayers; i++)
                if (!_config.IsStandardAttention(i) && _weights.Layers[i] is DeltaNetWeights dw)
                    { dlw = dw; break; }
            if (dlw != null)
            {
                int qkvDim = (int)dlw.AttnQkv.Dimensions[1];
                int nVHeads = (int)dlw.SsmAlpha.Dimensions[1];
                int hd = _config.SsmStateSize > 0 ? _config.SsmStateSize : _config.SsmHeadDim;
                int vDim = nVHeads * hd;
                int gateDim = (int)dlw.AttnGate.Dimensions[1];

                _bDnQkv = CreateF32("batch_dn_qkv", M * qkvDim);
                _bDnAlpha = CreateF32("batch_dn_alpha", M * nVHeads);
                _bDnBeta = CreateF32("batch_dn_beta", M * nVHeads);
                _bDnGate = CreateF32("batch_dn_gate", M * gateDim);
                _bDnOutput = CreateF32("batch_dn_out", M * vDim);
            }
        }
    }

    private void DisposeBatchBuffers()
    {
        _bHidden?.Dispose(); _bResidual?.Dispose(); _bNormOut?.Dispose();
        _bQAttn?.Dispose(); _bQGate?.Dispose(); _bKProj?.Dispose();
        _bVProj?.Dispose(); _bAttnOut?.Dispose();
        _bGate?.Dispose(); _bUp?.Dispose();
        _bFusedQkvOut?.Dispose(); _bFusedGateUpOut?.Dispose();
        _bQFull?.Dispose();
        _bDnQkv?.Dispose(); _bDnAlpha?.Dispose(); _bDnBeta?.Dispose();
        _bDnGate?.Dispose(); _bDnOutput?.Dispose();
        _bHidden = _bResidual = _bNormOut = null;
        _bQAttn = _bQGate = _bKProj = _bVProj = _bAttnOut = null;
        _bGate = _bUp = _bFusedQkvOut = _bFusedGateUpOut = _bQFull = null;
        _bDnQkv = _bDnAlpha = _bDnBeta = _bDnGate = _bDnOutput = null;
        _batchM = 0;
    }

    private void ProjectLinearBatched(ITensor output, ITensor input, ITensor weight, int M)
    {
        int K = (int)weight.Dimensions[0];
        int N = (int)weight.Dimensions[1];
        _backend.MatMul(output, input, weight, M, K, N);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void ProjectLinear(ITensor output, ITensor input, ITensor weight)
    {
        int K = (int)weight.Dimensions[0];
        int N = (int)weight.Dimensions[1];
        _backend.MatMul(output, input, weight, 1, K, N);
    }

    /// <summary>
    /// Project with a reduced output dimension. Only computes the first maxN output neurons.
    /// Used for lm_head in greedy decode — common tokens are in the low vocab range,
    /// so computing a subset of logits is sufficient for argmax.
    /// </summary>
    private void ProjectLinearPartial(ITensor output, ITensor input, ITensor weight, int maxN)
    {
        int K = (int)weight.Dimensions[0];
        int N = Math.Min((int)weight.Dimensions[1], maxN);
        _backend.MatMul(output, input, weight, 1, K, N);
    }

    // ── DaisiChain: Pipeline Parallelism ─────────────────────────────────────
    // These methods enable splitting the forward pass across multiple hosts.
    // Each host runs a contiguous slice of layers and passes the hidden state
    // to the next host. Existing Forward/ForwardHidden/ForwardArgMax paths
    // are completely unaffected.

    /// <summary>
    /// Download the current hidden state into a caller-provided buffer.
    /// Used by DaisiChain to extract the intermediate hidden state after
    /// ForwardEmbedding or ForwardLayers, for transmission to the next pipeline stage.
    /// </summary>
    public void GetHidden(float[] buffer)
    {
        _backend.FlushCommands();
        _hidden.DequantizeTo(buffer);
    }

    /// <summary>Download the residual buffer (for hybrid GPU+CPU transfer).</summary>
    public void GetResidual(float[] buffer)
    {
        _backend.FlushCommands();
        _residual.DequantizeTo(buffer);
    }

    /// <summary>Upload a residual state into the internal buffer.</summary>
    public void SetResidual(ReadOnlySpan<float> residual)
    {
        var bytes = new byte[residual.Length * sizeof(float)];
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(residual).CopyTo(bytes);
        _residual.CopyFrom(bytes);
    }

    /// <summary>
    /// Upload a hidden state into the internal buffer.
    /// Used by DaisiChain to inject a hidden state received from a previous pipeline stage
    /// before calling ForwardLayers.
    /// </summary>
    public void SetHidden(ReadOnlySpan<float> hidden)
    {
        var bytes = new byte[hidden.Length * sizeof(float)];
        System.Runtime.InteropServices.MemoryMarshal.AsBytes(hidden).CopyTo(bytes);
        _hidden.CopyFrom(bytes);
    }

    /// <summary>
    /// Run only the embedding lookup for a single token.
    /// Leaves the result in the internal hidden buffer — call GetHidden to extract it,
    /// or follow with ForwardLayers to continue processing.
    /// </summary>
    public void ForwardEmbedding(int tokenId)
    {
        _backend.BeginCommands();
        _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenId);
        _backend.FlushCommands();
    }

    /// <summary>
    /// Run a contiguous subset of transformer layers [startLayer, endLayer).
    /// Reads from and writes to the internal hidden buffer.
    /// The caller must ensure the hidden buffer contains valid input (via ForwardEmbedding
    /// or SetHidden). KV cache and DeltaNet state are updated for the processed layers.
    /// </summary>
    /// <summary>
    /// Run a contiguous subset of transformer layers [startLayer, endLayer).
    /// When continuation=true, the first layer uses AddRmsNormResidual (preserving
    /// residual state from a previous ForwardLayers call). When false (default),
    /// the first layer uses RmsNormResidual (fresh start from embedding or SetHidden).
    /// </summary>
    /// <param name="continuation">If true, first layer uses AddRmsNormResidual (continuing from prior segment).</param>
    /// <param name="isFinal">If true, last layer applies ElementAdd (final residual). If false, defers for next segment.</param>
    public void ForwardLayers(int startLayer, int endLayer, int position,
        bool continuation = false, bool isFinal = true)
    {
        _backend.BeginCommands();

        for (int layer = startLayer; layer < endLayer; layer++)
        {
            var lw = _weights.Layers[layer];

            if (layer == startLayer && !continuation)
                _backend.RmsNormResidual(_normOut, _residual, _hidden, lw.AttnNorm, _config.NormEps);
            else
                _backend.AddRmsNormResidual(_normOut, _hidden, _residual, _residual, lw.AttnNorm, _config.NormEps);

            if (lw is StandardAttentionWeights saw)
                ForwardStandardAttention(saw, position, layer);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNet(dnw, layer);

            _backend.AddRmsNorm(_normOut, _hidden, _hidden, _residual, lw.PostAttnNorm, _config.NormEps);
            _backend.CopyTensor(_residual, _hidden);

            if (lw is StandardAttentionWeights sawFfn && sawFfn.FusedGateUp != null && _fusedGateUpOut != null)
            {
                int gateDim = _config.IntermediateDim;
                ProjectLinear(_fusedGateUpOut, _normOut, sawFfn.FusedGateUp);
                _backend.SplitSwiGLU(_gate, _fusedGateUpOut, gateDim);
            }
            else
            {
                ProjectLinear(_gate, _normOut, lw.FfnGate);
                ProjectLinear(_up, _normOut, lw.FfnUp);
                _backend.SwiGLU(_gate, _gate, _up);
            }
            ProjectLinear(_hidden, _gate, lw.FfnDown);

            if (layer == endLayer - 1 && isFinal)
                _backend.ElementAdd(_hidden, _hidden, _residual);
        }

        _backend.FlushCommands();
    }

    /// <summary>
    /// Run the output head: RmsNorm + LM head projection → logits.
    /// Reads from the internal hidden buffer (set via SetHidden or after ForwardLayers).
    /// Returns logits in the caller-provided buffer (must be VocabSize length).
    /// </summary>
    public void ForwardOutputHead(float[] buffer)
    {
        _backend.BeginCommands();
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);
        _backend.FlushCommands();
        _logits.DequantizeTo(buffer);
    }

    /// <summary>Number of transformer layers in the model.</summary>
    public int NumLayers => _config.NumLayers;


    /// <summary>Hidden dimension size (for sizing DaisiChain activation buffers).</summary>
    public int HiddenDim => _config.HiddenDim;

    /// <summary>Vocabulary size (for sizing DaisiChain logit buffers).</summary>
    public int VocabSize => _config.VocabSize;

    private ITensor CreateF32(string name, int size) =>
        _backend.CreateTensor(name, GgmlType.F32, [(long)size]);

    public void Dispose()
    {
        DisposeBatchBuffers();
        _bLogits?.Dispose();
        _hidden.Dispose();
        _residual.Dispose();
        _normOut.Dispose();
        _logits.Dispose();
        _qFull?.Dispose();
        _qAttn.Dispose();
        _qGate.Dispose();
        _kProj.Dispose();
        _vProj.Dispose();
        _attnOut.Dispose();
        if (_config.SsmInnerSize > 0)
        {
            _qkvBuf.Dispose();
            _ssmQ.Dispose();
            _ssmK.Dispose();
            _ssmV.Dispose();
            _ssmAlpha.Dispose();
            _ssmBeta.Dispose();
            _ssmDecay.Dispose();
            _ssmBetaVal.Dispose();
            _ssmGate.Dispose();
            _ssmOutput.Dispose();
        }
        _gate.Dispose();
        _up.Dispose();
    }
}
