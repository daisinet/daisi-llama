using Daisi.Llogos;
using Daisi.Llogos.Chat;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.TelosDistill;

/// <summary>
/// Loads a GGUF teacher once, exposes a Generate method that runs a single-turn chat
/// against a fresh KV cache. Designed for batch distillation: load once, generate N times.
/// </summary>
public sealed class ModelHost : IDisposable
{
    private readonly IComputeBackend _backend;
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly BpeTokenizer _tokenizer;
    private readonly ChatTemplate _template;
    private readonly int _maxContext;
    private readonly int? _seed;

    public string ModelPath { get; }

    private ModelHost(
        IComputeBackend backend, ModelConfig config, ModelWeights weights,
        BpeTokenizer tokenizer, ChatTemplate template, int maxContext, int? seed, string modelPath)
    {
        _backend = backend;
        _config = config;
        _weights = weights;
        _tokenizer = tokenizer;
        _template = template;
        _maxContext = maxContext;
        _seed = seed;
        ModelPath = modelPath;
    }

    public static ModelHost Load(string modelPath, string backendName, int maxContext, int? seed)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"GGUF model not found: {modelPath}");

        IComputeBackend backend = backendName.ToLowerInvariant() switch
        {
            "cuda" => new CudaBackend(),
            _ => new CpuBackend(),
        };

        Console.Error.WriteLine($"[host] loading {Path.GetFileName(modelPath)} on {backend.Name}...");
        var t0 = DateTime.UtcNow;

        using var stream = File.OpenRead(modelPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        var template = ChatTemplate.FromGguf(gguf);

        var ctx = maxContext > 0 ? Math.Min(maxContext, config.MaxContext) : Math.Min(4096, config.MaxContext);
        var weights = MmapModelLoader.Load(gguf, modelPath, backend, config);

        Console.Error.WriteLine(
            $"[host] loaded in {(DateTime.UtcNow - t0).TotalSeconds:F1}s " +
            $"({config.Architecture}, {config.NumLayers}L, {config.HiddenDim}d, ctx={ctx})");

        return new ModelHost(backend, config, weights, tokenizer, template, ctx, seed, modelPath);
    }

    /// <summary>
    /// Run a one-shot chat: system + user → assistant string. Fresh KV cache each call.
    /// <paramref name="seedOverride"/> lets callers vary the seed per batch without
    /// reloading the model; pass null to use the host-level seed.
    /// </summary>
    public async Task<string> GenerateAsync(
        string systemPrompt, string userPrompt, int maxTokens, float temperature,
        int? seedOverride = null, CancellationToken ct = default)
    {
        var kvCache = new KvCache(_backend, _config, _maxContext);
        var deltaState = new DeltaNetState(_backend, _config, _weights);
        var forward = new ForwardPass(_backend, _config, _weights, kvCache, deltaState);

        try
        {
            var renderer = new ChatTemplateRenderer(_template);
            var stops = renderer.GetStopSequences();
            var callSeed = seedOverride ?? _seed;
            var session = new DaisiLlogosChatSession(forward, _tokenizer, renderer, stops, callSeed);

            if (!string.IsNullOrEmpty(systemPrompt))
                session.AddMessage(new ChatMessage("system", systemPrompt));

            var parameters = new GenerationParams
            {
                MaxTokens = maxTokens,
                Temperature = temperature,
                TopK = 40,
                TopP = 0.9f,
                RepetitionPenalty = 1.1f,
                Seed = callSeed,
            };

            var sb = new System.Text.StringBuilder();
            await foreach (var chunk in session.ChatAsync(
                new ChatMessage("user", userPrompt), parameters, ct))
            {
                sb.Append(chunk);
            }
            return sb.ToString();
        }
        finally
        {
            forward.Dispose();
            deltaState.Dispose();
            kvCache.Dispose();
        }
    }

    /// <summary>
    /// Low-level: render (system, user) via the model's chat template, prefill the
    /// KV cache with all tokens except the last, run a single forward on the last
    /// token, and return the full-vocab logits as a newly-allocated float[].
    ///
    /// This is the soft-target distillation (M14 Mode 2) entry point — callers
    /// pick the logits for the classes they care about (e.g. first-token ids for
    /// PERMIT / DENY / AMBIGUOUS), softmax over just those, and emit the
    /// resulting probability triple as a teacher row.
    ///
    /// Fresh KV cache + delta-state per call, matching <see cref="GenerateAsync"/>
    /// so batched emission stays stateless across intents.
    /// </summary>
    public Task<float[]> FirstResponseLogitsAsync(
        string systemPrompt, string userPrompt, CancellationToken ct = default)
    {
        return Task.Run(() =>
        {
            var kvCache = new KvCache(_backend, _config, _maxContext);
            var deltaState = new DeltaNetState(_backend, _config, _weights);
            var forward = new ForwardPass(_backend, _config, _weights, kvCache, deltaState);
            try
            {
                var renderer = new ChatTemplateRenderer(_template);
                var messages = new List<ChatMessage>();
                if (!string.IsNullOrEmpty(systemPrompt))
                    messages.Add(new ChatMessage("system", systemPrompt));
                messages.Add(new ChatMessage("user", userPrompt));
                // `addGenerationPrompt: true` appends the assistant-turn marker so the
                // next token we predict is the first response token — exactly where
                // the classifier's answer lives for a one-word completion prompt.
                var rendered = renderer.Render(messages, addGenerationPrompt: true);
                var tokenIds = _tokenizer.Encode(rendered);
                if (tokenIds.Length == 0)
                    throw new InvalidOperationException("empty tokenization of chat prompt");

                int prefillEnd = tokenIds.Length - 1;
                if (prefillEnd > 0 && forward.SupportsBatchedPrefill)
                {
                    ct.ThrowIfCancellationRequested();
                    forward.ForwardBatchedPrefill(tokenIds[..prefillEnd], 0);
                }
                else
                {
                    for (int i = 0; i < prefillEnd; i++)
                    {
                        ct.ThrowIfCancellationRequested();
                        forward.ForwardHidden(tokenIds[i], i);
                    }
                }

                ct.ThrowIfCancellationRequested();
                var logits = forward.Forward(tokenIds[prefillEnd], prefillEnd);
                return logits.ToArray();
            }
            finally
            {
                forward.Dispose();
                deltaState.Dispose();
                kvCache.Dispose();
            }
        }, ct);
    }

    /// <summary>
    /// Resolve a candidate word to the single token id a teacher would emit as
    /// the first response token. Tries several tokenizations (leading-space,
    /// as-is, lowercase) and returns the first one that produces a single token.
    /// Returns -1 when no candidate is a single token — the caller is expected
    /// to fall back to an empirical-sampling path or to bail with a warning.
    /// </summary>
    public int ResolveSingleToken(params string[] candidates)
    {
        foreach (var cand in candidates)
        {
            if (string.IsNullOrEmpty(cand)) continue;
            var ids = _tokenizer.Encode(cand);
            if (ids.Length == 1) return ids[0];
        }
        return -1;
    }

    /// <summary>
    /// Tokenizer access for callers that need to introspect the vocab.
    /// </summary>
    public BpeTokenizer Tokenizer => _tokenizer;

    public void Dispose()
    {
        _weights.Dispose();
        _backend.Dispose();
    }
}
