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

    public void Dispose()
    {
        _weights.Dispose();
        _backend.Dispose();
    }
}
