using System.Diagnostics;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Metal;
using Daisi.Llogos.Vulkan;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos;
using Daisi.Llogos.Inference.DaisiTurbo;
using Daisi.Llogos.Tokenizer;
using Daisi.Llogos.Training;
using Daisi.Llogos.Training.Lora;

// ── Subcommands ─────────────────────────────────────────────────────────────
if (args.Length > 0 && args[0] == "train")
{
    return RunTraining(args[1..]);
}
if (args.Length > 0 && args[0] == "split")
{
    return RunSplit(args[1..]);
}
if (args.Length > 0 && args[0] == "metal-bench")
{
    Daisi.Llogos.Metal.MetalBench.Run();
    return 0;
}
if (args.Length > 0 && args[0] == "metal-diff")
{
    return RunMetalDiff(args[1..]);
}

// Parse arguments
var options = ParseArgs(args);

if (options.ShowHelp || options.ModelPath == null || (options.Prompt == null && !options.Bench))
{
    PrintUsage();
    return options.ShowHelp ? 0 : 1;
}

if (!File.Exists(options.ModelPath))
{
    Console.Error.WriteLine($"Error: model file not found: {options.ModelPath}");
    return 1;
}

// Load model
Console.Error.WriteLine($"Loading model: {options.ModelPath}");
var loadSw = Stopwatch.StartNew();

using var stream = File.OpenRead(options.ModelPath);
var gguf = GgufFile.Read(stream);
var config = ModelConfig.FromGguf(gguf);
IComputeBackend backend = options.Backend switch
{
    "cuda" => new CudaBackend(),
    "vulkan" => new VulkanBackend(),
    "metal" => new MetalBackend(),
    _ => new CpuBackend(),
};

var tokenizer = TokenizerFactory.FromGguf(gguf);
bool isBitNet = config.Architecture.StartsWith("bitnet", StringComparison.OrdinalIgnoreCase);
bool isGemma4 = config.IsGemma4;

if (isBitNet)
{
    // ── BitNet path ─────────────────────────────────────────────────────────
    var weights = BitNetModelLoader.Load(gguf, stream, backend, config);
    var kvCache = new BitNetKvCache(backend, config, maxSeqLen: options.MaxContext);
    var forward = new BitNetForwardPass(backend, config, weights, kvCache);

    loadSw.Stop();
    Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
        $"({config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d, BitNet)");

    var generator = new BitNetTextGenerator(forward, tokenizer, options.Seed);
    RunGeneration(generator.Generate, options);

    forward.Dispose();
    kvCache.Dispose();
    weights.Dispose();
}
else if (isGemma4)
{
    // ── Gemma 4 path ────────────────────────────────────────────────────────
    var weights = MmapModelLoader.Load(gguf, options.ModelPath!, backend, config);
    var kvCache = new Gemma4KvCache(backend, config, maxSeqLen: options.MaxContext);
    var forward = new Gemma4ForwardPass(backend, config, weights, kvCache);

    loadSw.Stop();
    int numFull = 0;
    int numSwa = 0;
    for (int i = 0; i < config.NumLayers; i++)
    {
        if (config.IsSlidingLayer(i)) numSwa++; else numFull++;
    }
    Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
        $"({config.Architecture}, {config.NumLayers} layers [{numSwa} SWA + {numFull} full], " +
        $"{config.HiddenDim}d, PLE={config.PerLayerInputDim})");
    Console.Error.WriteLine($"Backend: {backend.Name}, Max tokens: {options.MaxTokens}");

    var generator = new Gemma4TextGenerator(forward, tokenizer, options.Seed);
    RunGeneration(generator.Generate, options);

    forward.Dispose();
    kvCache.Dispose();
    weights.Dispose();
}
else
{
    // ── Standard path (Qwen / hybrid) ───────────────────────────────────────
    // Build vocab remapper if partial vocab is active (vocab-limit > 1)
    // Build vocab remapper if partial vocab is active (vocab-limit > 1)
    // Same-family models (Qwen3.5) have identical vocabularies, so same remapper works for both
    VocabRemapper? remapper = null;
    // Disable remapper for speculative decoding and pipeline mode (both need full vocab space)
    int vocabDivisor = options.DraftModelPath != null ? 1 : (options.VocabLimit ?? 1);
    if (vocabDivisor > 1)
    {
        var tokens = gguf.GetMetadata<string[]>("tokenizer.ggml.tokens")!;
        remapper = new VocabRemapper(tokens);
        tokenizer.Vocabulary.ApplyRemapper(remapper);
    }

        // ── Pipeline mode: stream layers from shards for models > VRAM ──────────
    // Check BEFORE loading weights — pipeline loads its own embed/output from shards.
    if (options.Pipeline && backend is CudaBackend pipelineCuda)
    {
        var shardDir = options.ModelPath + ".shards";
        if (!Directory.Exists(shardDir))
        {
            Console.Error.WriteLine($"Shard directory not found: {shardDir}");
            Console.Error.WriteLine("Run: daisi-llogos split --model <path> --align-gpu");
            return 1;
        }

        var pipeForward = PipelinedForwardPass.Create(gguf, shardDir, config, pipelineCuda,
            maxContext: options.MaxContext);

        loadSw.Stop();
        Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
            $"({config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d, pipeline)");
        Console.Error.WriteLine($"Backend: {backend.Name}, Max tokens: {options.MaxTokens}");

        var pipeGenerator = new TextGenerator(pipeForward, tokenizer, options.Seed);

        if (options.Bench)
        {
            string benchPrompt = options.Prompt ?? "The meaning of life is";
            Console.Error.WriteLine($"Benchmarking with prompt: \"{benchPrompt}\"");
            Console.Error.WriteLine($"Backend: {backend.Name}, Max tokens: {options.MaxTokens}");
            Console.Error.WriteLine();
            var result = pipeGenerator.Benchmark(benchPrompt, options.MaxTokens);
            Console.Error.WriteLine("=== Benchmark Results ===");
            Console.Error.WriteLine($"  Prefill:  {result.PromptTokens,6} tokens in {result.PrefillTime.TotalMilliseconds,8:F1} ms  ({result.PrefillTokPerSec,8:F1} tok/s)");
            Console.Error.WriteLine($"  Decode:   {result.GeneratedTokens,6} tokens in {result.DecodeTime.TotalMilliseconds,8:F1} ms  ({result.DecodeTokPerSec,8:F1} tok/s)");
            Console.Error.WriteLine($"  Total:    {result.PromptTokens + result.GeneratedTokens,6} tokens in {result.TotalTime.TotalMilliseconds,8:F1} ms");
            Console.Error.WriteLine($"  Load:     {loadSw.Elapsed.TotalMilliseconds,8:F1} ms");
        }
        else
        {
            foreach (var token in pipeGenerator.Generate(options.Prompt!, new GenerationParams
            {
                MaxTokens = options.MaxTokens, Temperature = options.Temperature,
                TopK = options.TopK, TopP = options.TopP, RepetitionPenalty = options.RepeatPenalty,
            }))
            {
                if (token.IsDone)
                {
                    Console.Error.WriteLine();
                    Console.Error.WriteLine($"\n[prefill: {token.PrefillTokens} tokens, {token.PrefillTokensPerSecond:F1} tok/s | " +
                        $"decode: {token.TotalTokens} tokens, {token.TokensPerSecond:F1} tok/s]");
                }
                else Console.Write(token.Text);
            }
        }
        pipeForward.Dispose();
        backend.Dispose();
        return 0;
    }

    // ── Load full model weights (non-pipeline paths) ───────────────────────
    ModelWeights weights;
    if (options.LoraPaths.Count > 0)
    {
        var cpuBackend = new CpuBackend();
        weights = MmapModelLoader.Load(gguf, options.ModelPath, cpuBackend, config, remapper);
        foreach (var loraPath in options.LoraPaths)
        {
            Console.Error.Write($"Loading LoRA adapter: {loraPath}... ");
            var adapter = LoraInference.LoadAndMerge(loraPath, weights, cpuBackend, config);
            Console.Error.WriteLine($"done ({adapter.ParameterCount:N0} params, rank={adapter.Config.Rank})");
        }
        if (backend is not CpuBackend)
            LoraInference.UploadWeights(weights, backend, config);
    }
    else if (options.UseMmap)
        weights = MmapModelLoader.Load(gguf, options.ModelPath, backend, config, remapper);
    else
        weights = ModelLoader.Load(gguf, stream, backend, config);

    var strategy = AttentionStrategy.Parse(options.Attention);
    int maxContext = strategy.Mode != AttentionMode.Full && strategy.CacheCapacity > 0
        ? strategy.CacheCapacity
        : options.MaxContext;
    IKvCache kvCache;
    if (options.KvQuant != null)
    {
        var turboConfig = TurboQuantConfig.Parse(options.KvQuant);
        if (backend is CudaBackend cudaBackend)
            kvCache = new CudaTurboQuantKvCache(cudaBackend, config, maxSeqLen: maxContext,
                turboConfig: turboConfig, strategy: strategy);
        else
            kvCache = new TurboQuantKvCache(backend, config, maxSeqLen: maxContext,
                turboConfig: turboConfig, strategy: strategy);
        Console.Error.WriteLine($"  LLogos Turbo: {turboConfig.EffectiveBitsPerDim(config.KeyLength):F1} bits/dim " +
            $"(q{turboConfig.QuantBits}" +
            $"{(turboConfig.QjlProjectionDim is > 0 ? $"+qjl{turboConfig.QjlProjectionDim}" : turboConfig.QjlProjectionDim == 0 ? "+noqjl" : "+qjl")}" +
            $", {(backend is CudaBackend ? "CUDA" : "CPU")})");
    }
    else if (options.Paged)
        kvCache = new PagedKvCache(backend, config, maxSeqLen: maxContext, strategy: strategy,
            vramPageBudget: options.OffloadPages);
    else
        kvCache = new KvCache(backend, config, maxSeqLen: maxContext, strategy: strategy);
    var deltaState = new DeltaNetState(backend, config, weights);
    var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
    forward.ArgMaxVocabLimit = config.VocabSize / vocabDivisor;

    // Early exit profiling: measure at which layer the token prediction stabilizes
    if (options.ProfileEarlyExit)
    {
        forward.EarlyExitProfile = true;
        forward.EarlyExitTokens = new int[config.NumLayers];
        Array.Fill(forward.EarlyExitTokens, -1);
    }

    loadSw.Stop();
    var attnInfo = strategy.Mode switch
    {
        AttentionMode.Window => $", window:{strategy.WindowSize}",
        AttentionMode.Sinks => $", sinks:{strategy.SinkTokens},{strategy.WindowSize}",
        _ => ""
    };
    var pagedInfo = options.Paged ? $", paged{(options.OffloadPages > 0 ? $" offload>{options.OffloadPages}" : "")}" : "";
    Console.Error.WriteLine($"Model loaded in {loadSw.Elapsed.TotalSeconds:F1}s " +
        $"({config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d" +
        $"{(options.UseMmap ? ", mmap" : "")}{attnInfo}{pagedInfo})");

    // ── Speculative decoding (optional draft model) ─────────────────────────
    ForwardPass? draftForward = null;
    SpeculativeDecoder? specDecoder = null;
    if (options.DraftModelPath != null)
    {
        Console.Error.Write($"Loading draft model: {options.DraftModelPath}... ");
        using var draftStream = File.OpenRead(options.DraftModelPath);
        var draftGguf = GgufFile.Read(draftStream);
        var draftConfig = ModelConfig.FromGguf(draftGguf);

        // Draft uses NO remapper — its token IDs are in the original space.
        // The SpeculativeDecoder translates between remapped (target) and original (draft) IDs.
        ModelWeights draftWeights;
        if (options.UseMmap)
            draftWeights = MmapModelLoader.Load(draftGguf, options.DraftModelPath, backend, draftConfig, null);
        else
            draftWeights = ModelLoader.Load(draftGguf, draftStream, backend, draftConfig);

        var draftKvCache = new KvCache(backend, draftConfig, maxSeqLen: options.MaxContext);
        var draftDeltaState = new DeltaNetState(backend, draftConfig, draftWeights);
        draftForward = new ForwardPass(backend, draftConfig, draftWeights, draftKvCache, draftDeltaState);
        // Draft uses partial vocab (same raw token order as target since both un-remapped)
        // /4 gives ~62K tokens — generous enough for common tokens without remapper
        draftForward.ArgMaxVocabLimit = draftConfig.VocabSize / 4;

        specDecoder = new SpeculativeDecoder(forward, draftForward, tokenizer, options.SpecDepth, remapper)
        {
            BatchedVerify = options.BatchedVerify
        };
        Console.Error.WriteLine($"done ({draftConfig.Architecture}, {draftConfig.NumLayers}L, {draftConfig.HiddenDim}d)");
    }

    // Hybrid GPU+CPU: replace forward pass with split execution
    IForwardPass activeForward = forward;
    if (options.HybridLayers > 0 && backend is CudaBackend cudaForHybrid)
    {
        activeForward = HybridForwardPass.Create(gguf, options.ModelPath!, config,
            cudaForHybrid, options.HybridLayers, strategy);
    }
    var generator = new TextGenerator(activeForward, tokenizer, options.Seed);

    if (options.Bench)
    {
        string benchPrompt = options.Prompt ?? "The meaning of life is";
        Console.Error.WriteLine($"Benchmarking with prompt: \"{benchPrompt}\"");
        Console.Error.WriteLine($"Backend: {backend.Name}, Max tokens: {options.MaxTokens}");
        Console.Error.WriteLine();

        var result = generator.Benchmark(benchPrompt, options.MaxTokens);

        Console.Error.WriteLine("=== Benchmark Results ===");
        Console.Error.WriteLine($"  Prefill:  {result.PromptTokens,6} tokens in {result.PrefillTime.TotalMilliseconds,8:F1} ms  ({result.PrefillTokPerSec,8:F1} tok/s)");
        Console.Error.WriteLine($"  Decode:   {result.GeneratedTokens,6} tokens in {result.DecodeTime.TotalMilliseconds,8:F1} ms  ({result.DecodeTokPerSec,8:F1} tok/s)");
        Console.Error.WriteLine($"  Total:    {result.PromptTokens + result.GeneratedTokens,6} tokens in {result.TotalTime.TotalMilliseconds,8:F1} ms");
        Console.Error.WriteLine($"  Load:     {loadSw.Elapsed.TotalMilliseconds,8:F1} ms");
    }
    else
    {
        var parameters = new GenerationParams
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopK = options.TopK,
            TopP = options.TopP,
            RepetitionPenalty = options.RepeatPenalty,
            Seed = options.Seed,
        };

        var generateFn = specDecoder != null
            ? specDecoder.Generate(options.Prompt!, parameters)
            : generator.Generate(options.Prompt!, parameters);

        foreach (var token in generateFn)
        {
            if (token.IsDone)
            {
                Console.Error.WriteLine();
                var specInfo = specDecoder != null
                    ? $" | accept: {specDecoder.AcceptanceRate:P0} ({specDecoder.TotalAcceptedTokens}/{specDecoder.TotalDraftTokens})"
                    : "";
                Console.Error.WriteLine($"\n[prefill: {token.PrefillTokens} tokens, {token.PrefillTokensPerSecond:F1} tok/s | " +
                    $"decode: {token.TotalTokens} tokens, {token.TokensPerSecond:F1} tok/s{specInfo}]");
            }
            else
            {
                Console.Write(token.Text);
            }
        }
    }

    // Print early exit profiling results
    if (options.ProfileEarlyExit && forward.EarlyExitTokens != null)
    {
        Console.Error.WriteLine("\n[Early Exit Profile — token predicted at each layer]");
        int finalToken = forward.EarlyExitTokens[config.NumLayers - 1];
        int firstStableLayer = -1;
        for (int i = config.NumLayers / 4; i < config.NumLayers; i++)
        {
            int tok = forward.EarlyExitTokens[i];
            if (tok < 0) continue;
            string tokStr = tok < tokenizer.Vocabulary.Count ? tokenizer.Vocabulary.IdToToken(tok) : $"<{tok}>";
            bool isFinal = tok == finalToken;
            Console.Error.Write($"  L{i}: {tok}({tokStr}){(isFinal ? " ✓" : " ✗")}");
            if (isFinal && firstStableLayer < 0) firstStableLayer = i;
            if ((i + 1) % 4 == 0) Console.Error.WriteLine();
        }
        if (firstStableLayer >= 0)
            Console.Error.WriteLine($"\n  → Token stabilizes at layer {firstStableLayer}/{config.NumLayers} ({100 * firstStableLayer / config.NumLayers}% through)");
        Console.Error.WriteLine();
    }

    // Print LLogos Turbo compression stats
    TurboQuantStats? turboStats = kvCache switch
    {
        TurboQuantKvCache tq when tq.Length > 0 => tq.GetStats(),
        CudaTurboQuantKvCache ctq when ctq.Length > 0 => ctq.GetStats(),
        _ => null
    };
    if (turboStats is { } stats)
    {
        Console.Error.WriteLine($"\n[LLogos Turbo KV Cache]");
        Console.Error.WriteLine($"  Compressed:   {stats.CompressedBytes / 1024.0:F1} KB");
        Console.Error.WriteLine($"  Uncompressed: {stats.UncompressedBytes / 1024.0:F1} KB");
        Console.Error.WriteLine($"  Ratio:        {stats.CompressionRatio:F1}x ({stats.EffectiveBitsPerDim:F1} bits/dim)");
        Console.Error.WriteLine($"  Layers:       {stats.NumLayers}, Seq length: {stats.SeqLength}");
    }

    draftForward?.Dispose();
    forward.Dispose();
    deltaState.Dispose();
    kvCache.Dispose();
    weights.Dispose();
}

backend.Dispose();

return 0;

// ── Shared generation / bench logic ──────────────────────────────────────────

static void RunGeneration(
    Func<string, GenerationParams, IEnumerable<GenerationToken>> generateFn,
    CliArgs options)
{
    if (options.Bench)
    {
        string benchPrompt = options.Prompt ?? "The meaning of life is";
        Console.Error.WriteLine($"Benchmarking with prompt: \"{benchPrompt}\"");
        Console.Error.WriteLine($"Max tokens: {options.MaxTokens}");
        Console.Error.WriteLine();

        var parameters = new GenerationParams
        {
            MaxTokens = options.MaxTokens,
            Temperature = 0.7f,
            TopK = 40,
            TopP = 0.9f,
        };
        var sw = Stopwatch.StartNew();
        int totalTokens = 0;
        foreach (var token in generateFn(benchPrompt, parameters))
        {
            if (token.IsDone)
            {
                Console.Error.WriteLine("=== Benchmark Results ===");
                Console.Error.WriteLine($"  Prefill:  {token.PrefillTokens,6} tokens ({token.PrefillTokensPerSecond,8:F1} tok/s)");
                Console.Error.WriteLine($"  Decode:   {token.TotalTokens,6} tokens ({token.TokensPerSecond,8:F1} tok/s)");
                Console.Error.WriteLine($"  Total:    {sw.Elapsed.TotalMilliseconds,8:F1} ms");
            }
            totalTokens++;
        }
    }
    else
    {
        var parameters = new GenerationParams
        {
            MaxTokens = options.MaxTokens,
            Temperature = options.Temperature,
            TopK = options.TopK,
            TopP = options.TopP,
            RepetitionPenalty = options.RepeatPenalty,
            Seed = options.Seed,
        };

        foreach (var token in generateFn(options.Prompt!, parameters))
        {
            if (token.IsDone)
            {
                Console.Error.WriteLine();
                Console.Error.WriteLine($"\n[prefill: {token.PrefillTokens} tokens, {token.PrefillTokensPerSecond:F1} tok/s | " +
                    $"decode: {token.TotalTokens} tokens, {token.TokensPerSecond:F1} tok/s]");
            }
            else
            {
                Console.Write(token.Text);
            }
        }
    }
}

// ── Argument parsing ─────────────────────────────────────────────────────────

static CliArgs ParseArgs(string[] args)
{
    var result = new CliArgs();
    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--model" or "-m":
                result.ModelPath = NextArg(args, ref i);
                break;
            case "--prompt" or "-p":
                result.Prompt = NextArg(args, ref i);
                break;
            case "--max-tokens" or "-n":
                result.MaxTokens = int.Parse(NextArg(args, ref i));
                break;
            case "--max-context":
                result.MaxContext = int.Parse(NextArg(args, ref i));
                break;
            case "--temperature" or "-t":
                result.Temperature = float.Parse(NextArg(args, ref i));
                break;
            case "--top-k":
                result.TopK = int.Parse(NextArg(args, ref i));
                break;
            case "--top-p":
                result.TopP = float.Parse(NextArg(args, ref i));
                break;
            case "--repeat-penalty":
                result.RepeatPenalty = float.Parse(NextArg(args, ref i));
                break;
            case "--seed":
                result.Seed = int.Parse(NextArg(args, ref i));
                break;
            case "--backend" or "-b":
                result.Backend = NextArg(args, ref i);
                break;
            case "--bench":
                result.Bench = true;
                break;
            case "--no-mmap":
                result.UseMmap = false;
                break;
            case "--attention":
                result.Attention = NextArg(args, ref i);
                break;
            case "--vocab-limit":
                result.VocabLimit = int.Parse(NextArg(args, ref i));
                break;
            case "--profile-early-exit":
                result.ProfileEarlyExit = true;
                break;
            case "--paged":
                result.Paged = true;
                break;
            case "--offload-pages":
                result.Paged = true;
                result.OffloadPages = int.Parse(NextArg(args, ref i));
                break;
            case "--draft":
                result.DraftModelPath = NextArg(args, ref i);
                break;
            case "--spec-depth":
                result.SpecDepth = int.Parse(NextArg(args, ref i));
                break;
            case "--batched-verify":
                result.BatchedVerify = true;
                break;
            case "--kv-quant":
                result.KvQuant = NextArg(args, ref i);
                break;
            case "--hybrid-layers":
                result.HybridLayers = int.Parse(NextArg(args, ref i));
                break;
            case "--pipeline":
                result.Pipeline = true;
                break;
            case "--lora":
                result.LoraPaths.Add(NextArg(args, ref i));
                break;
            case "--help" or "-h":
                result.ShowHelp = true;
                break;
        }
    }
    return result;
}

static string NextArg(string[] args, ref int i) =>
    ++i < args.Length ? args[i] : throw new ArgumentException($"Missing value for {args[i - 1]}");

static void PrintUsage()
{
    Console.Error.WriteLine("""
        daisi-llogos - C# LLM inference engine

        Usage: daisi-llogos --model <path> --prompt <text> [options]

        Options:
          --model, -m <path>       Path to GGUF model file (required)
          --prompt, -p <text>      Input prompt (required for generate, optional for bench)
          --max-tokens, -n <n>     Maximum tokens to generate (default: 256)
          --max-context <n>        Maximum context length (default: 2048)
          --temperature, -t <f>    Sampling temperature, 0=greedy (default: 0.7)
          --top-k <n>              Top-k sampling, 0=disabled (default: 40)
          --top-p <f>              Top-p nucleus sampling (default: 0.9)
          --repeat-penalty <f>     Repetition penalty (default: 1.1)
          --seed <n>               Random seed for reproducibility
          --backend, -b <name>     Compute backend: cpu, cuda, or vulkan (default: cpu)
          --bench                  Run benchmark (prefill + decode timing)
          --vocab-limit <n>        Vocab divisor for greedy argmax (1=full, 32=3%, default: 32)
          --no-mmap                Disable memory-mapped loading (use stream loading)
          --attention <mode>       Attention strategy: full, window:<N>, sinks:<S>,<W> (default: full)
          --paged                  Use paged KV cache (dynamic allocation, grows with context)
          --offload-pages <n>      Enable RAM offloading: keep first N pages in VRAM, rest in RAM
          --draft <path>           Draft model for speculative decoding (smaller, same family)
          --spec-depth <n>         Speculation depth (default: 5)
          --batched-verify         Use batched verify (faster, higher acceptance, different FP from native)
          --kv-quant <mode>        KV cache compression: turbo, turbo:3, turbo:4, turbo:3+qjl32, turbo:3+noqjl
          --hybrid-layers <n>      GPU+CPU split: first N layers on GPU, rest on CPU
          --pipeline               Stream layers from shards (for models > VRAM, requires split first)
          --lora <path>            LoRA adapter file (.llra) to merge into model weights
          --help, -h               Show this help

        Splitting (DaisiChain shard format):
          daisi-llogos split --model <path> --output-dir <path>
          Split a GGUF into per-layer shard files for partial downloads.

        Training:
          daisi-llogos train --model <path> --data <path> [options]
          Run 'daisi-llogos train --help' for training options.
        """);
}

// ── Split Subcommand ────────────────────────────────────────────────────────

static int RunSplit(string[] args)
{
    string? modelPath = null;
    string? outputDir = null;
    bool alignGpu = false;
    bool showHelp = false;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--model" or "-m":
                modelPath = NextArg(args, ref i);
                break;
            case "--output-dir" or "-o":
                outputDir = NextArg(args, ref i);
                break;
            case "--align-gpu":
                alignGpu = true;
                break;
            case "--help" or "-h":
                showHelp = true;
                break;
        }
    }

    if (showHelp || modelPath == null)
    {
        Console.Error.WriteLine("""
            daisi-llogos split - Split GGUF into per-layer shards

            Usage: daisi-llogos split --model <path> [options]

            Required:
              --model, -m <path>       Path to GGUF model file

            Options:
              --output-dir, -o <path>  Output directory (default: {model}.shards/)
              --align-gpu              Pre-repack Q4_0/Q8_0 to GPU-aligned layout (faster pipelined inference)
              --help, -h               Show this help
            """);
        return showHelp ? 0 : 1;
    }

    if (!File.Exists(modelPath))
    {
        Console.Error.WriteLine($"Error: model file not found: {modelPath}");
        return 1;
    }

    outputDir ??= modelPath + ".shards";

    GgufSplitter.Split(modelPath, outputDir, msg => Console.Error.WriteLine($"  {msg}"), gpuAligned: alignGpu);
    return 0;
}

// ── Training Subcommand ──────────────────────────────────────────────────────

static int RunTraining(string[] args)
{
    string? modelPath = null;
    string? dataPath = null;
    string? outputPath = null;
    int rank = 8;
    float alpha = 16;
    float lr = 1e-4f;
    int epochs = 3;
    int seqLen = 512;
    int warmup = 50;
    int saveEvery = 100;
    int logEvery = 10;
    float weightDecay = 0.01f;
    float maxGradNorm = 1.0f;
    int gradAccum = 1;
    int seed = 42;
    string targets = "qkvo";
    string trainBackend = "cpu";
    bool showHelp = false;

    for (int i = 0; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--model" or "-m":
                modelPath = NextArg(args, ref i);
                break;
            case "--data" or "-d":
                dataPath = NextArg(args, ref i);
                break;
            case "--output" or "-o":
                outputPath = NextArg(args, ref i);
                break;
            case "--rank" or "-r":
                rank = int.Parse(NextArg(args, ref i));
                break;
            case "--alpha":
                alpha = float.Parse(NextArg(args, ref i));
                break;
            case "--lr":
                lr = float.Parse(NextArg(args, ref i));
                break;
            case "--epochs":
                epochs = int.Parse(NextArg(args, ref i));
                break;
            case "--seq-len":
                seqLen = int.Parse(NextArg(args, ref i));
                break;
            case "--warmup":
                warmup = int.Parse(NextArg(args, ref i));
                break;
            case "--save-every":
                saveEvery = int.Parse(NextArg(args, ref i));
                break;
            case "--log-every":
                logEvery = int.Parse(NextArg(args, ref i));
                break;
            case "--weight-decay":
                weightDecay = float.Parse(NextArg(args, ref i));
                break;
            case "--max-grad-norm":
                maxGradNorm = float.Parse(NextArg(args, ref i));
                break;
            case "--grad-accum":
                gradAccum = int.Parse(NextArg(args, ref i));
                break;
            case "--seed":
                seed = int.Parse(NextArg(args, ref i));
                break;
            case "--targets":
                targets = NextArg(args, ref i).ToLowerInvariant();
                break;
            case "--backend" or "-b":
                trainBackend = NextArg(args, ref i).ToLowerInvariant();
                break;
            case "--help" or "-h":
                showHelp = true;
                break;
        }
    }

    if (showHelp || modelPath == null || dataPath == null)
    {
        Console.Error.WriteLine("""
            daisi-llogos train - LoRA fine-tuning

            Usage: daisi-llogos train --model <path> --data <path> [options]

            Required:
              --model, -m <path>       Path to GGUF model file
              --data, -d <path>        Training data (text, .jsonl with "text" field, or .jsonl with "prompt"+"completion")
              --output, -o <path>      Output path for trained adapter (default: <model>.lora)

            LoRA:
              --rank, -r <n>           LoRA rank (default: 8)
              --alpha <f>              LoRA alpha scaling (default: 16)
              --targets <str>          Target projections: qkvo (default), qkvod (+ DeltaNet), all
              --backend, -b <name>     Compute backend: cpu (default), cuda

            Training:
              --lr <f>                 Learning rate (default: 1e-4)
              --epochs <n>             Number of epochs (default: 3)
              --seq-len <n>            Sequence length (default: 512)
              --warmup <n>             Warmup steps (default: 50)
              --weight-decay <f>       Weight decay (default: 0.01)
              --max-grad-norm <f>      Gradient clipping norm (default: 1.0)
              --grad-accum <n>         Gradient accumulation steps (default: 1)
              --seed <n>               Random seed (default: 42)
              --save-every <n>         Save checkpoint every N steps (default: 100)
              --log-every <n>          Log every N steps (default: 10)
              --help, -h               Show this help
            """);
        return showHelp ? 0 : 1;
    }

    outputPath ??= Path.ChangeExtension(modelPath, ".llra");

    var loraTargets = LoraTarget.None;
    if (targets == "all")
    {
        loraTargets = LoraTarget.AllLayers;
    }
    else
    {
        if (targets.Contains('q')) loraTargets |= LoraTarget.Q;
        if (targets.Contains('k')) loraTargets |= LoraTarget.K;
        if (targets.Contains('v')) loraTargets |= LoraTarget.V;
        if (targets.Contains('o')) loraTargets |= LoraTarget.O;
        if (targets.Contains('f')) loraTargets |= LoraTarget.AllFfn;
        if (targets.Contains('d')) loraTargets |= LoraTarget.DeltaQkv | LoraTarget.DeltaOut;
    }

    var config = new TrainingConfig
    {
        ModelPath = modelPath,
        DataPath = dataPath,
        OutputPath = outputPath,
        Lora = new LoraConfig { Rank = rank, Alpha = alpha, Targets = loraTargets },
        Epochs = epochs,
        LearningRate = lr,
        SeqLen = seqLen,
        WarmupSteps = warmup,
        WeightDecay = weightDecay,
        MaxGradNorm = maxGradNorm,
        GradientAccumulationSteps = gradAccum,
        Seed = seed,
        SaveEverySteps = saveEvery,
        LogEverySteps = logEvery,
    };

    IComputeBackend trainingBackend = trainBackend switch
    {
        "cuda" => new CudaTrainingBackend(),
        _ => new CpuBackend(),
    };

    using var session = new TrainingSession(config, trainingBackend);
    session.Run();
    return 0;
}

// ── metal-diff: compare per-op vs batched forward pass layer-by-layer ──
static int RunMetalDiff(string[] args)
{
    string? modelPath = null;
    string prompt = "Hello";
    int tokenCount = 1;
    for (int i = 0; i < args.Length; i++)
    {
        if (args[i] == "--model" && i + 1 < args.Length) modelPath = args[++i];
        else if (args[i] == "--prompt" && i + 1 < args.Length) prompt = args[++i];
        else if (args[i] == "--tokens" && i + 1 < args.Length) tokenCount = int.Parse(args[++i]);
    }
    if (modelPath == null) { Console.Error.WriteLine("metal-diff requires --model"); return 1; }

    Dictionary<string, double[]>? lastPerOp = null;

    double[] RunOnce(bool batched)
    {
        Environment.SetEnvironmentVariable("DAISI_METAL_BATCH", batched ? "1" : "0");
        Environment.SetEnvironmentVariable("DAISI_METAL_BATCH_SIZE", batched ? "1000000" : "");
        using var stream = File.OpenRead(modelPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new MetalBackend();
        var weights = MmapModelLoader.Load(gguf, modelPath, backend, config);
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        var kvCache = new KvCache(backend, config, maxSeqLen: 2048);
        var dnState = new DeltaNetState(backend, config);
        using var forward = new ForwardPass(backend, config, weights, kvCache, dnState);

        var tokenIds = tokenizer.Encode(prompt);
        var perLayerResults = new List<(int layer, string tag, float[] data)>();
        ForwardPass.DebugHook = (layer, tag, tensor) =>
        {
            var buf = new float[tensor.ElementCount];
            tensor.DequantizeTo(buf);
            perLayerResults.Add((layer, tag, buf));
        };

        try
        {
            // Prefill, then decode 1 token
            for (int i = 0; i < tokenIds.Length - 1; i++)
                forward.ForwardHidden(tokenIds[i], i);
            var logits = forward.Forward(tokenIds[tokenIds.Length - 1], tokenIds.Length - 1).ToArray();

            // Report hash per layer
            double[] hashes = new double[perLayerResults.Count];
            for (int i = 0; i < perLayerResults.Count; i++)
            {
                var (layer, tag, buf) = perLayerResults[i];
                // Compute max, min, L2 norm
                double sumSq = 0; float min = float.MaxValue, max = float.MinValue;
                for (int j = 0; j < buf.Length; j++)
                {
                    sumSq += (double)buf[j] * buf[j];
                    if (buf[j] < min) min = buf[j];
                    if (buf[j] > max) max = buf[j];
                }
                hashes[i] = Math.Sqrt(sumSq);
                Console.WriteLine($"[{(batched ? "batched" : "per-op")}] layer={layer} tag={tag} L2={hashes[i]:G6} min={min:G4} max={max:G4} first={buf[0]:G6} last={buf[buf.Length - 1]:G6}");
            }
            return hashes;
        }
        finally
        {
            ForwardPass.DebugHook = null;
            weights.Dispose();
            kvCache.Dispose();
            dnState.Dispose();
        }
    }

    Console.WriteLine("=== Running per-op (batch=0) ===");
    var perOp = RunOnce(false);
    Console.WriteLine("=== Running batched (batch=1, size=1000000) ===");
    var batched = RunOnce(true);

    Console.WriteLine("=== Diff ===");
    for (int i = 0; i < Math.Min(perOp.Length, batched.Length); i++)
    {
        double rel = Math.Abs(perOp[i] - batched[i]) / Math.Max(1e-9, Math.Abs(perOp[i]));
        string marker = rel > 1e-3 ? " <<< DIVERGES" : "";
        Console.WriteLine($"  layer {i}  per-op={perOp[i]:G6}  batched={batched[i]:G6}  relDiff={rel:G4}{marker}");
    }
    return 0;
}

class CliArgs
{
    public string? ModelPath;
    public string? Prompt;
    public int MaxTokens = 256;
    public int MaxContext = 2048;
    public float Temperature = 0.7f;
    public int TopK = 40;
    public float TopP = 0.9f;
    public float RepeatPenalty = 1.1f;
    public int? Seed;
    public string Backend = "cpu";
    public bool ShowHelp;
    public bool Bench;
    public bool UseMmap = true;
    public string Attention = "full";
    public bool Paged;
    public int OffloadPages;
    public int? VocabLimit;
    public bool ProfileEarlyExit;
    public string? DraftModelPath;
    public int SpecDepth = 5;
    public bool BatchedVerify;
    public string? KvQuant;
    public int HybridLayers;
    public List<string> LoraPaths = new();
    public bool Pipeline;
}
