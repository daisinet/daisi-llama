namespace Daisi.Llogos.TelosDistill;

/// <summary>
/// Options for <c>telos-distill distill-soft</c> — M14 Mode 2 teacher emit.
/// </summary>
public sealed class DistillSoftOptions
{
    public string ModelPath = "";
    public string Backend = "cpu";
    public string InputFile = "";         // one intent per line; comments (#) skipped
    public string OutputFile = "";        // .soft TSV destination
    public int? Sample;                   // cap on intents processed (after dedup)
    public int? Seed;
    public int? MaxContext;
    /// <summary>
    /// Optional logit-space temperature applied before softmax. Higher values
    /// soften the teacher's distribution (the thing Hinton distillation
    /// actually trades on); T=1 leaves logits unchanged. Defaults to 2.0 so
    /// the emitted rows carry useful uncertainty out of the box.
    /// </summary>
    public float Temperature = 2.0f;
    /// <summary>
    /// Progress check-in interval in intents. 0 disables mid-run progress lines.
    /// </summary>
    public int ProgressEvery = 25;
}

/// <summary>
/// Soft-target distillation emit (M14 Mode 2). For each intent the teacher
/// was asked to classify, extract its first-response-token logits for the
/// three class words (PERMIT / DENY / AMBIGUOUS), softmax over just those,
/// and write <c>prompt\tp_permit\tp_deny\tp_amb</c> rows matching the telos
/// side's <c>SoftTargetCorpus</c> loader.
/// </summary>
public static class DistillSoftCommand
{
    public static async Task<int> RunAsync(DistillSoftOptions o)
    {
        if (!(o.Temperature > 0.0f && float.IsFinite(o.Temperature)))
        {
            Console.Error.WriteLine($"[distill-soft] invalid temperature {o.Temperature}");
            return 1;
        }

        var lines = File.ReadAllLines(o.InputFile)
            .Where(l => !string.IsNullOrWhiteSpace(l) && !l.TrimStart().StartsWith('#'))
            .Select(l => l.Trim())
            .Distinct()
            .ToList();

        if (o.Sample is int n && n < lines.Count)
        {
            var rng = new Random(o.Seed ?? 42);
            lines = lines.OrderBy(_ => rng.Next()).Take(n).ToList();
        }

        Console.Error.WriteLine(
            $"[distill-soft] {o.InputFile} → {o.OutputFile} · n={lines.Count} · T={o.Temperature:0.##}");

        using var host = ModelHost.Load(o.ModelPath, o.Backend, o.MaxContext ?? 2048, o.Seed);

        // Resolve first-response token ids for each class. We try the common
        // tokenizations in order: leading-space-uppercase (what most BPE
        // tokenizers produce for "PERMIT" as a standalone word), then
        // uppercase-no-space, then lowercase variants. If *none* of them are
        // a single token, the class can't be cleanly read off the first
        // response token and we bail — this would need a multi-token
        // marginalization path, which is a real but out-of-scope project.
        var classes = new[] { "permit", "deny", "ambiguous" };
        var tokenIds = new int[3];
        for (int i = 0; i < classes.Length; i++)
        {
            var upper = classes[i].ToUpperInvariant();
            int id = host.ResolveSingleToken(
                " " + upper,          // " PERMIT"
                upper,                 // "PERMIT"
                " " + classes[i],     // " permit"
                classes[i]);           // "permit"
            if (id < 0)
            {
                Console.Error.WriteLine(
                    $"[distill-soft] cannot find single-token id for {upper} in this model's vocab." +
                    " Teacher's tokenizer splits the class word; Mode-2 emit with first-token" +
                    " logit extraction isn't supported for this model. Use Mode 1 (verify) instead.");
                return 2;
            }
            tokenIds[i] = id;
        }
        Console.Error.WriteLine(
            $"[distill-soft] class token ids: permit={tokenIds[0]} deny={tokenIds[1]} amb={tokenIds[2]}");

        Directory.CreateDirectory(Path.GetDirectoryName(o.OutputFile) ?? ".");
        await using var outStream = File.CreateText(o.OutputFile);
        await outStream.WriteLineAsync(
            $"# teacher: {Path.GetFileName(host.ModelPath)}; T={o.Temperature:0.##}; n={lines.Count}");
        await outStream.WriteLineAsync(
            "# prompt<TAB>p_permit<TAB>p_deny<TAB>p_ambiguous (raw teacher softmax over class tokens)");

        var t0 = DateTime.UtcNow;
        int done = 0;
        foreach (var intent in lines)
        {
            float[] logits;
            try
            {
                logits = await host.FirstResponseLogitsAsync(
                    Prompts.VerificationSystem,
                    Prompts.VerificationUser(intent));
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[distill-soft] forward failed on '{intent}': {ex.Message}");
                continue;
            }

            // Extract just the three class-token logits, apply temperature,
            // then softmax over those three. Anything the teacher would
            // emit that isn't one of {PERMIT, DENY, AMBIGUOUS} is
            // marginalized out — acceptable here because the verification
            // prompt pins the answer format to one of those three words.
            float lp = logits[tokenIds[0]];
            float ld = logits[tokenIds[1]];
            float la = logits[tokenIds[2]];
            float lpT = lp / o.Temperature;
            float ldT = ld / o.Temperature;
            float laT = la / o.Temperature;
            float max = MathF.Max(lpT, MathF.Max(ldT, laT));
            float ep = MathF.Exp(lpT - max);
            float ed = MathF.Exp(ldT - max);
            float ea = MathF.Exp(laT - max);
            float sum = ep + ed + ea;
            float pP = ep / sum;
            float pD = ed / sum;
            float pA = ea / sum;

            // Tab in the intent would break the TSV; collapse any stray
            // tabs to spaces (real intents shouldn't have them, but guard).
            var safe = intent.Replace('\t', ' ');
            await outStream.WriteLineAsync(
                $"{safe}\t{pP:0.######}\t{pD:0.######}\t{pA:0.######}");

            done++;
            if (o.ProgressEvery > 0 && done % o.ProgressEvery == 0)
            {
                var elapsed = DateTime.UtcNow - t0;
                var rate = done / Math.Max(1e-3, elapsed.TotalSeconds);
                Console.Error.WriteLine(
                    $"[distill-soft] {done}/{lines.Count} · {rate:0.0}/s · elapsed {elapsed.TotalSeconds:0.0}s");
            }
        }

        var total = DateTime.UtcNow - t0;
        Console.Error.WriteLine(
            $"[distill-soft] done. {done}/{lines.Count} rows in {total.TotalSeconds:0.0}s");
        return 0;
    }
}
