namespace Daisi.Llogos.TelosDistill;

public sealed class VerifyOptions
{
    public string ModelPath = "";
    public string Backend = "cpu";
    public string InputFile = "";
    public string ExpectedClass = "";   // inferred from path if empty
    public int? Sample;                 // null = all
    public float Temperature = 0.1f;
    public int? Seed;
    public int? MaxContext;
    public string? DisagreementsOut;    // optional file to dump disagreements
}

public static class VerifyCommand
{
    public static async Task<int> RunAsync(VerifyOptions o)
    {
        var expected = string.IsNullOrEmpty(o.ExpectedClass)
            ? InferClassFromPath(o.InputFile)
            : o.ExpectedClass.ToLowerInvariant();
        if (expected is null)
        {
            Console.Error.WriteLine(
                "[verify] cannot infer class from path; pass --expected permit|deny|ambiguous.");
            return 1;
        }

        var lines = File.ReadAllLines(o.InputFile)
            .Where(l => !string.IsNullOrWhiteSpace(l) && !l.TrimStart().StartsWith("#"))
            .Select(l => l.Trim())
            .ToList();

        if (o.Sample is int n && n < lines.Count)
        {
            var rng = new Random(o.Seed ?? 42);
            lines = lines.OrderBy(_ => rng.Next()).Take(n).ToList();
        }

        Console.Error.WriteLine(
            $"[verify] {o.InputFile} · expected={expected} · n={lines.Count}");

        using var host = ModelHost.Load(o.ModelPath, o.Backend, o.MaxContext ?? 2048, o.Seed);

        int agree = 0, disagree = 0, uncertain = 0;
        var disagreements = new List<(string intent, string predicted)>();

        for (int i = 0; i < lines.Count; i++)
        {
            // Reasoning-model teachers (Qwen3) emit a <think> block before the
            // answer, so allow headroom; ParseClass skips past the block.
            var response = await host.GenerateAsync(
                Prompts.VerificationSystem,
                Prompts.VerificationUser(lines[i]),
                maxTokens: 512,
                temperature: o.Temperature);

            var predicted = ParseClass(response);
            if (predicted is null)
            {
                uncertain++;
            }
            else if (predicted == expected)
            {
                agree++;
            }
            else
            {
                disagree++;
                disagreements.Add((lines[i], predicted));
            }

            if ((i + 1) % 10 == 0 || i + 1 == lines.Count)
            {
                Console.Error.WriteLine(
                    $"[verify] {i + 1}/{lines.Count} agree={agree} disagree={disagree} unparsed={uncertain}");
            }
        }

        double rate = lines.Count == 0 ? 0 : (double)agree / lines.Count;
        Console.Error.WriteLine(
            $"[verify] done. agreement={rate:P1} ({agree}/{lines.Count}) " +
            $"disagree={disagree} unparsed={uncertain}");

        if (disagreements.Count > 0)
        {
            Console.Error.WriteLine($"[verify] disagreements (expected {expected.ToUpperInvariant()}):");
            foreach (var (intent, pred) in disagreements.Take(20))
                Console.Error.WriteLine($"  [{pred.ToUpperInvariant(),-9}] {intent}");
            if (disagreements.Count > 20)
                Console.Error.WriteLine($"  ... {disagreements.Count - 20} more.");

            if (!string.IsNullOrEmpty(o.DisagreementsOut))
            {
                File.WriteAllLines(o.DisagreementsOut,
                    disagreements.Select(d => $"{d.predicted}\t{d.intent}"));
                Console.Error.WriteLine($"[verify] wrote disagreements to {o.DisagreementsOut}");
            }
        }

        return 0;
    }

    private static string? InferClassFromPath(string path)
    {
        var parts = Path.GetFullPath(path)
            .Replace('\\', '/')
            .Split('/', StringSplitOptions.RemoveEmptyEntries);
        foreach (var p in parts)
        {
            var l = p.ToLowerInvariant();
            if (l == "permit" || l == "deny" || l == "ambiguous") return l;
        }
        return null;
    }

    private static string? ParseClass(string response)
    {
        // Strip any Qwen-style <think>...</think> block; the answer comes after.
        var s = response;
        int close = s.IndexOf("</think>", StringComparison.OrdinalIgnoreCase);
        if (close >= 0) s = s[(close + "</think>".Length)..];

        // Scan the remaining text for the first PERMIT / DENY / AMBIGUOUS word boundary.
        var lower = s.ToLowerInvariant();
        int i = 0;
        while (i < lower.Length)
        {
            if (!char.IsLetter(lower[i])) { i++; continue; }
            int j = i;
            while (j < lower.Length && char.IsLetter(lower[j])) j++;
            var word = lower[i..j];
            if (word == "permit" || word == "deny" || word == "ambiguous")
                return word;
            i = j;
        }
        return null;
    }
}
