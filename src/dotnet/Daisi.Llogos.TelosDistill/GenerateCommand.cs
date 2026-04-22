namespace Daisi.Llogos.TelosDistill;

public sealed class GenerateOptions
{
    public string ModelPath = "";
    public string Backend = "cpu";
    public string TargetClass = "";
    public string TopicSlug = "";
    public string TopicDescription = "";
    public string CorpusDir = "";
    public int Count = 20;
    public int BatchSize = 10;     // intents requested per teacher call
    public float Temperature = 0.9f;
    public int MaxTokens = 1024;
    public int? Seed;
    public int? MaxContext;
}

public static class GenerateCommand
{
    public static async Task<int> RunAsync(GenerateOptions o)
    {
        ValidateClass(o.TargetClass);

        var classDir = Path.Combine(o.CorpusDir, o.TargetClass.ToLowerInvariant());
        Directory.CreateDirectory(classDir);

        var stamp = DateTime.UtcNow.ToString("yyyyMMdd");
        var outPath = Path.Combine(classDir, $"distilled_{Slugify(o.TopicSlug)}_{stamp}.txt");

        using var host = ModelHost.Load(o.ModelPath, o.Backend, o.MaxContext ?? 4096, o.Seed);

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var accepted = new List<string>(o.Count);
        int totalRaw = 0, totalDuplicate = 0, totalEmpty = 0;

        Console.Error.WriteLine(
            $"[gen] class={o.TargetClass} topic={o.TopicSlug} target={o.Count} batch={o.BatchSize}");
        Console.Error.WriteLine($"[gen] writing to {outPath}");

        // Append, not overwrite — re-runs add to the day's file.
        await using var writer = new StreamWriter(outPath, append: true);
        if (new FileInfo(outPath).Length == 0)
        {
            await writer.WriteLineAsync($"# {o.TargetClass.ToUpperInvariant()} · {o.TopicSlug}");
            await writer.WriteLineAsync($"# {o.TopicDescription}");
            await writer.WriteLineAsync($"# distilled on {stamp} — spot-check before shipping.");
            await writer.WriteLineAsync();
        }

        int callCount = 0;
        while (accepted.Count < o.Count)
        {
            callCount++;
            int want = Math.Min(o.BatchSize, o.Count - accepted.Count + o.BatchSize / 2);
            var user = Prompts.GenerationUser(o.TargetClass, o.TopicSlug, o.TopicDescription, want);
            // Vary seed per call so fresh KV caches don't replay the same output.
            var callSeed = o.Seed is int s ? s + callCount : (int?)null;
            var response = await host.GenerateAsync(
                Prompts.GenerationSystem, user, o.MaxTokens, o.Temperature,
                seedOverride: callSeed);

            var lines = ExtractLines(response);
            totalRaw += lines.Count;

            int addedThisCall = 0;
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) { totalEmpty++; continue; }
                var cleaned = Clean(line);
                if (cleaned.Length == 0) { totalEmpty++; continue; }
                if (!seen.Add(cleaned)) { totalDuplicate++; continue; }
                accepted.Add(cleaned);
                await writer.WriteLineAsync(cleaned);
                addedThisCall++;
                if (accepted.Count >= o.Count) break;
            }
            await writer.FlushAsync();

            Console.Error.WriteLine(
                $"[gen] call {callCount}: raw={lines.Count} kept={addedThisCall} " +
                $"total={accepted.Count}/{o.Count}");

            if (addedThisCall == 0 && callCount >= 3)
            {
                Console.Error.WriteLine("[gen] three empty calls in a row, stopping early.");
                break;
            }
        }

        Console.Error.WriteLine(
            $"[gen] done. kept={accepted.Count} raw={totalRaw} dupes={totalDuplicate} empty={totalEmpty}");
        Console.Error.WriteLine($"[gen] spot-check: {outPath}");
        return 0;
    }

    private static void ValidateClass(string c)
    {
        var l = c.ToLowerInvariant();
        if (l != "permit" && l != "deny" && l != "ambiguous")
            throw new ArgumentException($"--class must be permit|deny|ambiguous, got '{c}'");
    }

    private static string Slugify(string topic)
    {
        var chars = topic.ToLowerInvariant()
            .Select(ch => char.IsLetterOrDigit(ch) ? ch : '_')
            .ToArray();
        return new string(chars).Trim('_');
    }

    /// <summary>
    /// Split the response into candidate lines, stripping any Qwen-style reasoning blocks.
    /// Handles both &lt;think&gt;...&lt;/think&gt; and the rare case where only a closing tag appears.
    /// </summary>
    private static List<string> ExtractLines(string response)
    {
        var cleaned = StripThinkBlocks(response);
        return cleaned.Split('\n', StringSplitOptions.None).ToList();
    }

    private static string StripThinkBlocks(string s)
    {
        // Remove everything up to and including a closing </think> (covers open block at start + nested).
        int close = s.IndexOf("</think>", StringComparison.OrdinalIgnoreCase);
        if (close >= 0)
            s = s[(close + "</think>".Length)..];

        // Remove any remaining <think>...</think> pairs or dangling open <think>... to end.
        while (true)
        {
            int open = s.IndexOf("<think>", StringComparison.OrdinalIgnoreCase);
            if (open < 0) break;
            int end = s.IndexOf("</think>", open, StringComparison.OrdinalIgnoreCase);
            if (end < 0) { s = s[..open]; break; }
            s = s[..open] + s[(end + "</think>".Length)..];
        }
        return s;
    }

    /// <summary>
    /// Strip leading numbering, bullets, quotes, and trailing junk. Reject lines that look
    /// like the model talking about its output instead of producing it.
    /// </summary>
    private static string Clean(string line)
    {
        var s = line.Trim();
        if (s.Length == 0) return "";

        // Drop leading list markers: "1.", "1)", "-", "*", "•"
        while (s.Length > 0)
        {
            if (s[0] is '-' or '*' or '•' or '·' or '+')
            {
                s = s[1..].TrimStart();
                continue;
            }
            if (char.IsDigit(s[0]))
            {
                int i = 0;
                while (i < s.Length && char.IsDigit(s[i])) i++;
                if (i < s.Length && (s[i] == '.' || s[i] == ')' || s[i] == ':'))
                {
                    s = s[(i + 1)..].TrimStart();
                    continue;
                }
            }
            break;
        }

        // Strip surrounding quotes.
        if (s.Length >= 2 && ((s[0] == '"' && s[^1] == '"') || (s[0] == '\'' && s[^1] == '\'')))
            s = s[1..^1].Trim();

        // Reject lines that are clearly meta-commentary or stray markup.
        if (s.Length > 240) return "";
        if (s.Length < 3) return "";
        if (s.StartsWith('<') || s.StartsWith('>')) return "";   // drifting tags
        if (s.Contains("<think", StringComparison.OrdinalIgnoreCase) ||
            s.Contains("</think", StringComparison.OrdinalIgnoreCase))
            return "";
        var lower = s.ToLowerInvariant();
        if (lower.StartsWith("here are") || lower.StartsWith("here's") ||
            lower.StartsWith("sure,") || lower.StartsWith("certainly") ||
            lower.StartsWith("note:") || lower.StartsWith("category:") ||
            lower.StartsWith("topic:") || lower.StartsWith("example") ||
            lower.Contains("as an ai") || lower.Contains("i cannot"))
            return "";

        return s;
    }
}
