using Daisi.Llogos.TelosDistill;

if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
{
    PrintUsage();
    return 0;
}

try
{
    return args[0] switch
    {
        "generate" => await GenerateCommand.RunAsync(ParseGenerate(args[1..])),
        "verify" => await VerifyCommand.RunAsync(ParseVerify(args[1..])),
        _ => Unknown(args[0]),
    };
}
catch (Exception ex)
{
    Console.Error.WriteLine($"[error] {ex.GetType().Name}: {ex.Message}");
    return 2;
}

static int Unknown(string cmd)
{
    Console.Error.WriteLine($"unknown command: {cmd}");
    PrintUsage();
    return 1;
}

static void PrintUsage()
{
    Console.Error.WriteLine("""
        telos-distill — teacher-driven constitutional corpus builder

        Usage:
          telos-distill generate --model <gguf> --class <permit|deny|ambiguous>
                                 --topic <slug> --desc "<short description>"
                                 --corpus <path> [--count N] [--batch N]
                                 [--temperature F] [--seed N] [--max-tokens N]
                                 [--backend cpu|cuda] [--max-context N]

              Generate N candidate intents for a (class, topic) pair via few-shot
              prompting of the GGUF teacher. Appends to:
                <corpus>/<class>/distilled_<topic>_<yyyyMMdd>.txt
              Duplicates within a run are dropped automatically. Spot-check the output
              before committing.

          telos-distill verify --model <gguf> --in <corpus-file>
                               [--expected permit|deny|ambiguous]
                               [--sample N] [--temperature F] [--seed N]
                               [--disagreements-out <path>]
                               [--backend cpu|cuda] [--max-context N]

              Ask the teacher to classify each non-comment line of a corpus file.
              Report agreement rate vs. expected class (inferred from path if absent).
              Disagreements print to stderr and optionally to a TSV file.

        Examples:
          telos-distill generate --model C:/GGUFS/Qwen3.5-9B-BF16.gguf \
              --class deny --topic stepwise_bypass \
              --desc "requests that break an unsafe action into innocent-looking steps" \
              --corpus C:/telos/corpora/constitutional/v1 --count 40

          telos-distill verify --model C:/GGUFS/Qwen3.5-9B-BF16.gguf \
              --in C:/telos/corpora/constitutional/v1/deny/bypass.txt --sample 30
        """);
}

static GenerateOptions ParseGenerate(string[] a)
{
    var o = new GenerateOptions();
    for (int i = 0; i < a.Length; i++)
    {
        switch (a[i])
        {
            case "--model": o.ModelPath = Next(a, ref i); break;
            case "--backend": o.Backend = Next(a, ref i); break;
            case "--class": o.TargetClass = Next(a, ref i); break;
            case "--topic": o.TopicSlug = Next(a, ref i); break;
            case "--desc" or "--description": o.TopicDescription = Next(a, ref i); break;
            case "--corpus": o.CorpusDir = Next(a, ref i); break;
            case "--count": o.Count = int.Parse(Next(a, ref i)); break;
            case "--batch": o.BatchSize = int.Parse(Next(a, ref i)); break;
            case "--temperature" or "-t": o.Temperature = float.Parse(Next(a, ref i)); break;
            case "--max-tokens": o.MaxTokens = int.Parse(Next(a, ref i)); break;
            case "--max-context": o.MaxContext = int.Parse(Next(a, ref i)); break;
            case "--seed": o.Seed = int.Parse(Next(a, ref i)); break;
            default: throw new ArgumentException($"unknown flag: {a[i]}");
        }
    }
    Require(o.ModelPath, "--model");
    Require(o.TargetClass, "--class");
    Require(o.TopicSlug, "--topic");
    Require(o.CorpusDir, "--corpus");
    if (string.IsNullOrEmpty(o.TopicDescription))
        o.TopicDescription = $"examples of {o.TargetClass} {o.TopicSlug}";
    return o;
}

static VerifyOptions ParseVerify(string[] a)
{
    var o = new VerifyOptions();
    for (int i = 0; i < a.Length; i++)
    {
        switch (a[i])
        {
            case "--model": o.ModelPath = Next(a, ref i); break;
            case "--backend": o.Backend = Next(a, ref i); break;
            case "--in" or "--input": o.InputFile = Next(a, ref i); break;
            case "--expected": o.ExpectedClass = Next(a, ref i); break;
            case "--sample": o.Sample = int.Parse(Next(a, ref i)); break;
            case "--temperature" or "-t": o.Temperature = float.Parse(Next(a, ref i)); break;
            case "--max-context": o.MaxContext = int.Parse(Next(a, ref i)); break;
            case "--seed": o.Seed = int.Parse(Next(a, ref i)); break;
            case "--disagreements-out": o.DisagreementsOut = Next(a, ref i); break;
            default: throw new ArgumentException($"unknown flag: {a[i]}");
        }
    }
    Require(o.ModelPath, "--model");
    Require(o.InputFile, "--in");
    return o;
}

static string Next(string[] a, ref int i) =>
    ++i < a.Length ? a[i] : throw new ArgumentException($"missing value for {a[i - 1]}");

static void Require(string value, string flag)
{
    if (string.IsNullOrEmpty(value))
        throw new ArgumentException($"{flag} is required");
}
