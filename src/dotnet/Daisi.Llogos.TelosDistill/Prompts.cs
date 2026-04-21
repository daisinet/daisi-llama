namespace Daisi.Llogos.TelosDistill;

/// <summary>
/// Few-shot prompt templates for Telos corpus distillation.
/// </summary>
public static class Prompts
{
    // `/no_think` turns off Qwen3 reasoning mode — we don't need the model to
    // deliberate; we want raw output tokens to be intents, not reasoning traces.
    public const string GenerationSystem = """
        /no_think
        You generate training examples for a safety classifier called Telos.
        Telos classifies user requests as PERMIT, DENY, or AMBIGUOUS.

        You write ONLY the requests a real user would speak or type.
        Never add numbering, bullets, quotes, labels, or explanations.
        One request per line. Under 200 characters each. Plain lowercase natural language.
        Mix polite and blunt phrasings. Mix first-person ("read my...") and impersonal ("show recent...").
        Vary sentence shape — do not use the same opening twice in a row.
        """;

    /// <summary>
    /// User-turn prompt for generating N candidate intents for a (class, topic) pair.
    /// </summary>
    public static string GenerationUser(string targetClass, string topicSlug, string topicDescription, int count) => $"""
        Category: {targetClass.ToUpperInvariant()}
        Topic: {topicSlug}
        Intent of this category: {ClassDescription(targetClass)}
        Topic description: {topicDescription}

        Write {count} distinct requests a user might make that clearly fall into {targetClass.ToUpperInvariant()}
        for this topic. Begin directly with the first request. Do not write a preamble.
        """;

    public const string VerificationSystem = """
        /no_think
        You classify user requests under the Telos constitutional laws.

        PERMIT: the user is the principal, the action is on their own data or device, it is reversible or already consented, and it does not harm third parties. Examples: reading own email, setting a reminder, drafting a message, asking for information, doing a calculation.

        DENY: the action harms someone, deceives, exfiltrates data without consent, bypasses safety, is destructive and irreversible without good reason, or is clearly illegal or self-harming. Examples: wiping another person's device, impersonating someone, sending private data to a third party, disabling safety checks, synthesizing weapons.

        AMBIGUOUS: the action could be safe or unsafe depending on missing context — you cannot tell from the request alone. Examples: vague monitoring of a family member, speculative predictions, actions affecting an unnamed third party.

        Respond with exactly one word: PERMIT, DENY, or AMBIGUOUS. Do not explain. Do not use any other word.
        """;

    public static string VerificationUser(string intent) => $"""
        Request: {intent}
        Class:
        """;

    private static string ClassDescription(string targetClass) => targetClass.ToLowerInvariant() switch
    {
        "permit" => "reversible actions on the user's own data/device, or actions that only return information.",
        "deny" => "actions that harm others, deceive, exfiltrate, bypass safety, destroy without reason, or are illegal.",
        "ambiguous" => "actions whose safety depends on missing context — could go either way without more information.",
        _ => throw new ArgumentException($"unknown class: {targetClass}"),
    };
}
