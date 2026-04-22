using System.Diagnostics;
using System.Runtime.InteropServices;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Metal;

/// <summary>
/// Entry-point helper so we can micro-benchmark the Metal dispatch path from
/// the CLI. Not referenced from MetalBackend itself.
/// </summary>
public static class MetalBench
{
    public static void Run()
    {
        using var b = new MetalBackend();

        // ── Cross-dispatch correctness test (write then read same buffer) ──
        // Chain of 10 element_add ops writing the same buffer.
        //   acc[i] = 0
        //   for step in 0..9: acc = acc + ones  →  acc should be 10
        {
            int Nc = 1024;
            using var acc = (MetalTensor)b.CreateTensor("acc", GgmlType.F32, new long[] { Nc });
            using var ones = (MetalTensor)b.CreateTensor("ones", GgmlType.F32, new long[] { Nc });
            var os = ones.AsFloatSpan();
            for (int i = 0; i < Nc; i++) os[i] = 1.0f;
            b.ZeroTensor(acc);
            for (int step = 0; step < 10; step++) b.ElementAdd(acc, acc, ones);
            b.FlushCommands();
            var result = new float[Nc];
            acc.AsFloatSpan().CopyTo(result);
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < Nc; i++) { if (result[i] < min) min = result[i]; if (result[i] > max) max = result[i]; }
            Console.WriteLine($"[chain-test] 10 chained ElementAdd, acc should be 10.0 — min={min}, max={max}");
        }

        // ── Differential test: batched vs per-op should match bit-exact ─────
        {
            using var bCpu = new Daisi.Llogos.Cpu.CpuBackend();
            int Kx = 4096, Nx = 4096;
            var rng2 = new Random(123);
            var wBytes = new byte[(Nx * Kx / 32) * 18];
            rng2.NextBytes(wBytes);
            var inBytes = new byte[Kx * 4];
            var inFlo = MemoryMarshal.Cast<byte, float>((Span<byte>)inBytes);
            for (int i = 0; i < Kx; i++) inFlo[i] = (float)(rng2.NextDouble() - 0.5);

            // Per-op reference
            Environment.SetEnvironmentVariable("DAISI_METAL_BATCH", "0");
            using var bRef = new MetalBackend();
            using var wRef = (MetalTensor)bRef.LoadTensor("w", GgmlType.Q4_0, new long[] { Kx, Nx }, wBytes);
            using var inRef = (MetalTensor)bRef.LoadTensor("i", GgmlType.F32, new long[] { Kx }, inBytes);
            using var oRef = (MetalTensor)bRef.CreateTensor("o", GgmlType.F32, new long[] { Nx });
            using var normW = (MetalTensor)bRef.CreateTensor("nw", GgmlType.F32, new long[] { Kx });
            for (int i = 0; i < Kx; i++) normW.AsFloatSpan()[i] = 1.0f;
            using var normOut = (MetalTensor)bRef.CreateTensor("no", GgmlType.F32, new long[] { Kx });

            // Realistic attention-layer chain: rmsnorm → matmul → per-head-rmsnorm → rope
            bRef.RmsNorm(normOut, inRef, normW, 1e-6f);
            bRef.MatMul(oRef, normOut, wRef, 1, Kx, Nx);
            bRef.PerHeadRmsNorm(oRef, normW, 32, 128, 1e-6f);
            // Note: we pass oRef as both q and k to exercise a single buffer
            bRef.RoPE(oRef, oRef, 128, 128, 5, 10000.0f);
            bRef.FlushCommands();
            var refResult = new float[Nx];
            oRef.AsFloatSpan().CopyTo(refResult);

            // Batched
            Environment.SetEnvironmentVariable("DAISI_METAL_BATCH", "1");
            using var bBatch = new MetalBackend();
            using var wB = (MetalTensor)bBatch.LoadTensor("w", GgmlType.Q4_0, new long[] { Kx, Nx }, wBytes);
            using var inB = (MetalTensor)bBatch.LoadTensor("i", GgmlType.F32, new long[] { Kx }, inBytes);
            using var oB = (MetalTensor)bBatch.CreateTensor("o", GgmlType.F32, new long[] { Nx });
            using var nwB = (MetalTensor)bBatch.CreateTensor("nw", GgmlType.F32, new long[] { Kx });
            for (int i = 0; i < Kx; i++) nwB.AsFloatSpan()[i] = 1.0f;
            using var noB = (MetalTensor)bBatch.CreateTensor("no", GgmlType.F32, new long[] { Kx });

            bBatch.RmsNorm(noB, inB, nwB, 1e-6f);
            bBatch.MatMul(oB, noB, wB, 1, Kx, Nx);
            bBatch.PerHeadRmsNorm(oB, nwB, 32, 128, 1e-6f);
            bBatch.RoPE(oB, oB, 128, 128, 5, 10000.0f);
            bBatch.FlushCommands();
            var batchResult = new float[Nx];
            oB.AsFloatSpan().CopyTo(batchResult);

            int diffs = 0;
            float maxDiff = 0;
            for (int i = 0; i < Nx; i++)
            {
                float d = MathF.Abs(refResult[i] - batchResult[i]);
                if (d > 1e-3f) diffs++;
                if (d > maxDiff) maxDiff = d;
            }
            Console.WriteLine($"[diff-test] rmsnorm→matmul perOp vs batched: diffs>1e-3 = {diffs}/{Nx}, maxDiff = {maxDiff}");
        }

        // ── Extended diff-test: simulate first ~12 ops of a Qwen3.5 attention layer ──
        // Goal: find the exact op index where batched diverges from per-op.
        // Chain: rmsnormresidual → 3 matmuls → de_interleave_q → 2 per_head_rmsnorm → rope → kv_write → …
        {
            int Hx = 4096;
            int NumHeads = 32;
            int NumKv = 4;
            int KeyLen = 128;
            int ValLen = 128;
            int QFullDim = NumHeads * KeyLen * 2; // interleaved attn+gate
            int QAttnDim = NumHeads * KeyLen;
            int KDim = NumKv * KeyLen;
            int VDim = NumKv * ValLen;
            int MaxSeq = 32;

            var rngX = new Random(77);
            byte[] MakeQ4(int K, int N) { var w = new byte[(K * N / 32) * 18]; rngX.NextBytes(w); return w; }
            byte[] MakeF32(int n) { var b2 = new byte[n * 4]; var s = MemoryMarshal.Cast<byte, float>((Span<byte>)b2); for (int i = 0; i < n; i++) s[i] = (float)(rngX.NextDouble() - 0.5); return b2; }

            var hiddenInit = MakeF32(Hx);
            var attnNormW = MakeF32(Hx);   // replaces the per-layer norm weight
            var qNormW = MakeF32(KeyLen);
            var kNormW = MakeF32(KeyLen);
            var wQ = MakeQ4(Hx, NumHeads * KeyLen * 2);  // gated Q: 2× dim
            var wK = MakeQ4(Hx, NumKv * KeyLen);
            var wV = MakeQ4(Hx, NumKv * ValLen);

            float[] RunChain(bool batched, int opsToDo)
            {
                Environment.SetEnvironmentVariable("DAISI_METAL_BATCH", batched ? "1" : "0");
                using var backendImpl = new MetalBackend();
                Daisi.Llogos.IComputeBackend backend = backendImpl;
                var hidden = (MetalTensor)backend.LoadTensor("hidden", GgmlType.F32, new long[] { Hx }, hiddenInit);
                var residual = (MetalTensor)backend.CreateTensor("res", GgmlType.F32, new long[] { Hx });
                var normOut = (MetalTensor)backend.CreateTensor("norm", GgmlType.F32, new long[] { Hx });
                var attnNorm = (MetalTensor)backend.LoadTensor("attnNorm", GgmlType.F32, new long[] { Hx }, attnNormW);
                var qNorm = (MetalTensor)backend.LoadTensor("qN", GgmlType.F32, new long[] { KeyLen }, qNormW);
                var kNorm = (MetalTensor)backend.LoadTensor("kN", GgmlType.F32, new long[] { KeyLen }, kNormW);
                var qW = (MetalTensor)backend.LoadTensor("wQ", GgmlType.Q4_0, new long[] { Hx, NumHeads * KeyLen * 2 }, wQ);
                var kW = (MetalTensor)backend.LoadTensor("wK", GgmlType.Q4_0, new long[] { Hx, KDim }, wK);
                var vW = (MetalTensor)backend.LoadTensor("wV", GgmlType.Q4_0, new long[] { Hx, VDim }, wV);

                var qFull = (MetalTensor)backend.CreateTensor("qFull", GgmlType.F32, new long[] { QFullDim });
                var qAttn = (MetalTensor)backend.CreateTensor("qAttn", GgmlType.F32, new long[] { QAttnDim });
                var qGate = (MetalTensor)backend.CreateTensor("qGate", GgmlType.F32, new long[] { QAttnDim });
                var kProj = (MetalTensor)backend.CreateTensor("kP", GgmlType.F32, new long[] { KDim });
                var vProj = (MetalTensor)backend.CreateTensor("vP", GgmlType.F32, new long[] { VDim });
                var kCache = (MetalTensor)backend.CreateTensor("kC", GgmlType.F32, new long[] { NumKv * MaxSeq * KeyLen });
                var vCache = (MetalTensor)backend.CreateTensor("vC", GgmlType.F32, new long[] { NumKv * MaxSeq * ValLen });

                backend.BeginCommands();
                int op = 0;
                if (opsToDo > op++) backend.RmsNormResidual(normOut, residual, hidden, attnNorm, 1e-6f);
                if (opsToDo > op++) backend.MatMul(kProj, normOut, kW, 1, Hx, KDim);
                if (opsToDo > op++) backend.MatMul(vProj, normOut, vW, 1, Hx, VDim);
                if (opsToDo > op++) backend.MatMul(qFull, normOut, qW, 1, Hx, QFullDim);
                if (opsToDo > op++) backend.DeInterleaveQ(qAttn, qGate, qFull, NumHeads, KeyLen);
                if (opsToDo > op++) backend.PerHeadRmsNorm(qAttn, qNorm, NumHeads, KeyLen, 1e-6f);
                if (opsToDo > op++) backend.PerHeadRmsNorm(kProj, kNorm, NumKv, KeyLen, 1e-6f);
                if (opsToDo > op++) backend.RoPE(qAttn, kProj, KeyLen, KeyLen, 5, 10000.0f);
                if (opsToDo > op++) backend.KvCacheWrite(kCache, vCache, kProj, vProj, NumKv, KeyLen, ValLen, MaxSeq, 0);
                backend.FlushCommands();

                // Return concatenation of all interesting output states for comparison
                var outBuf = new float[QAttnDim + QAttnDim + KDim + VDim + NumKv * MaxSeq * KeyLen];
                int off = 0;
                qAttn.AsFloatSpan().CopyTo(outBuf.AsSpan(off, QAttnDim)); off += QAttnDim;
                qGate.AsFloatSpan().CopyTo(outBuf.AsSpan(off, QAttnDim)); off += QAttnDim;
                kProj.AsFloatSpan().CopyTo(outBuf.AsSpan(off, KDim)); off += KDim;
                vProj.AsFloatSpan().CopyTo(outBuf.AsSpan(off, VDim)); off += VDim;
                kCache.AsFloatSpan().CopyTo(outBuf.AsSpan(off, NumKv * MaxSeq * KeyLen));

                hidden.Dispose(); residual.Dispose(); normOut.Dispose();
                attnNorm.Dispose(); qNorm.Dispose(); kNorm.Dispose();
                qW.Dispose(); kW.Dispose(); vW.Dispose();
                qFull.Dispose(); qAttn.Dispose(); qGate.Dispose();
                kProj.Dispose(); vProj.Dispose(); kCache.Dispose(); vCache.Dispose();
                return outBuf;
            }

            // Binary search across opsToDo: 1..9
            string[] opNames = {
                "rmsnormresidual", "matmul(k)", "matmul(v)", "matmul(qFull)",
                "de_interleave_q", "perHeadRms(q)", "perHeadRms(k)", "rope", "kv_write"
            };
            for (int nOps = 1; nOps <= 9; nOps++)
            {
                var refBuf = RunChain(false, nOps);
                var batchBuf = RunChain(true, nOps);
                int diffs = 0; float maxDiff = 0;
                for (int i = 0; i < refBuf.Length; i++)
                {
                    float d = MathF.Abs(refBuf[i] - batchBuf[i]);
                    if (d > 1e-3f) diffs++;
                    if (d > maxDiff) maxDiff = d;
                }
                Console.WriteLine($"[attn-diff] nOps={nOps} last={opNames[nOps - 1],-22} diffs>1e-3 = {diffs,6}/{refBuf.Length,-6}, maxDiff = {maxDiff:G3}");
            }
        }

        // Tensor sizes typical of decode-path ops (4096-dim).
        int N = 4096;
        using var x = (MetalTensor)b.CreateTensor("x", GgmlType.F32, new long[] { N });
        using var y = (MetalTensor)b.CreateTensor("y", GgmlType.F32, new long[] { N });
        using var z = (MetalTensor)b.CreateTensor("z", GgmlType.F32, new long[] { N });
        var xs = x.AsFloatSpan();
        var ys = y.AsFloatSpan();
        for (int i = 0; i < N; i++) { xs[i] = 1.0f; ys[i] = 2.0f; }

        const int warmup = 100;
        const int iters = 1000;

        // Warm-up.
        for (int i = 0; i < warmup; i++) b.ElementAdd(z, x, y);
        b.FlushCommands();

        // ── Per-op (commit+wait every call) ─────────────────────────────────
        Environment.SetEnvironmentVariable("_BENCH_NOBATCH", "1");
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
        {
            b.ElementAdd(z, x, y);
            b.FlushCommands(); // commit+wait per op
        }
        sw.Stop();
        Console.WriteLine($"[per-op]   {iters} ElementAdd + flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / iters:F3} ms/op");

        // ── Batched (commit once per N ops) ────────────────────────────────
        sw.Restart();
        for (int i = 0; i < iters; i++) b.ElementAdd(z, x, y);
        b.FlushCommands();
        sw.Stop();
        Console.WriteLine($"[batched]  {iters} ElementAdd, 1 flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / iters:F3} ms/op");

        // ── Q4_0 matmul (4096 × 4096 = attn/ffn projection size) ────────────
        const int K = 4096, Nmat = 4096;
        var rng = new Random(42);
        var w = new byte[(Nmat * K / 32) * 18];
        rng.NextBytes(w);
        using var weight = (MetalTensor)b.LoadTensor("w", GgmlType.Q4_0, new long[] { K, Nmat }, w);
        using var inp = (MetalTensor)b.CreateTensor("inp", GgmlType.F32, new long[] { K });
        using var outp = (MetalTensor)b.CreateTensor("out", GgmlType.F32, new long[] { Nmat });
        var ispan = inp.AsFloatSpan();
        for (int i = 0; i < K; i++) ispan[i] = (float)(rng.NextDouble() - 0.5);

        // Warm
        for (int i = 0; i < 50; i++) b.MatMul(outp, inp, weight, 1, K, Nmat);
        b.FlushCommands();

        const int mmIters = 200;
        sw.Restart();
        for (int i = 0; i < mmIters; i++) { b.MatMul(outp, inp, weight, 1, K, Nmat); b.FlushCommands(); }
        sw.Stop();
        Console.WriteLine($"[per-op]   {mmIters} Q4_0 matmul 4096x4096 + flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / mmIters:F3} ms/op");

        sw.Restart();
        for (int i = 0; i < mmIters; i++) b.MatMul(outp, inp, weight, 1, K, Nmat);
        b.FlushCommands();
        sw.Stop();
        Console.WriteLine($"[batched]  {mmIters} Q4_0 matmul 4096x4096, 1 flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / mmIters:F3} ms/op");

        // RmsNorm
        using var rmsOut = (MetalTensor)b.CreateTensor("rms", GgmlType.F32, new long[] { K });
        using var rmsW = (MetalTensor)b.CreateTensor("rmsW", GgmlType.F32, new long[] { K });
        for (int i = 0; i < K; i++) rmsW.AsFloatSpan()[i] = 1.0f;
        for (int i = 0; i < 50; i++) b.RmsNorm(rmsOut, inp, rmsW, 1e-6f);
        b.FlushCommands();

        sw.Restart();
        for (int i = 0; i < iters; i++) { b.RmsNorm(rmsOut, inp, rmsW, 1e-6f); b.FlushCommands(); }
        sw.Stop();
        Console.WriteLine($"[per-op]   {iters} RmsNorm 4096 + flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / iters:F3} ms/op");

        sw.Restart();
        for (int i = 0; i < iters; i++) b.RmsNorm(rmsOut, inp, rmsW, 1e-6f);
        b.FlushCommands();
        sw.Stop();
        Console.WriteLine($"[batched]  {iters} RmsNorm 4096, 1 flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / iters:F3} ms/op");

        // Interleaved: matmul, rmsnorm, element_add — mimics real forward pass pattern.
        for (int i = 0; i < 50; i++)
        {
            b.MatMul(outp, inp, weight, 1, K, Nmat);
            b.RmsNorm(rmsOut, inp, rmsW, 1e-6f);
            b.ElementAdd(z, x, y);
        }
        b.FlushCommands();

        const int iIters = 200;
        sw.Restart();
        for (int i = 0; i < iIters; i++)
        {
            b.MatMul(outp, inp, weight, 1, K, Nmat);
            b.FlushCommands();
            b.RmsNorm(rmsOut, inp, rmsW, 1e-6f);
            b.FlushCommands();
            b.ElementAdd(z, x, y);
            b.FlushCommands();
        }
        sw.Stop();
        Console.WriteLine($"[per-op interleaved]  {iIters * 3} ops (mm+rms+add) with flush each: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / (iIters * 3):F3} ms/op");

        sw.Restart();
        for (int i = 0; i < iIters; i++)
        {
            b.MatMul(outp, inp, weight, 1, K, Nmat);
            b.RmsNorm(rmsOut, inp, rmsW, 1e-6f);
            b.ElementAdd(z, x, y);
        }
        b.FlushCommands();
        sw.Stop();
        Console.WriteLine($"[batched interleaved] {iIters * 3} ops (mm+rms+add) 1 flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / (iIters * 3):F3} ms/op");

        // Simulate multi-layer access: rotate through 64 distinct weight tensors to
        // evict L2 cache each iteration. Mimics decode reading ALL model weights.
        const int NumLayers = 64;
        var weightsPool = new MetalTensor[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            rng.NextBytes(w);
            weightsPool[i] = (MetalTensor)b.LoadTensor($"w{i}", GgmlType.Q4_0, new long[] { K, Nmat }, w);
        }

        for (int i = 0; i < 10; i++) b.MatMul(outp, inp, weightsPool[0], 1, K, Nmat);
        b.FlushCommands();

        const int rotIters = 200;
        sw.Restart();
        for (int i = 0; i < rotIters; i++)
        {
            b.MatMul(outp, inp, weightsPool[i % NumLayers], 1, K, Nmat);
            b.FlushCommands();
        }
        sw.Stop();
        Console.WriteLine($"[per-op rotating weights] {rotIters} Q4_0 matmul (cache-evicting): {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / rotIters:F3} ms/op");

        sw.Restart();
        for (int i = 0; i < rotIters; i++)
        {
            b.MatMul(outp, inp, weightsPool[i % NumLayers], 1, K, Nmat);
        }
        b.FlushCommands();
        sw.Stop();
        Console.WriteLine($"[batched rotating weights] {rotIters} Q4_0 matmul 1 flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / rotIters:F3} ms/op");

        // With CPU-side work between dispatches (simulates .NET / p-invoke overhead).
        sw.Restart();
        for (int i = 0; i < rotIters; i++)
        {
            b.MatMul(outp, inp, weightsPool[i % NumLayers], 1, K, Nmat);
            b.FlushCommands();
            // simulate some CPU work
            double x2 = 0;
            for (int j = 0; j < 100; j++) x2 += Math.Sqrt(j + i);
            if (x2 < 0) Console.Write("");
        }
        sw.Stop();
        Console.WriteLine($"[per-op rotating + CPU work] {rotIters} Q4_0 matmul: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / rotIters:F3} ms/op");

        // With a long CPU sleep between dispatches (exposes GPU idle wake-up).
        sw.Restart();
        for (int i = 0; i < 50; i++)
        {
            b.MatMul(outp, inp, weightsPool[i % NumLayers], 1, K, Nmat);
            b.FlushCommands();
            System.Threading.Thread.Sleep(1); // ensure GPU idles
        }
        sw.Stop();
        Console.WriteLine($"[per-op with 1ms sleep]   50 Q4_0 matmul: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / 50:F3} ms/op");

        foreach (var t in weightsPool) t.Dispose();

        // lm_head-sized matmul: 4096 × 152064 (vocab). Q4_0 weight = ~300 MB.
        const int Nvocab = 152064;
        var wvocab = new byte[(Nvocab * K / 32) * 18];
        rng.NextBytes(wvocab);
        using var wVocab = (MetalTensor)b.LoadTensor("wv", GgmlType.Q4_0, new long[] { K, Nvocab }, wvocab);
        using var logits = (MetalTensor)b.CreateTensor("lg", GgmlType.F32, new long[] { Nvocab });

        for (int i = 0; i < 5; i++) b.MatMul(logits, inp, wVocab, 1, K, Nvocab);
        b.FlushCommands();

        const int vIters = 20;
        sw.Restart();
        for (int i = 0; i < vIters; i++) { b.MatMul(logits, inp, wVocab, 1, K, Nvocab); b.FlushCommands(); }
        sw.Stop();
        Console.WriteLine($"[per-op]   {vIters} Q4_0 matmul 4096x{Nvocab} + flush: {sw.Elapsed.TotalMilliseconds:F1} ms total, {sw.Elapsed.TotalMilliseconds / vIters:F1} ms/op");
    }
}
