using System.Runtime.InteropServices;
using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Metal;

/// <summary>
/// GPU-visible tensor backed by a shared-storage MTLBuffer. On Apple Silicon the
/// buffer's contents live in unified memory, so CPU and GPU kernels read/write
/// the same physical bytes — no upload/download required.
/// </summary>
public sealed class MetalTensor : ITensor
{
    private readonly long[] _dimensions;
    private readonly MetalDevice _dev;
    internal MetalBuffer Buffer { get; private set; }
    internal bool IsAlignedQ8_0 { get; }
    internal bool IsAlignedQ4_0 { get; }
    internal bool IsAlignedQ6_K { get; }
    // When true, the backing MTLBuffer stores F16 values even though the
    // ITensor's declared Type is F32. Used for activation tensors in fp16
    // mode — halves memory bandwidth on activation-heavy paths.
    internal bool IsF16Backed { get; }
    private bool _disposed;

    internal MetalTensor(MetalDevice dev, string name, GgmlType type, ReadOnlySpan<long> dimensions)
        : this(dev, name, type, dimensions, f16Backed: false) { }

    internal MetalTensor(MetalDevice dev, string name, GgmlType type, ReadOnlySpan<long> dimensions, bool f16Backed)
    {
        _dev = dev;
        Name = name;
        Type = type;
        IsF16Backed = f16Backed && type == GgmlType.F32;
        _dimensions = dimensions.ToArray();
        ElementCount = ComputeElementCount(dimensions);
        // The declared Type is F32 but we allocate half-sized storage.
        // ByteSize returned to external callers still reflects the F32 size
        // so their accounting is unchanged; GPU kernels operating on this
        // tensor are dispatched via Metal-specific paths that know about
        // the half-precision backing and use half-sized offsets.
        ByteSize = (long)GgmlTypeInfo.ByteSize(type, (ulong)ElementCount);
        long bufferBytes = IsF16Backed ? ElementCount * 2 : ByteSize;
        Buffer = new MetalBuffer(dev, bufferBytes);
    }

    internal MetalTensor(MetalDevice dev, string name, GgmlType type, ReadOnlySpan<long> dimensions, ReadOnlySpan<byte> data,
        bool isAlignedQ8_0 = false, bool isAlignedQ4_0 = false, bool isAlignedQ6_K = false)
    {
        _dev = dev;
        Name = name;
        Type = type;
        IsAlignedQ8_0 = isAlignedQ8_0;
        IsAlignedQ4_0 = isAlignedQ4_0;
        IsAlignedQ6_K = isAlignedQ6_K;
        _dimensions = dimensions.ToArray();
        ElementCount = ComputeElementCount(dimensions);
        ByteSize = isAlignedQ8_0 ? (ElementCount / 32) * 36
                 : isAlignedQ4_0 ? (ElementCount / 32) * 20
                 : isAlignedQ6_K ? (ElementCount / 256) * 224
                 : (long)GgmlTypeInfo.ByteSize(type, (ulong)ElementCount);
        // LoadTensor-created tensors carry initial data — almost always model
        // weights that are read-only after load. Skip hazard tracking since
        // there are no writes to serialize against.
        Buffer = new MetalBuffer(dev, data.Slice(0, (int)ByteSize), untracked: true);
    }

    public string Name { get; }
    public GgmlType Type { get; }
    public ReadOnlySpan<long> Dimensions => _dimensions;
    public long ElementCount { get; }
    public long ByteSize { get; }

    /// <inheritdoc />
    public void CopyFrom(ReadOnlySpan<byte> data)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (data.Length != ByteSize)
            throw new ArgumentException($"Data length {data.Length} does not match tensor byte size {ByteSize}.");
        if (IsF16Backed)
        {
            // Incoming data is F32; convert to F16 into the half-sized buffer.
            var floats = MemoryMarshal.Cast<byte, float>(data);
            var halfSpan = MemoryMarshal.Cast<byte, Half>(Buffer.AsByteSpan().Slice(0, (int)(ElementCount * 2)));
            for (int i = 0; i < floats.Length; i++) halfSpan[i] = (Half)floats[i];
            return;
        }
        data.CopyTo(Buffer.AsByteSpan());
    }

    /// <inheritdoc />
    public void DequantizeTo(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length < ElementCount)
            throw new ArgumentException("Destination too small.");

        // Unified memory: read bytes directly out of the MTLBuffer, dequantize via CPU routines.
        // The Metal backend must have flushed any in-flight GPU writes before this is called.
        var bytes = Buffer.AsByteSpan();
        if (IsF16Backed)
        {
            var halves = MemoryMarshal.Cast<byte, Half>(bytes).Slice(0, (int)ElementCount);
            for (int i = 0; i < halves.Length; i++) destination[i] = (float)halves[i];
            return;
        }
        if (Type == GgmlType.F32)
        {
            MemoryMarshal.Cast<byte, float>(bytes).Slice(0, (int)ElementCount).CopyTo(destination);
            return;
        }

        if (IsAlignedQ8_0)
        {
            // Unpack 36→34 back into CPU layout for dequant
            int blocks = (int)(ElementCount / 32);
            var unpacked = new byte[blocks * 34];
            for (int i = 0; i < blocks; i++)
            {
                int src = i * 36;
                int dst = i * 34;
                unpacked[dst] = bytes[src];
                unpacked[dst + 1] = bytes[src + 1];
                bytes.Slice(src + 4, 32).CopyTo(unpacked.AsSpan(dst + 2, 32));
            }
            using var cpuTmp = new Cpu.CpuTensor(Name + "_tmp", Type, _dimensions, unpacked);
            cpuTmp.DequantizeTo(destination);
            return;
        }

        if (IsAlignedQ4_0)
        {
            int blocks = (int)(ElementCount / 32);
            var unpacked = new byte[blocks * 18];
            for (int i = 0; i < blocks; i++)
            {
                int src = i * 20;
                int dst = i * 18;
                unpacked[dst] = bytes[src];
                unpacked[dst + 1] = bytes[src + 1];
                bytes.Slice(src + 4, 16).CopyTo(unpacked.AsSpan(dst + 2, 16));
            }
            using var cpuTmp = new Cpu.CpuTensor(Name + "_tmp", Type, _dimensions, unpacked);
            cpuTmp.DequantizeTo(destination);
            return;
        }

        if (IsAlignedQ6_K)
        {
            // Reverse the repack back to canonical 210-byte Q6_K layout.
            int blocks = (int)(ElementCount / 256);
            var unpacked = new byte[blocks * 210];
            for (int i = 0; i < blocks; i++)
            {
                int src = i * 224;
                int dst = i * 210;
                // ql (128 bytes) at aligned[20..147] → src[0..127]
                bytes.Slice(src + 20, 128).CopyTo(unpacked.AsSpan(dst, 128));
                // qh (64 bytes) at aligned[148..211] → src[128..191]
                bytes.Slice(src + 148, 64).CopyTo(unpacked.AsSpan(dst + 128, 64));
                // scales at aligned[4..19] → src[192..207]
                bytes.Slice(src + 4, 16).CopyTo(unpacked.AsSpan(dst + 192, 16));
                // d at aligned[0..1] → src[208..209]
                unpacked[dst + 208] = bytes[src];
                unpacked[dst + 209] = bytes[src + 1];
            }
            using var cpuTmp = new Cpu.CpuTensor(Name + "_tmp", Type, _dimensions, unpacked);
            cpuTmp.DequantizeTo(destination);
            return;
        }

        using var cpuTensor = new Cpu.CpuTensor(Name + "_tmp", Type, _dimensions, bytes);
        cpuTensor.DequantizeTo(destination);
    }

    /// <summary>
    /// Direct float-span view of the buffer for F32 tensors (unified memory zero-copy).
    /// The backend must flush pending GPU work before the caller reads/writes this span.
    /// </summary>
    public Span<float> AsFloatSpan()
    {
        if (Type != GgmlType.F32)
            throw new InvalidOperationException($"AsFloatSpan only valid for F32 tensors (got {Type}).");
        if (IsF16Backed)
            throw new InvalidOperationException($"AsFloatSpan not supported on F16-backed tensor '{Name}'. Use DequantizeTo.");
        return Buffer.AsFloatSpan().Slice(0, (int)ElementCount);
    }

    /// <summary>
    /// Direct half-span view for F16-backed tensors. Caller must flush GPU first.
    /// </summary>
    internal Span<Half> AsHalfSpan()
    {
        if (!IsF16Backed)
            throw new InvalidOperationException("AsHalfSpan only valid on F16-backed tensors.");
        return MemoryMarshal.Cast<byte, Half>(Buffer.AsByteSpan()).Slice(0, (int)ElementCount);
    }

    /// <summary>Raw byte-span view. Must flush GPU work first.</summary>
    internal Span<byte> RawBytes() => Buffer.AsByteSpan().Slice(0, (int)ByteSize);

    public void CopyRawTo(Span<byte> destination)
    {
        Buffer.AsByteSpan().Slice(0, (int)ByteSize).CopyTo(destination);
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Buffer?.Dispose();
            _disposed = true;
        }
    }

    private static long ComputeElementCount(ReadOnlySpan<long> dims)
    {
        long count = 1;
        for (int i = 0; i < dims.Length; i++)
            count *= dims[i];
        return count;
    }
}
