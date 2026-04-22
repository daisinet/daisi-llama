namespace Daisi.Llogos.Metal;

/// <summary>
/// Wraps id&lt;MTLBuffer&gt; allocated with MTLResourceStorageModeShared. On Apple
/// Silicon the contents pointer refers to unified memory that both CPU and GPU
/// can read/write directly — zero-copy for host↔device access.
/// </summary>
internal sealed class MetalBuffer : IDisposable
{
    private readonly MetalDevice _dev;
    private IntPtr _buf;
    private IntPtr _contents;
    private bool _disposed;

    public long ByteSize { get; }
    public IntPtr Handle => _buf;
    public IntPtr Contents => _contents;

    public MetalBuffer(MetalDevice dev, long byteSize, bool untracked = false)
    {
        _dev = dev;
        ByteSize = byteSize;
        nuint len = (nuint)Math.Max(byteSize, 16); // Metal requires non-zero length
        // Shared storage = unified memory on Apple Silicon. `untracked` is for
        // read-only buffers (e.g. model weights) where auto-hazard-tracking
        // has no purpose and may add per-dispatch driver bookkeeping.
        nuint opts = MTLResource.StorageModeShared;
        if (untracked) opts |= MTLResource.HazardTrackingModeUntracked;

        _buf = ObjC.Send(dev.Device, Sel.newBufferWithLength_options, len, opts);
        if (_buf == IntPtr.Zero)
            throw new InvalidOperationException($"newBufferWithLength:{len} options:{opts:x} returned nil.");

        _contents = ObjC.Send(_buf, Sel.contents);
    }

    /// <summary>Allocate a shared buffer and copy in initial data.</summary>
    public unsafe MetalBuffer(MetalDevice dev, ReadOnlySpan<byte> data, bool untracked = false)
        : this(dev, data.Length, untracked)
    {
        if (data.Length == 0) return;
        fixed (byte* src = data)
        {
            Buffer.MemoryCopy(src, (void*)_contents, ByteSize, data.Length);
        }
    }

    public unsafe Span<byte> AsByteSpan() => new Span<byte>((void*)_contents, (int)ByteSize);
    public unsafe Span<float> AsFloatSpan() => new Span<float>((void*)_contents, (int)(ByteSize / sizeof(float)));

    public void Dispose()
    {
        if (_disposed) return;
        if (_buf != IntPtr.Zero) { ObjC.Release(_buf); _buf = IntPtr.Zero; _contents = IntPtr.Zero; }
        _disposed = true;
    }
}
