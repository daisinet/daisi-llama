using System.Text;

namespace Daisi.Llogos.Metal;

/// <summary>
/// Wraps id&lt;MTLDevice&gt; and its default command queue. The device is discovered
/// via MTLCreateSystemDefaultDevice(). On Apple Silicon this is the unified-memory
/// GPU; on Intel Macs it's the integrated/discrete GPU the system picks.
/// </summary>
public sealed class MetalDevice : IDisposable
{
    internal IntPtr Device { get; private set; }
    internal IntPtr CommandQueue { get; private set; }
    internal IntPtr DefaultLibrary { get; private set; }
    public string DeviceName { get; }
    public bool IsAppleSilicon { get; }
    private bool _disposed;

    public MetalDevice()
    {
        Device = ObjC.MTLCreateSystemDefaultDevice();
        if (Device == IntPtr.Zero)
            throw new InvalidOperationException("MTLCreateSystemDefaultDevice returned nil — no Metal-capable GPU available.");

        CommandQueue = ObjC.Send(Device, Sel.newCommandQueue);
        if (CommandQueue == IntPtr.Zero)
            throw new InvalidOperationException("Failed to create MTLCommandQueue.");

        DeviceName = ObjC.NSStringToManaged(ObjC.Send(Device, Sel.name)) ?? "Metal device";

        // MTLGPUFamilyApple1 = 1001; any Apple family ≥ 1 means unified memory.
        IsAppleSilicon = ObjC.Send(Device, Sel.supportsFamily, (nuint)1001) != IntPtr.Zero;

        DefaultLibrary = IntPtr.Zero; // Loaded lazily once kernel source is compiled
    }

    /// <summary>
    /// Compile all MSL source (embedded resource) into a single MTLLibrary.
    /// Called lazily by MetalBackend after construction so we only pay the cost once.
    /// </summary>
    internal void LoadLibrary(string source)
    {
        if (DefaultLibrary != IntPtr.Zero) return;

        IntPtr nsSource = ObjC.NSStringUtf8(source);
        IntPtr err = IntPtr.Zero;
        IntPtr lib;
        unsafe
        {
            // [device newLibraryWithSource:source options:nil error:&err]
            lib = ObjC.Send(Device, Sel.newLibraryWithSource_options_error, nsSource, IntPtr.Zero, &err);
        }
        ObjC.Release(nsSource);

        if (lib == IntPtr.Zero)
        {
            string msg = err != IntPtr.Zero
                ? ObjC.NSStringToManaged(ObjC.Send(err, Sel.localizedDescription)) ?? "unknown MSL compile error"
                : "nil library returned from newLibraryWithSource:";
            throw new InvalidOperationException($"Metal shader compilation failed: {msg}");
        }
        DefaultLibrary = lib;
    }

    /// <summary>Create a compute pipeline for a named MSL function.</summary>
    internal IntPtr NewComputePipeline(string functionName)
    {
        if (DefaultLibrary == IntPtr.Zero)
            throw new InvalidOperationException("Shader library not loaded yet.");

        IntPtr nsName = ObjC.NSStringUtf8(functionName);
        IntPtr fn = ObjC.Send(DefaultLibrary, Sel.newFunctionWithName, nsName);
        ObjC.Release(nsName);
        if (fn == IntPtr.Zero)
            throw new InvalidOperationException($"MSL function '{functionName}' not found in library.");

        IntPtr err = IntPtr.Zero;
        IntPtr pso;
        unsafe
        {
            pso = ObjC.Send(Device, Sel.newComputePipelineStateWithFunction_error, fn, IntPtr.Zero, &err);
        }
        ObjC.Release(fn);

        if (pso == IntPtr.Zero)
        {
            string msg = err != IntPtr.Zero
                ? ObjC.NSStringToManaged(ObjC.Send(err, Sel.localizedDescription)) ?? "pipeline creation failed"
                : "nil pipeline";
            throw new InvalidOperationException($"Compute pipeline '{functionName}' failed: {msg}");
        }
        return pso;
    }

    public void Dispose()
    {
        if (_disposed) return;
        if (DefaultLibrary != IntPtr.Zero) { ObjC.Release(DefaultLibrary); DefaultLibrary = IntPtr.Zero; }
        if (CommandQueue != IntPtr.Zero) { ObjC.Release(CommandQueue); CommandQueue = IntPtr.Zero; }
        if (Device != IntPtr.Zero) { ObjC.Release(Device); Device = IntPtr.Zero; }
        _disposed = true;
    }
}
