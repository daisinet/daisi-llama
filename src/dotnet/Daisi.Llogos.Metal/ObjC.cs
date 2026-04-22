using System.Runtime.InteropServices;

namespace Daisi.Llogos.Metal;

/// <summary>
/// P/Invoke bindings to the Objective-C runtime and Metal framework.
/// Apple Silicon arm64 uses regular C calling convention for objc_msgSend
/// (unlike x86_64 which needs objc_msgSend_stret for struct returns).
/// </summary>
internal static unsafe class ObjC
{
    private const string ObjCLib = "/usr/lib/libobjc.A.dylib";
    private const string MetalFw = "/System/Library/Frameworks/Metal.framework/Metal";
    private const string FoundationFw = "/System/Library/Frameworks/Foundation.framework/Foundation";

    // ── Class/Selector lookup ─────────────────────────────────────────────

    [DllImport(ObjCLib, EntryPoint = "objc_getClass", CharSet = CharSet.Ansi)]
    public static extern IntPtr objc_getClass([MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport(ObjCLib, EntryPoint = "sel_registerName", CharSet = CharSet.Ansi)]
    public static extern IntPtr sel_registerName([MarshalAs(UnmanagedType.LPStr)] string name);

    // ── objc_msgSend overloads ────────────────────────────────────────────
    // arm64: single entry point, C calling convention, variable args per signature.

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a, IntPtr b);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a, IntPtr b, IntPtr c);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, nuint a);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, nuint a, nuint b);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, void* a, nuint b, nuint c);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a, nuint b, IntPtr* error);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a, IntPtr b, IntPtr* error);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a, nuint b, nuint c);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern IntPtr Send(IntPtr receiver, IntPtr sel, IntPtr a, nuint b);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, IntPtr a);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, IntPtr a, IntPtr b);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, IntPtr a, nuint b, nuint c);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, void* a, nuint b, nuint c);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, MTLSize a, MTLSize b);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, IntPtr a, MTLSize b, MTLSize c);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, nuint a);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoid(IntPtr receiver, IntPtr sel, uint a);

    // Returns a double (MTLCommandBuffer.GPUStartTime / GPUEndTime, CFTimeInterval).
    // On arm64 Apple's ABI returns doubles in d0, accessible via regular objc_msgSend.
    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern double SendDouble(IntPtr receiver, IntPtr sel);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SendVoidIp(IntPtr receiver, IntPtr sel, IntPtr a, nuint b);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SetBytes(IntPtr receiver, IntPtr sel, void* bytes, nuint length, nuint idx);

    [DllImport(ObjCLib, EntryPoint = "objc_msgSend")]
    public static extern void SetBuffer(IntPtr receiver, IntPtr sel, IntPtr buffer, nuint offset, nuint idx);

    // ── Object lifecycle ──────────────────────────────────────────────────

    public static void Release(IntPtr obj) { if (obj != IntPtr.Zero) SendVoid(obj, Sel.release); }
    public static IntPtr Retain(IntPtr obj) => obj == IntPtr.Zero ? IntPtr.Zero : Send(obj, Sel.retain);

    // ── Metal framework entrypoints ───────────────────────────────────────

    [DllImport(MetalFw, EntryPoint = "MTLCreateSystemDefaultDevice")]
    public static extern IntPtr MTLCreateSystemDefaultDevice();

    // CFAbsoluteTimeGetCurrent is tied to NSDate (1 Jan 2001 epoch, seconds).
    // Not what we want. For GPU-time correlation we want CACurrentMediaTime()
    // which reads mach_absolute_time (ns resolution, monotonic) converted to
    // seconds — the same clock GPUStartTime/GPUEndTime use.
    [DllImport("/System/Library/Frameworks/QuartzCore.framework/QuartzCore",
        EntryPoint = "CACurrentMediaTime")]
    public static extern double CACurrentMediaTime();

    // ── Foundation helpers ────────────────────────────────────────────────

    public static IntPtr NSStringUtf8(string s)
    {
        // [NSString stringWithUTF8String:]
        var cls = objc_getClass("NSString");
        var sel = sel_registerName("stringWithUTF8String:");
        var bytes = System.Text.Encoding.UTF8.GetBytes(s + "\0");
        fixed (byte* p = bytes)
        {
            return Send(cls, sel, (IntPtr)p);
        }
    }

    public static string? NSStringToManaged(IntPtr nsString)
    {
        if (nsString == IntPtr.Zero) return null;
        var sel = sel_registerName("UTF8String");
        var cstr = Send(nsString, sel);
        if (cstr == IntPtr.Zero) return null;
        return Marshal.PtrToStringUTF8(cstr);
    }
}

/// <summary>Cached selectors — avoid per-call sel_registerName lookups.</summary>
internal static class Sel
{
    public static readonly IntPtr retain = ObjC.sel_registerName("retain");
    public static readonly IntPtr release = ObjC.sel_registerName("release");
    public static readonly IntPtr alloc = ObjC.sel_registerName("alloc");
    public static readonly IntPtr init = ObjC.sel_registerName("init");
    public static readonly IntPtr name = ObjC.sel_registerName("name");

    // MTLDevice
    public static readonly IntPtr newCommandQueue = ObjC.sel_registerName("newCommandQueue");
    public static readonly IntPtr newBufferWithLength_options = ObjC.sel_registerName("newBufferWithLength:options:");
    public static readonly IntPtr newBufferWithBytes_length_options = ObjC.sel_registerName("newBufferWithBytes:length:options:");
    public static readonly IntPtr newBufferWithBytesNoCopy_length_options_deallocator = ObjC.sel_registerName("newBufferWithBytesNoCopy:length:options:deallocator:");
    public static readonly IntPtr newLibraryWithSource_options_error = ObjC.sel_registerName("newLibraryWithSource:options:error:");
    public static readonly IntPtr newComputePipelineStateWithFunction_error = ObjC.sel_registerName("newComputePipelineStateWithFunction:error:");
    public static readonly IntPtr supportsFamily = ObjC.sel_registerName("supportsFamily:");
    public static readonly IntPtr recommendedMaxWorkingSetSize = ObjC.sel_registerName("recommendedMaxWorkingSetSize");
    public static readonly IntPtr maxThreadsPerThreadgroup = ObjC.sel_registerName("maxThreadsPerThreadgroup");

    // MTLLibrary
    public static readonly IntPtr newFunctionWithName = ObjC.sel_registerName("newFunctionWithName:");

    // MTLBuffer
    public static readonly IntPtr contents = ObjC.sel_registerName("contents");
    public static readonly IntPtr length = ObjC.sel_registerName("length");

    // MTLCommandQueue
    public static readonly IntPtr commandBuffer = ObjC.sel_registerName("commandBuffer");

    // MTLCommandBuffer
    public static readonly IntPtr computeCommandEncoder = ObjC.sel_registerName("computeCommandEncoder");
    public static readonly IntPtr blitCommandEncoder = ObjC.sel_registerName("blitCommandEncoder");
    public static readonly IntPtr commit = ObjC.sel_registerName("commit");
    public static readonly IntPtr waitUntilCompleted = ObjC.sel_registerName("waitUntilCompleted");
    public static readonly IntPtr error = ObjC.sel_registerName("error");
    public static readonly IntPtr status = ObjC.sel_registerName("status");
    public static readonly IntPtr GPUStartTime = ObjC.sel_registerName("GPUStartTime");
    public static readonly IntPtr GPUEndTime = ObjC.sel_registerName("GPUEndTime");
    public static readonly IntPtr kernelStartTime = ObjC.sel_registerName("kernelStartTime");
    public static readonly IntPtr kernelEndTime = ObjC.sel_registerName("kernelEndTime");
    public static readonly IntPtr enqueue = ObjC.sel_registerName("enqueue");

    // MTLComputeCommandEncoder
    public static readonly IntPtr setComputePipelineState = ObjC.sel_registerName("setComputePipelineState:");
    public static readonly IntPtr setBuffer_offset_atIndex = ObjC.sel_registerName("setBuffer:offset:atIndex:");
    public static readonly IntPtr setBytes_length_atIndex = ObjC.sel_registerName("setBytes:length:atIndex:");
    public static readonly IntPtr dispatchThreadgroups_threadsPerThreadgroup = ObjC.sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:");
    public static readonly IntPtr dispatchThreads_threadsPerThreadgroup = ObjC.sel_registerName("dispatchThreads:threadsPerThreadgroup:");
    public static readonly IntPtr endEncoding = ObjC.sel_registerName("endEncoding");
    public static readonly IntPtr memoryBarrierWithScope = ObjC.sel_registerName("memoryBarrierWithScope:");

    // MTLBlitCommandEncoder
    public static readonly IntPtr copyFromBuffer = ObjC.sel_registerName("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:");
    public static readonly IntPtr fillBuffer_range_value = ObjC.sel_registerName("fillBuffer:range:value:");

    // NSError
    public static readonly IntPtr localizedDescription = ObjC.sel_registerName("localizedDescription");
}

// MTLSize — 3 nuints (width, height, depth). arm64: passed in registers.
[StructLayout(LayoutKind.Sequential)]
internal struct MTLSize
{
    public nuint width;
    public nuint height;
    public nuint depth;

    public MTLSize(nuint w, nuint h, nuint d) { width = w; height = h; depth = d; }
}

// MTLResourceOptions bits (we only care about storage mode).
internal static class MTLResource
{
    public const nuint StorageModeShared = 0 << 4;    // unified memory on Apple Silicon
    public const nuint StorageModeManaged = 1 << 4;   // Intel Macs
    public const nuint StorageModePrivate = 2 << 4;
    public const nuint HazardTrackingModeUntracked = 1 << 8;
}

// MTLCommandBufferStatus
internal enum MTLCommandBufferStatus : ulong
{
    NotEnqueued = 0,
    Enqueued = 1,
    Committed = 2,
    Scheduled = 3,
    Completed = 4,
    Error = 5,
}
