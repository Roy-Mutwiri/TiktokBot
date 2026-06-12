/*
 * capture.c — RoyCam capture pin.
 *
 * Lifecycle:  Create -> SetDataFormat -> SetDeviceState(RUN) starts a 30fps
 * KTIMER whose DPC calls KsPinAttemptProcessing; Process fills each leading-edge
 * stream pointer with a frame (simulated in stage 1, service back buffer in
 * stage 3) and advances it.
 *
 * NOTE: this is a skeleton. It follows the documented AVStream contract but has
 * not yet been compiled against the WDK (the VS<->WDK toolset integration is
 * pending). Treat per-API details as needing a compile/SDV pass.
 */
#include "roycam.h"

/* ------------------------------------------------------------------ timer -- */
static KDEFERRED_ROUTINE RoycFrameDpc;

_Use_decl_annotations_
static VOID RoycFrameDpc(PKDPC Dpc, PVOID Context, PVOID Arg1, PVOID Arg2)
{
    PKSPIN pin = (PKSPIN)Context;
    UNREFERENCED_PARAMETER(Dpc);
    UNREFERENCED_PARAMETER(Arg1);
    UNREFERENCED_PARAMETER(Arg2);
    if (pin != NULL)
        KsPinAttemptProcessing(pin, TRUE);
}

static VOID RoycStartTimer(PROYC_PIN_CONTEXT ctx, PKSPIN pin)
{
    LARGE_INTEGER due;
    if (InterlockedExchange(&ctx->Running, 1) == 1)
        return;
    due.QuadPart = -ctx->FramePeriod100ns;   /* relative, first tick */
    KeInitializeDpc(&ctx->FrameDpc, RoycFrameDpc, pin);
    KeInitializeTimerEx(&ctx->FrameTimer, SynchronizationTimer);
    /* Period in ms; 33 ms ~ 30fps. */
    KeSetTimerEx(&ctx->FrameTimer, due,
                 (LONG)(ctx->FramePeriod100ns / 10000), &ctx->FrameDpc);
}

static VOID RoycStopTimer(PROYC_PIN_CONTEXT ctx)
{
    if (InterlockedExchange(&ctx->Running, 0) == 0)
        return;
    KeCancelTimer(&ctx->FrameTimer);
    KeFlushQueuedDpcs();
}

/* ---------------------------------------------------------------- helpers -- */
static VOID RoycApplyFormat(PROYC_PIN_CONTEXT ctx, const KS_VIDEOINFOHEADER *vih,
                            const GUID *subFormat)
{
    ctx->Format    = RoycFormatFromSubtype(subFormat);
    ctx->Width     = (ULONG)vih->bmiHeader.biWidth;
    ctx->Height    = (ULONG)((vih->bmiHeader.biHeight < 0)
                             ? -vih->bmiHeader.biHeight : vih->bmiHeader.biHeight);
    ctx->Stride    = RoycComputeStride(ctx->Format, ctx->Width);
    ctx->ImageSize = RoycComputeImageSize(ctx->Format, ctx->Width, ctx->Height);
    if (ctx->FramePeriod100ns == 0)
        ctx->FramePeriod100ns = (vih->AvgTimePerFrame != 0)
                                 ? vih->AvgTimePerFrame : 333333;
}

/* ----------------------------------------------------------------- create -- */
NTSTATUS RoycPinCreate(PKSPIN Pin, PIRP Irp)
{
    PROYC_PIN_CONTEXT ctx;
    UNREFERENCED_PARAMETER(Irp);

    ctx = (PROYC_PIN_CONTEXT)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(ROYC_PIN_CONTEXT), ROYC_POOL_TAG);
    if (ctx == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;
    RtlZeroMemory(ctx, sizeof(*ctx));
    KeInitializeSpinLock(&ctx->BackBufferLock);
    ctx->FramePeriod100ns = 333333;   /* 30fps default until SetDataFormat */

    /* Negotiated format lives in Pin->ConnectionFormat. */
    if (Pin->ConnectionFormat != NULL &&
        Pin->ConnectionFormat->FormatSize >= sizeof(KS_DATAFORMAT_VIDEOINFOHEADER)) {
        PKS_DATAFORMAT_VIDEOINFOHEADER df =
            (PKS_DATAFORMAT_VIDEOINFOHEADER)Pin->ConnectionFormat;
        RoycApplyFormat(ctx, &df->VideoInfoHeader, &df->DataFormat.SubFormat);
    } else {
        ctx->Format    = RoycFmtNV12;
        ctx->Width     = 1280;
        ctx->Height    = 720;
        ctx->Stride    = RoycComputeStride(RoycFmtNV12, 1280);
        ctx->ImageSize = RoycComputeImageSize(RoycFmtNV12, 1280, 720);
    }

    /* Preallocate the service-fed back buffer at the max payload. */
    ctx->BackBufferSize = ROYC_MAX_PAYLOAD;
    ctx->BackBuffer = (PUCHAR)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, ctx->BackBufferSize, ROYC_POOL_TAG);
    if (ctx->BackBuffer == NULL) {
        ExFreePoolWithTag(ctx, ROYC_POOL_TAG);
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    Pin->Context = ctx;
    RoycInfo("PinCreate fmt=%d %lux%lu image=%lu", ctx->Format,
             ctx->Width, ctx->Height, ctx->ImageSize);
    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------------ close -- */
NTSTATUS RoycPinClose(PKSPIN Pin, PIRP Irp)
{
    PROYC_PIN_CONTEXT ctx = (PROYC_PIN_CONTEXT)Pin->Context;
    UNREFERENCED_PARAMETER(Irp);
    if (ctx != NULL) {
        RoycStopTimer(ctx);
        if (ctx->BackBuffer != NULL)
            ExFreePoolWithTag(ctx->BackBuffer, ROYC_POOL_TAG);
        ExFreePoolWithTag(ctx, ROYC_POOL_TAG);
        Pin->Context = NULL;
    }
    return STATUS_SUCCESS;
}

/* ------------------------------------------------------------ set format --- */
NTSTATUS RoycPinSetDataFormat(PKSPIN Pin, PKSDATAFORMAT OldFormat,
                              PKSMULTIPLE_ITEM OldAttributeList,
                              const KSDATARANGE *DataRange,
                              const KSATTRIBUTE_LIST *AttributeRange)
{
    PROYC_PIN_CONTEXT ctx = (PROYC_PIN_CONTEXT)Pin->Context;
    UNREFERENCED_PARAMETER(OldFormat);
    UNREFERENCED_PARAMETER(OldAttributeList);
    UNREFERENCED_PARAMETER(DataRange);
    UNREFERENCED_PARAMETER(AttributeRange);

    if (ctx == NULL || Pin->ConnectionFormat == NULL)
        return STATUS_INVALID_PARAMETER;
    if (Pin->ConnectionFormat->FormatSize <
        sizeof(KS_DATAFORMAT_VIDEOINFOHEADER))
        return STATUS_INVALID_PARAMETER;

    {
        PKS_DATAFORMAT_VIDEOINFOHEADER df =
            (PKS_DATAFORMAT_VIDEOINFOHEADER)Pin->ConnectionFormat;
        RoycApplyFormat(ctx, &df->VideoInfoHeader, &df->DataFormat.SubFormat);
    }
    RoycInfo("SetDataFormat fmt=%d %lux%lu image=%lu", ctx->Format,
             ctx->Width, ctx->Height, ctx->ImageSize);
    return STATUS_SUCCESS;
}

/* ----------------------------------------------------------- device state -- */
NTSTATUS RoycPinSetDeviceState(PKSPIN Pin, KSSTATE ToState, KSSTATE FromState)
{
    PROYC_PIN_CONTEXT ctx = (PROYC_PIN_CONTEXT)Pin->Context;
    if (ctx == NULL)
        return STATUS_INVALID_PARAMETER;

    if (ToState == KSSTATE_RUN && FromState != KSSTATE_RUN) {
        ctx->FrameIndex = 0;
        ctx->Pts = 0;
        RoycStartTimer(ctx, Pin);
        RoycInfo("Pin RUN");
    } else if (ToState != KSSTATE_RUN && FromState == KSSTATE_RUN) {
        RoycStopTimer(ctx);
        RoycInfo("Pin STOP");
    }
    return STATUS_SUCCESS;
}

/* ---------------------------------------------------------------- process -- */
NTSTATUS RoycPinProcess(PKSPIN Pin)
{
    PROYC_PIN_CONTEXT ctx = (PROYC_PIN_CONTEXT)Pin->Context;
    PKSSTREAM_POINTER sp;
    PROYC_FILTER_CONTEXT fctx;

    if (ctx == NULL)
        return STATUS_UNSUCCESSFUL;
    fctx = (PROYC_FILTER_CONTEXT)KsPinGetParentFilter(Pin)->Context;

    sp = KsPinGetLeadingEdgeStreamPointer(Pin, KSSTREAM_POINTER_STATE_LOCKED);
    while (sp != NULL) {
        PKSSTREAM_POINTER next;
        PKSSTREAM_HEADER hdr = sp->StreamHeader;
        PUCHAR dst = (PUCHAR)sp->Data;
        ULONG cap = sp->OffsetOut.Remaining;

        if (dst != NULL && cap >= ctx->ImageSize) {
            BOOLEAN served = FALSE;

            /* Stage 3: prefer a fresh service-fed frame if present. */
            if (fctx != NULL && fctx->SourceMode == RoycSourceService) {
                KIRQL irql;
                KeAcquireSpinLock(&ctx->BackBufferLock, &irql);
                if (ctx->BackBufferValid >= ctx->ImageSize) {
                    RtlCopyMemory(dst, ctx->BackBuffer, ctx->ImageSize);
                    served = TRUE;
                }
                KeReleaseSpinLock(&ctx->BackBufferLock, irql);
            }
            /* Stage 1 fallback: generate a simulated frame. */
            if (!served) {
                LONG tick = InterlockedIncrement(
                    fctx ? &fctx->FrameCounter : &ctx->Running);
                RoycFillSimulatedFrame(ctx, dst, cap, tick);
            }

            hdr->DataUsed         = ctx->ImageSize;
            hdr->PresentationTime.Time      = (LONGLONG)ctx->Pts;
            hdr->PresentationTime.Numerator = 1;
            hdr->PresentationTime.Denominator = 1;
            hdr->Duration         = ctx->FramePeriod100ns;
            hdr->OptionsFlags     = KSSTREAM_HEADER_OPTIONSF_TIMEVALID |
                                    KSSTREAM_HEADER_OPTIONSF_DURATIONVALID;
            ctx->Pts             += ctx->FramePeriod100ns;
            ctx->FrameIndex++;
        } else {
            hdr->DataUsed = 0;
        }

        next = KsStreamPointerGetNextClone(sp);
        KsStreamPointerDelete(sp);
        if (next == NULL) {
            /* Advance the leading edge; STATUS_DEVICE_NOT_READY when drained. */
            NTSTATUS st = KsStreamPointerAdvance(sp);
            if (st != STATUS_SUCCESS)
                break;
        }
        sp = next;
    }
    return STATUS_PENDING;
}

/* ------------------------------------------------------- intersect handler - */
NTSTATUS RoycIntersectHandler(PVOID Context, PIRP Irp, PKSP_PIN PinInstance,
                              PKSDATARANGE CallerDataRange,
                              PKSDATARANGE DescriptorDataRange,
                              ULONG BufferSize, PVOID Data, PULONG DataSize)
{
    PKS_DATARANGE_VIDEO descV = (PKS_DATARANGE_VIDEO)DescriptorDataRange;
    ULONG needed = sizeof(KS_DATAFORMAT_VIDEOINFOHEADER);

    UNREFERENCED_PARAMETER(Context);
    UNREFERENCED_PARAMETER(Irp);
    UNREFERENCED_PARAMETER(PinInstance);
    UNREFERENCED_PARAMETER(CallerDataRange);

    *DataSize = needed;
    if (BufferSize == 0)
        return STATUS_BUFFER_OVERFLOW;     /* size query */
    if (BufferSize < needed)
        return STATUS_BUFFER_TOO_SMALL;

    {
        PKS_DATAFORMAT_VIDEOINFOHEADER out =
            (PKS_DATAFORMAT_VIDEOINFOHEADER)Data;
        ULONG imgSize;
        ROYC_FORMAT fmt;

        RtlZeroMemory(out, needed);
        out->DataFormat = descV->DataRange;
        out->DataFormat.FormatSize = needed;
        out->VideoInfoHeader = descV->VideoInfoHeader;

        fmt = RoycFormatFromSubtype(&descV->DataRange.SubFormat);
        imgSize = RoycComputeImageSize(fmt,
                      (ULONG)descV->VideoInfoHeader.bmiHeader.biWidth,
                      (ULONG)descV->VideoInfoHeader.bmiHeader.biHeight);
        out->DataFormat.SampleSize = imgSize;
        out->VideoInfoHeader.bmiHeader.biSizeImage = imgSize;
    }
    return STATUS_SUCCESS;
}
