/*
 * framesource.c — frame providers.
 *
 *  Stage 1 (active):   RoycFillSimulatedFrame — driver-generated color bars
 *                      with a moving marker, in the pin's negotiated format.
 *  Stage 3 (wired):    RoycSubmitServiceFrame — validated copy of a service-fed
 *                      frame into the pin's preallocated back buffer.
 *
 * Integer-only (no FP at DISPATCH_LEVEL). Bar colors are precomputed in both
 * BGRA and BT.601 YUV.
 */
#include "roycam.h"

#define ROYC_BARS 8

/* BGRA bar colors: white, yellow, cyan, green, magenta, red, blue, black. */
static const UCHAR RoycBarsBGRA[ROYC_BARS][4] = {
    {255,255,255,255}, {0,255,255,255}, {255,255,0,255}, {0,255,0,255},
    {255,0,255,255},   {0,0,255,255},   {255,0,0,255},   {0,0,0,255},
};
/* Matching BT.601 {Y,U,V}. */
static const UCHAR RoycBarsYUV[ROYC_BARS][3] = {
    {255,128,128}, {226,0,148}, {178,170,0}, {149,44,21},
    {105,212,235}, {76,85,255}, {29,255,107}, {0,128,128},
};

static __forceinline ULONG RoycBarIndex(ULONG x, ULONG width)
{
    ULONG bw = width / ROYC_BARS;
    if (bw == 0) bw = 1;
    ULONG i = x / bw;
    return (i >= ROYC_BARS) ? (ROYC_BARS - 1) : i;
}

static VOID RoycFillRGB32(PROYC_PIN_CONTEXT pin, PUCHAR dst, LONG tick)
{
    ULONG w = pin->Width, h = pin->Height;
    ULONG marker = (ULONG)((tick * 6)) % (w ? w : 1);
    for (ULONG y = 0; y < h; ++y) {
        PUCHAR row = dst + (SIZE_T)y * pin->Stride;
        for (ULONG x = 0; x < w; ++x) {
            const UCHAR *c = RoycBarsBGRA[RoycBarIndex(x, w)];
            PUCHAR px = row + (SIZE_T)x * 4;
            if (x >= marker && x < marker + 8) {
                px[0] = 40; px[1] = 40; px[2] = 40; px[3] = 255;
            } else {
                px[0] = c[0]; px[1] = c[1]; px[2] = c[2]; px[3] = c[3];
            }
        }
    }
}

static VOID RoycFillNV12(PROYC_PIN_CONTEXT pin, PUCHAR dst, LONG tick)
{
    ULONG w = pin->Width, h = pin->Height;
    ULONG marker = (ULONG)((tick * 6)) % (w ? w : 1);
    PUCHAR yPlane = dst;
    PUCHAR uvPlane = dst + (SIZE_T)w * h;
    for (ULONG y = 0; y < h; ++y) {
        PUCHAR yr = yPlane + (SIZE_T)y * w;
        for (ULONG x = 0; x < w; ++x) {
            const UCHAR *c = RoycBarsYUV[RoycBarIndex(x, w)];
            yr[x] = (x >= marker && x < marker + 8) ? 40 : c[0];
        }
    }
    for (ULONG y = 0; y < h / 2; ++y) {
        PUCHAR uv = uvPlane + (SIZE_T)y * w;
        for (ULONG x = 0; x < w / 2; ++x) {
            const UCHAR *c = RoycBarsYUV[RoycBarIndex(x * 2, w)];
            uv[x * 2 + 0] = c[1];   /* U */
            uv[x * 2 + 1] = c[2];   /* V */
        }
    }
}

static VOID RoycFillYUY2(PROYC_PIN_CONTEXT pin, PUCHAR dst, LONG tick)
{
    ULONG w = pin->Width, h = pin->Height;
    ULONG marker = (ULONG)((tick * 6)) % (w ? w : 1);
    for (ULONG y = 0; y < h; ++y) {
        PUCHAR row = dst + (SIZE_T)y * pin->Stride;
        for (ULONG x = 0; x < w; x += 2) {
            const UCHAR *c0 = RoycBarsYUV[RoycBarIndex(x, w)];
            const UCHAR *c1 = RoycBarsYUV[RoycBarIndex(x + 1, w)];
            UCHAR y0 = (x >= marker && x < marker + 8) ? 40 : c0[0];
            UCHAR y1 = (x + 1 >= marker && x + 1 < marker + 8) ? 40 : c1[0];
            PUCHAR p = row + (SIZE_T)x * 2;
            p[0] = y0; p[1] = c0[1];   /* Y0 U */
            p[2] = y1; p[3] = c0[2];   /* Y1 V */
        }
    }
}

VOID RoycFillSimulatedFrame(PROYC_PIN_CONTEXT pin, PUCHAR dst, ULONG dstSize,
                            LONG tick)
{
    if (dst == NULL || dstSize < pin->ImageSize)
        return;
    switch (pin->Format) {
    case RoycFmtRGB32: RoycFillRGB32(pin, dst, tick); break;
    case RoycFmtYUY2:  RoycFillYUY2(pin, dst, tick);  break;
    case RoycFmtNV12:
    default:           RoycFillNV12(pin, dst, tick);  break;
    }
}

/* Stage 3: service-fed frame. Validates, then copies into the back buffer. */
NTSTATUS RoycSubmitServiceFrame(PROYC_PIN_CONTEXT pin,
                                const ROYC_FRAME_SUBMIT *meta,
                                PUCHAR payload, ULONG payloadSize)
{
    KIRQL irql;

    if (pin == NULL || meta == NULL || payload == NULL)
        return STATUS_INVALID_PARAMETER;
    if (meta->width > ROYC_MAX_WIDTH || meta->height > ROYC_MAX_HEIGHT)
        return STATUS_INVALID_PARAMETER;
    if (meta->data_size != payloadSize)
        return STATUS_INVALID_BUFFER_SIZE;
    if (payloadSize == 0 || payloadSize > pin->BackBufferSize)
        return STATUS_BUFFER_TOO_SMALL;

    KeAcquireSpinLock(&pin->BackBufferLock, &irql);
    RtlCopyMemory(pin->BackBuffer, payload, payloadSize);
    pin->BackBufferValid = payloadSize;
    KeReleaseSpinLock(&pin->BackBufferLock, irql);
    return STATUS_SUCCESS;
}
