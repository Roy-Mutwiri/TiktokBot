/*
 * formats.c — capture-pin media format tables (KS_DATARANGE_VIDEO) and the
 * size/stride helpers. RoyCam advertises:
 *     NV12  1280x720 @ 30fps   (primary)
 *     YUY2  1280x720 @ 30fps   (secondary)
 *     RGB32  640x480 @ 30fps   (debug)
 *
 * Subtype GUIDs are the standard FOURCC/MEDIASUBTYPE identifiers (open format
 * IDs, not any vendor's proprietary signature). Everything here is compile-time
 * const so the pin descriptor can reference it directly.
 */
#include "roycam.h"

#define ROYC_FPS          30
#define ROYC_FRAME_100NS  (10000000 / ROYC_FPS)   /* 333333 */

#define ROYC_GUID_VIDEO \
    { 0x73646976,0x0000,0x0010,{0x80,0x00,0x00,0xAA,0x00,0x38,0x9B,0x71} }
#define ROYC_GUID_VIDEOINFO \
    { 0x05589f80,0xc356,0x11ce,{0xbf,0x01,0x00,0xaa,0x00,0x55,0x59,0x5a} }
#define ROYC_GUID_NV12 \
    { 0x3231564e,0x0000,0x0010,{0x80,0x00,0x00,0xAA,0x00,0x38,0x9B,0x71} }
#define ROYC_GUID_YUY2 \
    { 0x32595559,0x0000,0x0010,{0x80,0x00,0x00,0xAA,0x00,0x38,0x9B,0x71} }
#define ROYC_GUID_RGB32 \
    { 0xe436eb7e,0x524f,0x11ce,{0x9f,0x53,0x00,0x20,0xaf,0x0b,0xa7,0x70} }

#define FOURCC_NV12  0x3231564E   /* 'NV12' */
#define FOURCC_YUY2  0x32595559   /* 'YUY2' */

/* Fully-initialized const KS_DATARANGE_VIDEO. */
#define ROYC_DRV(subGuid, bits, comp, cx, cy, sampleBytes)                     \
{                                                                              \
    {   /* .DataRange (KSDATARANGE) */                                        \
        sizeof(KS_DATARANGE_VIDEO), 0, (sampleBytes), 0,                       \
        ROYC_GUID_VIDEO, subGuid, ROYC_GUID_VIDEOINFO                          \
    },                                                                         \
    TRUE, FALSE, 0, 0,                                                         \
    {   /* KS_VIDEO_STREAM_CONFIG_CAPS */                                      \
        ROYC_GUID_VIDEO, KS_AnalogVideo_None,                                  \
        { (cx),(cy) }, { (cx),(cy) }, { (cx),(cy) }, 1, 1, 1, 1,               \
        { (cx),(cy) }, { (cx),(cy) }, 1, 1, 0, 0, 0, 0,                        \
        ROYC_FRAME_100NS, ROYC_FRAME_100NS,                                    \
        (LONG)((LONGLONG)(cx)*(cy)*(bits)*ROYC_FPS),                           \
        (LONG)((LONGLONG)(cx)*(cy)*(bits)*ROYC_FPS)                            \
    },                                                                         \
    {   /* KS_VIDEOINFOHEADER */                                               \
        { 0,0,(cx),(cy) }, { 0,0,(cx),(cy) },                                  \
        (DWORD)((LONGLONG)(cx)*(cy)*(bits)*ROYC_FPS), 0, ROYC_FRAME_100NS,     \
        {   /* KS_BITMAPINFOHEADER */                                          \
            sizeof(KS_BITMAPINFOHEADER), (cx), (cy), 1, (WORD)(bits),          \
            (comp), (DWORD)(sampleBytes), 0, 0, 0, 0                           \
        }                                                                      \
    }                                                                          \
}

static const KS_DATARANGE_VIDEO RoycDataRangeNV12 =
    ROYC_DRV(ROYC_GUID_NV12,  12, FOURCC_NV12, 1280, 720, (1280*720*3/2));
static const KS_DATARANGE_VIDEO RoycDataRangeYUY2 =
    ROYC_DRV(ROYC_GUID_YUY2,  16, FOURCC_YUY2, 1280, 720, (1280*720*2));
static const KS_DATARANGE_VIDEO RoycDataRangeRGB32 =
    ROYC_DRV(ROYC_GUID_RGB32, 32, 0 /*BI_RGB*/, 640, 480, (640*480*4));

/* Const data-range pointer table for the pin descriptor. */
const PKSDATARANGE RoycPinDataRanges[RoycFmtCount] = {
    (PKSDATARANGE)&RoycDataRangeNV12,
    (PKSDATARANGE)&RoycDataRangeYUY2,
    (PKSDATARANGE)&RoycDataRangeRGB32,
};

static const GUID Royc_SUB_NV12  = ROYC_GUID_NV12;
static const GUID Royc_SUB_YUY2  = ROYC_GUID_YUY2;
static const GUID Royc_SUB_RGB32 = ROYC_GUID_RGB32;

ULONG RoycComputeStride(ROYC_FORMAT fmt, ULONG width)
{
    switch (fmt) {
    case RoycFmtNV12:  return width;
    case RoycFmtYUY2:  return width * 2;
    case RoycFmtRGB32: return width * 4;
    default:           return 0;
    }
}

ULONG RoycComputeImageSize(ROYC_FORMAT fmt, ULONG width, ULONG height)
{
    switch (fmt) {
    case RoycFmtNV12:  return width * height + (width * height) / 2;
    case RoycFmtYUY2:  return width * height * 2;
    case RoycFmtRGB32: return width * height * 4;
    default:           return 0;
    }
}

ROYC_FORMAT RoycFormatFromSubtype(const GUID *subFormat)
{
    if (RtlEqualMemory(subFormat, &Royc_SUB_YUY2,  sizeof(GUID)))
        return RoycFmtYUY2;
    if (RtlEqualMemory(subFormat, &Royc_SUB_RGB32, sizeof(GUID)))
        return RoycFmtRGB32;
    return RoycFmtNV12;
}
