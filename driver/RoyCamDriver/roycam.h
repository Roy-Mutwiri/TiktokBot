/*
 * roycam.h — RoyCam HD Camera AVStream minidriver, common definitions.
 *
 * RoyCam is a LEGITIMATE, OWN-BRANDED software camera. It is root-enumerated
 * under our own hardware-ID namespace (Root\RoyCamHDCamera), advertises
 * KSCATEGORY_VIDEO_CAMERA + KSCATEGORY_CAPTURE, and exposes app-generated
 * frames on a capture pin. It does NOT impersonate any vendor, spoof any
 * VID/PID/hardware-ID/signature, and does NOT claim to be physical USB
 * hardware.
 *
 * Frame source stages (the skeleton ships stage 1; 2/3 are wired but inert):
 *   1. SIMULATED  — driver-generated test pattern (proves the pipeline)
 *   2. STATIC     — a single embedded/static image
 *   3. SERVICE    — frames fed from the RoyCam Frame Bridge via IOCTL
 */
#ifndef ROYCAM_H
#define ROYCAM_H

#include <ntddk.h>
#include <wdf.h>
#include <ks.h>
#include <ksmedia.h>

#include "trace.h"
#include "..\..\shared\roycam_ioctl.h"

/* ---- branding ------------------------------------------------------------ */
#define ROYC_CAMERA_NAME    L"RoyCam HD Camera"
#define ROYC_POOL_TAG       'CyoR'   /* 'RoyC' */

/* ---- capture format identifiers (internal index into the data-range table) */
typedef enum _ROYC_FORMAT {
    RoycFmtNV12 = 0,
    RoycFmtYUY2,
    RoycFmtRGB32,
    RoycFmtCount
} ROYC_FORMAT;

/* ---- frame source mode --------------------------------------------------- */
typedef enum _ROYC_SOURCE_MODE {
    RoycSourceSimulated = 0,
    RoycSourceStatic,
    RoycSourceService
} ROYC_SOURCE_MODE;

/* ---- per-filter context -------------------------------------------------- */
typedef struct _ROYC_FILTER_CONTEXT {
    ROYC_SOURCE_MODE SourceMode;
    LONG             FrameCounter;     /* drives the simulated pattern        */
} ROYC_FILTER_CONTEXT, *PROYC_FILTER_CONTEXT;

/* ---- per-pin context ----------------------------------------------------- */
typedef struct _ROYC_PIN_CONTEXT {
    KS_VIDEOINFOHEADER *VideoInfo;     /* negotiated format (owned copy)      */
    ULONG               Width;
    ULONG               Height;
    ULONG               Stride;
    ULONG               ImageSize;     /* payload bytes per frame             */
    ROYC_FORMAT         Format;
    ULONGLONG           FrameIndex;
    ULONGLONG           Pts;           /* 100ns presentation clock            */

    /* 30fps pacing: a periodic timer kicks KsPinAttemptProcessing.           */
    KTIMER              FrameTimer;
    KDPC                FrameDpc;
    LONG                Running;
    LONGLONG            FramePeriod100ns;

    /* Preallocated double buffer for the service-fed hot path (stage 3).     */
    PUCHAR              BackBuffer;
    ULONG               BackBufferSize;
    ULONG               BackBufferValid;
    KSPIN_LOCK          BackBufferLock;
} ROYC_PIN_CONTEXT, *PROYC_PIN_CONTEXT;

/* ---- descriptors exported across the translation units ------------------- */
extern const KSDEVICE_DISPATCH    RoycDeviceDispatch;
extern const KSDEVICE_DESCRIPTOR  RoycDeviceDescriptor;
extern const KSFILTER_DESCRIPTOR  RoycFilterDescriptor;
extern const KSFILTER_DESCRIPTOR *RoycFilterDescriptors[1];
extern const KSPIN_DESCRIPTOR_EX  RoycPinDescriptors[1];

/* ---- formats.c ----------------------------------------------------------- */
extern const PKSDATARANGE RoycPinDataRanges[RoycFmtCount];
ULONG RoycComputeImageSize(ROYC_FORMAT fmt, ULONG width, ULONG height);
ULONG RoycComputeStride(ROYC_FORMAT fmt, ULONG width);
ROYC_FORMAT RoycFormatFromSubtype(const GUID *subFormat);

/* ---- framesource.c ------------------------------------------------------- */
VOID RoycFillSimulatedFrame(PROYC_PIN_CONTEXT pin, PUCHAR dst, ULONG dstSize,
                            LONG tick);
NTSTATUS RoycSubmitServiceFrame(PROYC_PIN_CONTEXT pin,
                                const ROYC_FRAME_SUBMIT *meta,
                                PUCHAR payload, ULONG payloadSize);

/* ---- capture.c ----------------------------------------------------------- */
NTSTATUS RoycPinCreate(PKSPIN Pin, PIRP Irp);
NTSTATUS RoycPinClose(PKSPIN Pin, PIRP Irp);
NTSTATUS RoycPinProcess(PKSPIN Pin);
NTSTATUS RoycPinSetDeviceState(PKSPIN Pin, KSSTATE ToState, KSSTATE FromState);
NTSTATUS RoycPinSetDataFormat(PKSPIN Pin, PKSDATAFORMAT OldFormat,
                              PKSMULTIPLE_ITEM OldAttributeList,
                              const KSDATARANGE *DataRange,
                              const KSATTRIBUTE_LIST *AttributeRange);
NTSTATUS RoycIntersectHandler(PVOID Context, PIRP Irp, PKSP_PIN PinInstance,
                              PKSDATARANGE CallerDataRange,
                              PKSDATARANGE DescriptorDataRange,
                              ULONG BufferSize, PVOID Data,
                              PULONG DataSize);

#endif /* ROYCAM_H */
