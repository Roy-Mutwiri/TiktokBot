/*
 * ioctl.c — service-fed (stage 3) IOCTL handling.
 *
 * This is the validated boundary between the RoyCam Frame Bridge service and
 * the driver. It is COMPILED but not yet wired into an IRP dispatch path; the
 * device Add hook (device.c) will create a control device and route
 * IRP_MJ_DEVICE_CONTROL here once stage 3 is activated. Every field is bounds-
 * checked before any payload byte is touched.
 *
 * pinContext is the target capture pin's ROYC_PIN_CONTEXT (resolved from the
 * active filter/pin when the control device is wired up).
 */
#include "roycam.h"

NTSTATUS RoycHandleControlIoctl(PVOID pinContext, ULONG code,
                                PVOID inBuf, ULONG inLen,
                                PVOID outBuf, ULONG outLen,
                                PUCHAR payload, ULONG payloadLen,
                                PULONG bytesReturned)
{
    PROYC_PIN_CONTEXT pin = (PROYC_PIN_CONTEXT)pinContext;
    *bytesReturned = 0;

    switch (code) {
    case ROYC_IOCTL_SUBMIT_FRAME: {
        const ROYC_FRAME_SUBMIT *meta = (const ROYC_FRAME_SUBMIT *)inBuf;
        if (pin == NULL || inBuf == NULL || inLen < sizeof(ROYC_FRAME_SUBMIT))
            return STATUS_INVALID_PARAMETER;
        if (meta->format != ROYC_IOCTL_FMT_NV12 &&
            meta->format != ROYC_IOCTL_FMT_YUY2 &&
            meta->format != ROYC_IOCTL_FMT_RGB32)
            return STATUS_INVALID_PARAMETER;
        return RoycSubmitServiceFrame(pin, meta, payload, payloadLen);
    }

    case ROYC_IOCTL_SET_FORMAT: {
        const ROYC_SET_FORMAT *fmt = (const ROYC_SET_FORMAT *)inBuf;
        if (inBuf == NULL || inLen < sizeof(ROYC_SET_FORMAT))
            return STATUS_INVALID_PARAMETER;
        if (fmt->width > ROYC_MAX_WIDTH || fmt->height > ROYC_MAX_HEIGHT)
            return STATUS_INVALID_PARAMETER;
        /* Stage 3: re-size buffers / re-advertise. Inert for now. */
        return STATUS_SUCCESS;
    }

    case ROYC_IOCTL_QUERY_STATUS: {
        ROYC_STATUS *st = (ROYC_STATUS *)outBuf;
        if (outBuf == NULL || outLen < sizeof(ROYC_STATUS))
            return STATUS_BUFFER_TOO_SMALL;
        RtlZeroMemory(st, sizeof(*st));
        if (pin != NULL) {
            st->active_format = (ROYC_U32)pin->Format;
            st->active_width  = pin->Width;
            st->active_height = pin->Height;
            st->frames_delivered = pin->FrameIndex;
        }
        st->source_state = 1;   /* active */
        *bytesReturned = sizeof(ROYC_STATUS);
        return STATUS_SUCCESS;
    }

    default:
        return STATUS_INVALID_DEVICE_REQUEST;
    }
}
