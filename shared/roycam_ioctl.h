/*
 * roycam_ioctl.h — IOCTL contract between the RoyCam Frame Bridge service
 * (user mode) and the RoyCam HD Camera AVStream driver (kernel mode).
 *
 * Shared by both sides. Self-contained: usable from kernel (wdm.h) and user
 * (windows.h) without pulling stdint.h. All bounded copies; the driver
 * validates every field before touching payload bytes.
 *
 * This describes a SOFTWARE camera. It does not model physical USB hardware.
 */
#ifndef ROYCAM_IOCTL_H
#define ROYCAM_IOCTL_H

#if defined(_KERNEL_MODE)
#include <ntddk.h>
#else
#include <windows.h>
#include <winioctl.h>
#endif

/* Fixed-width aliases that exist in both km and um without stdint.h. */
typedef unsigned int       ROYC_U32;
typedef unsigned __int64   ROYC_U64;

/* Custom device type (FILE_DEVICE_* user range, not a vendor type). */
#define ROYC_DEVICE_TYPE   0x8123u

/* Pixel formats — must match shared/roycam_frame_format.h. */
#define ROYC_IOCTL_FMT_NV12   1u
#define ROYC_IOCTL_FMT_YUY2   2u
#define ROYC_IOCTL_FMT_RGB32  3u

/* Submit one pre-formatted frame to the driver's back buffer (latest wins). */
#define ROYC_IOCTL_SUBMIT_FRAME \
    CTL_CODE(ROYC_DEVICE_TYPE, 0x800, METHOD_IN_DIRECT, FILE_WRITE_ACCESS)
/* Declare the active capture format/resolution (driver pre-sizes buffers).   */
#define ROYC_IOCTL_SET_FORMAT \
    CTL_CODE(ROYC_DEVICE_TYPE, 0x801, METHOD_BUFFERED, FILE_WRITE_ACCESS)
/* Query driver/source status (drives the app's status panel).                */
#define ROYC_IOCTL_QUERY_STATUS \
    CTL_CODE(ROYC_DEVICE_TYPE, 0x802, METHOD_BUFFERED, FILE_READ_ACCESS)

/* Hard caps the driver enforces on any submitted frame. */
#define ROYC_MAX_WIDTH    1920u
#define ROYC_MAX_HEIGHT   1080u
#define ROYC_MAX_PAYLOAD  (ROYC_MAX_WIDTH * ROYC_MAX_HEIGHT * 4u)  /* RGB32 */

/* Input buffer for ROYC_IOCTL_SUBMIT_FRAME (the METHOD_IN_DIRECT MDL carries
 * the payload separately; this is the AssociatedIrp.SystemBuffer metadata).  */
typedef struct _ROYC_FRAME_SUBMIT {
    ROYC_U32 format;        /* ROYC_IOCTL_FMT_*                               */
    ROYC_U32 width;         /* <= ROYC_MAX_WIDTH                              */
    ROYC_U32 height;        /* <= ROYC_MAX_HEIGHT                            */
    ROYC_U32 stride;        /* bytes per row of plane 0                       */
    ROYC_U32 data_size;     /* payload bytes in the OutBuffer MDL             */
    ROYC_U32 flags;         /* reserved (fallback bit etc.)                   */
    ROYC_U64 frame_index;   /* monotonic                                     */
    ROYC_U64 timestamp_qpc; /* producer QPC                                  */
} ROYC_FRAME_SUBMIT, *PROYC_FRAME_SUBMIT;

/* Input buffer for ROYC_IOCTL_SET_FORMAT. */
typedef struct _ROYC_SET_FORMAT {
    ROYC_U32 format;
    ROYC_U32 width;
    ROYC_U32 height;
    ROYC_U32 fps_num;
    ROYC_U32 fps_den;
} ROYC_SET_FORMAT, *PROYC_SET_FORMAT;

/* Output buffer for ROYC_IOCTL_QUERY_STATUS. */
typedef struct _ROYC_STATUS {
    ROYC_U32 source_state;    /* 0 none, 1 active, 2 paused                   */
    ROYC_U32 last_status;     /* RoycStatus code                             */
    ROYC_U32 active_format;
    ROYC_U32 active_width;
    ROYC_U32 active_height;
    ROYC_U64 frames_delivered;
    ROYC_U64 frames_dropped;
} ROYC_STATUS, *PROYC_STATUS;

#endif /* ROYCAM_IOCTL_H */
