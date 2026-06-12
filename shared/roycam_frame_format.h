/*
 * roycam_frame_format.h  —  RoyCam shared frame contract (v2)
 * -----------------------------------------------------------------------------
 * Single source of truth for the App (Python ctypes mirror), the Service, and the
 * future AVStream driver. Defines the shared-memory layout: one control block +
 * a triple-buffer ring of [frame header + payload]. Little-endian, fixed sizes.
 *
 * This is a SOFTWARE camera contract. It carries app-generated frames; it does
 * not describe physical USB hardware.
 */
#ifndef ROYCAM_FRAME_FORMAT_H
#define ROYCAM_FRAME_FORMAT_H

#include <stdint.h>

#define ROYC_MAGIC        0x43594F52u   /* 'R''O''Y''C' (little-endian)        */
#define ROYC_VERSION      2u
#define ROYC_HEADER_SZ    128u          /* per-slot frame header size          */
#define ROYC_CONTROL_SZ   64u           /* control block size at offset 0      */
#define ROYC_RING_SLOTS   3u            /* triple buffer                       */

/* Pixel formats the camera advertises (user mode converts to these).          */
typedef enum {
    ROYC_FMT_NV12  = 1,   /* primary  : Y plane + interleaved UV, 12 bpp       */
    ROYC_FMT_YUY2  = 2,   /* secondary: packed 4:2:2 (YUYV), 16 bpp           */
    ROYC_FMT_RGB32 = 3    /* debug    : BGRA top-down, 32 bpp                  */
} RoycPixelFormat;

/* Source state (drives fallback in the service/driver).                       */
typedef enum {
    ROYC_SRC_NONE   = 0,  /* no source -> serve fallback                       */
    ROYC_SRC_ACTIVE = 1,  /* app is producing frames                          */
    ROYC_SRC_PAUSED = 2
} RoycSourceState;

/* status_flags bits */
#define ROYC_FLAG_READY     0x1u   /* slot holds a complete, valid frame       */
#define ROYC_FLAG_FALLBACK  0x2u   /* this is a fallback frame                 */

/* Control block — lives at shared-memory offset 0. Size == ROYC_CONTROL_SZ.   */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;            /* ROYC_MAGIC                                   */
    uint32_t version;          /* ROYC_VERSION                                */
    uint32_t header_size;      /* ROYC_HEADER_SZ                              */
    uint32_t ring_slots;       /* ROYC_RING_SLOTS                            */
    uint32_t slot_stride;      /* ROYC_HEADER_SZ + max payload bytes per slot  */
    uint32_t max_width;        /* capacity, e.g. 1920                         */
    uint32_t max_height;       /* capacity, e.g. 1080                         */
    volatile uint32_t write_index;  /* atomic: index of last fully-written slot*/
    volatile uint32_t source_state; /* RoycSourceState                        */
    uint32_t reserved[7];      /* pad to 64 bytes                             */
} RoycControl;

/* Per-slot frame header — precedes each slot's payload. Size == ROYC_HEADER_SZ.*/
typedef struct {
    uint32_t magic;            /* ROYC_MAGIC (per-slot sanity)                 */
    uint32_t format;           /* RoycPixelFormat                             */
    uint32_t width;            /* active frame width                          */
    uint32_t height;           /* active frame height                         */
    uint32_t stride;           /* bytes per row of plane 0                     */
    uint32_t data_size;        /* payload bytes actually written               */
    uint32_t fps_num;          /* e.g. 30                                     */
    uint32_t fps_den;          /* e.g. 1                                      */
    uint64_t frame_index;      /* monotonic                                   */
    uint64_t timestamp_qpc;    /* QueryPerformanceCounter at write             */
    volatile uint64_t seq;     /* SEQLOCK: odd = writing, even = stable        */
    uint32_t status_flags;     /* ROYC_FLAG_*                                  */
    uint32_t checksum;         /* optional CRC32 of payload (0 = not computed) */
    uint32_t reserved[16];     /* pad header to 128 bytes (64 used + 64 pad)   */
} RoycFrameHeader;
#pragma pack(pop)

/* Expected payload size for a given format/width/height (overflow-safe callers
 * must still validate width*height first).                                    */
static __inline uint64_t RoycPayloadSize(uint32_t fmt, uint32_t w, uint32_t h) {
    uint64_t px = (uint64_t)w * (uint64_t)h;
    switch (fmt) {
        case ROYC_FMT_NV12:  return px + (px / 2);   /* 12 bpp */
        case ROYC_FMT_YUY2:  return px * 2;          /* 16 bpp */
        case ROYC_FMT_RGB32: return px * 4;          /* 32 bpp */
        default:             return 0;
    }
}

#endif /* ROYCAM_FRAME_FORMAT_H */
