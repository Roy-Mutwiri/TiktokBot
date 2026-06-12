/*
 * roycam_status_codes.h  —  shared status/error enum (app <-> service <-> driver)
 * These are surfaced in the app's Camera Output status panel.
 */
#ifndef ROYCAM_STATUS_CODES_H
#define ROYCAM_STATUS_CODES_H

typedef enum {
    ROYC_OK                  = 0,
    ROYC_DRIVER_NOT_INSTALLED= 1,
    ROYC_SERVICE_STOPPED     = 2,
    ROYC_NO_SOURCE           = 3,
    ROYC_FRAME_TOO_LARGE     = 4,
    ROYC_UNSUPPORTED_FORMAT  = 5,
    ROYC_RESOLUTION_MISMATCH = 6,
    ROYC_IPC_TIMEOUT         = 7,
    ROYC_CLIENT_BUSY         = 8,
    ROYC_FALLBACK_ACTIVE     = 9,
    ROYC_BAD_HEADER          = 10
} RoycStatus;

#endif /* ROYCAM_STATUS_CODES_H */
