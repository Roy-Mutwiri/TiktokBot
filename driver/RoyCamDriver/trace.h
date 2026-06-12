/*
 * trace.h — lightweight debug tracing for the RoyCam driver.
 *
 * Uses DbgPrintEx so traces are visible in WinDbg / DebugView without WPP
 * tooling. Swap to WPP for production telemetry if desired.
 */
#ifndef ROYCAM_TRACE_H
#define ROYCAM_TRACE_H

#include <ntddk.h>

#define ROYC_TRACE_COMP   DPFLTR_IHVDRIVER_ID

#if DBG
#define RoycTrace(level, fmt, ...) \
    DbgPrintEx(ROYC_TRACE_COMP, (level), "[RoyCam] " fmt "\n", __VA_ARGS__)
#else
#define RoycTrace(level, fmt, ...) ((void)0)
#endif

#define RoycInfo(fmt, ...)  RoycTrace(DPFLTR_INFO_LEVEL,    fmt, __VA_ARGS__)
#define RoycWarn(fmt, ...)  RoycTrace(DPFLTR_WARNING_LEVEL, fmt, __VA_ARGS__)
#define RoycErr(fmt, ...)   RoycTrace(DPFLTR_ERROR_LEVEL,   fmt, __VA_ARGS__)

#endif /* ROYCAM_TRACE_H */
