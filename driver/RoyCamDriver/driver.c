/*
 * driver.c — DriverEntry for the RoyCam HD Camera AVStream minidriver.
 *
 * Hands the device descriptor to KsInitializeDriver, which wires up the KS
 * dispatch, builds the filter factory, and registers the device interfaces
 * (KSCATEGORY_VIDEO_CAMERA / _CAPTURE / _VIDEO) declared in filter.c.
 */
#include "roycam.h"

DRIVER_INITIALIZE DriverEntry;

#ifdef ALLOC_PRAGMA
#pragma alloc_text(INIT, DriverEntry)
#endif

NTSTATUS DriverEntry(PDRIVER_OBJECT DriverObject, PUNICODE_STRING RegistryPath)
{
    NTSTATUS status;

    RoycInfo("RoyCam HD Camera minidriver loading");
    status = KsInitializeDriver(DriverObject, RegistryPath,
                                &RoycDeviceDescriptor);
    if (!NT_SUCCESS(status))
        RoycErr("KsInitializeDriver failed 0x%08x", status);
    return status;
}
