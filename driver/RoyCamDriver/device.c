/*
 * device.c — KS device dispatch + the top-level device descriptor.
 *
 * RoyCam is root-enumerated (see the INF). KsInitializeDriver builds the filter
 * factory from RoycFilterDescriptor automatically; the device Add hook is where
 * a control device for the service-fed (stage 3) IOCTL path will be created.
 */
#include "roycam.h"

static NTSTATUS RoycDeviceAdd(PKSDEVICE Device)
{
    UNREFERENCED_PARAMETER(Device);
    RoycInfo("DeviceAdd: RoyCam HD Camera");
    /* Stage 3 TODO: IoCreateDevice + symbolic link \\.\RoyCamControl and route
     * IRP_MJ_DEVICE_CONTROL to RoycHandleControlIoctl(). Inert in stage 1. */
    return STATUS_SUCCESS;
}

/* Only Add is needed; the rest of the PnP/power dispatch is zero-filled and
 * KS supplies sensible defaults for a software-enumerated device. */
const KSDEVICE_DISPATCH RoycDeviceDispatch = {
    RoycDeviceAdd,   /* Add */
    NULL,            /* Start */
};

const KSDEVICE_DESCRIPTOR RoycDeviceDescriptor = {
    &RoycDeviceDispatch,
    1,                              /* FilterDescriptorsCount */
    RoycFilterDescriptors,
    0,                              /* Version */
};
