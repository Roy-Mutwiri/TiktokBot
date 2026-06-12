/*
 * filter.c — filter + pin descriptors, dispatch tables, and categories.
 *
 * The filter advertises KSCATEGORY_VIDEO_CAMERA + KSCATEGORY_CAPTURE +
 * KSCATEGORY_VIDEO so the Frame Server / Media Foundation surface it as a
 * camera named "RoyCam HD Camera" (from the INF). One capture (output) pin.
 */
#include "roycam.h"

/* ---- allocator framing: 4 frames, system non-paged, sized to max payload -- */
DECLARE_SIMPLE_FRAMING_EX(
    RoycAllocatorFraming,
    STATICGUIDOF(KSMEMORY_TYPE_KERNEL_NONPAGED),
    KSALLOCATOR_REQUIREMENTF_SYSTEM_MEMORY |
        KSALLOCATOR_REQUIREMENTF_PREFERENCES_ONLY,
    4,
    FILE_QUAD_ALIGNMENT,
    ROYC_MAX_PAYLOAD,
    ROYC_MAX_PAYLOAD);

/* ---- pin dispatch -------------------------------------------------------- */
static const KSPIN_DISPATCH RoycPinDispatch = {
    RoycPinCreate,          /* Create        */
    RoycPinClose,           /* Close         */
    RoycPinProcess,         /* Process       */
    NULL,                   /* Reset         */
    RoycPinSetDataFormat,   /* SetDataFormat */
    RoycPinSetDeviceState,  /* SetDeviceState*/
    NULL,                   /* Connect       */
    NULL,                   /* Disconnect    */
    NULL,                   /* Clock         */
    NULL,                   /* Allocator     */
};

/* ---- pin category/name (video capture output) ---------------------------- */
static const GUID RoycPinCategoryCapture = { 0xfb6c4281, 0x0353, 0x11d1,
    { 0x90, 0x5f, 0x00, 0x00, 0xc0, 0xcc, 0x16, 0xba } };  /* PINNAME_VIDEO_CAPTURE */

/* ---- capture pin descriptor ---------------------------------------------- */
const KSPIN_DESCRIPTOR_EX RoycPinDescriptors[1] = {
    {
        &RoycPinDispatch,
        NULL,                                   /* AutomationTable */
        {
            0, NULL,                            /* Interfaces      */
            0, NULL,                            /* Mediums         */
            RoycFmtCount,                       /* DataRangesCount */
            (const PKSDATARANGE *)RoycPinDataRanges,
            KSPIN_DATAFLOW_OUT,
            KSPIN_COMMUNICATION_BOTH,
            &RoycPinCategoryCapture,            /* Category */
            &RoycPinCategoryCapture,            /* Name     */
            0
        },
        KSPIN_FLAG_PROCESS_IN_RUN_STATE_ONLY |
            KSPIN_FLAG_FIXED_FORMAT,
        1,                                      /* InstancesPossible  */
        0,                                      /* InstancesNecessary */
        &RoycAllocatorFraming,
        RoycIntersectHandler
    }
};

/* ---- filter create: pick the frame source mode --------------------------- */
static NTSTATUS RoycFilterCreate(PKSFILTER Filter, PIRP Irp)
{
    PROYC_FILTER_CONTEXT fctx;
    UNREFERENCED_PARAMETER(Irp);

    fctx = (PROYC_FILTER_CONTEXT)ExAllocatePool2(
        POOL_FLAG_NON_PAGED, sizeof(ROYC_FILTER_CONTEXT), ROYC_POOL_TAG);
    if (fctx == NULL)
        return STATUS_INSUFFICIENT_RESOURCES;
    RtlZeroMemory(fctx, sizeof(*fctx));
    fctx->SourceMode  = RoycSourceSimulated;   /* stage 1 ships active */
    fctx->FrameCounter = 0;
    Filter->Context = fctx;
    RoycInfo("FilterCreate (source=simulated)");
    return STATUS_SUCCESS;
}

static NTSTATUS RoycFilterClose(PKSFILTER Filter, PIRP Irp)
{
    UNREFERENCED_PARAMETER(Irp);
    if (Filter->Context != NULL) {
        ExFreePoolWithTag(Filter->Context, ROYC_POOL_TAG);
        Filter->Context = NULL;
    }
    return STATUS_SUCCESS;
}

/* ---- filter dispatch ----------------------------------------------------- */
static const KSFILTER_DISPATCH RoycFilterDispatch = {
    RoycFilterCreate,   /* Create  */
    RoycFilterClose,    /* Close   */
    NULL,               /* Process */
    NULL,               /* Reset   */
};

/* ---- categories ---------------------------------------------------------- */
static const GUID RoycFilterCategories[] = {
    STATICGUIDOF(KSCATEGORY_VIDEO_CAMERA),
    STATICGUIDOF(KSCATEGORY_CAPTURE),
    STATICGUIDOF(KSCATEGORY_VIDEO),
};

/* ---- filter descriptor --------------------------------------------------- */
const KSFILTER_DESCRIPTOR RoycFilterDescriptor = {
    &RoycFilterDispatch,
    NULL,                                   /* AutomationTable */
    KSFILTER_DESCRIPTOR_VERSION,
    0,                                      /* Flags */
    &KSNAME_Filter,                         /* ReferenceGuid */
    DEFINE_KSFILTER_PIN_DESCRIPTORS(RoycPinDescriptors),
    DEFINE_KSFILTER_CATEGORIES(RoycFilterCategories),
    0, 0, NULL,                             /* Nodes */
    0, NULL,                                /* Connections (default) */
    NULL                                    /* ComponentId */
};

const KSFILTER_DESCRIPTOR *RoycFilterDescriptors[1] = {
    &RoycFilterDescriptor
};
