//
// AvatarCamFilter.cpp - DirectShow software camera "Avatar Studio Camera"
// -----------------------------------------------------------------------------
// A DirectShow video capture source filter (CSource/CSourceStream). Unlike the
// MFCreateVirtualCamera device, a DirectShow source is NOT tagged with the
// "(Windows Virtual Camera)" suffix and carries NO virtual-camera flag, so apps
// that enumerate DirectShow (OBS, Zoom, Discord, Chrome/Edge, TikTok Live
// Studio) see it as an ordinary webcam.
//
// Frames come from the same shared memory-mapped file the avatar writes
// (SharedFrame.h / avatar_sharedframe.py). Output is RGB24 512x512 @ 30fps.
//
#include <streams.h>
#include <dvdmedia.h>
#include <vector>
#include "SharedFrame.h"
#include <initguid.h>   // must precede our DEFINE_GUID so it allocates storage

#define VCAM_WIDTH  512
#define VCAM_HEIGHT 512
#define VCAM_FPS    30
static const REFERENCE_TIME VCAM_FRAME_TIME = 10000000 / VCAM_FPS; // 100ns units

// {6F2B8C10-1A3D-4E9F-AB12-7C9E10000001}  - our filter's CLSID
DEFINE_GUID(CLSID_AvatarCam,
    0x6f2b8c10, 0x1a3d, 0x4e9f, 0xab, 0x12, 0x7c, 0x9e, 0x10, 0x00, 0x00, 0x01);

static const wchar_t* kFilterName = L"Avatar Studio Camera";

// ---------------------------------------------------------------------------
class CAvatarStream;

class CAvatarCam : public CSource
{
public:
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN lpunk, HRESULT* phr);
private:
    CAvatarCam(LPUNKNOWN lpunk, HRESULT* phr);
};

class CAvatarStream : public CSourceStream, public IKsPropertySet, public IAMStreamConfig
{
public:
    CAvatarStream(HRESULT* phr, CAvatarCam* pParent, LPCWSTR pPinName);
    ~CAvatarStream();

    HRESULT FillBuffer(IMediaSample* pms);
    HRESULT DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* pProp);
    HRESULT CheckMediaType(const CMediaType* pmt);
    HRESULT GetMediaType(int iPosition, CMediaType* pmt);
    HRESULT SetMediaType(const CMediaType* pmt);
    HRESULT OnThreadCreate();
    STDMETHODIMP Notify(IBaseFilter* pSender, Quality q) { return E_NOTIMPL; }

    // IUnknown - expose extra interfaces on the pin, delegate refcount to owner
    STDMETHODIMP QueryInterface(REFIID riid, void** ppv);
    STDMETHODIMP_(ULONG) AddRef()  { return GetOwner()->AddRef(); }
    STDMETHODIMP_(ULONG) Release() { return GetOwner()->Release(); }

    // IKsPropertySet - lets capture apps see this as a CAPTURE pin
    STDMETHODIMP Set(REFGUID g, DWORD id, LPVOID pi, DWORD ci, LPVOID pd, DWORD cd);
    STDMETHODIMP Get(REFGUID g, DWORD id, LPVOID pi, DWORD ci, LPVOID pd, DWORD cd, DWORD* pcb);
    STDMETHODIMP QuerySupported(REFGUID g, DWORD id, DWORD* pSupport);

    // IAMStreamConfig - lets apps query the (single, fixed) capture format
    STDMETHODIMP SetFormat(AM_MEDIA_TYPE* pmt);
    STDMETHODIMP GetFormat(AM_MEDIA_TYPE** ppmt);
    STDMETHODIMP GetNumberOfCapabilities(int* piCount, int* piSize);
    STDMETHODIMP GetStreamCaps(int iIndex, AM_MEDIA_TYPE** ppmt, BYTE* pSCC);

private:
    int m_iFrame = 0;
    REFERENCE_TIME m_rtStream = 0;
    DWORD m_dwLastTick = 0;
    avatarcam::SharedFrameReader m_reader;
    std::vector<BYTE> m_bgra;       // scratch: top-down BGRA pulled from shared mem
};

// ---------------------------------------------------------------------------
CAvatarCam::CAvatarCam(LPUNKNOWN lpunk, HRESULT* phr)
    : CSource(L"AvatarStudioCamera", lpunk, CLSID_AvatarCam)
{
    new CAvatarStream(phr, this, L"Output");   // adds itself to the pin list
}

CUnknown* WINAPI CAvatarCam::CreateInstance(LPUNKNOWN lpunk, HRESULT* phr)
{
    CAvatarCam* p = new CAvatarCam(lpunk, phr);
    if (!p && phr) *phr = E_OUTOFMEMORY;
    return p;
}

// ---------------------------------------------------------------------------
CAvatarStream::CAvatarStream(HRESULT* phr, CAvatarCam* pParent, LPCWSTR pPinName)
    : CSourceStream(L"AvatarStudioStream", phr, pParent, pPinName)
{
    m_bgra.resize((size_t)VCAM_WIDTH * VCAM_HEIGHT * 4);
}

CAvatarStream::~CAvatarStream() {}

STDMETHODIMP CAvatarStream::QueryInterface(REFIID riid, void** ppv)
{
    if (riid == IID_IKsPropertySet) {
        *ppv = static_cast<IKsPropertySet*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == IID_IAMStreamConfig) {
        *ppv = static_cast<IAMStreamConfig*>(this);
        AddRef();
        return S_OK;
    }
    return CSourceStream::QueryInterface(riid, ppv);
}

STDMETHODIMP CAvatarStream::SetFormat(AM_MEDIA_TYPE* pmt)
{
    // We expose exactly one fixed format; accept a matching request, ignore else.
    return S_OK;
}

STDMETHODIMP CAvatarStream::GetFormat(AM_MEDIA_TYPE** ppmt)
{
    if (!ppmt) return E_POINTER;
    *ppmt = CreateMediaType(&m_mt);
    return (*ppmt) ? S_OK : E_OUTOFMEMORY;
}

STDMETHODIMP CAvatarStream::GetNumberOfCapabilities(int* piCount, int* piSize)
{
    if (!piCount || !piSize) return E_POINTER;
    *piCount = 1;
    *piSize = sizeof(VIDEO_STREAM_CONFIG_CAPS);
    return S_OK;
}

STDMETHODIMP CAvatarStream::GetStreamCaps(int iIndex, AM_MEDIA_TYPE** ppmt, BYTE* pSCC)
{
    if (iIndex < 0) return E_INVALIDARG;
    if (iIndex > 0) return S_FALSE;
    if (!ppmt || !pSCC) return E_POINTER;

    CMediaType mt;
    GetMediaType(0, &mt);
    *ppmt = CreateMediaType(&mt);
    if (!*ppmt) return E_OUTOFMEMORY;

    VIDEO_STREAM_CONFIG_CAPS* caps = (VIDEO_STREAM_CONFIG_CAPS*)pSCC;
    ZeroMemory(caps, sizeof(*caps));
    caps->guid = FORMAT_VideoInfo;
    caps->VideoStandard = 0;
    caps->InputSize.cx = VCAM_WIDTH;  caps->InputSize.cy = VCAM_HEIGHT;
    caps->MinCroppingSize = caps->InputSize;
    caps->MaxCroppingSize = caps->InputSize;
    caps->CropGranularityX = caps->CropGranularityY = 1;
    caps->MinOutputSize = caps->InputSize;
    caps->MaxOutputSize = caps->InputSize;
    caps->OutputGranularityX = caps->OutputGranularityY = 1;
    caps->MinFrameInterval = VCAM_FRAME_TIME;
    caps->MaxFrameInterval = VCAM_FRAME_TIME;
    caps->MinBitsPerSecond = caps->MaxBitsPerSecond =
        (LONG)((LONGLONG)VCAM_WIDTH * VCAM_HEIGHT * 3 * 8 * VCAM_FPS);
    return S_OK;
}

STDMETHODIMP CAvatarStream::Set(REFGUID, DWORD, LPVOID, DWORD, LPVOID, DWORD)
{
    return E_NOTIMPL;
}

STDMETHODIMP CAvatarStream::Get(REFGUID guidPropSet, DWORD dwPropID, LPVOID,
                                DWORD, LPVOID pPropData, DWORD cbPropData,
                                DWORD* pcbReturned)
{
    if (guidPropSet != AMPROPSETID_Pin) return E_PROP_SET_UNSUPPORTED;
    if (dwPropID != AMPROPERTY_PIN_CATEGORY) return E_PROP_ID_UNSUPPORTED;
    if (pPropData == NULL && pcbReturned == NULL) return E_POINTER;
    if (pcbReturned) *pcbReturned = sizeof(GUID);
    if (pPropData == NULL) return S_OK;
    if (cbPropData < sizeof(GUID)) return E_UNEXPECTED;
    *(GUID*)pPropData = PIN_CATEGORY_CAPTURE;
    return S_OK;
}

STDMETHODIMP CAvatarStream::QuerySupported(REFGUID guidPropSet, DWORD dwPropID,
                                           DWORD* pTypeSupport)
{
    if (guidPropSet != AMPROPSETID_Pin) return E_PROP_SET_UNSUPPORTED;
    if (dwPropID != AMPROPERTY_PIN_CATEGORY) return E_PROP_ID_UNSUPPORTED;
    if (pTypeSupport) *pTypeSupport = KSPROPERTY_SUPPORT_GET;
    return S_OK;
}

HRESULT CAvatarStream::OnThreadCreate()
{
    m_iFrame = 0; m_rtStream = 0; m_dwLastTick = 0;
    m_reader.Open();   // ok if it fails; FillBuffer retries
    return S_OK;
}

HRESULT CAvatarStream::GetMediaType(int iPosition, CMediaType* pmt)
{
    CAutoLock lock(m_pFilter->pStateLock());
    if (iPosition < 0) return E_INVALIDARG;
    if (iPosition > 0) return VFW_S_NO_MORE_ITEMS;

    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pmt->AllocFormatBuffer(sizeof(VIDEOINFOHEADER));
    if (!pvi) return E_OUTOFMEMORY;
    ZeroMemory(pvi, sizeof(VIDEOINFOHEADER));
    pvi->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    pvi->bmiHeader.biWidth = VCAM_WIDTH;
    pvi->bmiHeader.biHeight = VCAM_HEIGHT;       // positive => bottom-up RGB24
    pvi->bmiHeader.biPlanes = 1;
    pvi->bmiHeader.biBitCount = 24;
    pvi->bmiHeader.biCompression = BI_RGB;
    pvi->bmiHeader.biSizeImage = GetBitmapSize(&pvi->bmiHeader);
    pvi->AvgTimePerFrame = VCAM_FRAME_TIME;
    pvi->dwBitRate = (DWORD)((LONGLONG)pvi->bmiHeader.biSizeImage * 8 * VCAM_FPS);
    SetRectEmpty(&pvi->rcSource);
    SetRectEmpty(&pvi->rcTarget);

    pmt->SetType(&MEDIATYPE_Video);
    pmt->SetFormatType(&FORMAT_VideoInfo);
    pmt->SetTemporalCompression(FALSE);
    const GUID sub = GetBitmapSubtype(&pvi->bmiHeader);  // MEDIASUBTYPE_RGB24
    pmt->SetSubtype(&sub);
    pmt->SetSampleSize(pvi->bmiHeader.biSizeImage);
    return S_OK;
}

HRESULT CAvatarStream::CheckMediaType(const CMediaType* pmt)
{
    if (*pmt->Type() != MEDIATYPE_Video) return E_INVALIDARG;
    if (*pmt->Subtype() != MEDIASUBTYPE_RGB24) return E_INVALIDARG;
    if (*pmt->FormatType() != FORMAT_VideoInfo) return E_INVALIDARG;
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)pmt->Format();
    if (!pvi) return E_INVALIDARG;
    if (pvi->bmiHeader.biWidth != VCAM_WIDTH ||
        abs(pvi->bmiHeader.biHeight) != VCAM_HEIGHT) return E_INVALIDARG;
    return S_OK;
}

HRESULT CAvatarStream::SetMediaType(const CMediaType* pmt)
{
    CAutoLock lock(m_pFilter->pStateLock());
    return CSourceStream::SetMediaType(pmt);
}

HRESULT CAvatarStream::DecideBufferSize(IMemAllocator* pAlloc, ALLOCATOR_PROPERTIES* pProp)
{
    CAutoLock lock(m_pFilter->pStateLock());
    VIDEOINFOHEADER* pvi = (VIDEOINFOHEADER*)m_mt.Format();
    pProp->cBuffers = 1;
    pProp->cbBuffer = pvi->bmiHeader.biSizeImage;
    ALLOCATOR_PROPERTIES actual;
    HRESULT hr = pAlloc->SetProperties(pProp, &actual);
    if (FAILED(hr)) return hr;
    if (actual.cbBuffer < pProp->cbBuffer) return E_FAIL;
    return S_OK;
}

HRESULT CAvatarStream::FillBuffer(IMediaSample* pms)
{
    BYTE* pData = NULL;
    if (FAILED(pms->GetPointer(&pData))) return E_FAIL;
    const long lSize = pms->GetSize();
    const int W = VCAM_WIDTH, H = VCAM_HEIGHT;
    const int rgbStride = ((W * 3 + 3) & ~3);          // DWORD-aligned RGB24 stride
    if (lSize < rgbStride * H) return E_FAIL;

    // Pull the latest avatar frame (top-down BGRA) from shared memory.
    bool got = false;
    if (!m_reader.IsOpen()) m_reader.Open();
    if (m_reader.IsOpen())
        got = m_reader.ReadBGRA(m_bgra.data(), (DWORD)m_bgra.size(), W * 4, W, H);

    if (got) {
        // BGRA top-down -> RGB24 bottom-up (DShow stores RGB bottom-up).
        for (int y = 0; y < H; ++y) {
            const BYTE* src = m_bgra.data() + (size_t)y * W * 4;
            BYTE* dst = pData + (size_t)(H - 1 - y) * rgbStride;
            for (int x = 0; x < W; ++x) {
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; // B,G,R
                dst += 3; src += 4;
            }
        }
    } else {
        // standby: calm dark slate
        for (int y = 0; y < H; ++y) {
            BYTE* dst = pData + (size_t)y * rgbStride;
            BYTE v = (BYTE)(10 + (y * 18) / H);
            for (int x = 0; x < W; ++x) { dst[0]=(BYTE)(v+6); dst[1]=v; dst[2]=v; dst += 3; }
        }
    }

    // timestamps
    REFERENCE_TIME rtStart = m_rtStream;
    m_rtStream += VCAM_FRAME_TIME;
    REFERENCE_TIME rtStop = m_rtStream;
    pms->SetTime(&rtStart, &rtStop);
    pms->SetSyncPoint(TRUE);

    // pace ~30fps in wall-clock so consumers see a steady live stream
    DWORD now = GetTickCount();
    if (m_dwLastTick != 0) {
        DWORD elapsed = now - m_dwLastTick;
        DWORD target = 1000 / VCAM_FPS;
        if (elapsed < target) Sleep(target - elapsed);
    }
    m_dwLastTick = GetTickCount();
    m_iFrame++;
    return S_OK;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
const AMOVIESETUP_MEDIATYPE AMSMediaType = { &MEDIATYPE_Video, &MEDIASUBTYPE_NULL };
const AMOVIESETUP_PIN AMSPin = {
    L"Output", FALSE, TRUE, FALSE, FALSE, &CLSID_NULL, NULL, 1, &AMSMediaType };
const AMOVIESETUP_FILTER AMSFilter = {
    &CLSID_AvatarCam, L"Avatar Studio Camera", MERIT_DO_NOT_USE, 1, &AMSPin };

CFactoryTemplate g_Templates[] = {
    { L"Avatar Studio Camera", &CLSID_AvatarCam, CAvatarCam::CreateInstance, NULL, &AMSFilter }
};
int g_cTemplates = sizeof(g_Templates) / sizeof(g_Templates[0]);

STDAPI DllRegisterServer()
{
    HRESULT hr = AMovieDllRegisterServer2(TRUE);
    if (FAILED(hr)) return hr;

    IFilterMapper2* pFM2 = NULL;
    hr = CoCreateInstance(CLSID_FilterMapper2, NULL, CLSCTX_INPROC_SERVER,
                          IID_IFilterMapper2, (void**)&pFM2);
    if (FAILED(hr)) return hr;

    REGFILTER2 rf2;
    rf2.dwVersion = 1;
    rf2.dwMerit = MERIT_DO_NOT_USE;
    rf2.cPins = 1;
    rf2.rgPins = &AMSPin;
    hr = pFM2->RegisterFilter(CLSID_AvatarCam, kFilterName, NULL,
                              &CLSID_VideoInputDeviceCategory, NULL, &rf2);
    pFM2->Release();
    return hr;
}

STDAPI DllUnregisterServer()
{
    IFilterMapper2* pFM2 = NULL;
    HRESULT hr = CoCreateInstance(CLSID_FilterMapper2, NULL, CLSCTX_INPROC_SERVER,
                                  IID_IFilterMapper2, (void**)&pFM2);
    if (SUCCEEDED(hr) && pFM2) {
        pFM2->UnregisterFilter(&CLSID_VideoInputDeviceCategory, NULL, CLSID_AvatarCam);
        pFM2->Release();
    }
    return AMovieDllRegisterServer2(FALSE);
}

extern "C" BOOL WINAPI DllEntryPoint(HINSTANCE, ULONG, LPVOID);
BOOL APIENTRY DllMain(HANDLE hModule, DWORD dwReason, LPVOID lpReserved)
{
    return DllEntryPoint((HINSTANCE)hModule, dwReason, lpReserved);
}
