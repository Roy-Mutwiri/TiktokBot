//
// vcam_host.cpp  —  Avatar Studio Camera host (MFCreateVirtualCamera)
// -----------------------------------------------------------------------------
// Creates a Windows 11 virtual camera backed by our registered media source DLL
// and holds it open. Lifetime = Session, so the camera DEVICE appears in every
// app's camera list the moment this process starts and DISAPPEARS when it exits
// (Ctrl+C, window close, or the parent avatar_camera.py terminating it).
//
// The frames themselves come from the media source DLL, which reads the live
// avatar out of the shared file (SharedFrame.h) written by avatar_sharedframe.py.
//
//   vcam_host.exe "Avatar Studio Camera"
//
// Built by native_camera\build.ps1 after the media source DLL is registered.
//
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfvirtualcamera.h>
#include <wrl/client.h>
#include <cstdio>

#pragma comment(lib, "mfsensorgroup.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")

using Microsoft::WRL::ComPtr;

// CLSID of the registered Avatar Studio Camera media source (regsvr32'd DLL).
// Must match VirtualCameraMediaSource's CLSID_VirtualCameraMediaSource.
static const wchar_t* kSourceId = L"{7B89B92E-FE71-42D0-8A41-E137D06EA184}";

static volatile bool g_run = true;
static BOOL WINAPI CtrlHandler(DWORD) { g_run = false; return TRUE; }

int wmain(int argc, wchar_t** argv)
{
    const wchar_t* name = (argc > 1) ? argv[1] : L"Avatar Studio Camera";
    SetConsoleCtrlHandler(CtrlHandler, TRUE);

    HRESULT hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) { wprintf(L"MFStartup failed 0x%08X\n", hr); return 1; }

    ComPtr<IMFVirtualCamera> vcam;
    hr = MFCreateVirtualCamera(
        MFVirtualCameraType_SoftwareCameraSource,
        MFVirtualCameraLifetime_Session,         // <- vanishes when we exit
        MFVirtualCameraAccess_CurrentUser,
        name,
        kSourceId,
        nullptr,    // category list
        0,          // category count
        &vcam);
    if (FAILED(hr) || !vcam)
    {
        wprintf(L"MFCreateVirtualCamera failed 0x%08X\n", hr);
        if (hr == HRESULT_FROM_WIN32(ERROR_NOT_FOUND) || hr == REGDB_E_CLASSNOTREG)
            wprintf(L"  -> is the media source DLL registered?  run install.ps1\n");
        MFShutdown();
        return 1;
    }

    // Start() is REQUIRED for the camera to surface to consumers / the DShow
    // bridge. It returns E_ACCESSDENIED in some contexts but the camera still
    // works, so we log-and-continue rather than abort.
    hr = vcam->Start(nullptr);
    if (FAILED(hr))
        wprintf(L"(Start returned 0x%08X - continuing)\n", hr);

    wprintf(L"Avatar Studio Camera LIVE as \"%s\". Close this to remove it.\n", name);
    fflush(stdout);

    while (g_run) Sleep(150);

    vcam->Stop();
    vcam->Remove();
    vcam.Reset();
    MFShutdown();
    wprintf(L"Avatar Studio Camera removed.\n");
    return 0;
}
