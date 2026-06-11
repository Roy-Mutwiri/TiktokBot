//
// SharedFrame.h  —  native reader for the avatar shared-frame file
// -----------------------------------------------------------------------------
// Mirror of avatar_sharedframe.py. The Python avatar process writes BGRA frames
// into a memory-mapped file under C:\Users\Public; this reader (compiled into
// the virtual-camera media source DLL, which the Windows Camera Frame Server
// loads) maps the same file and copies the latest tear-free frame out.
//
// A plain file is used on purpose: creating a Global\ section needs admin
// privilege, but a file in the world-readable Public folder is reachable by the
// Frame Server with no special rights, even across sessions.
//
#pragma once
#include <windows.h>
#include <cstdint>
#include <string>

namespace avatarcam
{
    static const uint32_t kMagic = 0x31435641; // 'AVC1'
    static const uint32_t kVersion = 1;
    static const uint32_t kHeaderSize = 64;
    static const uint32_t kSeqOffset = 24;

#pragma pack(push, 1)
    struct Header
    {
        uint32_t magic;     // 0
        uint32_t version;   // 4
        uint32_t width;     // 8
        uint32_t height;    // 12
        uint32_t fourcc;    // 16  (0 = RGB32 / BGRA top-down)
        uint32_t _pad;      // 20
        uint64_t seq;       // 24  seqlock: odd = writing, even = stable
    };
#pragma pack(pop)

    // Default agreed path. Resolve %PUBLIC% at runtime to be robust.
    inline std::wstring DefaultPath()
    {
        wchar_t buf[MAX_PATH] = {};
        DWORD n = GetEnvironmentVariableW(L"PUBLIC", buf, MAX_PATH);
        std::wstring base = (n > 0 && n < MAX_PATH) ? std::wstring(buf)
                                                    : L"C:\\Users\\Public";
        return base + L"\\AvatarStudioCamera\\frame.bin";
    }

    // Maps the shared file read-only and serves the most recent frame as BGRA.
    class SharedFrameReader
    {
    public:
        SharedFrameReader() = default;
        ~SharedFrameReader() { Close(); }

        // Open the mapping. Returns false if the file is not present/sized yet;
        // callers should retry (the avatar may start after the camera opens).
        bool Open(const std::wstring& path = DefaultPath())
        {
            Close();
            m_path = path;
            m_file = CreateFileW(path.c_str(), GENERIC_READ,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (m_file == INVALID_HANDLE_VALUE) return false;

            LARGE_INTEGER sz{};
            if (!GetFileSizeEx(m_file, &sz) || sz.QuadPart < (LONGLONG)kHeaderSize)
            {
                Close(); return false;
            }
            m_size = (size_t)sz.QuadPart;
            m_mapping = CreateFileMappingW(m_file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (!m_mapping) { Close(); return false; }
            m_view = (BYTE*)MapViewOfFile(m_mapping, FILE_MAP_READ, 0, 0, 0);
            if (!m_view) { Close(); return false; }
            return true;
        }

        bool IsOpen() const { return m_view != nullptr; }

        // Copy the latest BGRA frame into pDst (len bytes, stride dstPitch,
        // top-down). Returns true if a valid, stable frame was produced.
        // width/height must match the negotiated media type.
        bool ReadBGRA(BYTE* pDst, DWORD len, LONG dstPitch, UINT width, UINT height)
        {
            if (!m_view) return false;
            volatile Header* h = reinterpret_cast<volatile Header*>(m_view);
            if (h->magic != kMagic) return false;
            if (h->width != width || h->height != height) return false;

            const size_t frameBytes = (size_t)width * height * 4;
            if (m_size < kHeaderSize + frameBytes) return false;

            // seqlock read: retry until the sequence is stable and even.
            for (int attempt = 0; attempt < 8; ++attempt)
            {
                uint64_t s1 = h->seq;
                if (s1 == 0 || (s1 & 1)) { Sleep(0); continue; }

                const BYTE* src = m_view + kHeaderSize;
                const LONG srcPitch = (LONG)width * 4;
                for (UINT row = 0; row < height; ++row)
                {
                    memcpy(pDst + row * dstPitch, src + row * srcPitch, srcPitch);
                }

                uint64_t s2 = h->seq;
                if (s1 == s2) return true;     // no writer mid-update -> good
            }
            return false;
        }

        void Close()
        {
            if (m_view) { UnmapViewOfFile(m_view); m_view = nullptr; }
            if (m_mapping) { CloseHandle(m_mapping); m_mapping = nullptr; }
            if (m_file != INVALID_HANDLE_VALUE) { CloseHandle(m_file); m_file = INVALID_HANDLE_VALUE; }
            m_size = 0;
        }

    private:
        std::wstring m_path;
        HANDLE m_file = INVALID_HANDLE_VALUE;
        HANDLE m_mapping = nullptr;
        BYTE* m_view = nullptr;
        size_t m_size = 0;
    };
}
