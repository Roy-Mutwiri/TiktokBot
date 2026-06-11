# =============================================================================
# brand_camera.py  —  rename the virtual camera to YOUR brand (Phase 2)
# -----------------------------------------------------------------------------
# avatar_camera.py publishes into the system's virtual-camera device, which
# Windows ships labelled "OBS Virtual Camera". This tool rebrands that device so
# every app's camera menu shows YOUR name instead (e.g. "Avatar Studio Camera").
#
# It only edits the device's display name in the registry — it does NOT touch
# the driver, so the camera keeps working and avatar_camera.py automatically
# reports the new name. Fully reversible: the original names are saved to
# brand_camera_backup.json and restored with --restore.
#
#   python brand_camera.py "Avatar Studio Camera"   # set the brand
#   python brand_camera.py --show                    # show current name(s)
#   python brand_camera.py --restore                 # put the old name back
#
# Needs administrator rights (it writes HKLM). If you launch it un-elevated it
# re-launches itself with a UAC prompt automatically.
#
# Note: re-installing / repairing OBS, or toggling OBS's own virtual camera, can
# reset the name to "OBS Virtual Camera". Just run this again to re-apply.
# =============================================================================

import os
import sys
import json
import ctypes
import winreg

# DirectShow "Video Capture Sources" filter category — every webcam (real or
# virtual) registers an Instance under here; FriendlyName is what apps display.
VIDEO_CAT = "{860BB310-5D01-11d0-BD3B-00A0C911CE86}"

# The device we rebrand is whichever instance currently carries this name.
DEFAULT_SOURCE_NAME = "OBS Virtual Camera"

BACKUP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "brand_camera_backup.json")

# Both registry views so 64-bit AND 32-bit apps see the brand.
_VIEWS = (
    ("64", r"SOFTWARE\Classes\CLSID"),
    ("32", r"SOFTWARE\Classes\WOW6432Node\CLSID"),
)


def _is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _elevate():
    """Re-launch this script elevated (UAC prompt), passing the same args."""
    params = " ".join(f'"{a}"' for a in sys.argv)
    rc = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable, params, None, 1)
    # ShellExecuteW returns >32 on success.
    return rc > 32


def _find_instances(source_name):
    """Return [instance_clsid, ...] whose FriendlyName == source_name (64-bit view)."""
    base = r"SOFTWARE\Classes\CLSID\%s\Instance" % VIDEO_CAT
    out = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except FileNotFoundError:
        return out
    try:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            try:
                k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base + "\\" + sub)
                fn, _ = winreg.QueryValueEx(k, "FriendlyName")
                winreg.CloseKey(k)
                if fn == source_name:
                    out.append(sub)
            except OSError:
                continue
    finally:
        winreg.CloseKey(root)
    return out


def _targets(instance_clsid):
    """All (hive_path, value_name) pairs holding the display name for one device.

    value_name "" means the key's default value.
    """
    t = []
    for _tag, clsid_base in _VIEWS:
        inst = r"%s\%s\Instance\%s" % (clsid_base, VIDEO_CAT, instance_clsid)
        t.append((inst, "FriendlyName"))
        t.append((r"%s\%s" % (clsid_base, instance_clsid), ""))  # filter's own name
    return t


def _read(path, value_name):
    try:
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
    except FileNotFoundError:
        return None
    try:
        v, _ = winreg.QueryValueEx(k, value_name)
        return v
    except FileNotFoundError:
        return None
    finally:
        winreg.CloseKey(k)


def _write(path, value_name, new_value):
    k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_SET_VALUE)
    try:
        winreg.SetValueEx(k, value_name, 0, winreg.REG_SZ, new_value)
    finally:
        winreg.CloseKey(k)


def show(source_name):
    insts = _find_instances(source_name)
    base = r"SOFTWARE\Classes\CLSID\%s\Instance" % VIDEO_CAT
    print("Virtual-camera devices registered on this PC:")
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            fn = _read(base + "\\" + sub, "FriendlyName")
            print(f"   {sub}  =>  {fn!r}")
        winreg.CloseKey(root)
    except FileNotFoundError:
        print("   (none)")
    print(f"\nMatching source name {source_name!r}: "
          f"{len(insts)} device(s) -> {insts or '[]'}")


def apply_brand(new_name, source_name):
    insts = _find_instances(source_name)
    if not insts:
        print(f"[brand] no device currently named {source_name!r} was found.")
        print("[brand] is OBS / the virtual camera installed? "
              "Run:  python avatar_camera.py --list")
        return 1

    backup = {"source_name": source_name, "values": []}
    changed = 0
    for inst in insts:
        for path, vname in _targets(inst):
            cur = _read(path, vname)
            if cur is None:
                continue
            backup["values"].append({"path": path, "value": vname, "old": cur})
            try:
                _write(path, vname, new_name)
                changed += 1
            except PermissionError:
                print(f"[brand] ACCESS DENIED writing {path} — run as administrator.")
                return 1
            except OSError as exc:
                print(f"[brand] could not write {path}\\{vname or '(default)'}: {exc}")

    # Only overwrite the backup if we don't already have a pristine one, so a
    # second re-brand doesn't lose the TRUE original name.
    if not os.path.exists(BACKUP_PATH):
        with open(BACKUP_PATH, "w", encoding="utf-8") as f:
            json.dump(backup, f, indent=2)
    print(f"[brand] renamed {len(insts)} device(s) to {new_name!r} "
          f"({changed} registry value(s) updated).")
    print(f"[brand] original names saved to {os.path.basename(BACKUP_PATH)} "
          f"(use --restore to undo).")
    print("[brand] Re-open the camera menu in your app to see the new name.")
    return 0


def restore():
    if not os.path.exists(BACKUP_PATH):
        print(f"[brand] no backup file at {BACKUP_PATH} — nothing to restore.")
        return 1
    with open(BACKUP_PATH, "r", encoding="utf-8") as f:
        backup = json.load(f)
    n = 0
    for item in backup.get("values", []):
        try:
            _write(item["path"], item["value"], item["old"])
            n += 1
        except PermissionError:
            print("[brand] ACCESS DENIED — run as administrator.")
            return 1
        except OSError as exc:
            print(f"[brand] restore skip {item['path']}: {exc}")
    print(f"[brand] restored {n} value(s) to the original name "
          f"({backup.get('source_name')!r}).")
    try:
        os.remove(BACKUP_PATH)
    except OSError:
        pass
    return 0


def main(argv):
    args = argv[1:]
    if "--help" in args or "-h" in args or not args:
        print(__doc__)
        return 0

    source_name = DEFAULT_SOURCE_NAME
    if "--from" in args:
        i = args.index("--from")
        if i + 1 < len(args):
            source_name = args[i + 1]
            del args[i:i + 2]

    # --show doesn't need admin.
    if "--show" in args:
        show(source_name)
        return 0

    # Everything below writes HKLM -> require elevation, self-elevate if needed.
    if not _is_admin():
        print("[brand] needs administrator rights — relaunching with a UAC prompt...")
        if _elevate():
            return 0
        print("[brand] could not elevate. Right-click your terminal -> "
              "'Run as administrator' and rerun this command.")
        return 1

    if "--restore" in args:
        return restore()

    # First positional non-flag arg is the new brand name.
    name = next((a for a in args if not a.startswith("-")), None)
    if not name:
        print('[brand] give a name, e.g.:  python brand_camera.py "Avatar Studio Camera"')
        return 2
    return apply_brand(name, source_name)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
