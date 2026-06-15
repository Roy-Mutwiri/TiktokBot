import os
import sys


bundle_dir = getattr(
    sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
playwright_browsers = os.path.join(bundle_dir, "ms-playwright")
if os.path.isdir(playwright_browsers):
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = playwright_browsers
