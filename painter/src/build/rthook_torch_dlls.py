"""
PyInstaller runtime hook: add DLL search paths on Windows.

torch.__init__ loads DLLs from torch/lib/ using LoadLibraryExW with
restricted search flags (LOAD_LIBRARY_SEARCH_DEFAULT_DIRS). In a frozen
app, dependencies like the VC++ runtime end up in _internal/ which is
outside those restricted paths. This hook registers both _internal/ and
_internal/torch/lib/ via os.add_dll_directory so all deps are findable.
"""
import os
import sys

if sys.platform == "win32":
    base = getattr(sys, "_MEIPASS", None)
    if base:
        # _internal/ itself (contains VC++ runtime, ucrtbase, etc.)
        os.add_dll_directory(base)
        # torch/lib/ (contains c10.dll, CUDA DLLs, etc.)
        torch_lib = os.path.join(base, "torch", "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
        # Also prepend to PATH as a fallback for older-style lookups
        os.environ["PATH"] = (
            torch_lib + os.pathsep + base + os.pathsep
            + os.environ.get("PATH", "")
        )
