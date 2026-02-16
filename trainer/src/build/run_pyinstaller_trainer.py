from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import PyInstaller.__main__

HERE = Path(__file__).resolve().parent          # trainer/src/build
SRC  = HERE.parent                               # trainer/src
ENTRY = SRC / "main.py"

DIST = SRC / "dist"
WORK = SRC / "build" / "_pyi_work"

# ---------------------------------------------------------------------------
# Shared helpers for building CUDA stub .so files.
# ---------------------------------------------------------------------------

def _get_needed_symbols(lib_path, soname):
    """Extract undefined symbols that reference a given shared library."""
    result = subprocess.run(
        ["nm", "-D", str(lib_path)],
        capture_output=True, text=True, check=True,
    )
    symbols = []
    tag = f"@{soname}"
    for line in result.stdout.splitlines():
        if " U " in line and tag in line:
            # line looks like: "                 U cusparseCreate@libcusparse.so.12"
            sym = line.split()[-1].split("@")[0]
            symbols.append(sym)
    return symbols


def _build_stub(soname, symbols, out_path):
    """Compile a minimal .so that exports the given symbols as no-ops."""
    out_path = Path(out_path)
    stub_c = out_path.with_suffix(".c")
    ver_script = out_path.with_suffix(".ver")

    with open(stub_c, "w") as f:
        f.write(f"// Auto-generated stub for {soname}\n")
        for sym in symbols:
            f.write(f"void {sym}(void) {{ }}\n")

    with open(ver_script, "w") as f:
        f.write(f"{soname} {{ global: *; }};\n")

    subprocess.run([
        "gcc", "-shared", "-o", str(out_path), str(stub_c),
        f"-Wl,-soname,{soname}",
        f"-Wl,--version-script,{ver_script}",
    ], check=True)

    stub_c.unlink()
    ver_script.unlink()


def _find_libtorch_cuda():
    """Find libtorch_cuda.so in the installed torch package.

    Uses importlib to locate the torch package directory without actually
    importing torch (which would fail if CUDA stubs are missing).
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("torch")
        if spec and spec.origin:
            torch_lib = Path(spec.origin).parent / "lib"
            candidate = torch_lib / "libtorch_cuda.so"
            if candidate.exists():
                return candidate
    except (ImportError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------
# Ensure nvidia pip package .so files are discoverable by PyInstaller.
#
# When building without a system CUDA toolkit (e.g. CI runners), the CUDA
# shared libraries live inside site-packages/nvidia/*/lib/.  PyInstaller
# needs them on LD_LIBRARY_PATH to resolve binary dependencies.
# ---------------------------------------------------------------------------
def _nvidia_lib_dirs():
    """Find all lib/ dirs inside installed nvidia pip packages."""
    try:
        import nvidia
        nvidia_root = Path(nvidia.__path__[0])
    except (ImportError, AttributeError):
        return []
    return sorted(str(p) for p in nvidia_root.rglob("lib") if p.is_dir())

_nv_dirs = _nvidia_lib_dirs()
if _nv_dirs:
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(_nv_dirs + ([existing] if existing else []))
    print(f"Added {len(_nv_dirs)} nvidia lib dirs to LD_LIBRARY_PATH")

# ---------------------------------------------------------------------------
# Create stub .so files for CUDA libraries that aren't installed (cusparse,
# cufft, cusolver).  torch tries to dlopen these at import time; without them
# PyInstaller's analysis fails with "Failed to collect submodules for torch"
# and it misses dynamically-imported modules like torch._dynamo.polyfills.fx.
#
# The stubs must export the actual symbols that libtorch_cuda.so references
# (not just be empty .so files), otherwise the dynamic linker raises
# "undefined symbol" errors when torch tries to import.
# ---------------------------------------------------------------------------
_STUB_SONAMES = ["libcusparse.so.12", "libcufft.so.11", "libcusolver.so.11"]
_stub_dir = None

if sys.platform == "linux" and shutil.which("gcc") and shutil.which("nm"):
    _stub_dir = tempfile.mkdtemp(prefix="cuda_stubs_")
    _libtorch = _find_libtorch_cuda()

    for _soname in _STUB_SONAMES:
        _stub_path = os.path.join(_stub_dir, _soname)
        # Only create stub if the real library isn't already on LD_LIBRARY_PATH
        if not any(os.path.exists(os.path.join(d, _soname)) for d in
                   os.environ.get("LD_LIBRARY_PATH", "").split(":")):
            # Extract the symbols libtorch_cuda.so needs from this library
            symbols = []
            if _libtorch:
                symbols = _get_needed_symbols(_libtorch, _soname)
            if symbols:
                _build_stub(_soname, symbols, _stub_path)
                print(f"Created pre-analysis stub: {_soname} ({len(symbols)} symbols)")
            else:
                # Fallback: empty stub (enough for dlopen, but not symbol resolution)
                _stub_c = _stub_path + ".c"
                with open(_stub_c, "w") as _f:
                    _f.write("// empty stub\n")
                subprocess.run([
                    "gcc", "-shared", "-o", _stub_path, _stub_c,
                    f"-Wl,-soname,{_soname}",
                ], check=True)
                os.unlink(_stub_c)
                print(f"Created pre-analysis stub: {_soname} (empty, no libtorch_cuda found)")

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = _stub_dir + ":" + existing

PyInstaller.__main__.run([
    "--noconfirm",
    "--clean",
    "--name", "RootPainterTrainer",
    "--console",

    # CRITICAL: make trainer/src importable during analysis (so 'trainer.py' is found)
    "--paths", str(SRC),

    # Ensure local module is bundled
    "--hidden-import", "trainer",

    # Torch bits
    "--hidden-import", "torch",
    "--hidden-import", "torchvision",
    "--hidden-import", "torchvision.ops",
    "--hidden-import", "torchvision.io",
    "--hidden-import", "torchvision.transforms",

    # Exclude unused CUDA/torch packages to reduce build size
    # Note: with the custom minimal PyTorch wheel, nvidia.nccl/nvshmem/cufft/
    # cusparselt/cusparse/triton are not installed at all. These excludes are
    # kept as defensive no-ops in case the build ever uses stock PyTorch.
    "--exclude-module", "triton",
    "--exclude-module", "nvidia.nccl",
    "--exclude-module", "nvidia.nvshmem",
    "--exclude-module", "nvidia.cufft",
    "--exclude-module", "nvidia.cusparselt",
    "--exclude-module", "nvidia.cusparse",

    # Note: torch subpackages (distributed, testing, _inductor) cannot be safely
    # excluded — they are imported internally by torch's own modules.  The real
    # size savings come from the custom wheel (USE_DISTRIBUTED=0 etc.) and the
    # post-build stub replacement below.

    "--distpath", str(DIST),
    "--workpath", str(WORK),
    str(ENTRY),
])

BUNDLE = DIST / "RootPainterTrainer"
print("Built:", BUNDLE)

# ---------------------------------------------------------------------------
# Post-build: replace unused CUDA shared libraries with tiny stubs.
#
# The custom PyTorch wheel is built with USE_CUSPARSE=0 / USE_CUFFT=0, so
# RootPainter never calls into these libraries.  However libtorch_cuda.so
# still has DT_NEEDED entries for them (transitive deps from cuBLAS/cuDNN),
# so the dynamic linker requires them at load time.
#
# We replace the real libraries (~870 MB total) with tiny stubs (~30 KB)
# that export the required symbol names.  The stubs are never called.
# ---------------------------------------------------------------------------
STUB_LIBS = {
    # soname -> search pattern for the bundled file
    "libcusparse.so.12": "libcusparse.so.12",
    "libcufft.so.11": "libcufft.so.11",
}


def replace_with_stubs(bundle_dir):
    """Find real CUDA libs in the bundle and replace with stubs."""
    internal = bundle_dir / "_internal"
    if not internal.is_dir():
        print("  _internal/ not found, skipping stub replacement")
        return

    # Find libtorch_cuda.so in the bundle
    libtorch_candidates = list(internal.rglob("libtorch_cuda.so"))
    if not libtorch_candidates:
        print("  libtorch_cuda.so not found in bundle, skipping")
        return
    libtorch = libtorch_candidates[0]

    saved = 0
    for soname, pattern in STUB_LIBS.items():
        # Get the symbols libtorch_cuda.so needs from this library
        symbols = _get_needed_symbols(libtorch, soname)
        if not symbols:
            print(f"  {soname}: no symbols referenced, skipping")
            continue

        # Find the real library in the bundle (if present)
        matches = list(internal.glob(pattern))
        if matches:
            real_lib = matches[0]
            old_size = real_lib.stat().st_size
        else:
            # Library not bundled (e.g. CI without CUDA toolkit) — create stub
            real_lib = internal / soname
            old_size = 0

        # Build stub (replace real lib, or create from scratch)
        _build_stub(soname, symbols, real_lib)
        new_size = real_lib.stat().st_size
        saved += old_size - new_size
        if old_size > 0:
            print(f"  {soname}: {old_size // 1024 // 1024}MB -> {new_size // 1024}KB "
                  f"({len(symbols)} stub symbols)")
        else:
            print(f"  {soname}: created stub {new_size // 1024}KB "
                  f"({len(symbols)} symbols)")

    # Also remove libcusolver if present (not in NEEDED, purely transitive)
    for extra in ["libcusolver.so.11"]:
        matches = list(internal.glob(extra))
        for m in matches:
            sz = m.stat().st_size
            m.unlink()
            saved += sz
            print(f"  {extra}: deleted ({sz // 1024 // 1024}MB)")

    print(f"  Total saved: {saved // 1024 // 1024}MB")


def deduplicate_nvidia_libs(bundle_dir):
    """Remove duplicate .so files under _internal/nvidia/ that also exist at _internal/.

    PyInstaller bundles nvidia pip package .so files twice:
      1. _internal/*.so  (binary dependency resolution, used via RPATH)
      2. _internal/nvidia/*/lib/*.so  (package data collection)
    The top-level copies are the ones the dynamic linker finds via RPATH.
    The nvidia/ copies are dead weight — delete them to save space.
    """
    internal = bundle_dir / "_internal"
    nvidia_dir = internal / "nvidia"
    if not nvidia_dir.is_dir():
        print("  _internal/nvidia/ not found, skipping dedup")
        return

    saved = 0
    deduped = 0
    for so_file in sorted(nvidia_dir.rglob("*.so*")):
        if not so_file.is_file():
            continue
        # Check if same filename exists at _internal/ top level
        top_level = internal / so_file.name
        if top_level.is_file() and top_level != so_file:
            sz = so_file.stat().st_size
            so_file.unlink()
            saved += sz
            deduped += 1

    print(f"  Deduplicated {deduped} nvidia .so files, saved {saved // 1024 // 1024}MB")


print()
if sys.platform == "linux":
    if not shutil.which("gcc") or not shutil.which("nm"):
        print("WARNING: gcc or nm not found — skipping CUDA stub replacement.")
        print("  Install with: sudo apt-get install build-essential binutils")
        print("  Bundle will work but be ~870MB larger than necessary.")
    else:
        print("Replacing unused CUDA libraries with stubs...")
        replace_with_stubs(BUNDLE)
        print("Done.")

    print("\nDeduplicating nvidia .so files...")
    deduplicate_nvidia_libs(BUNDLE)
    print("Done.")
else:
    print("Stub replacement is Linux-only, skipping on", sys.platform)
