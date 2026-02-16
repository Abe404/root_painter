from pathlib import Path
import os
import shutil
import subprocess
import sys
import PyInstaller.__main__

HERE = Path(__file__).resolve().parent          # trainer/src/build
SRC  = HERE.parent                               # trainer/src
ENTRY = SRC / "main.py"

DIST = SRC / "dist"
WORK = SRC / "build" / "_pyi_work"

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


def _get_needed_symbols(libtorch_path, soname):
    """Extract undefined symbols that reference a given shared library."""
    result = subprocess.run(
        ["nm", "-D", str(libtorch_path)],
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
else:
    print("Stub replacement is Linux-only, skipping on", sys.platform)
