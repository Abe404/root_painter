from pathlib import Path
import PyInstaller.__main__

HERE = Path(__file__).resolve().parent          # trainer/src/build
SRC  = HERE.parent                               # trainer/src
ENTRY = SRC / "main.py"

DIST = SRC / "dist"
WORK = SRC / "build" / "_pyi_work"

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

    # Exclude unused CUDA/torch packages to reduce build size (~2.3GB savings)
    "--exclude-module", "triton",         # torch.compile JIT, not needed
    "--exclude-module", "nvidia.nccl",    # multi-GPU communication
    "--exclude-module", "nvidia.nvshmem", # multi-node communication
    "--exclude-module", "nvidia.cufft",   # FFT operations
    "--exclude-module", "nvidia.cusparselt",  # sparse matrix ops
    "--exclude-module", "nvidia.cusparse",    # sparse matrix ops

    "--distpath", str(DIST),
    "--workpath", str(WORK),
    str(ENTRY),
])

print("Built:", DIST / "RootPainterTrainer")
