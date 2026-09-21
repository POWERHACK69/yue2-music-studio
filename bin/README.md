Place your compiled audiocpp_cli binary here (or ensure it is on PATH).
On Windows also copy ALL sibling DLLs (ggml backends, VCOMP140/VCRUNTIME,
plus cublas/cufft/cudart for CUDA) — a bare exe will not run elsewhere.

Build only the yue2 family, with DISTRIBUTABLE CPU flags
(never ship `-CpuArch native` / `--native-cpu ON` publicly —
they emit AVX-512 and crash with 0xC000001D on other PCs):
  Linux:  ./scripts/build_linux.sh --backend cpu --native-cpu OFF --model-set custom --models yue2 --target audiocpp_cli
  macOS:  ./scripts/build_metal.sh --native-cpu OFF --model-set custom --models yue2 --target audiocpp_cli
  Windows CPU (PowerShell):
    .\scripts\build_windows.ps1 -Preset windows-cpu-release -CpuArch avx2 -ModelSet custom -Models "yue2" -Target audiocpp_cli
  Windows CUDA, portable multi-arch (includes CPU backend):
    .\scripts\build_windows.ps1 -Preset windows-cuda-release -CpuArch avx2 -CudaArchitectures default -ModelSet custom -Models "yue2" -Target audiocpp_cli
