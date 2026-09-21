Place your compiled audiocpp_cli binary here (or ensure it is on PATH).
On Windows also copy ALL sibling DLLs (if any) plus the CUDA runtime DLLs
(cudart/cublas/cublasLt/cufft) and MSVC/OpenMP runtimes — see the release
workflow for the exact bundling steps.

Each release binary has ALL backends compiled in (Windows/Linux: CPU+CUDA+Vulkan,
macOS: CPU+Metal). Upstream helper scripts only allow ONE GPU backend per
preset, so the all-in-one binary is configured with direct CMake
(CUDA+Vulkan together is allowed — only CUDA+HIP is forbidden).
Never ship `native`-tuned CPU kernels publicly (they emit AVX-512 and crash
with 0xC000001D on other PCs):

  Windows (CPU+CUDA+Vulkan; needs CUDA Toolkit + Vulkan SDK):
    cmake -S audio-cpp -B build/windows-all-release -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_NATIVE_CPU=OFF `
      -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF `
      -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF `
      -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=yue2
    cmake --build build/windows-all-release --target audiocpp_cli

  Linux (CPU+CUDA+Vulkan; needs CUDA Toolkit + libvulkan-dev + glslc):
    export CC=gcc-13 CXX=g++-13
    cmake -S audio-cpp -B build/linux-all-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_NATIVE_CPU=OFF \
      -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=yue2
    cmake --build build/linux-all-release --target audiocpp_cli

  macOS (CPU+Metal; helper script is fine — one GPU backend anyway):
    ./scripts/build_metal.sh --native-cpu OFF --model-set custom --models yue2 --target audiocpp_cli
