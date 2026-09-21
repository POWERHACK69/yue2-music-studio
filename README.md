# YuE2 Music Studio (audio.cpp) — Standalone Desktop App

Standalone desktop GUI for **YuE2-3B** song generation powered by the native
[audio.cpp](https://github.com/0xShug0/audio.cpp) inference engine (`audiocpp_cli`).

- **GUI front-end:** CustomTkinter (dark mode, no Qt display-server issues)
- **Inference engine:** bundled `audiocpp_cli` with **all backends compiled into one binary**
  - Windows (`Yue2Studio-Windows-x64.zip`): **CPU + CUDA + Vulkan** — leave the backend on `auto` or pick explicitly
  - Linux (`Yue2Studio-Linux-x64.*`): **CPU + CUDA + Vulkan**
  - macOS Apple Silicon → **CPU + Metal**
- **Model manager:** resumable downloads with per-file + overall progress bars, verify/import for manually downloaded files, free-space pre-check. Weights come from `audio-cpp/Yue2-3B-GGUF` (`q4_0` ~2.6GB / `q8_0` ~4.2GB / `bf16` ~7.2GB, VAE f16 ~265MB / f32 ~531MB + 4 sidecars).
- **Full YuE2 surface:** `cot` off/melody/full, ABC score paste/file (covers & edits), `guidance_scale`, all ABC + semantic sampling knobs, VAE choice, `weight_type`/`attention`, AR/NAR LoRA adapters, `--out-dir` score.abc export, engine diagnostics (`--list-devices`), lyrics load/save + built-in examples.
- **Packaging:** PyInstaller in GitHub Actions — all compilation happens in CI, so you never build locally.

## Downloads (Releases)

| File | Use |
|------|-----|
| `Yue2Studio-Windows-x64.zip` | All backends: `--backend cuda` on NVIDIA (driver 580+, RTX 20+), `--backend vulkan` on AMD/Intel, `--backend cpu` anywhere. CUDA runtime DLLs bundled — no Toolkit needed. |
| `Yue2Studio-Linux-x64.tar.gz` | Portable: `tar -xzf ... && ./Yue2Studio/Yue2Studio` (needs `libsndfile1`: `sudo apt install libsndfile1`) |
| `Yue2Studio-Linux-x64.AppImage` | Portable: `chmod +x ...AppImage && ./Yue2Studio-Linux-x64.AppImage` (needs FUSE: `sudo apt install libfuse2` on newer Ubuntu) |
| `Yue2Studio-Linux-x64.rpm` | Install: `sudo dnf install ./Yue2Studio-Linux-x64.rpm`, then run `Yue2Studio` |
| `Yue2Studio-macOS-arm64.zip` | Metal `.app` for Apple Silicon |

Models are **not** bundled — they download on first run from inside the app.

To save CI time/space: GitHub repo → **Actions** → *Build & Release* → **Run workflow** and tick only the platforms you need (Windows, Linux, macOS, AppImage, RPM, CPU arch `avx2`/`baseline`).

## Project layout

```text
yue2-music-studio/
├── .github/workflows/build-release.yml
├── app/main.py
├── requirements.txt
└── README.md
```

Runtime folders (auto-created, git-ignored except placeholders):

```text
bin/        # place compiled audiocpp_cli[.exe] here for local runs
models/     # downloaded GGUF weights (Yue2-3B-GGUF)
outputs/    # generated .wav files (+ yue2_<seed>_artifacts/score.abc)
```

## A. Run locally

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Place your compiled `audiocpp_cli` binary into `bin/` (or have it on `PATH`).
   Build only the `yue2` family to keep it small. Upstream helper scripts
   (`build_windows.ps1` presets, `build_linux.sh --backend`) each allow only
   ONE GPU backend, so the all-in-one binary is configured with direct CMake
   (CUDA+Vulkan together is allowed — only CUDA+HIP is forbidden):

```powershell
# Windows (CPU+CUDA+Vulkan, portable — needs CUDA Toolkit + Vulkan SDK installed)
cmake -S audio-cpp -B build/windows-all-release -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_NATIVE_CPU=OFF `
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF `
  -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF -DGGML_AVX_VNNI=OFF `
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=yue2
cmake --build build/windows-all-release --target audiocpp_cli
```

```bash
# Linux (CPU+CUDA+Vulkan, portable — needs CUDA Toolkit + libvulkan-dev + glslc)
export CC=gcc-13 CXX=g++-13
cmake -S audio-cpp -B build/linux-all-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_VULKAN=ON -DENGINE_ENABLE_NATIVE_CPU=OFF \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS=yue2
cmake --build build/linux-all-release --target audiocpp_cli
```

```bash
# macOS (Metal) — helper script is fine, it only has one GPU backend anyway
./scripts/build_metal.sh --native-cpu OFF --model-set custom --models yue2 --target audiocpp_cli
```

   Copy **all** `bin/*.dll` next to the exe too (ggml backends, VCOMP140/VCRUNTIME,
   plus cublas/cufft/cudart for CUDA) — a bare exe is not enough on Windows.

3. Run the studio:

```bash
python app/main.py
```

4. In the app: pick quantization + VAE → **Download Model** (watch the progress bars) or **Verify / Import Models** if you already fetched files from Hugging Face (`audio-cpp/Yue2-3B-GGUF`) manually → enter style + lyrics → **Generate Music**.

### CLI invocation used by the GUI

```text
audiocpp_cli --task gen --family yue2 --model <models/Yue2-3B-GGUF>
  --backend <cuda|cpu|vulkan|metal> --threads <N>
  --lyrics "<lyrics>"
  --request-option style="<style>"
  --request-option cot=<off|melody|full>
  --request-option seed=<N> --seed <N>
  --request-option num_inference_steps=<N>
  --request-option guidance_scale=<f>            # alias cfg_scale
  [--request-option abc_file=<score.abc> | --request-option abc=<inline ABC>]
  [--request-option abc_temperature/top_p/top_k/…]
  [--request-option semantic_temperature/top_p/top_k/…]
  --session-option yue2.model_gguf=<q4_0|q8_0|bf16 file>
  --session-option yue2.vae_gguf=<f16|f32 file>
  [--session-option yue2.ar_lora=<file> --session-option yue2.ar_lora_scale=<f>]
  [--session-option yue2.nar_lora=<file> --session-option yue2.nar_lora_scale=<f>]
  [--session-option yue2.weight_type=<native|…> --session-option yue2.attention=<auto|flash|eager>]
  --out outputs/yue2_<seed>.wav [--out-dir outputs/yue2_<seed>_artifacts] [--log]
```

Backend defaults to `auto` in the sidebar (Metal on macOS, CUDA when `nvidia-smi`
exists, otherwise CPU). Use **Engine Info / List Devices** to confirm which
backends your binary actually supports.

Covers: transcribe with SheetSage2, save melody ABC **without** chords, set
`cot=melody`, attach the `.abc` in *Score & Cover*. Edits: run once with
`cot=full` + score export, edit the exported `score.abc`, re-attach it with
`cot=full` and your revised style.

## B. Build with GitHub Actions

1. Push to GitHub, then trigger by tag **or** manual dispatch:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Or: GitHub repo → **Actions** → *Build & Release Yue2 Studio Standalone App* → **Run workflow** (choose Windows CPU/CUDA, Linux, macOS, AppImage/RPM, `avx2` vs `baseline`).

2. Download artifacts from the workflow run (**Artifacts**) or from **Releases**
   (on tag pushes).

### Notes

- CPU kernels are portable by default (`-CpuArch avx2` balance on Windows /
  `--native-cpu OFF` elsewhere: no AVX-512-only instructions). Pick `baseline`
  only for very old CPUs. Never ship `native` builds publicly.
- CUDA builds for RTX 30/40 by default for fast CI — widen coverage in the
  Run workflow picker (`cuda_arch`: `rtx203040`, `gtx10plus`, `legacy` for
  GTX 9xx-era, `wide`, or `full`), or pick `nocuda` for a CPU+Vulkan-only
  build with no toolkit/nvcc (fastest; select `--backend cpu/vulkan` manually
  since `auto` still prefers CUDA when `nvidia-smi` exists) (NOT local-GPU auto-detect),
  bundles the CUDA runtime DLLs on Windows, and still runs `--backend cpu` when
  no NVIDIA GPU is present. Vulkan needs a Vulkan 1.1+ driver (AMD/NVIDIA/Intel).
- If `pygame` cannot initialise audio (headless CI), the GUI still runs and logs
  a warning; generated WAVs remain playable from `outputs/`.
- Tkinter updates from worker threads are marshalled via `after()` so log
  streaming and progress bars are thread-safe.

## Troubleshooting

- **Exit 3221225501 (`0xC000001D`) right after the TRACE lines, model files present.**
  `STATUS_ILLEGAL_INSTRUCTION`: the old `native`-tuned binary used AVX-512 your CPU
  lacks. Solution: re-download the current release (built `-CpuArch avx2` /
  `--native-cpu OFF`), or use the CUDA build. The app now explains this inline
  in the logs instead of failing silently.
- **Backend rejected / slow GPU.** `auto` picks CUDA when `nvidia-smi` exists,
  else CPU — AMD/Intel users should pick `vulkan` manually. Use
  **Engine Info / List Devices** to confirm the binary reports
  `cuda, cpu, vulkan`. CUDA needs driver 580+ (RTX 20+); Vulkan needs a
  Vulkan 1.1+ driver; otherwise use `cpu`.
- **Download stalls / fails.** The app streams with resume + progress bars and falls
  back to the Hub client (install `hf_xet`/`hf_transfer` via `requirements.txt`).
  Partial files live as `.part` and resume automatically. If HF is blocked, download
  from `audio-cpp/Yue2-3B-GGUF` in a browser, then sidebar → **Verify / Import Models**
  and point at that folder — required layout: `<models>/Yue2-3B-GGUF/<quant>.gguf`,
  `<models>/Yue2-3B-GGUF/yue2-vae-*.gguf`, `<models>/Yue2-3B-GGUF/sidecars/*` (4 files).
- **Crash / OOM on CPU.** Verify models first, try `q4_0` + VAE `f16`, lower threads
  and `num_inference_steps`, close other apps. `cot=off` is cheapest; `full` plans first.
