# YuE2 Music Studio (audio.cpp) — Standalone Desktop App

Standalone desktop GUI for **YuE2-3B** song generation powered by the native
[audio.cpp](https://github.com/0xShug0/audio.cpp) inference engine (`audiocpp_cli`).

- **GUI front-end:** CustomTkinter (dark mode, no Qt display-server issues)
- **Inference engine:** bundled `audiocpp_cli` (built with portable CPU kernels)
  - Windows: **CUDA build** (`Yue2Studio-Windows-x64-CUDA.zip`, includes CPU fallback) and **CPU-only portable build** (`Yue2Studio-Windows-x64.zip`)
  - macOS Apple Silicon → **Metal**
  - Linux → **CPU portable**
- **Model manager:** resumable downloads with per-file + overall progress bars, verify/import for manually downloaded files, free-space pre-check. Weights come from `audio-cpp/Yue2-3B-GGUF` (`q4_0` ~2.6GB / `q8_0` ~4.2GB / `bf16` ~7.2GB, VAE f16 ~265MB / f32 ~531MB + 4 sidecars).
- **Full YuE2 surface:** `cot` off/melody/full, ABC score paste/file (covers & edits), `guidance_scale`, all ABC + semantic sampling knobs, VAE choice, `weight_type`/`attention`, AR/NAR LoRA adapters, `--out-dir` score.abc export, engine diagnostics (`--list-devices`), lyrics load/save + built-in examples.
- **Packaging:** PyInstaller in GitHub Actions — all compilation happens in CI, so you never build locally.

## Downloads (Releases)

| File | Use |
|------|-----|
| `Yue2Studio-Windows-x64-CUDA.zip` | **Recommended for NVIDIA GPUs** (RTX 20/30/40/50, driver 580+). Runs `--backend cuda`, falls back to `--backend cpu`. Needs no CUDA Toolkit (runtime DLLs bundled). |
| `Yue2Studio-Windows-x64.zip` | CPU-only portable build. Use when you have no NVIDIA GPU or are short on disk. |
| `Yue2Studio-Linux-x64.tar.gz` | Portable: `tar -xzf ... && ./Yue2Studio/Yue2Studio` (needs `libsndfile1`: `sudo apt install libsndfile1`) |
| `Yue2Studio-Linux-x64.AppImage` | Portable: `chmod +x ...AppImage && ./Yue2Studio-Linux-x64.AppImage` (needs FUSE: `sudo apt install libfuse2` on newer Ubuntu) |
| `Yue2Studio-Linux-x64.rpm` | Install: `sudo dnf install ./Yue2Studio-Linux-x64.rpm`, then run `Yue2Studio` |
| `Yue2Studio-macOS-arm64.zip` | Metal `.app` for Apple Silicon |

Models are **not** bundled — they download on first run from inside the app.

To save CI time/space: GitHub repo → **Actions** → *Build & Release* → **Run workflow** and tick only the platforms you need (Windows CPU/CUDA, Linux, macOS, AppImage, RPM, CPU arch `avx2`/`baseline`).

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
   Build only the `yue2` family to keep it small, and use **distributable**
   CPU flags (never `native` for releases):

```bash
# Linux (CPU portable)
./scripts/build_linux.sh --backend cpu --native-cpu OFF --model-set custom --models yue2 --target audiocpp_cli

# macOS (Metal)
./scripts/build_metal.sh --native-cpu OFF --model-set custom --models yue2 --target audiocpp_cli
```

```powershell
# Windows (CPU portable — the crash fix for 0xC000001D)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 -Preset windows-cpu-release -CpuArch avx2 -ModelSet custom -Models "yue2" -Target audiocpp_cli

# Windows (CUDA, includes CPU backend; portable multi-arch, not local-GPU-only)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 -Preset windows-cuda-release -CpuArch avx2 -CudaArchitectures default -ModelSet custom -Models "yue2" -Target audiocpp_cli
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

- Windows releases use `-CpuArch avx2` (**balance**) by default: no AVX-512-only
  instructions, runs on most modern PCs. Pick `baseline` (+ slowest, most
  compatible) only for very old CPUs. Never ship `-CpuArch native` publicly.
- Linux builds use `--native-cpu OFF`; macOS Metal uses `--native-cpu OFF`.
- Windows CUDA uses `-CudaArchitectures default` (portable 75/80/86/89/120a/121a
  list), bundles `ggml-cuda`, MSVC/OpenMP and CUDA runtime DLLs, and still runs
  `--backend cpu` when no NVIDIA GPU is present.
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
- **CUDA backend missing** (`--backend cuda` rejected, no `*cuda*.dll` next to the exe).
  You have the CPU-only zip. Download `Yue2Studio-Windows-x64-CUDA.zip`, keep all
  DLLs next to the exe, update the NVIDIA driver (580+), and pick `cuda` (or `auto`
  with `nvidia-smi` present).
- **Download stalls / fails.** The app streams with resume + progress bars and falls
  back to the Hub client (install `hf_xet`/`hf_transfer` via `requirements.txt`).
  Partial files live as `.part` and resume automatically. If HF is blocked, download
  from `audio-cpp/Yue2-3B-GGUF` in a browser, then sidebar → **Verify / Import Models**
  and point at that folder — required layout: `<models>/Yue2-3B-GGUF/<quant>.gguf`,
  `<models>/Yue2-3B-GGUF/yue2-vae-*.gguf`, `<models>/Yue2-3B-GGUF/sidecars/*` (4 files).
- **Crash / OOM on CPU.** Verify models first, try `q4_0` + VAE `f16`, lower threads
  and `num_inference_steps`, close other apps. `cot=off` is cheapest; `full` plans first.
