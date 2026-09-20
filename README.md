# YuE2 Music Studio (audio.cpp) — Standalone Desktop App

Standalone desktop GUI for **YuE2-3B** song generation powered by the native
[audio.cpp](https://github.com/0xShug0/audio.cpp) inference engine (`audiocpp_cli`).

- **GUI front-end:** CustomTkinter (dark mode, no Qt display-server issues)
- **Inference engine:** bundled `audiocpp_cli`
  - macOS Apple Silicon → **Metal**
  - Windows → **CUDA / CPU** (auto-detected)
  - Linux → **CUDA / CPU** (auto-detected)
- **Model manager:** one-click download of GGUF weights (`q4_0`, `q8_0`, `bf16`) + sidecars/VAE from `audio-cpp/Yue2-3B-GGUF` via `huggingface_hub`
- **Packaging:** PyInstaller in GitHub Actions
  - Windows: `Yue2Studio-Windows-x64.zip` (portable `.exe`)
  - macOS: `Yue2Studio-macOS-arm64.zip` (`.app`, Metal)
  - Linux: `Yue2Studio-Linux-x64.tar.gz` (portable), `Yue2Studio-Linux-x64.AppImage`, `Yue2Studio-Linux-x64.rpm`

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
outputs/    # generated .wav files
```

## A. Run locally

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Place your compiled `audiocpp_cli` binary into `bin/` (or have it on `PATH`).
   Build only the `yue2` family to keep it small, e.g.:

```bash
# Linux (CPU)
./scripts/build_linux.sh --backend cpu --model-set custom --models yue2 --target audiocpp_cli

# Linux (CUDA)
./scripts/build_linux.sh --backend cuda --model-set custom --models yue2 --target audiocpp_cli

# macOS (Metal)
./scripts/build_metal.sh --model-set custom --models yue2 --target audiocpp_cli
```

```powershell
# Windows (CPU)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build_windows.ps1 -Preset windows-cpu-release -ModelSet custom -Models "yue2" -Target audiocpp_cli
```

3. Run the studio:

```bash
python app/main.py
```

4. In the app: pick a quantization → **Download / Check Model** → enter style + lyrics → **Generate Music**.

### CLI invocation used by the GUI

```text
audiocpp_cli --task gen --family yue2 --model <models/Yue2-3B-GGUF>
  --backend <metal|cuda|cpu> --threads <N>
  --text "<lyrics>"
  --request-option style="<style>"
  --request-option cot=<off|full|melody>
  --request-option seed=<N>
  --request-option num_inference_steps=<N>
  --session-option yue2.model_gguf=<q4_0|q8_0|bf16 file>
  --session-option yue2.vae_gguf=yue2-vae-f16.gguf
  --out outputs/yue2_<seed>.wav --log
```

Backend defaults to `auto` in the sidebar (Metal on macOS, CUDA when `nvidia-smi`
exists, otherwise CPU). Override it explicitly if needed.

## B. Build with GitHub Actions

1. Push to GitHub:

```bash
git init
git add .
git commit -m "feat: Initial commit for YuE2 Studio standalone desktop app"
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

2. Trigger the build by pushing a tag **or** manual dispatch:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Or: GitHub repo → **Actions** → *Build & Release Yue2 Studio Standalone App* → **Run workflow**.

3. Download artifacts from the workflow run (**Artifacts**) or from **Releases**
   (on tag pushes). Linux ships three formats:

   | File | How to use |
   |------|------------|
   | `Yue2Studio-Linux-x64.tar.gz` | Portable: `tar -xzf ... && ./Yue2Studio/Yue2Studio` |
   | `Yue2Studio-Linux-x64.AppImage` | Portable: `chmod +x ...AppImage && ./Yue2Studio-Linux-x64.AppImage` (needs FUSE: `sudo apt install libfuse2` on newer Ubuntu) |
   | `Yue2Studio-Linux-x64.rpm` | Install: `sudo dnf install ./Yue2Studio-Linux-x64.rpm`, then run `Yue2Studio` |

   Models are **not** bundled — they download on first run
   (~2.6 GB for `q4_0`, ~4.2 GB for `q8_0`, ~7.2 GB for `bf16`).

### Notes

- Linux CI builds on Ubuntu 22.04 (older glibc for wider compatibility) with
  GCC 13 from the Ubuntu toolchain PPA + `libsndfile1-dev`; Windows CI uses the
  `windows-cpu-release` preset; macOS CI (`macos-14`) builds the Metal backend.
- If `pygame` cannot initialise audio (headless CI), the GUI still runs and logs
  a warning; generated WAVs remain playable from `outputs/`.
- Tkinter updates from worker threads are marshalled via `after()` so log
  streaming is thread-safe.
