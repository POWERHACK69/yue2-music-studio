"""YuE2 Music Studio — standalone desktop GUI front-end for audio.cpp.

Covers the full YuE2 (audio.cpp `--family yue2`) surface:
  text-to-music (cot off / melody / full), ABC score conditioning for
  covers & edits, guidance scale, full ABC + semantic sampling controls,
  VAE selection, weight types, AR/NAR LoRA adapters, score.abc export,
  engine diagnostics, and a resumable model downloader with progress bars.

Run locally:
    pip install -r requirements.txt
    python app/main.py

Frozen (PyInstaller) layout:
    <BASE_DIR>/bin/audiocpp_cli[.exe] (+ backend DLLs on Windows)
    <BASE_DIR>/models/Yue2-3B-GGUF
    <BASE_DIR>/outputs
"""

import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import webbrowser  # noqa: F401 (kept for future help-link use)

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    from huggingface_hub import HfApi, hf_hub_url
except Exception:  # huggingface_hub not installed yet
    HfApi = None  # type: ignore
    hf_hub_url = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore

# ---------------------------------------------------------------------------
# Portable audio playback (guarded so the GUI still works headless / without
# audio device at import time).
# ---------------------------------------------------------------------------
try:
    import pygame

    try:
        pygame.mixer.init()
        _AUDIO_AVAILABLE = True
    except Exception:  # no audio device in CI / headless boxes
        _AUDIO_AVAILABLE = False
except ImportError:  # pygame not installed
    pygame = None  # type: ignore
    _AUDIO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths — work both normally and as a PyInstaller frozen bundle.
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
    try:
        BUNDLE_DIR = sys._MEIPASS  # type: ignore[attr-defined]
    except AttributeError:
        BUNDLE_DIR = BASE_DIR
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BUNDLE_DIR = BASE_DIR

MODELS_DIR = os.path.join(BASE_DIR, "models", "Yue2-3B-GGUF")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

HF_REPO_ID = "audio-cpp/Yue2-3B-GGUF"

# ---------------------------------------------------------------------------
# Model manifest. Sizes are approximate display values (bytes) used for the
# pre-download space check and progress totals when the Hub API is offline.
# ---------------------------------------------------------------------------
SIDECAR_FILES = [
    "sidecars/yue2-model-config.json",
    "sidecars/yue2-generation-config.json",
    "sidecars/yue2-qwen.tiktoken",
    "sidecars/yue2-vae-config.json",
]

QUANT_CHOICES = [
    "yue2-3b-q4_0.gguf (Fast / ~2.6GB)",
    "yue2-3b-q8_0.gguf (Quality / ~4.2GB)",
    "yue2-3b-bf16.gguf (Full / ~7.2GB)",
]

VAE_CHOICES = [
    "yue2-vae-f16.gguf (Compact / ~265MB)",
    "yue2-vae-f32.gguf (Full / ~531MB)",
]

# NOTE: the Hub reports decimal units (1 GB = 1e9 bytes): q4_0 is exactly
# 2,665,632,320 bytes ("2.67 GB"). A 1024**3 multiplier here overshoots by
# ~7.5%, which put every complete download under the 95% truncation gate in
# check_model_files and reported it "missing or truncated". Keep decimal.
APPROX_SIZES = {
    "yue2-3b-q4_0.gguf": int(2.67 * 1000**3),
    "yue2-3b-q8_0.gguf": int(4.26 * 1000**3),
    "yue2-3b-bf16.gguf": int(7.26 * 1000**3),
    "yue2-vae-f16.gguf": 265 * 1000**2,
    "yue2-vae-f32.gguf": 531 * 1000**2,
}

# Known Windows NTSTATUS crashes mapped to actionable advice.
# 3221225501 == 0xC000001D STATUS_ILLEGAL_INSTRUCTION (the reported failure).
EXIT_CODE_ADVICE = {
    3221225501: (
        "STATUS_ILLEGAL_INSTRUCTION — the bundled CPU kernels used an "
        "instruction your CPU does not support (typically AVX-512 from a "
        "'native'-tuned build running on a CPU without it).\n"
        "Fix: use a portable/balanced build (Windows '-CpuArch avx2', Linux "
        "'--native-cpu OFF'), or the CUDA build on an NVIDIA GPU. "
        "See the Engine tab → 'Engine Info' and the release notes."
    ),
    3221225477: (
        "STATUS_ACCESS_VIOLATION — often out-of-memory or a corrupt/truncated "
        "model file. Verify models (sidebar → Verify), re-download any file "
        "with a size mismatch, try q4_0, and lower CPU threads."
    ),
    3221225781: (
        "STATUS_DLL_NOT_FOUND — a runtime DLL is missing. Keep all DLLs next "
        "to the .exe (VCOMP140.DLL, MSVCP140/VCRUNTIME, plus cublas/cufft for "
        "CUDA builds) or install the VC++ redistributable."
    ),
}


def _exe_name():
    return "audiocpp_cli.exe" if platform.system() == "Windows" else "audiocpp_cli"


def find_audiocpp_cli():
    """Locate audiocpp_cli binary in bundled dir, local bin, or system PATH.

    PyInstaller `--add-binary "bin/x;bin"` lands under `_internal/bin` in
    onedir builds, so both `<exe>/bin` and `<exe>/_internal/bin` are checked,
    plus the frozen bundle dir and PATH.
    """
    bin_name = _exe_name()
    candidates = [
        os.path.join(BASE_DIR, "bin", bin_name),
        os.path.join(BASE_DIR, "_internal", "bin", bin_name),
        os.path.join(BUNDLE_DIR, "bin", bin_name),
        os.path.join(BUNDLE_DIR, "_internal", "bin", bin_name),
        os.path.join(BASE_DIR, bin_name),
        os.path.join(BUNDLE_DIR, bin_name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            if platform.system() == "Windows" or os.access(candidate, os.X_OK):
                return candidate
    return shutil.which(bin_name)


def list_bundled_dlls():
    """Return backend DLLs sitting next to the CLI (Windows diagnostics)."""
    cli = find_audiocpp_cli()
    if not cli:
        return []
    d = os.path.dirname(cli)
    try:
        return sorted(f for f in os.listdir(d) if f.lower().endswith(".dll"))
    except OSError:
        return []


def detect_backend() -> str:
    """Pick a sensible default audio.cpp backend for this machine."""
    system = platform.system()
    if system == "Darwin":
        return "metal"
    if system == "Windows":
        if shutil.which("nvidia-smi") is not None:
            return "cuda"
        return "cpu"
    # Linux
    if os.path.exists("/proc/driver/nvidia") or shutil.which("nvidia-smi") is not None:
        return "cuda"
    return "cpu"


def fmt_mb(n):
    try:
        return f"{n / 1024 / 1024:.1f} MB"
    except Exception:
        return "?"


def explain_exit_code(code):
    if code in EXIT_CODE_ADVICE:
        return EXIT_CODE_ADVICE[code]
    if code is not None and code < 0:
        return f"Process was killed by signal {-code} (often OOM)."
    return ""


def check_model_files(selected_quant, selected_vae):
    """Return {rel_path: (exists_ok, local_bytes, expected_or_None)}."""
    wanted = list(SIDECAR_FILES) + [selected_vae, selected_quant]
    report = {}
    for rel in wanted:
        path = os.path.join(MODELS_DIR, rel)
        if os.path.isfile(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                size = -1
            base = os.path.basename(rel)
            exp = APPROX_SIZES.get(base)
            ok = True
            if exp and size >= 0:
                # Allow 2% tolerance for display-size drift; sidecars are tiny.
                if size < 1024 and rel.startswith("sidecars/"):
                    ok = size > 0
                elif abs(size - exp) / max(exp, 1) > 0.05 and size < exp * 0.95:
                    ok = False  # likely truncated download
            report[rel] = (ok, size, exp)
        else:
            base = os.path.basename(rel)
            report[rel] = (False, -1, APPROX_SIZES.get(base))
    return report


class ModelDownloader(threading.Thread):
    """Resumable file-by-file downloader with GUI progress callbacks.

    Uses plain HTTPS streaming (Range resume to `<dest>.part`) so progress is
    exact and large GGUFs survive restarts. Falls back to
    `huggingface_hub.hf_hub_download` (which handles XET via hf_xet when
    installed) if plain streaming fails.
    """

    def __init__(self, files, on_file_start, on_progress, on_log, cancel_event):
        super().__init__(daemon=True)
        self.files = files
        self.on_file_start = on_file_start  # (index, total, rel, total_bytes|None)
        self.on_progress = on_progress  # (rel, done_bytes, total_bytes|None)
        self.on_log = on_log
        self.cancel_event = cancel_event

    def _stream_one(self, rel, total_hint):
        from huggingface_hub import hf_hub_download as _hf_dl  # local import

        if hf_hub_url is None or requests is None:
            raise RuntimeError("huggingface_hub/requests not available")
        url = hf_hub_url(HF_REPO_ID, rel)
        dest = os.path.join(MODELS_DIR, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {}
        if done > 0:
            headers["Range"] = f"bytes={done}-"
        # Fast path: already complete.
        if os.path.isfile(dest):
            base = os.path.basename(rel)
            exp = total_hint or APPROX_SIZES.get(base)
            if exp and abs(os.path.getsize(dest) - exp) / max(exp, 1) < 0.05:
                self.on_progress(rel, os.path.getsize(dest), exp)
                return True
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {rel}")
                total = r.headers.get("Content-Length")
                total = (int(total) + done) if total is not None else total_hint
                mode = "ab" if (r.status_code == 206 and done > 0) else "wb"
                if mode == "wb":
                    done = 0
                self.on_file_start(rel, total)
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if self.cancel_event.is_set():
                            return False
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        self.on_progress(rel, done, total)
            os.replace(part, dest)
            return True
        except Exception as e:
            self.on_log(f"[Download] streaming failed for {rel}: {e} — trying Hub client…")
            # Fallback: Hub client (XET-aware when hf_xet is installed).
            try:
                _hf_dl(
                    repo_id=HF_REPO_ID,
                    filename=rel,
                    local_dir=MODELS_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
                if os.path.isfile(dest):
                    self.on_progress(rel, os.path.getsize(dest), total_hint)
                    return True
                return False
            except Exception as e2:
                self.on_log(f"[Download Error] {rel}: {e2}")
                return False

    def run(self):
        ok_all = True
        for rel in self.files:
            if self.cancel_event.is_set():
                self.on_log("[Download] cancelled.")
                return
            base = os.path.basename(rel)
            if not self._stream_one(rel, APPROX_SIZES.get(base)):
                ok_all = False
                if self.cancel_event.is_set():
                    return
        self.on_log("[Download] done." if ok_all else "[Download] finished with errors — see above.")


class Yue2StudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YuE2 Music Studio (audio.cpp)")
        self.geometry("1180x830")
        self.minsize(1000, 720)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.cli_path = find_audiocpp_cli()
        self.current_audio_file = None
        self.generated_score_text = None
        self.is_generating = False
        self._proc = None
        self._dl_cancel = threading.Event()
        self._dl_running = False

        self.setup_ui()
        self.check_initial_state()

    # ================= UI =================
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- LEFT SIDEBAR ----------
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(30, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="\U0001f3b5 YuE2 Studio", font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(20, 6), sticky="w")

        self.lbl_engine_status = ctk.CTkLabel(
            self.sidebar, text="Engine: Searching...", font=ctk.CTkFont(size=12)
        )
        self.lbl_engine_status.grid(row=1, column=0, padx=20, pady=2, sticky="w")

        self.btn_engine_info = ctk.CTkButton(
            self.sidebar, text="Engine Info / List Devices", height=28, command=self.show_engine_info
        )
        self.btn_engine_info.grid(row=2, column=0, padx=20, pady=(2, 6), sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Model Quantization:", font=ctk.CTkFont(weight="bold")).grid(
            row=3, column=0, padx=20, pady=(10, 2), sticky="w"
        )
        self.opt_quant = ctk.CTkOptionMenu(self.sidebar, values=QUANT_CHOICES, command=lambda _: self.refresh_model_status())
        self.opt_quant.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        ctk.CTkLabel(self.sidebar, text="VAE:", font=ctk.CTkFont(weight="bold")).grid(
            row=5, column=0, padx=20, pady=(6, 2), sticky="w"
        )
        self.opt_vae = ctk.CTkOptionMenu(self.sidebar, values=VAE_CHOICES, command=lambda _: self.refresh_model_status())
        self.opt_vae.set(VAE_CHOICES[0])
        self.opt_vae.grid(row=6, column=0, padx=20, pady=5, sticky="ew")

        self.lbl_model_status = ctk.CTkLabel(
            self.sidebar, text="Models: unknown", font=ctk.CTkFont(size=11), wraplength=250, justify="left"
        )
        self.lbl_model_status.grid(row=7, column=0, padx=20, pady=2, sticky="w")

        self.btn_download_model = ctk.CTkButton(
            self.sidebar, text="\u2b07 Download Model", command=self.start_model_download
        )
        self.btn_download_model.grid(row=8, column=0, padx=20, pady=5, sticky="ew")

        self.btn_verify_model = ctk.CTkButton(
            self.sidebar, text="Verify / Import Models", fg_color="transparent",
            border_width=1, command=self.verify_or_import_models,
        )
        self.btn_verify_model.grid(row=9, column=0, padx=20, pady=2, sticky="ew")

        self.btn_cancel_dl = ctk.CTkButton(
            self.sidebar, text="Cancel Download", fg_color="gray30", command=self.cancel_download
        )
        self.btn_cancel_dl.grid(row=10, column=0, padx=20, pady=2, sticky="ew")

        # Overall + per-file download progress.
        self.lbl_dl = ctk.CTkLabel(self.sidebar, text="Idle.", font=ctk.CTkFont(size=11))
        self.lbl_dl.grid(row=11, column=0, padx=20, pady=(8, 0), sticky="w")
        self.prog_overall = ctk.CTkProgressBar(self.sidebar)
        self.prog_overall.set(0)
        self.prog_overall.grid(row=12, column=0, padx=20, pady=2, sticky="ew")
        self.prog_file = ctk.CTkProgressBar(self.sidebar)
        self.prog_file.set(0)
        self.prog_file.grid(row=13, column=0, padx=20, pady=2, sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Planning Mode (cot):", font=ctk.CTkFont(weight="bold")).grid(
            row=14, column=0, padx=20, pady=(10, 2), sticky="w"
        )
        self.opt_cot = ctk.CTkSegmentedButton(self.sidebar, values=["off", "full", "melody"])
        self.opt_cot.set("off")
        self.opt_cot.grid(row=15, column=0, padx=20, pady=5, sticky="ew")

        ctk.CTkLabel(self.sidebar, text="Inference Steps:").grid(
            row=16, column=0, padx=20, pady=(6, 2), sticky="w"
        )
        self.slider_steps = ctk.CTkSlider(self.sidebar, from_=4, to=30, number_of_steps=26)
        self.slider_steps.set(8)
        self.slider_steps.grid(row=17, column=0, padx=20, pady=2, sticky="ew")
        self.lbl_steps_val = ctk.CTkLabel(self.sidebar, text="8 steps", font=ctk.CTkFont(size=11))
        self.lbl_steps_val.grid(row=18, column=0, padx=20, pady=0, sticky="w")
        self.slider_steps.configure(command=lambda v: self.lbl_steps_val.configure(text=f"{int(float(v))} steps"))

        ctk.CTkLabel(self.sidebar, text="Guidance Scale:").grid(
            row=19, column=0, padx=20, pady=(6, 2), sticky="w"
        )
        self.slider_guid = ctk.CTkSlider(self.sidebar, from_=0, to=2, number_of_steps=40)
        self.slider_guid.set(1.0)
        self.slider_guid.grid(row=20, column=0, padx=20, pady=2, sticky="ew")
        self.lbl_guid_val = ctk.CTkLabel(self.sidebar, text="1.00 (default 1.0 planned / 1.01 direct)", font=ctk.CTkFont(size=11))
        self.lbl_guid_val.grid(row=21, column=0, padx=20, pady=0, sticky="w")
        self.slider_guid.configure(command=lambda v: self.lbl_guid_val.configure(text=f"{float(v):.2f}"))

        self._cpu_count = os.cpu_count() or 8
        ctk.CTkLabel(self.sidebar, text="CPU Threads:").grid(
            row=22, column=0, padx=20, pady=(6, 2), sticky="w"
        )
        self.slider_threads = ctk.CTkSlider(
            self.sidebar, from_=1, to=max(2, min(32, self._cpu_count)), number_of_steps=max(1, min(31, self._cpu_count - 1))
        )
        self.slider_threads.set(min(8, self._cpu_count))
        self.slider_threads.grid(row=23, column=0, padx=20, pady=2, sticky="ew")
        self.lbl_threads_val = ctk.CTkLabel(
            self.sidebar, text=f"{int(self.slider_threads.get())} threads", font=ctk.CTkFont(size=11)
        )
        self.lbl_threads_val.grid(row=24, column=0, padx=20, pady=0, sticky="w")
        self.slider_threads.configure(
            command=lambda v: self.lbl_threads_val.configure(text=f"{int(float(v))} threads")
        )

        ctk.CTkLabel(self.sidebar, text="Backend:", font=ctk.CTkFont(weight="bold")).grid(
            row=25, column=0, padx=20, pady=(6, 2), sticky="w"
        )
        self.opt_backend = ctk.CTkOptionMenu(
            self.sidebar, values=["auto", "cuda", "cpu", "vulkan", "metal"]
        )
        self.opt_backend.set("auto")
        self.opt_backend.grid(row=26, column=0, padx=20, pady=5, sticky="ew")

        self.btn_open_folder = ctk.CTkButton(
            self.sidebar, text="\U0001f4c2 Open Output Folder", fg_color="transparent",
            border_width=1, command=self.open_output_dir,
        )
        self.btn_open_folder.grid(row=27, column=0, padx=20, pady=(8, 20), sticky="sew")

        # ---------- MAIN WORKSPACE ----------
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=15)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.tabs = ctk.CTkTabview(self.main_frame)
        self.tabs.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        for name in ("Compose", "Score & Cover", "Advanced"):
            self.tabs.add(name)
            self.tabs.tab(name).grid_columnconfigure(0, weight=1)

        self._build_compose_tab()
        self._build_score_tab()
        self._build_advanced_tab()

        # Controls & seed (always visible below tabs)
        self.ctrl_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.ctrl_frame.grid(row=1, column=0, sticky="ew", pady=5)
        self.ctrl_frame.grid_columnconfigure(1, weight=1)

        self.seed_frame = ctk.CTkFrame(self.ctrl_frame, fg_color="transparent")
        self.seed_frame.grid(row=0, column=0, padx=(0, 15), sticky="w")
        self.chk_random_seed = ctk.CTkCheckBox(self.seed_frame, text="Random Seed")
        self.chk_random_seed.select()
        self.chk_random_seed.pack(side="left")
        self.ent_seed = ctk.CTkEntry(self.seed_frame, width=110, placeholder_text="seed")
        self.ent_seed.insert(0, "42")
        self.ent_seed.pack(side="left", padx=(8, 0))

        self.btn_generate = ctk.CTkButton(
            self.ctrl_frame, text="\U0001f3bc Generate Music", height=42,
            font=ctk.CTkFont(size=15, weight="bold"), command=self.start_generation,
        )
        self.btn_generate.grid(row=0, column=1, sticky="ew")

        # Terminal / logs
        ctk.CTkLabel(self.main_frame, text="Execution Logs:", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=2, column=0, sticky="w", pady=(6, 2)
        )
        self.txt_logs = ctk.CTkTextbox(self.main_frame, height=130, font=ctk.CTkFont(family="Courier", size=11))
        self.txt_logs.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        # Audio player bar
        self.player_bar = ctk.CTkFrame(self.main_frame)
        self.player_bar.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.lbl_now_playing = ctk.CTkLabel(self.player_bar, text="Ready.")
        self.lbl_now_playing.pack(side="left", padx=15, pady=10)
        self.btn_play = ctk.CTkButton(self.player_bar, text="\u25b6 Play", width=80, command=self.play_audio)
        self.btn_play.pack(side="right", padx=5, pady=10)
        self.btn_stop = ctk.CTkButton(
            self.player_bar, text="\u23f9 Stop", width=80, fg_color="gray30", command=self.stop_audio
        )
        self.btn_stop.pack(side="right", padx=5, pady=10)

    def _build_compose_tab(self):
        tab = self.tabs.tab("Compose")
        ctk.CTkLabel(tab, text="Music Style & Genre Prompt:", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", pady=(8, 2))
        self.txt_style = ctk.CTkEntry(
            tab, placeholder_text="e.g. English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished mix",
        )
        self.txt_style.insert(0, "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished mix")
        self.txt_style.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.grid(row=2, column=0, sticky="ew", pady=(0, 2))
        ctk.CTkLabel(top, text="Lyrics & Structure ([Verse], [Chorus]):", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        ctk.CTkButton(top, text="Load .txt/.json", width=110, command=self.load_lyrics_file).pack(side="right", padx=4)
        ctk.CTkButton(top, text="Save", width=70, command=self.save_lyrics_file).pack(side="right", padx=4)
        ctk.CTkButton(top, text="Examples", width=90, command=self.show_example_menu).pack(side="right", padx=4)

        self.txt_lyrics = ctk.CTkTextbox(tab, height=170)
        self.txt_lyrics.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        self.txt_lyrics.insert(
            "1.0",
            "[Verse]\nSoft morning light is touching the window.\nI hear the city waking below.\n\n"
            "[Chorus]\nStay with the rhythm, let it carry us home.\nSing with the sunrise, we are never alone.",
        )
        ctk.CTkLabel(
            tab,
            text="Tip: cot=full writes an editable melody+chord plan first (default for new songs). "
                 "cot=melody keeps melody, frees accompaniment (best for covers). cot=off goes direct.",
            font=ctk.CTkFont(size=11), wraplength=700, justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(0, 8))

    def _build_score_tab(self):
        tab = self.tabs.tab("Score & Cover")
        ctk.CTkLabel(tab, text="ABC Score Conditioning (covers & edits):", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", pady=(8, 2))
        self.opt_abc_src = ctk.CTkSegmentedButton(tab, values=["none", "paste", "file"])
        self.opt_abc_src.set("none")
        self.opt_abc_src.grid(row=1, column=0, sticky="w", pady=4)

        file_row = ctk.CTkFrame(tab, fg_color="transparent")
        file_row.grid(row=2, column=0, sticky="ew", pady=2)
        file_row.grid_columnconfigure(0, weight=1)
        self.ent_abc_file = ctk.CTkEntry(file_row, placeholder_text="Path to .abc score file (requires cot=melody/full)")
        self.ent_abc_file.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(file_row, text="Browse", width=90, command=self.browse_abc_file).grid(row=0, column=1)

        self.txt_abc = ctk.CTkTextbox(tab, height=120, font=ctk.CTkFont(family="Courier", size=11))
        self.txt_abc.grid(row=3, column=0, sticky="ew", pady=6)
        self.txt_abc.insert("1.0", "% Paste ABC here when source=paste (e.g. transcribed melody without chords for covers).")

        lora = ctk.CTkFrame(tab, fg_color="transparent")
        lora.grid(row=4, column=0, sticky="ew", pady=4)
        lora.grid_columnconfigure(0, weight=1)
        lora.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(lora, text="AR LoRA (planning):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(lora, text="NAR LoRA (acoustic):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, sticky="w")
        self.ent_ar_lora = ctk.CTkEntry(lora, placeholder_text="optional .safetensors")
        self.ent_ar_lora.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.ent_nar_lora = ctk.CTkEntry(lora, placeholder_text="optional .safetensors")
        self.ent_nar_lora.grid(row=1, column=1, sticky="ew")
        ctk.CTkButton(lora, text="Browse AR", width=90, command=lambda: self._browse_lora(self.ent_ar_lora)).grid(row=2, column=0, sticky="w", pady=2)
        ctk.CTkButton(lora, text="Browse NAR", width=90, command=lambda: self._browse_lora(self.ent_nar_lora)).grid(row=2, column=1, sticky="w", pady=2)
        self.slider_ar_scale = ctk.CTkSlider(lora, from_=0, to=1.5, number_of_steps=30)
        self.slider_ar_scale.set(1.0)
        self.slider_ar_scale.grid(row=3, column=0, sticky="ew", padx=(0, 8))
        self.slider_nar_scale = ctk.CTkSlider(lora, from_=0, to=1.5, number_of_steps=30)
        self.slider_nar_scale.set(1.0)
        self.slider_nar_scale.grid(row=3, column=1, sticky="ew")
        self.lbl_lora = ctk.CTkLabel(lora, text="AR scale 1.00 · NAR scale 1.00 (0 disables)", font=ctk.CTkFont(size=11))
        self.lbl_lora.grid(row=4, column=0, columnspan=2, sticky="w")
        self.slider_ar_scale.configure(command=lambda *_: self._refresh_lora_lbl())
        self.slider_nar_scale.configure(command=lambda *_: self._refresh_lora_lbl())

        btn_row = ctk.CTkFrame(tab, fg_color="transparent")
        btn_row.grid(row=5, column=0, sticky="ew", pady=6)
        ctk.CTkButton(btn_row, text="Save Generated Score", command=self.save_generated_score).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(btn_row, text="After a planned run, score.abc is exported via --out-dir automatically.", font=ctk.CTkFont(size=11)).pack(side="left")

    def _build_advanced_tab(self):
        tab = self.tabs.tab("Advanced")
        ctk.CTkLabel(tab, text="ABC planner sampling:", font=ctk.CTkFont(size=13, weight="bold")).grid(row=0, column=0, sticky="w", pady=(8, 2))
        self.abc_entries = {}
        for i, (key, default) in enumerate([
            ("abc_temperature", "0.7"), ("abc_top_p", "0.9"), ("abc_top_k", "30"),
            ("abc_repetition_penalty", "1.005"), ("abc_min_tokens", "32"), ("abc_max_tokens", "4096"),
        ]):
            r, c = 1 + i // 3, (i % 3) * 2
            ctk.CTkLabel(tab, text=key + ":").grid(row=r, column=c, sticky="e", padx=(0, 4))
            e = ctk.CTkEntry(tab, width=110)
            e.insert(0, default)
            e.grid(row=r, column=c + 1, sticky="w", padx=(0, 12), pady=2)
            self.abc_entries[key] = e
        ctk.CTkLabel(tab, text="Semantic sampling:", font=ctk.CTkFont(size=13, weight="bold")).grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.sem_entries = {}
        for i, (key, default) in enumerate([
            ("semantic_temperature", "1.0"), ("semantic_top_p", "0.95"), ("semantic_top_k", "100"),
            ("semantic_repetition_penalty", "1.2"), ("semantic_min_tokens", "200"), ("semantic_max_tokens", "9000"),
        ]):
            r, c = 4 + i // 3, (i % 3) * 2
            ctk.CTkLabel(tab, text=key + ":").grid(row=r, column=c, sticky="e", padx=(0, 4))
            e = ctk.CTkEntry(tab, width=110)
            e.insert(0, default)
            e.grid(row=r, column=c + 1, sticky="w", padx=(0, 12), pady=2)
            self.sem_entries[key] = e

        misc = ctk.CTkFrame(tab, fg_color="transparent")
        misc.grid(row=6, column=0, columnspan=6, sticky="ew", pady=8)
        ctk.CTkLabel(misc, text="weight_type:").pack(side="left")
        self.opt_wtype = ctk.CTkOptionMenu(misc, values=["native", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_k"], width=110)
        self.opt_wtype.set("native")
        self.opt_wtype.pack(side="left", padx=6)
        ctk.CTkLabel(misc, text="attention:").pack(side="left", padx=(12, 0))
        self.opt_attn = ctk.CTkOptionMenu(misc, values=["auto", "flash", "eager"], width=110)
        self.opt_attn.set("auto")
        self.opt_attn.pack(side="left", padx=6)
        self.chk_score_export = ctk.CTkCheckBox(misc, text="Export score.abc (--out-dir)")
        self.chk_score_export.select()
        self.chk_score_export.pack(side="left", padx=12)
        self.chk_verbose = ctk.CTkCheckBox(misc, text="Verbose (--log)")
        self.chk_verbose.select()
        self.chk_verbose.pack(side="left", padx=6)
        ctk.CTkLabel(
            tab,
            text="Leave sampling boxes at defaults unless you know what you do. "
                 "num_inference_steps (NAR ODE steps) and guidance live in the sidebar/Compose tab.",
            font=ctk.CTkFont(size=11), wraplength=700, justify="left",
        ).grid(row=7, column=0, columnspan=6, sticky="w")

    # ================= Helpers (thread-safe) =================
    def log(self, text):
        def _append():
            try:
                self.txt_logs.insert("end", text + "\n")
                self.txt_logs.see("end")
            except Exception:
                pass
        try:
            self.after(0, _append)
        except Exception:
            _append()

    def set_status(self, text, color=None):
        def _set():
            try:
                self.lbl_engine_status.configure(text=text, **({"text_color": color} if color else {}))
            except Exception:
                pass
        try:
            self.after(0, _set)
        except Exception:
            _set()

    def check_initial_state(self):
        if self.cli_path:
            self.lbl_engine_status.configure(
                text=f"Engine: Ready ({os.path.basename(self.cli_path)})", text_color="green"
            )
            dlls = list_bundled_dlls()
            if dlls:
                self.log(f"[Engine] {self.cli_path}")
                self.log(f"[Engine] bundled DLLs: {', '.join(dlls[:12])}" + (" …" if len(dlls) > 12 else ""))
            else:
                self.log(f"[Engine] {self.cli_path} (no sibling DLLs seen)")
        else:
            self.lbl_engine_status.configure(text="Engine: audiocpp_cli not found!", text_color="red")
            self.log("[Warning] audiocpp_cli was not found. Place it in 'bin/' or add it to PATH.")
        if not _AUDIO_AVAILABLE:
            self.log("[Warning] Audio playback unavailable (pygame mixer could not initialise).")
            self.log("          Generation still works; play the WAV files from the output folder.")
        self.refresh_model_status()
        det = detect_backend()
        self.log(f"[Info] Suggested backend for this machine: {det} (override in sidebar).")

    # ================= Engine diagnostics =================
    def show_engine_info(self):
        if not self.cli_path:
            messagebox.showerror("Engine", "audiocpp_cli not found.")
            return

        def worker():
            self.log(f"[Engine] binary: {self.cli_path}")
            dlls = list_bundled_dlls()
            self.log(f"[Engine] sibling DLLs ({len(dlls)}): {', '.join(dlls) if dlls else '(none — static build, backends compiled in)'}")
            # Authoritative backend check: ask the binary itself. Static builds
            # have no ggml-*.dll files, so DLL names prove nothing.
            try:
                p = subprocess.run([self.cli_path, "--list-devices"], capture_output=True, text=True, timeout=30)
                out = (p.stdout or "") + (p.stderr or "")
                found = sorted({b for b in ("cuda", "vulkan", "cpu", "metal", "hip") if b in out.lower()})
                self.log(f"[Engine] --list-devices (exit {p.returncode}):\n{out[:3000]}")
                self.log(f"[Engine] compiled backends detected: {', '.join(found) if found else '(unparseable — see above)'}")
                if "cuda" not in found and "vulkan" not in found and platform.system() != "Darwin":
                    self.log("[Engine] No GPU backend compiled in → CPU-only binary. "
                             "Re-download the release (Windows/Linux zips ship CPU+CUDA+Vulkan).")
            except Exception as e:
                self.log(f"[Engine] --list-devices failed: {e}")
            try:
                p = subprocess.run([self.cli_path, "--help"], capture_output=True, text=True, timeout=30)
                out = (p.stdout or "") + (p.stderr or "")
                self.log(f"[Engine] --help (exit {p.returncode}):\n{out[:3000]}")
            except Exception as e:
                self.log(f"[Engine] --help failed: {e}")
            smi = shutil.which("nvidia-smi")
            self.log(f"[Engine] nvidia-smi: {'found' if smi else 'NOT found — CPU expected'}")
            if smi:
                try:
                    p = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total",
                                        "--format=csv"], capture_output=True, text=True, timeout=15)
                    self.log("[nvidia-smi]\n" + ((p.stdout or "") + (p.stderr or ""))[:1500])
                except Exception as e:
                    self.log(f"[nvidia-smi] query failed: {e}")

        threading.Thread(target=worker, daemon=True).start()

    # ================= Model status / download =================
    def _selected_files(self):
        quant = self.opt_quant.get().split()[0]
        vae = self.opt_vae.get().split()[0]
        return quant, vae

    def refresh_model_status(self):
        quant, vae = self._selected_files()
        try:
            report = check_model_files(quant, vae)
        except Exception:
            return
        missing = [k for k, (ok, _, _) in report.items() if not ok]
        if not missing:
            self.lbl_model_status.configure(text=f"Models: OK ({quant}, {vae})", text_color="green")
        else:
            self.lbl_model_status.configure(text=f"Models: {len(missing)} file(s) missing", text_color="orange")

    def verify_or_import_models(self):
        quant, vae = self._selected_files()
        report = check_model_files(quant, vae)
        lines = ["Model folder:", MODELS_DIR, ""]
        for rel, (ok, size, exp) in report.items():
            have = fmt_mb(size) if size >= 0 else "missing"
            want = fmt_mb(exp) if exp else "?"
            flag = "OK " if ok else "MISS" if size < 0 else "SIZE?"
            lines.append(f"[{flag}] {rel} — have {have}, expected ~{want}")
        self.log("\n".join(lines))
        missing = [k for k, (ok, _, _) in report.items() if not ok]
        if not missing:
            try:
                self.after(0, lambda: messagebox.showinfo("Models", "All required files verified."))
            except Exception:
                pass
            self.refresh_model_status()
            return
        ans = messagebox.askyesno(
            "Models incomplete",
            f"{len(missing)} file(s) missing or truncated (see logs).\n\n"
            "Yes = pick a folder to IMPORT manually-downloaded files from\n"
            "      (e.g. from Hugging Face: audio-cpp/Yue2-3B-GGUF).\n"
            "No = just close this dialog.",
        )
        if not ans:
            return
        src = filedialog.askdirectory(title="Select folder containing downloaded GGUF + sidecars")
        if not src:
            return
        copied = 0
        for rel, (ok, _, _) in report.items():
            if ok:
                continue
            for cand in (os.path.join(src, rel), os.path.join(src, os.path.basename(rel))):
                if os.path.isfile(cand):
                    dst = os.path.join(MODELS_DIR, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    try:
                        shutil.copy2(cand, dst)
                        copied += 1
                        self.log(f"[Import] {cand} -> {dst}")
                    except Exception as e:
                        self.log(f"[Import Error] {e}")
                    break
        self.log(f"[Import] copied {copied} file(s). Re-running verification…")
        self.verify_or_import_models()

    def cancel_download(self):
        if self._dl_running:
            self._dl_cancel.set()
            self.log("[Download] cancelling…")

    def start_model_download(self):
        if self._dl_running:
            messagebox.showinfo("Download", "A download is already running.")
            return
        quant, vae = self._selected_files()
        files = list(SIDECAR_FILES) + [vae, quant]
        # Pre-check disk space.
        need = sum(APPROX_SIZES.get(os.path.basename(f), 0) for f in files)
        try:
            free = shutil.disk_usage(MODELS_DIR).free
            self.log(f"[Download] need ~{fmt_mb(need)}, free {fmt_mb(free)} at {MODELS_DIR}")
            if free < need:
                if not messagebox.askyesno("Low disk space",
                        f"Need ~{fmt_mb(need)} but only {fmt_mb(free)} free.\nContinue anyway?"):
                    return
        except Exception:
            pass

        self._dl_cancel.clear()
        self._dl_running = True
        try:
            self.btn_download_model.configure(state="disabled", text="Downloading… (see bars)")
        except Exception:
            pass
        self._dl_total = files
        self._dl_done_files = 0
        self._dl_current_total = None

        def on_file_start(rel, total):
            self._dl_current_total = total
            def _u():
                try:
                    self.lbl_dl.configure(text=f"{rel} ({fmt_mb(total) if total else 'size?'})")
                    self.prog_file.set(0)
                except Exception:
                    pass
            try:
                self.after(0, _u)
            except Exception:
                pass
            self.log(f"[Download] {rel} …")

        def on_progress(rel, done, total):
            frac_file = (done / total) if total else 0
            frac_all = (self._dl_done_files + min(frac_file, 1.0)) / max(len(self._dl_total), 1)
            def _u():
                try:
                    self.prog_file.set(max(0.0, min(frac_file, 1.0)))
                    self.prog_overall.set(max(0.0, min(frac_all, 1.0)))
                    t = fmt_mb(total) if total else "?"
                    self.lbl_dl.configure(text=f"{rel}: {fmt_mb(done)} / {t}")
                except Exception:
                    pass
            try:
                self.after(0, _u)
            except Exception:
                pass

        def on_log(msg):
            self.log(msg)

        def worker():
            try:
                dl = ModelDownloader(files, on_file_start, on_progress, on_log, self._dl_cancel)
                # Track per-file completion for the overall bar.
                orig_prog = dl.on_progress
                def counting(rel, done, total):
                    if total and done >= total * 0.999:
                        pass
                    orig_prog(rel, done, total)
                dl.on_progress = counting
                dl.run()
                # Re-count finished files for overall bar correctness.
                self._dl_done_files = sum(
                    1 for f in files if os.path.isfile(os.path.join(MODELS_DIR, f)))
                def _fin():
                    try:
                        self.prog_overall.set(self._dl_done_files / max(len(files), 1))
                    except Exception:
                        pass
                try:
                    self.after(0, _fin)
                except Exception:
                    pass
                if self._dl_cancel.is_set():
                    self.log("[Download] cancelled by user.")
                else:
                    ok = all(os.path.isfile(os.path.join(MODELS_DIR, f)) for f in files)
                    if ok:
                        self.log("[Download] All model weights verified and ready!")
                        try:
                            self.after(0, lambda: messagebox.showinfo("Success", "Model weights downloaded successfully!"))
                        except Exception:
                            pass
                    else:
                        self.log("[Download] Some files are still missing — use Verify/Import or retry.")
            except Exception as e:
                self.log(f"[Download Error] {e}")
            finally:
                self._dl_running = False
                try:
                    self.after(0, lambda: self.btn_download_model.configure(state="normal", text="\u2b07 Download Model"))
                    self.after(0, lambda: self.lbl_dl.configure(text="Idle."))
                except Exception:
                    pass
                try:
                    self.after(0, self.refresh_model_status)
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # ================= Compose helpers =================
    EXAMPLES = {
        "Direct indie pop (cot=off)": (
            "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished demo mix",
            "[Verse]\nSoft morning light is touching the window.\nI hear the city waking below.\n\n"
            "[Chorus]\nStay with the rhythm, let it carry us home.\nSing with the sunrise, we are never alone.",
            "off",
        ),
        "Piano pop full plan (cot=full)": (
            "English, piano pop, clear lead vocal, gentle bass, soft drums, warm chorus harmonies",
            "[Verse]\nA quiet piano opens the evening.\nWarm city lights are starting to glow.\n\n"
            "[Chorus]\nLift every voice into the skyline.\nLet the melody carry us home.",
            "full",
        ),
        "Jazz-funk cover (cot=melody + ABC)": (
            "English, jazz funk cover, warm Rhodes, round bass, light drums, relaxed vocal, clean live band feel",
            "[Verse]\nWe follow the melody line.\nThe rhythm keeps everything fine.\n\n"
            "[Chorus]\nHold that shape and make it new.\nChange the color, keep the view.",
            "melody",
        ),
    }

    def show_example_menu(self):
        names = list(self.EXAMPLES.keys())
        msg = "Examples:\n\n" + "\n".join(f"• {n}" for n in names)
        messagebox.showinfo("Examples", msg + "\n\nChoose one from the next dialog.")
        choice = messagebox.askquestion("Example 1/3", f"Load '{names[0]}'?")
        idx = 0 if choice == "yes" else None
        if idx is None:
            choice = messagebox.askquestion("Example 2/3", f"Load '{names[1]}'?")
            idx = 1 if choice == "yes" else None
        if idx is None:
            choice = messagebox.askquestion("Example 3/3", f"Load '{names[2]}'?")
            idx = 2 if choice == "yes" else None
        if idx is None:
            return
        style, lyrics, cot = self.EXAMPLES[names[idx]]
        self.txt_style.delete(0, "end")
        self.txt_style.insert(0, style)
        self.txt_lyrics.delete("1.0", "end")
        self.txt_lyrics.insert("1.0", lyrics)
        self.opt_cot.set(cot)
        self.log(f"[Example] loaded '{names[idx]}' (cot={cot}). For the cover template, add your .abc in Score & Cover.")

    def load_lyrics_file(self, path=None):
        p = path or filedialog.askopenfilename(
            title="Load lyrics", filetypes=[("Text/JSON", "*.txt *.json *.lyr"), ("All", "*.*")])
        if not p or not os.path.isfile(p):
            return
        try:
            txt = open(p, encoding="utf-8").read()
            import json as _json
            if p.lower().endswith(".json"):
                try:
                    obj = _json.loads(txt)
                    if isinstance(obj, dict) and "lyrics" in obj:
                        if "style" in obj and obj["style"]:
                            self.txt_style.delete(0, "end")
                            self.txt_style.insert(0, str(obj["style"]))
                        txt = str(obj["lyrics"])
                except Exception:
                    pass
            self.txt_lyrics.delete("1.0", "end")
            self.txt_lyrics.insert("1.0", txt)
            self.log(f"[Lyrics] loaded {p}")
        except Exception as e:
            messagebox.showerror("Lyrics", str(e))

    def save_lyrics_file(self):
        p = filedialog.asksaveasfilename(title="Save lyrics", defaultextension=".txt",
                                         filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not p:
            return
        try:
            open(p, "w", encoding="utf-8").write(self.txt_lyrics.get("1.0", "end"))
            self.log(f"[Lyrics] saved {p}")
        except Exception as e:
            messagebox.showerror("Lyrics", str(e))

    def browse_abc_file(self):
        p = filedialog.askopenfilename(title="Select ABC score", filetypes=[("ABC", "*.abc"), ("All", "*.*")])
        if p:
            self.ent_abc_file.delete(0, "end")
            self.ent_abc_file.insert(0, p)
            self.opt_abc_src.set("file")
            try:
                preview = open(p, encoding="utf-8").read()[:2000]
                self.txt_abc.delete("1.0", "end")
                self.txt_abc.insert("1.0", preview)
            except Exception:
                pass

    def _browse_lora(self, entry):
        p = filedialog.askopenfilename(title="Select LoRA adapter",
                                       filetypes=[("SafeTensors", "*.safetensors"), ("All", "*.*")])
        if p:
            entry.delete(0, "end")
            entry.insert(0, p)

    def _refresh_lora_lbl(self):
        try:
            self.lbl_lora.configure(
                text=f"AR scale {float(self.slider_ar_scale.get()):.2f} · NAR scale {float(self.slider_nar_scale.get()):.2f} (0 disables)")
        except Exception:
            pass

    def save_generated_score(self):
        if not self.generated_score_text:
            messagebox.showinfo("Score", "No generated score yet. Run with cot=melody/full and score export on.")
            return
        p = filedialog.asksaveasfilename(title="Save score.abc", defaultextension=".abc",
                                         filetypes=[("ABC", "*.abc")])
        if not p:
            return
        try:
            open(p, "w", encoding="utf-8").write(self.generated_score_text)
            self.log(f"[Score] saved {p}")
        except Exception as e:
            messagebox.showerror("Score", str(e))

    # ================= Generation =================
    def _collect_generation_args(self):
        lyrics = self.txt_lyrics.get("1.0", "end").strip()
        style = self.txt_style.get().strip()
        if not lyrics:
            messagebox.showwarning("Warning", "Please provide lyrics.")
            return None
        if self.chk_random_seed.get():
            seed = random.randint(10000000, 99999999)
        else:
            try:
                seed = int(self.ent_seed.get().strip() or "42")
            except ValueError:
                seed = 42
        quant, vae = self._selected_files()
        cot = self.opt_cot.get()
        steps = int(float(self.slider_steps.get()))
        threads = int(float(self.slider_threads.get()))
        try:
            guidance = float(self.slider_guid.get())
        except Exception:
            guidance = 1.0
        requested_backend = self.opt_backend.get()
        backend = requested_backend if requested_backend != "auto" else detect_backend()

        abc_mode = self.opt_abc_src.get()
        abc_file = self.ent_abc_file.get().strip()
        abc_inline = self.txt_abc.get("1.0", "end").strip()
        if abc_mode != "none" and cot == "off":
            messagebox.showwarning("ABC", "ABC conditioning requires cot=melody or cot=full. Switch planning mode or set ABC source to none.")
            return None
        if abc_mode == "file" and not abc_file:
            messagebox.showwarning("ABC", "ABC source is 'file' but no file selected.")
            return None
        if abc_mode == "file" and not os.path.isfile(abc_file):
            messagebox.showwarning("ABC", f"ABC file not found:\n{abc_file}")
            return None

        return {
            "lyrics": lyrics, "style": style, "seed": seed, "quant": quant, "vae": vae,
            "cot": cot, "steps": steps, "threads": threads, "guidance": guidance,
            "backend": backend, "abc_mode": abc_mode, "abc_file": abc_file,
            "abc_inline": abc_inline if abc_mode == "paste" else "",
        }

    def start_generation(self):
        if self.is_generating:
            return
        if not self.cli_path:
            messagebox.showerror("Error", "audiocpp_cli binary not found in path or 'bin/' directory!")
            return
        args = self._collect_generation_args()
        if args is None:
            return

        # Pre-flight model check with actionable report.
        report = check_model_files(args["quant"], args["vae"])
        missing = [k for k, (ok, _, _) in report.items() if not ok]
        if missing:
            detail = "\n".join(f"• {m}" for m in missing)
            messagebox.showerror(
                "Models missing",
                f"These files are missing or truncated:\n{detail}\n\n"
                f"Folder: {MODELS_DIR}\n\nUse sidebar → Download Model or Verify/Import\n"
                "(you can copy files downloaded from Hugging Face manually).")
            self.log(f"[Preflight] missing: {missing}")
            return

        out_wav = os.path.join(OUTPUT_DIR, f"yue2_{args['seed']}.wav")
        out_dir = os.path.join(OUTPUT_DIR, f"yue2_{args['seed']}_artifacts")

        cmd = [
            self.cli_path,
            "--task", "gen",
            "--family", "yue2",
            "--model", MODELS_DIR,
            "--backend", args["backend"],
            "--threads", str(args["threads"]),
            "--lyrics", args["lyrics"],
            "--request-option", f"style={args['style']}",
            "--request-option", f"cot={args['cot']}",
            "--request-option", f"seed={args['seed']}",
            "--request-option", f"num_inference_steps={args['steps']}",
            "--request-option", f"guidance_scale={args['guidance']:.2f}",
            "--seed", str(args["seed"]),
            "--session-option", f"yue2.model_gguf={args['quant']}",
            "--session-option", f"yue2.vae_gguf={args['vae']}",
            "--out", out_wav,
        ]
        # ABC conditioning.
        if args["abc_mode"] == "file" and args["abc_file"]:
            cmd += ["--request-option", f"abc_file={args['abc_file']}"]
        elif args["abc_mode"] == "paste" and args["abc_inline"]:
            cmd += ["--request-option", f"abc={args['abc_inline']}"]
        # LoRA adapters.
        ar_lora = self.ent_ar_lora.get().strip()
        nar_lora = self.ent_nar_lora.get().strip()
        if ar_lora:
            cmd += ["--session-option", f"yue2.ar_lora={ar_lora}",
                    "--session-option", f"yue2.ar_lora_scale={float(self.slider_ar_scale.get()):.3f}"]
        if nar_lora:
            cmd += ["--session-option", f"yue2.nar_lora={nar_lora}",
                    "--session-option", f"yue2.nar_lora_scale={float(self.slider_nar_scale.get()):.3f}"]
        # Sampling overrides (only send when changed from defaults to keep logs clean).
        def maybe_request(key, entry, default):
            val = entry.get().strip()
            if val and val != default:
                cmd.extend(["--request-option", f"{key}={val}"])
        for k, d in [("abc_temperature", "0.7"), ("abc_top_p", "0.9"), ("abc_top_k", "30"),
                     ("abc_repetition_penalty", "1.005"), ("abc_min_tokens", "32"), ("abc_max_tokens", "4096")]:
            maybe_request(k, self.abc_entries[k], d)
        for k, d in [("semantic_temperature", "1.0"), ("semantic_top_p", "0.95"), ("semantic_top_k", "100"),
                     ("semantic_repetition_penalty", "1.2"), ("semantic_min_tokens", "200"), ("semantic_max_tokens", "9000")]:
            maybe_request(k, self.sem_entries[k], d)
        if self.opt_wtype.get() != "native":
            cmd += ["--session-option", f"yue2.weight_type={self.opt_wtype.get()}"]
        if self.opt_attn.get() != "auto":
            cmd += ["--session-option", f"yue2.attention={self.opt_attn.get()}"]
        if self.chk_score_export.get():
            cmd += ["--out-dir", out_dir]
        if self.chk_verbose.get():
            cmd += ["--log"]

        seed = args["seed"]

        def run_thread():
            self.is_generating = True
            try:
                self.after(0, lambda: self.btn_generate.configure(state="disabled", text="Generating… (see logs)"))
            except Exception:
                pass
            # Log a safe preview (lyrics/ABC can be huge).
            preview = " ".join(cmd[:12]) + f" … --out {out_wav}"
            self.log(f"\n--- Starting Generation [Seed: {seed}] ---")
            self.log(f"Backend={args['backend']} cot={args['cot']} quant={args['quant']} vae={args['vae']}")
            self.log(f"Command: {preview}\n(full argv in output artifacts; lyrics/ABC omitted from log)")

            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                )
                self._proc = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    self.log(line.rstrip())
                proc.wait()

                if proc.returncode == 0 and os.path.exists(out_wav):
                    self.current_audio_file = out_wav
                    self.generated_score_text = None
                    score_path = os.path.join(out_dir, "score.abc")
                    if os.path.isfile(score_path):
                        try:
                            self.generated_score_text = open(score_path, encoding="utf-8").read()
                            self.log(f"[Score] exported {score_path} ({len(self.generated_score_text)} chars). "
                                     "Use Score & Cover → Save Generated Score to keep a copy anywhere.")
                        except Exception as e:
                            self.log(f"[Score] could not read {score_path}: {e}")

                    def _done():
                        self.lbl_now_playing.configure(text=f"Generated: {os.path.basename(out_wav)}")
                    try:
                        self.after(0, _done)
                    except Exception:
                        pass
                    self.log(f"\n[Success] File saved: {out_wav}")
                    self.play_audio()
                else:
                    self.log(f"\n[Failed] Process exited with code {proc.returncode}")
                    advice = explain_exit_code(proc.returncode)
                    if advice:
                        self.log(f"[Diagnosis] {advice}")
                    if proc.returncode == 3221225501:
                        self.log("[Diagnosis] Quick checks: 1) Engine Info → confirm CPU vs CUDA build. "
                                 "2) Re-download the release built with '-CpuArch avx2' / '--native-cpu OFF'. "
                                 "3) If your CPU predates AVX2, use the portable build.")
            except Exception as ex:
                self.log(f"[Execution Error] {str(ex)}")
            finally:
                self._proc = None
                self.is_generating = False
                try:
                    self.after(
                        0, lambda: self.btn_generate.configure(state="normal", text="\U0001f3bc Generate Music")
                    )
                except Exception:
                    pass

        threading.Thread(target=run_thread, daemon=True).start()

    # ================= Playback =================
    def play_audio(self):
        if not self.current_audio_file or not os.path.exists(self.current_audio_file):
            return
        if pygame is None or not _AUDIO_AVAILABLE:
            self.log("[Player] pygame mixer unavailable — open the WAV from the output folder.")
            return
        try:
            pygame.mixer.music.load(self.current_audio_file)
            pygame.mixer.music.play()

            def _label():
                self.lbl_now_playing.configure(
                    text=f"Playing: {os.path.basename(self.current_audio_file)}"
                )
            try:
                self.after(0, _label)
            except Exception:
                pass
        except Exception as e:
            self.log(f"[Player Error] {e}")

    def stop_audio(self):
        if pygame is not None and _AUDIO_AVAILABLE:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def open_output_dir(self):
        system = platform.system()
        try:
            if system == "Windows":
                os.startfile(OUTPUT_DIR)  # type: ignore[attr-defined]
            elif system == "Darwin":
                subprocess.run(["open", OUTPUT_DIR])
            else:
                subprocess.run(["xdg-open", OUTPUT_DIR])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output folder:\n{e}")


if __name__ == "__main__":
    app = Yue2StudioApp()
    app.mainloop()
