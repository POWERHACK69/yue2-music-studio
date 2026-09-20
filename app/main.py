"""YuE2 Music Studio — standalone desktop GUI front-end for audio.cpp.

Handles model downloading, CLI path auto-detection, lyrics structuring,
real-time log streaming, and audio playback.

Run locally:
    pip install -r requirements.txt
    python app/main.py

Frozen (PyInstaller) layout:
    <BASE_DIR>/bin/audiocpp_cli[.exe]  +  <BASE_DIR>/models  +  <BASE_DIR>/outputs
"""

import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import webbrowser  # noqa: F401  (kept for future help-link use)

import customtkinter as ctk
from tkinter import filedialog, messagebox  # noqa: F401 (filedialog kept for future use)

from huggingface_hub import hf_hub_download, snapshot_download

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

QUANT_CHOICES = [
    "yue2-3b-q4_0.gguf (Fast / ~2.6GB)",
    "yue2-3b-q8_0.gguf (Quality / ~4.2GB)",
    "yue2-3b-bf16.gguf (Full / ~7.2GB)",
]


def find_audiocpp_cli():
    """Locate audiocpp_cli binary in bundled dir, local bin, or system PATH."""
    bin_name = "audiocpp_cli.exe" if platform.system() == "Windows" else "audiocpp_cli"
    candidates = [
        os.path.join(BUNDLE_DIR, "bin", bin_name),
        os.path.join(BUNDLE_DIR, bin_name),
        os.path.join(BASE_DIR, "bin", bin_name),
        os.path.join(BASE_DIR, bin_name),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            if platform.system() == "Windows" or os.access(candidate, os.X_OK):
                return candidate
    return shutil.which(bin_name)


def detect_backend() -> str:
    """Pick a sensible default audio.cpp backend for this machine."""
    system = platform.system()
    if system == "Darwin":
        return "metal"
    if system == "Windows":
        # Use CUDA only when an NVIDIA driver looks present; else CPU.
        if shutil.which("nvidia-smi") is not None:
            return "cuda"
        return "cpu"
    # Linux
    if os.path.exists("/proc/driver/nvidia") or shutil.which("nvidia-smi") is not None:
        return "cuda"
    return "cpu"


class Yue2StudioApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("YuE2 Music Studio (audio.cpp)")
        self.geometry("1100x780")
        self.minsize(950, 700)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")

        self.cli_path = find_audiocpp_cli()
        self.current_audio_file = None
        self.is_generating = False
        self._proc = None

        self.setup_ui()
        self.check_initial_state()

    # ================= UI =================
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------- LEFT SIDEBAR ----------
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(
            self.sidebar, text="\U0001f3b5 YuE2 Studio", font=ctk.CTkFont(size=22, weight="bold")
        ).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        # Engine status
        self.lbl_engine_status = ctk.CTkLabel(
            self.sidebar, text="Engine: Searching...", font=ctk.CTkFont(size=12)
        )
        self.lbl_engine_status.grid(row=1, column=0, padx=20, pady=2, sticky="w")

        # Model quantization selector
        ctk.CTkLabel(self.sidebar, text="Model Quantization:", font=ctk.CTkFont(weight="bold")).grid(
            row=2, column=0, padx=20, pady=(15, 2), sticky="w"
        )
        self.opt_quant = ctk.CTkOptionMenu(self.sidebar, values=QUANT_CHOICES)
        self.opt_quant.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

        # Download model button
        self.btn_download_model = ctk.CTkButton(
            self.sidebar, text="\u2b07 Download / Check Model", command=self.start_model_download
        )
        self.btn_download_model.grid(row=4, column=0, padx=20, pady=5, sticky="ew")

        # Planning mode (cot)
        ctk.CTkLabel(self.sidebar, text="Planning Mode (cot):", font=ctk.CTkFont(weight="bold")).grid(
            row=5, column=0, padx=20, pady=(15, 2), sticky="w"
        )
        self.opt_cot = ctk.CTkSegmentedButton(self.sidebar, values=["off", "full", "melody"])
        self.opt_cot.set("off")
        self.opt_cot.grid(row=6, column=0, padx=20, pady=5, sticky="ew")

        # Inference steps
        ctk.CTkLabel(self.sidebar, text="Inference Steps:").grid(
            row=7, column=0, padx=20, pady=(10, 2), sticky="w"
        )
        self.slider_steps = ctk.CTkSlider(self.sidebar, from_=4, to=30, number_of_steps=26)
        self.slider_steps.set(8)
        self.slider_steps.grid(row=8, column=0, padx=20, pady=2, sticky="ew")
        self.lbl_steps_val = ctk.CTkLabel(self.sidebar, text="8 steps", font=ctk.CTkFont(size=11))
        self.lbl_steps_val.grid(row=9, column=0, padx=20, pady=0, sticky="w")
        self.slider_steps.configure(command=lambda v: self.lbl_steps_val.configure(text=f"{int(float(v))} steps"))

        # CPU threads
        self._cpu_count = os.cpu_count() or 8
        ctk.CTkLabel(self.sidebar, text="CPU Threads:").grid(
            row=10, column=0, padx=20, pady=(10, 2), sticky="w"
        )
        self.slider_threads = ctk.CTkSlider(
            self.sidebar, from_=1, to=max(2, min(32, self._cpu_count)), number_of_steps=max(1, min(31, self._cpu_count - 1))
        )
        self.slider_threads.set(min(8, self._cpu_count))
        self.slider_threads.grid(row=11, column=0, padx=20, pady=2, sticky="ew")
        self.lbl_threads_val = ctk.CTkLabel(
            self.sidebar, text=f"{int(self.slider_threads.get())} threads", font=ctk.CTkFont(size=11)
        )
        self.lbl_threads_val.grid(row=12, column=0, padx=20, pady=0, sticky="w")
        self.slider_threads.configure(
            command=lambda v: self.lbl_threads_val.configure(text=f"{int(float(v))} threads")
        )

        # Backend selector
        ctk.CTkLabel(self.sidebar, text="Backend:", font=ctk.CTkFont(weight="bold")).grid(
            row=13, column=0, padx=20, pady=(10, 2), sticky="w"
        )
        self.opt_backend = ctk.CTkOptionMenu(
            self.sidebar, values=["auto", "metal", "cuda", "vulkan", "cpu"]
        )
        self.opt_backend.set("auto")
        self.opt_backend.grid(row=14, column=0, padx=20, pady=5, sticky="ew")

        # Open output folder
        self.btn_open_folder = ctk.CTkButton(
            self.sidebar,
            text="\U0001f4c2 Open Output Folder",
            fg_color="transparent",
            border_width=1,
            command=self.open_output_dir,
        )
        self.btn_open_folder.grid(row=15, column=0, padx=20, pady=(10, 20), sticky="sew")

        # ---------- MAIN WORKSPACE ----------
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=15)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=0)

        # Style prompt input
        ctk.CTkLabel(
            self.main_frame, text="Music Style & Genre Prompt:", font=ctk.CTkFont(size=14, weight="bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.txt_style = ctk.CTkEntry(
            self.main_frame,
            placeholder_text="e.g. English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished mix",
        )
        self.txt_style.insert(
            0, "English, indie pop, bright acoustic guitar, soft drums, warm lead vocal, polished mix"
        )
        self.txt_style.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # Lyrics input
        ctk.CTkLabel(
            self.main_frame,
            text="Lyrics & Structure ([Verse], [Chorus]):",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).grid(row=2, column=0, sticky="nw", pady=(5, 2))
        self.txt_lyrics = ctk.CTkTextbox(self.main_frame, height=180)
        self.txt_lyrics.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.txt_lyrics.insert(
            "1.0",
            "[Verse]\nSoft morning light is touching the window.\nI hear the city waking below.\n\n"
            "[Chorus]\nStay with the rhythm, let it carry us home.\nSing with the sunrise, we are never alone.",
        )

        # Controls & seed
        self.ctrl_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.ctrl_frame.grid(row=4, column=0, sticky="ew", pady=5)
        self.ctrl_frame.grid_columnconfigure(1, weight=1)

        self.chk_random_seed = ctk.CTkCheckBox(self.ctrl_frame, text="Random Seed")
        self.chk_random_seed.select()
        self.chk_random_seed.grid(row=0, column=0, padx=(0, 15), sticky="w")

        self.btn_generate = ctk.CTkButton(
            self.ctrl_frame,
            text="\U0001f3bc Generate Music",
            height=40,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.start_generation,
        )
        self.btn_generate.grid(row=0, column=1, sticky="ew")

        # Terminal / logs
        ctk.CTkLabel(self.main_frame, text="Execution Logs:", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=5, column=0, sticky="w", pady=(10, 2)
        )
        self.txt_logs = ctk.CTkTextbox(self.main_frame, height=130, font=ctk.CTkFont(family="Courier", size=11))
        self.txt_logs.grid(row=6, column=0, sticky="ew", pady=(0, 10))

        # Audio player bar
        self.player_bar = ctk.CTkFrame(self.main_frame)
        self.player_bar.grid(row=7, column=0, sticky="ew", pady=(5, 0))

        self.lbl_now_playing = ctk.CTkLabel(self.player_bar, text="Ready.")
        self.lbl_now_playing.pack(side="left", padx=15, pady=10)

        self.btn_play = ctk.CTkButton(self.player_bar, text="\u25b6 Play", width=80, command=self.play_audio)
        self.btn_play.pack(side="right", padx=5, pady=10)

        self.btn_stop = ctk.CTkButton(
            self.player_bar, text="\u23f9 Stop", width=80, fg_color="gray30", command=self.stop_audio
        )
        self.btn_stop.pack(side="right", padx=5, pady=10)

    # ================= Helpers (thread-safe) =================
    def log(self, text):
        """Thread-safe log append (safe to call from worker threads)."""

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
        else:
            self.lbl_engine_status.configure(text="Engine: audiocpp_cli not found!", text_color="red")
            self.log("[Warning] audiocpp_cli was not found. Place it in 'bin/' or add it to PATH.")

        if not _AUDIO_AVAILABLE:
            self.log("[Warning] Audio playback unavailable (pygame mixer could not initialise).")
            self.log("          Generation still works; play the WAV files from the output folder.")

    # ================= Model download =================
    def start_model_download(self):
        def worker():
            try:
                self.after(0, lambda: self.btn_download_model.configure(state="disabled", text="Downloading..."))
            except Exception:
                pass
            self.log("[Download] Verifying/downloading sidecars and base weights...")
            try:
                # 1. Download sidecars and VAE
                snapshot_download(
                    repo_id=HF_REPO_ID,
                    allow_patterns=["sidecars/*", "yue2-vae-f16.gguf"],
                    local_dir=MODELS_DIR,
                    local_dir_use_symlinks=False,
                )
                # 2. Download selected quantization
                selected_model = self.opt_quant.get().split()[0]
                self.log(f"[Download] Downloading {selected_model}...")
                hf_hub_download(repo_id=HF_REPO_ID, filename=selected_model, local_dir=MODELS_DIR)
                self.log("[Download] All model weights verified and ready!")
                try:
                    self.after(0, lambda: messagebox.showinfo("Success", "Model weights downloaded successfully!"))
                except Exception:
                    pass
            except Exception as e:
                self.log(f"[Download Error] {str(e)}")
                try:
                    self.after(0, lambda: messagebox.showerror("Download Error", str(e)))
                except Exception:
                    pass
            finally:
                try:
                    self.after(
                        0,
                        lambda: self.btn_download_model.configure(
                            state="normal", text="\u2b07 Download / Check Model"
                        ),
                    )
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

    # ================= Generation =================
    def start_generation(self):
        if self.is_generating:
            return
        if not self.cli_path:
            messagebox.showerror("Error", "audiocpp_cli binary not found in path or 'bin/' directory!")
            return

        lyrics = self.txt_lyrics.get("1.0", "end").strip()
        style = self.txt_style.get().strip()
        if not lyrics:
            messagebox.showwarning("Warning", "Please provide lyrics.")
            return

        seed = random.randint(10000000, 99999999) if self.chk_random_seed.get() else 42
        selected_model = self.opt_quant.get().split()[0]
        cot = self.opt_cot.get()
        steps = int(float(self.slider_steps.get()))
        threads = int(float(self.slider_threads.get()))
        requested_backend = self.opt_backend.get()
        backend = requested_backend if requested_backend != "auto" else detect_backend()
        out_wav = os.path.join(OUTPUT_DIR, f"yue2_{seed}.wav")

        cmd = [
            self.cli_path,
            "--task", "gen",
            "--family", "yue2",
            "--model", MODELS_DIR,
            "--backend", backend,
            "--threads", str(threads),
            "--text", lyrics,
            "--request-option", f"style={style}",
            "--request-option", f"cot={cot}",
            "--request-option", f"seed={seed}",
            "--request-option", f"num_inference_steps={steps}",
            "--session-option", f"yue2.model_gguf={selected_model}",
            "--session-option", "yue2.vae_gguf=yue2-vae-f16.gguf",
            "--out", out_wav,
            "--log",
        ]

        def run_thread():
            self.is_generating = True
            try:
                self.after(0, lambda: self.btn_generate.configure(state="disabled", text="Generating..."))
            except Exception:
                pass
            self.log(f"\n--- Starting Generation [Seed: {seed}] ---")
            self.log(f"Command: {' '.join(cmd)}\n")

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
