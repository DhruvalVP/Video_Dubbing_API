import os
import gc
import torch
import torchaudio
from demucs.api import Separator
from df.enhance import enhance, init_df, load_audio, save_audio
import warnings

warnings.filterwarnings("ignore")


class AudioCleaner:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            print("[INFO] 🟢 NVIDIA GPU (CUDA) detected.")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("[INFO] 🟢 Apple Silicon GPU (MPS) detected.")
        else:
            self.device = "cpu"
            print("[INFO] 🟡 No GPU detected. Using CPU.")

        print("[INFO] Loading Demucs model...")
        self.separator = Separator("htdemucs", device=self.device)

        print("[INFO] Loading DeepFilterNet model...")
        self.df_model, self.df_state, _ = init_df()
        self.df_model = self.df_model.to(self.device)
        self.df_model.eval()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

    def unload_demucs(self):
        """Free Demucs from RAM/VRAM. Call after stem separation is done."""
        if hasattr(self, "separator"):
            del self.separator
            self.separator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self):
        """Free all models from RAM/VRAM."""
        if hasattr(self, "separator") and self.separator is not None:
            del self.separator
            self.separator = None
        if hasattr(self, "df_model"):
            del self.df_model
            self.df_model = None
        if hasattr(self, "df_state"):
            del self.df_state
            self.df_state = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def process_audio(self, filename: str, input_path: str, output_dir: str):
        """Separates vocals and background using Demucs, then denoises vocals."""
        os.makedirs(os.path.join(output_dir, "raw_vocals"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "background"), exist_ok=True)

        raw_vocals_path = os.path.join(output_dir, "raw_vocals", f"{filename}.wav")
        background_path = os.path.join(output_dir, "background", f"{filename}.wav")

        print(f"\n[STEP 1] Separating stems for: {input_path}")
        _, separated = self.separator.separate_audio_file(input_path)

        vocals_tensor = separated["vocals"]
        background_tensor = separated["drums"] + separated["bass"] + separated["other"]

        demucs_sr = self.separator.samplerate
        torchaudio.save(raw_vocals_path, vocals_tensor, demucs_sr)
        torchaudio.save(background_path, background_tensor, demucs_sr)
        print(f"         ✔️ Background track saved: {background_path}")

        return raw_vocals_path, background_path

    def enhance(self, filename: str, raw_vocals_path: str, output_dir: str):
        """Denoises vocals using DeepFilterNet."""
        print("[STEP 2] Denoising vocals...")
        os.makedirs(os.path.join(output_dir, "clean_vocals"), exist_ok=True)

        audio, _ = load_audio(raw_vocals_path, sr=self.df_state.sr())
        audio = audio.to(self.device)

        with torch.no_grad():
            enhanced_vocals = enhance(self.df_model, self.df_state, audio)

        max_abs_val = enhanced_vocals.abs().max()
        if max_abs_val > 0:
            enhanced_vocals = (enhanced_vocals / max_abs_val) * 0.95

        clean_vocals_path = os.path.join(output_dir, "clean_vocals", f"{filename}.wav")
        save_audio(clean_vocals_path, enhanced_vocals.cpu(), self.df_state.sr())
        print(f"        ✔️ Clean vocals saved: {clean_vocals_path}")
        return clean_vocals_path

    def process_vocals_only(self, filename: str, input_path: str, output_dir: str) -> str:
        """
        Denoises audio using DeepFilterNet only — no Demucs stem separation.
        Use this for TTS chunks which are already pure vocals.
        """
        os.makedirs(os.path.join(output_dir, "raw_vocals"), exist_ok=True)
        output_path = os.path.join(output_dir, "raw_vocals", f"{filename}.wav")

        audio, _ = load_audio(input_path, sr=self.df_state.sr())
        audio = audio.to(self.device)

        with torch.no_grad():
            enhanced = enhance(self.df_model, self.df_state, audio)

        max_abs_val = enhanced.abs().max()
        if max_abs_val > 0:
            enhanced = (enhanced / max_abs_val) * 0.95

        save_audio(output_path, enhanced.cpu(), self.df_state.sr())
        return output_path
