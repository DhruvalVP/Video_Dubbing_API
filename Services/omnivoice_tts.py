"""
omnivoice_robust.py — Enhanced with Dynamic Speed Estimation & Batch Processing
==========================================================================

Fixes the leading "uuh/ssuh" artifact that OmniVoice produces by:

  1. Prepending "salt" words to the input.
  2. Using Whisper as the EXCLUSIVE detector to find exact word boundaries.
  3. AGGRESSIVE RETRY: If the salt is missed, regenerates the audio (with
     a new seed) up to 5 times until the salt is detected.
  4. DYNAMIC CASCADE FALLBACK: Calculates the speaker's actual speech rate.
  5. Adding a safety margin (150ms) past the last detected salt word.
"""

import os
import re
import gc
import warnings
import logging
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
from difflib import SequenceMatcher

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s'
)
logger = logging.getLogger("RobustTTS")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RobustConfig:
    """All tunables in one place."""

    # --- Salt words ---
    salt_words: Tuple[str, ...] = ("chocolate",)
    num_salt_words: int = 1

    # --- Generation parameters ---
    num_step: int = 64
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 0.5
    class_temperature: float = 0.0
    denoise: bool = False
    preprocess_prompt: bool = True
    postprocess_output: bool = False

    # --- Long-form chunking ---
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0

    # --- Detection ---
    fuzzy_match_threshold: float = 0.70
    safety_margin_ms: float = 150.0  # Pushed past last detected salt word
    max_slice_ratio: float = 0.80    # Never slice past this fraction
    min_slice_ms: float = 200.0      # Never slice before this time

    # --- Smoothing ---
    fade_in_ms: float = 15.0

    # --- Detector selection ---
    use_whisper: bool = True

    # --- Models ---
    whisper_model: str = "openai/whisper-medium.en"

    # --- Device ---
    prefer_gpu: bool = True

    def get_device(self) -> str:
        if self.prefer_gpu and torch.cuda.is_available():
            return "cuda"
        if self.prefer_gpu and hasattr(torch.backends, "mps") and \
           torch.backends.mps.is_available():
            return "mps"
        return "cpu"


# ============================================================================
# SALT MANAGER
# ============================================================================

class SaltManager:
    """Manages the salt prefix."""

    def __init__(self, config: RobustConfig):
        self.config = config
        self.salt_words = list(config.salt_words[:config.num_salt_words])
        logger.info(f"Salt words configured: {self.salt_words}")

    def make_salted_text(self, text: str) -> str:
        """Prepend salt to the user's text with clear delimiters."""
        return ", ".join(self.salt_words) + ", " + text


# ============================================================================
# DETECTOR
# ============================================================================

class WhisperWordDetector:
    """Primary detector using Whisper with word-level timestamps."""

    def __init__(self, config: RobustConfig):
        self.config = config
        self._pipe = None

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_loaded(self):
        if self._pipe is not None:
            return
        from transformers import pipeline as hf_pipeline
        device = self.config.get_device()
        try:
            logger.info(f"Loading Whisper '{self.config.whisper_model}' on {device}")
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self.config.whisper_model,
                device=device,
                torch_dtype=torch.float32,
                return_timestamps="word",
            )
        except Exception as e:
            logger.warning(f"Failed {self.config.whisper_model}: {e}. Using tiny.en")
            self._pipe = hf_pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small.en",
                device=device,
                torch_dtype=torch.float32,
                return_timestamps="word",
            )

    def _fuzzy_match(self, w1: str, w2: str) -> bool:
        if w1 == w2:
            return True
        return SequenceMatcher(None, w1, w2).ratio() >= \
               self.config.fuzzy_match_threshold

    def detect_aggressive(
        self,
        audio: np.ndarray,
        sr: int,
        salt_manager: SaltManager,
        real_text: str,
    ) -> Tuple[Dict[str, Tuple[float, float]], Optional[Tuple[float, float]]]:
        self._ensure_loaded()

        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        audio = audio.astype(np.float32)

        try:
            result = self._pipe(
                {"array": audio, "sampling_rate": sr},
                return_timestamps="word",
                chunk_length_s=30,
            )
        except Exception as e:
            logger.warning(f"Whisper failed: {e}")
            return {}, None

        chunks = result.get("chunks", [])
        detected = {}
        anchor = None
        salt_set = {s.lower() for s in salt_manager.salt_words}

        for chunk in chunks:
            word = chunk.get("text", "").strip().lower().rstrip('.,!?;:()[]"')
            ts = chunk.get("timestamp", (0.0, 0.0))
            if not word or ts[0] is None or ts[1] is None:
                continue

            for salt in salt_set:
                if salt not in detected:
                    # Aggressive Matching: Check for exact fuzzy match OR substring match
                    if self._fuzzy_match(word, salt) or salt in word:
                        detected[salt] = (float(ts[0]), float(ts[1]))
                        break

        # Secondary Backup: Find the real text start (anchor)
        real_words = [w.lower().rstrip('.,!?;:()[]"') for w in real_text.split()[:3]]
        real_words = [w for w in real_words if w]

        if real_words:
            for chunk in chunks:
                word = chunk.get("text", "").strip().lower().rstrip('.,!?;:()[]"')
                ts = chunk.get("timestamp", (0.0, 0.0))
                if not word or ts[0] is None:
                    continue
                if self._fuzzy_match(word, real_words[0]) or real_words[0] in word:
                    anchor = (float(ts[0]), float(ts[1]))
                    break

        return detected, anchor


# ============================================================================
# CASCADE LOGIC — Dynamic Speed Estimation
# ============================================================================

def find_slice_point_cascade(
    detected: Dict[str, Tuple[float, float]],
    salt_manager: SaltManager,
    audio_duration: float,
    config: RobustConfig,
) -> Tuple[float, str]:
    """
    Determines slice point dynamically:
      1. Filters overlapping/bad timestamps.
      2. Estimates speaker's actual speed based on clean word detections.
      3. Calculates trim point accurately even if words are missed.
    """
    
    # Ultimate Fallback: Bypass salt entirely and slice right before the real text starts
    if '__REAL_ANCHOR__' in detected:
        start_ts = detected['__REAL_ANCHOR__'][0]
        slice_time = max(0.0, start_ts - 0.08)
        reason = "REAL_TEXT_ANCHOR__bypassed_salt"
        
        min_time = config.min_slice_ms / 1000.0
        max_time = audio_duration * config.max_slice_ratio
        slice_time = max(min_time, min(slice_time, max_time))
        return slice_time, reason

    # 1. Filter out invalid/overlapping timestamps
    valid_detections = {}
    last_end = -1.0
    
    for salt in salt_manager.salt_words:
        w = salt.lower()
        if w in detected:
            start, end = detected[w]
            if end > start and end > (last_end - 0.3):
                valid_detections[w] = (start, end)
                last_end = end
            else:
                logger.warning(f"Ignored invalid/overlapping timestamp for '{w}': {start:.2f}-{end:.2f}s")

    # 2. Estimate dynamic speech speed
    pure_durations = []
    for i in range(1, len(salt_manager.salt_words)):
        w = salt_manager.salt_words[i].lower()
        if w in valid_detections:
            start, end = valid_detections[w]
            dur = end - start
            if 0.1 < dur < 1.5:
                pure_durations.append(dur)
                
    if pure_durations:
        avg_word_dur = sum(pure_durations) / len(pure_durations)
        logger.info(f"⏱️ Dynamic speed estimate: {avg_word_dur:.3f}s per word (based on clear salt words)")
    else:
        first_w = salt_manager.salt_words[0].lower()
        if first_w in valid_detections:
            start, end = valid_detections[first_w]
            dur = end - start
            avg_word_dur = min(0.7, max(0.3, dur * 0.35))
            logger.info(f"⏱️ Dynamic speed estimate: {avg_word_dur:.3f}s per word (derived from first word)")
        else:
            avg_word_dur = 0.5
            logger.info(f"⏱️ Dynamic speed estimate: {avg_word_dur:.3f}s (default)")

    # 3. Find the LAST valid detected salt word
    last_detected_idx = -1
    last_detected_end = None
    
    for i in range(len(salt_manager.salt_words) - 1, -1, -1):
        salt = salt_manager.salt_words[i].lower()
        if salt in valid_detections:
            last_detected_idx = i
            last_detected_end = valid_detections[salt][1]
            break

    # 4. Cascade calculation
    if last_detected_idx == -1:
        slice_time = len(salt_manager.salt_words) * (avg_word_dur + 0.15)
        reason = "NO_SALT_DETECTED__dynamic_prediction"
    else:
        slice_time = last_detected_end
        detected_word = salt_manager.salt_words[last_detected_idx]
        reason = f"DETECTED_{detected_word.upper()}"
        
        missed = []
        for i in range(last_detected_idx + 1, len(salt_manager.salt_words)):
            missed_word = salt_manager.salt_words[i]
            slice_time += (avg_word_dur + 0.1)
            missed.append(missed_word)
            
        if missed:
            reason += f"__INFERRED_MISSED_{'+'.join(missed)}"

    # Add safety margin
    slice_time += config.safety_margin_ms / 1000.0
    
    # Enforce reasonable bounds
    min_time = config.min_slice_ms / 1000.0
    max_time = audio_duration * config.max_slice_ratio
    slice_time = max(min_time, min(slice_time, max_time))

    return slice_time, reason


# ============================================================================
# AUDIO SLICING
# ============================================================================

def slice_audio_clean(
    audio: np.ndarray,
    sr: int,
    slice_time: float,
    fade_in_ms: float = 15.0,
) -> np.ndarray:
    sample_idx = int(slice_time * sr)
    sample_idx = max(0, min(sample_idx, len(audio) - 1))
    sliced = audio[sample_idx:].copy()

    fade_samples = int(fade_in_ms * 0.001 * sr)
    if len(sliced) > fade_samples > 0:
        fade = 0.5 * (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
        sliced[:fade_samples] *= fade

    return sliced


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RobustTTSPipeline:
    def __init__(self, config: Optional[RobustConfig] = None):
        self.config = config or RobustConfig()
        self.salt_manager = SaltManager(self.config)
        self.whisper_det = (
            WhisperWordDetector(self.config)
            if self.config.use_whisper else None
        )
        self._model = None
        self._sr = 24000

    def unload(self):
        """Explicitly free OmniVoice and Whisper detector from RAM/VRAM."""
        if self.whisper_det is not None:
            self.whisper_det.unload()
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _load_model(self):
        if self._model is not None:
            return
        from omnivoice import OmniVoice
        device = self.config.get_device()
        dtype = torch.float32

        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        logger.info(f"Loading OmniVoice on {device} (dtype={dtype})...")
        self._model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
        )
        self._sr = self._model.sampling_rate
        logger.info(f"OmniVoice loaded. Sampling rate: {self._sr} Hz")

    def generate(
        self,
        text: Union[str, List[str]],
        ref_audio: Optional[Union[str, List[str]]] = None,
        ref_text: Optional[Union[str, List[str]]] = None,
        instruct: Optional[Union[str, List[str]]] = None,
        language: Optional[Union[str, List[str]]] = "en",
        duration: Optional[Union[float, List[Optional[float]]]] = None,
        speed: Optional[Union[float, List[Optional[float]]]] = None,
        num_step: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        t_shift: Optional[float] = None,
        layer_penalty_factor: Optional[float] = None,
        position_temperature: Optional[float] = None,
        class_temperature: Optional[float] = None,
        denoise: Optional[bool] = None,
        preprocess_prompt: Optional[bool] = None,
        audio_chunk_duration: Optional[float] = None,
        audio_chunk_threshold: Optional[float] = None,
        output_path: Optional[str] = None,
    ) -> Union[Tuple[np.ndarray, int], Tuple[List[np.ndarray], int]]:
        self._load_model()

        # Resolve parameters falling back to config
        _num_step = num_step if num_step is not None else self.config.num_step
        _guidance = guidance_scale if guidance_scale is not None else self.config.guidance_scale
        _t_shift = t_shift if t_shift is not None else self.config.t_shift
        _layer_pen = layer_penalty_factor if layer_penalty_factor is not None else self.config.layer_penalty_factor
        _pos_temp = position_temperature if position_temperature is not None else self.config.position_temperature
        _cls_temp = class_temperature if class_temperature is not None else self.config.class_temperature
        _denoise = denoise if denoise is not None else self.config.denoise
        _preprocess = preprocess_prompt if preprocess_prompt is not None else self.config.preprocess_prompt
        _chunk_dur = audio_chunk_duration if audio_chunk_duration is not None else self.config.audio_chunk_duration
        _chunk_thr = audio_chunk_threshold if audio_chunk_threshold is not None else self.config.audio_chunk_threshold

        # --- Normalize Inputs to Lists ---
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = list(text)

        batch_size = len(text_list)
        salted_text_list = [self.salt_manager.make_salted_text(t) for t in text_list]

        if duration is not None:
            if isinstance(duration, (int, float)):
                duration_list = [float(duration)] * batch_size
            else:
                duration_list = list(duration)
        else:
            duration_list = None

        if speed is not None:
            if isinstance(speed, (int, float)):
                speed_list = [float(speed)] * batch_size
            else:
                speed_list = list(speed)
        else:
            speed_list = None

        logger.info(
            f"🔊 Generating batch of {batch_size} item(s) "
            f"(num_step={_num_step}, duration={duration_list}, speed={speed_list})..."
        )
        
        # Initial Batch Generate
        audio_list = self._model.generate(
            text=salted_text_list,
            ref_audio=ref_audio,
            ref_text=ref_text,
            instruct=instruct,
            language=language,
            duration=duration_list,
            speed=speed_list,
            num_step=_num_step,
            guidance_scale=_guidance,
            t_shift=_t_shift,
            layer_penalty_factor=_layer_pen,
            position_temperature=_pos_temp,
            class_temperature=_cls_temp,
            denoise=_denoise,
            preprocess_prompt=_preprocess,
            postprocess_output=self.config.postprocess_output,
            audio_chunk_duration=_chunk_dur,
            audio_chunk_threshold=_chunk_thr,
        )

        results = []

        # Loop through each generated chunk
        for idx in range(batch_size):
            salted_t = salted_text_list[idx]
            real_t = text_list[idx]
            
            raw_audio = None
            detected = {}
            anchor = None
            
            max_audio_retries = 10
            max_whisper_retries = 5

            # --- Retry Loop Phase ---
            for attempt in range(max_audio_retries):
                if attempt == 0:
                    current_audio = audio_list[idx]
                else:
                    logger.info(f"🔄 Item {idx}: Salt not detected. Regenerating audio (Attempt {attempt + 1}/{max_audio_retries})...")
                    set_deterministic_seed(42 + attempt)

                    single_dur = [duration_list[idx]] if duration_list else None
                    single_spd = [speed_list[idx]] if speed_list else None
                    single_ref_audio = ref_audio[idx] if isinstance(ref_audio, list) else ref_audio
                    single_ref_text = ref_text[idx] if isinstance(ref_text, list) else ref_text
                    single_instruct = instruct[idx] if isinstance(instruct, list) else instruct
                    single_lang = language[idx] if isinstance(language, list) else language

                    single_audio_list = self._model.generate(
                        text=[salted_t],
                        ref_audio=single_ref_audio,
                        ref_text=single_ref_text,
                        instruct=single_instruct,
                        language=single_lang,
                        duration=single_dur,
                        speed=single_spd,
                        num_step=_num_step,
                        guidance_scale=_guidance,
                        t_shift=_t_shift,
                        layer_penalty_factor=_layer_pen,
                        position_temperature=_pos_temp,
                        class_temperature=_cls_temp,
                        denoise=_denoise,
                        preprocess_prompt=_preprocess,
                        postprocess_output=self.config.postprocess_output,
                        audio_chunk_duration=_chunk_dur,
                        audio_chunk_threshold=_chunk_thr,
                    )
                    current_audio = single_audio_list[0]
                    set_deterministic_seed(42)

                if isinstance(current_audio, torch.Tensor):
                    current_audio = current_audio.cpu().numpy()
                current_audio = current_audio.astype(np.float32)

                if self.whisper_det is None:
                    raw_audio = current_audio
                    break

                # Try detecting with Whisper up to max_whisper_retries times on this audio
                whisper_success = False
                for k in range(max_whisper_retries):
                    print(f"Item {idx} Audio attempt {attempt + 1}: Running Whisper detection pass {k + 1}/{max_whisper_retries}...")
                    try:
                        w_detected, current_anchor = self.whisper_det.detect_aggressive(current_audio, self._sr, self.salt_manager, real_t)
                        if w_detected:
                            detected = w_detected
                            anchor = current_anchor
                            raw_audio = current_audio
                            logger.info(f"🗣️  Item {idx} Whisper found: { {k: f'{v[0]:.2f}-{v[1]:.2f}s' for k,v in w_detected.items()} }")
                            whisper_success = True
                            break
                        else:
                            logger.warning(f"⚠️  Item {idx}: Whisper missed salt (audio attempt {attempt + 1}, whisper pass {k + 1}).")
                            if current_anchor:
                                anchor = current_anchor
                            raw_audio = current_audio
                    except Exception as e:
                        logger.warning(f"Item {idx} Whisper error on pass {k + 1}: {e}")
                        raw_audio = current_audio

                if whisper_success:
                    break  # Salt detected — no need for more audio regeneration

            # Fallback if all attempts exhausted
            if not detected and anchor is not None:
                detected['__REAL_ANCHOR__'] = anchor
                logger.warning(f"🚨 Item {idx}: Salt vanished entirely after {max_audio_retries} audio attempts! Falling back to Real Text Anchor at {anchor[0]:.2f}s")
            elif not detected:
                logger.error(f"❌ Item {idx}: Salt vanished and no anchor found after {max_audio_retries} audio attempts.")

            audio_duration = len(raw_audio) / self._sr
            logger.info(f"🎵 Item {idx}: Final audio {audio_duration:.2f}s, peak={np.abs(raw_audio).max():.4f}")

            # --- Cascade Trimming Phase ---
            slice_time, reason = find_slice_point_cascade(
                detected, self.salt_manager, audio_duration, self.config
            )
            logger.info(
                f"✂️  Item {idx} Slice at {slice_time:.3f}s [{reason}] "
                f"(removes first {slice_time:.2f}s, keeps {audio_duration - slice_time:.2f}s)"
            )

            final_audio = slice_audio_clean(raw_audio, self._sr, slice_time, self.config.fade_in_ms)

            peak = np.abs(final_audio).max()
            if peak > 0.99:
                final_audio = final_audio * (0.95 / peak)
            
            results.append(final_audio)

        # Handle batch saving & returning
        if output_path:
            if len(results) == 1:
                sf.write(output_path, results[0], self._sr)
                logger.info(f"💾 Saved: {output_path}")
            else:
                base, ext = os.path.splitext(output_path)
                for i, r in enumerate(results):
                    p = f"{base}_{i}{ext}"
                    sf.write(p, r, self._sr)
                logger.info(
                    f"💾 Saved {len(results)} files: "
                    f"{base}_0{ext} ... {base}_{len(results)-1}{ext}"
                )

        if len(results) == 1:
            return results[0], self._sr
        return results, self._sr