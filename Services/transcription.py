import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from faster_whisper import WhisperModel
import warnings

warnings.filterwarnings("ignore")

class Transcriber:
    def __init__(self, model_size="large-v3"):
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
        elif torch.backends.mps.is_available():
            self.device = "auto"
            self.compute_type = "float16"
        else:
            self.device = "cpu"
            self.compute_type = "int8"
            
        self.whisper = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = utils[0]

    def load_audio_robust(self, audio_path, target_sr=16000):
        wav, sr = torchaudio.load(audio_path)
        
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
            
        if sr != target_sr:
            resampler = T.Resample(sr, target_sr)
            wav = resampler(wav)
            
        return wav.squeeze(0)

    def transcribe_long_audio(self, filename, audio_path: str, gender: str, output_dir: str):
        new_output_dir = os.path.join(output_dir, "transcriptions")
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)

        json_output_path = os.path.join(new_output_dir, f"{filename}.json")

        sampling_rate = 16000
        wav = self.load_audio_robust(audio_path, target_sr=sampling_rate)
        
        timestamps = self.get_speech_timestamps(
            wav,
            self.vad_model,
            sampling_rate=sampling_rate,
            min_silence_duration_ms=300
        )

        # 1. Identify safe cut points in the exact middle of detected silences
        cut_points = [0]
        for i in range(len(timestamps) - 1):
            midpoint = (timestamps[i]['end'] + timestamps[i+1]['start']) // 2
            cut_points.append(midpoint)
        cut_points.append(len(wav))

        # 2. Create GAPLESS chunks (10s to 25s). 
        # This covers 100% of the audio array, so zero words are accidentally dropped by strict VAD limits.
        min_chunk_samples = 10 * sampling_rate
        max_chunk_samples = 25 * sampling_rate
        
        chunk_boundaries = [0]
        last_cut = 0
        
        for cp in cut_points[1:-1]:
            # If gap to next natural silence is too long, force a cut at 20s
            while cp - last_cut > max_chunk_samples:
                forced_cut = last_cut + (20 * sampling_rate)
                chunk_boundaries.append(forced_cut)
                last_cut = forced_cut
                
            # Take the natural cut point if it's long enough
            if cp - last_cut >= min_chunk_samples:
                chunk_boundaries.append(cp)
                last_cut = cp
                
        # Handle the remaining audio at the very end of the file
        while len(wav) - last_cut > max_chunk_samples:
            forced_cut = last_cut + (20 * sampling_rate)
            chunk_boundaries.append(forced_cut)
            last_cut = forced_cut
            
        if last_cut < len(wav):
            chunk_boundaries.append(len(wav))
            
        # Ensure list is clean
        chunk_boundaries = sorted(list(set(chunk_boundaries)))

        final_transcripts = []
        detected_language = None

        print(f"         ✔️ Audio mapped into {len(chunk_boundaries)-1} gapless segments.")

        for i in range(len(chunk_boundaries) - 1):
            chunk_start_sample = chunk_boundaries[i]
            chunk_end_sample = chunk_boundaries[i+1]
            
            # Extract audio array (100% sequential, no gaps)
            audio_chunk = wav[chunk_start_sample:chunk_end_sample].numpy()
            global_offset_sec = chunk_start_sample / sampling_rate
            
            # Artificial Silence Padding to flush trailing words from Whisper's memory
            silence_pad = np.zeros(int(1.0 * sampling_rate), dtype=np.float32)
            audio_chunk = np.concatenate([audio_chunk, silence_pad])
            
            with torch.inference_mode():
                segments, info = self.whisper.transcribe(
                    audio_chunk,
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    word_timestamps=False
                )
            
            if detected_language is None:
                detected_language = info.language

            for segment in segments:
                # Use standard segment boundaries to encapsulate full mouth activity
                exact_start = segment.start + global_offset_sec
                exact_end = segment.end + global_offset_sec
                
                final_transcripts.append({
                    detected_language: segment.text.strip(),
                    "start": round(exact_start, 3),
                    "end": round(exact_end, 3)
                })

        transcription_list = [item[detected_language] for item in final_transcripts]
        
        final_data = [
            {
                "filename": f"{filename}.wav",
                "audio-language": detected_language,
                "gender": gender,
                "transcription": final_transcripts,
                "transcription-list": transcription_list,
            }
        ]

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
        
        return json_output_path, detected_language