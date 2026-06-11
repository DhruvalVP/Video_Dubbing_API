import os
import json
import subprocess
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment

class AudioDubbingMuxer:
    def __init__(self, filename: str, TARGET_CODE: str,  json_path: str, tts_dir: str, bg_audio_path: str, orig_video_path: str):
        self.filename = filename
        self.TARGET_CODE = TARGET_CODE
        self.json_path = json_path
        self.tts_dir = tts_dir
        self.bg_audio_path = bg_audio_path
        self.orig_video_path = orig_video_path

    def adjust_speed_high_quality(self, input_wav: str, target_duration: float, output_wav: str):
        """
        Uses pyrubberband for studio-quality time stretching without robotic artifacts.
        """
        # Load the TTS audio array and its sample rate
        y, sr = sf.read(input_wav)
        
        # Calculate the exact current duration in seconds
        current_duration = len(y) / sr
        
        # Calculate EXACT stretch rate to fit the target duration perfectly.
        rate = current_duration / target_duration
        
        # FIX: Removed the max/min safety limits (e.g., rate = max(0.65, min(1.35, rate)))
        # By allowing PyRubberband to scale precisely to `rate`, the audio will 
        # exactly fill the gap from the start timestamp to the end timestamp.

        # Apply high-quality phase vocoder stretching
        y_stretched = pyrb.time_stretch(y, sr, rate)
        
        # Save the polished audio
        sf.write(output_wav, y_stretched, sr)

    def process(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Load JSON Data
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)[0]
        
        transcription_data = data["transcription"]
        
        # 2. Get Background Audio to determine total canvas length
        bg_audio = AudioSegment.from_file(self.bg_audio_path)
        total_duration_ms = len(bg_audio)
        
        # Create a silent canvas for the new vocals
        dubbed_vocals_canvas = AudioSegment.silent(duration=total_duration_ms)

        print("[INFO] Starting Exact Audio Speed Adjustment & Timeline Assembly...")
        temp_dir = os.path.join(output_dir, "temp_stretched_chunks")
        os.makedirs(temp_dir, exist_ok=True)

        for idx, segment in enumerate(transcription_data):
            start_time_sec = segment["start"]
            end_time_sec = segment["end"]
            target_duration_sec = end_time_sec - start_time_sec
            
            # Assuming TTS files are named sequentially like chunk_0.wav, chunk_1.wav, etc.
            tts_file = os.path.join(self.tts_dir, f"{self.filename}_chunk_{idx}.wav")
            
            if not os.path.exists(tts_file):
                print(f"[WARNING] Missing TTS chunk: {tts_file}. Skipping.")
                continue
                
            stretched_file = os.path.join(temp_dir, f"stretched_{idx}.wav")
            
            # Stretch audio EXACTLY to target duration
            self.adjust_speed_high_quality(tts_file, target_duration_sec, stretched_file)
            
            # Load stretched audio and overlay onto the exact millisecond start time
            stretched_audio = AudioSegment.from_file(stretched_file)
            start_time_ms = int(start_time_sec * 1000)
            
            dubbed_vocals_canvas = dubbed_vocals_canvas.overlay(stretched_audio, position=start_time_ms)

        # 3. Mix dubbed vocals with the original background
        print("[INFO] Mixing dubbed vocals with original background audio...")
        final_mixed_audio = bg_audio.overlay(dubbed_vocals_canvas)
        final_audio_path = os.path.join(output_dir, f"{self.filename}_{self.TARGET_CODE}.wav")
        final_mixed_audio.export(final_audio_path, format="wav")

        # 4. Mux final mixed audio with original video
        print("[INFO] Muxing final audio into video...")
        final_video_path = os.path.join(output_dir, f"{self.filename}_{self.TARGET_CODE}.mp4")
        
        mux_cmd = [
            'ffmpeg', '-y',
            '-i', self.orig_video_path,
            '-i', final_audio_path,
            '-c:v', 'copy',           # Copy original video codec exactly (no quality loss)
            '-c:a', 'aac',            # Encode audio to standard AAC
            '-map', '0:v:0',          # Take video stream from file 0
            '-map', '1:a:0',          # Take audio stream from file 1
            final_video_path
        ]
        
        subprocess.run(mux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        print(f"[SUCCESS] Final Video Generated: {final_video_path}")
        
        # Cleanup temp directory
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)