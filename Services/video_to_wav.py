from moviepy import VideoFileClip
import os

def convert_mp4_to_wav(filename, video_input, output_dir):
    os.makedirs(f"{output_dir}/wav_output_batch", exist_ok=True)
    audio_path = f"{output_dir}/wav_output_batch/{filename}.wav"
    video = VideoFileClip(video_input)
    try:
        # Extract audio and write to file
        video.audio.write_audiofile(audio_path, logger=None)
        print(f"Conversion complete: {audio_path}")
    finally:
        # Properly close video and audio resources
        video.close()
    
    return audio_path
# Usage
# convert_mp4_to_wav(r"D:\Linkedin Stuff\My Videos\02_Ollama+Python_Simple-ChatBot\Final Export.mp4", "output_audio.wav")