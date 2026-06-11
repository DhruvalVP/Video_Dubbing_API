from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os

def trim_end_silence(input_file, output_file, silence_threshold=-50.0):
    """
    Trims silence from the end of a .wav file.
    """
    if not os.path.exists(input_file):
        return
        
    print(f"Loading {input_file}...")
    audio = AudioSegment.from_wav(input_file)

    non_silent_parts = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_threshold)

    if non_silent_parts:
        last_end_time = non_silent_parts[-1][1]
        trimmed_audio = audio[:last_end_time]
        trimmed_audio.export(output_file, format="wav")
        print(f"Success! Trimmed audio saved to: {output_file}")
    else:
        print("Could not find any audio above the silence threshold. File may be completely blank.")