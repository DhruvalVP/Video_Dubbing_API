import os
import json
import wave

def prepare_tts_sample(filename: str, input_path: str, json_data: str, output_path: str):
    sample_dir = os.path.join(output_path, "sample_speech")
    os.makedirs(sample_dir, exist_ok=True)
    output_path = os.path.join(sample_dir, f"{filename}.wav")

    with open(json_data, "r", encoding="utf-8") as f:
        data = json.load(f)

    start_sec = data[0]["transcription"][0]["start"]
    end_sec = data[0]["transcription"][0]["end"]
    """
    Trims a WAV audio file from start_sec to end_sec and saves it.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file {input_path} was not found.")
        
    if start_sec >= end_sec:
        raise ValueError("Start time must be less than end time.")

    # Generate a default output filename if none is provided
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_trimmed{ext}"

    # Open the input WAV file
    with wave.open(input_path, 'rb') as wav_in:
        # Extract audio parameters (channels, sample width, framerate, etc.)
        params = wav_in.getparams()
        frame_rate = wav_in.getframerate()
        total_frames = wav_in.getnframes()
        
        # Calculate exact frames based on floating point seconds
        start_frame = int(start_sec * frame_rate)
        end_frame = int(end_sec * frame_rate)
        frames_to_keep = end_frame - start_frame
        
        # Validation
        if start_frame >= total_frames:
            raise ValueError("Start time is beyond the total length of the audio.")
            
        # Ensure we don't try to read past the end of the file
        if end_frame > total_frames:
            frames_to_keep = total_frames - start_frame
            
        # Move the reading pointer to the start frame
        wav_in.setpos(start_frame)
        
        # Read the raw audio data for the specified duration
        audio_data = wav_in.readframes(frames_to_keep)
        
    # Write the extracted audio data to a new WAV file
    with wave.open(output_path, 'wb') as wav_out:
        wav_out.setparams(params)
        wav_out.setnframes(frames_to_keep)
        wav_out.writeframes(audio_data)
        
    print(f"Success! Audio trimmed and saved to: {output_path}")
    return output_path