import os
import gc
import torch
import json
from datetime import datetime
from Services.video_to_wav import convert_mp4_to_wav
from Services.speech_seprator_enhancer import AudioCleaner
from Services.transcription import Transcriber
from Services.json_formatter import calculate_transcription_durations
from Services.translate import translate_list
from Services.speech_sampler import prepare_tts_sample
from Services.tts_engine import TTSBatchProcessor
from Services.trim_trailing_silence import trim_end_silence
from Services.video_audio_muxer import AudioDubbingMuxer
from Services.merge_video_audio import merge_video_audio


STATUS_FILE = "dubbing_status.json"

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def status(msg):
    """Print status message and save to JSON file with timestamp"""
    print("\n" + "="*60)
    print(msg)
    print("="*60)
    
    # Create status entry with timestamp
    status_entry = {
        "timestamp": datetime.now().isoformat(),
        "message": msg
    }
    
    # Load existing status list or create new one
    status_list = []
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r') as f:
                status_list = json.load(f)
        except (json.JSONDecodeError, IOError):
            status_list = []
    
    # Append new status and save
    status_list.append(status_entry)
    
    try:
        with open(STATUS_FILE, 'w') as f:
            json.dump(status_list, f, indent=2)
    except IOError as e:
        status(f"Warning: Could not save status to file: {e}")


def main(source_video_path: str, TARGET_CODE: str, gender: str, output_dir="intermediate"):

    filename = os.path.basename(source_video_path).split('.')[0]

    # Step 1: Convert video to WAV
    status("STEP 1: VIDEO TO AUDIO CONVERSION")
    audio_path = convert_mp4_to_wav(filename, source_video_path, output_dir)
    cleanup_memory()

    # Step 2: Separate speech and background — load AudioCleaner once, keep alive for Step 7
    status("STEP 2: AUDIO SEPARATION & ENHANCEMENT")
    audio_separator = AudioCleaner()
    raw_speech_path, background_path = audio_separator.process_audio(filename, audio_path, output_dir)
    clean_speech_path = audio_separator.enhance(filename, raw_speech_path, output_dir)
    # Demucs is no longer needed — free it now, keep DeepFilterNet alive for Step 7
    audio_separator.unload_demucs()

    # Step 3: Transcribe
    status("STEP 3: SPEECH TRANSCRIPTION")
    transcriber = Transcriber(model_size="large-v3")
    json_path, SOURCE_CODE = transcriber.transcribe_long_audio(filename, clean_speech_path, gender, output_dir)
    del transcriber.whisper
    del transcriber.vad_model
    del transcriber
    cleanup_memory()

    # Step 4: Calculate transcription durations
    translate_prompt_transcription = calculate_transcription_durations(filename, json_path, SOURCE_CODE, output_dir)
    print(translate_prompt_transcription)

    # Step 5: Translate
    status("STEP 5: TEXT TRANSLATION")
    translate_list(filename, output_dir, json_path, translate_prompt_transcription, TARGET_CODE, gender, model_name="gemma4:31b-cloud")
    cleanup_memory()

    # Step 5.5: Prepare TTS reference sample
    sample_speech = prepare_tts_sample(filename, clean_speech_path, json_path, output_dir)

    # Step 6: TTS generation — pipeline.unload() is called inside TTSBatchProcessor
    status("STEP 6: TEXT-TO-SPEECH GENERATION")
    tts_chunk_paths = TTSBatchProcessor(filename, sample_speech, json_path, output_dir, TARGET_CODE)
    cleanup_memory()

    # Step 7: Denoise TTS chunks using DeepFilterNet only (Demucs already freed in Step 2)
    status("STEP 7: TTS CHUNK DENOISING")
    for chunk_path in tts_chunk_paths:
        status(f"\t- Processing - {chunk_path}")
        chunk_name = os.path.basename(chunk_path).split('.')[0]
        denoised_path = audio_separator.process_vocals_only(
            chunk_name, chunk_path, os.path.join(output_dir, "tts_chunks", filename)
        )
        trim_end_silence(denoised_path, denoised_path, silence_threshold=-46.0)

    # All audio processing done — free DeepFilterNet
    audio_separator.unload()
    del audio_separator

    # Step 8: Dynamic Timeline Assembly
    status("STEP 8: DYNAMIC AUDIO/VIDEO MUXING")
    muxer = AudioDubbingMuxer(
        filename=filename,
        TARGET_CODE=TARGET_CODE,
        json_path=os.path.join(output_dir, "transcriptions", f"{filename}.json"),
        tts_dir=os.path.join(output_dir, "tts_chunks", filename, "raw_vocals"),
        bg_audio_path=os.path.join(output_dir, "background", f"{filename}.wav"),
        orig_video_path=source_video_path,
    )
    muxer.process(output_dir=os.path.join("mp4_translated_output", f"{filename}_{TARGET_CODE}"))

    status("✅ PIPELINE COMPLETED SUCCESSFULLY")



def main_enhance(source_video_path: str, output_dir="enhance"):

    filename = os.path.basename(source_video_path).split('.')[0]

    # Step 1: Convert video to WAV
    status("STEP 1: VIDEO TO AUDIO CONVERSION")
    audio_path = convert_mp4_to_wav(filename, source_video_path, output_dir)
    cleanup_memory()

    # Step 2: Separate speech and background noise from audio
    status("STEP 2: AUDIO SEPARATION & ENHANCEMENT")
    audio_separator = AudioCleaner()
    raw_speech_path, background_path = audio_separator.process_audio(filename, audio_path, output_dir)
    clean_speech_path = audio_separator.enhance(filename, raw_speech_path, output_dir)
    
    # Unload audio separator models immediately
    del audio_separator.separator
    del audio_separator.df_model
    del audio_separator.df_state
    del audio_separator
    cleanup_memory()

    status("STEP 3: DYNAMIC AUDIO/VIDEO MUXING")
    merge_video_audio(source_video_path, clean_speech_path, f"mp4_enhanced_output\\enhanced_{filename}.mp4")
    

    status("✅ PIPELINE COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    pass
    # source_video_path = "mp4_queue/shreya.mp4"
    # gender = "female"
    # TARGET_CODE = "hi"
    # main(source_video_path, TARGET_CODE, gender)
