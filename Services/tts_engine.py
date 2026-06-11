from Services.omnivoice_tts import RobustTTSPipeline
import json
import os

def chunk_duration_calculator(start_time, end_time):
    estimated_salt_duration = 1.25
    return end_time - start_time + estimated_salt_duration

def TTSBatchProcessor(filename, sample_speech_path, json_path, output_dir, target_code):

    output_dir = os.path.join(output_dir, "tts_chunks", filename)
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_language = str(data[0]["audio-language"])
    print(f"Source language detected: {source_language}")
    print(f"Reference Text = {data[0]['transcription'][0][source_language]}")

    pipeline = RobustTTSPipeline()

    chunks_path = []
    for i, text in enumerate(data[0]["transcription"]):
        try:
            print(f"\n{'='*60}")
            print(f"Chunk {i}: \"{text[target_code]}\"")
            print('='*60)
            audio, sr = pipeline.generate(
                text=str(text[target_code]),
                ref_audio=sample_speech_path,
                ref_text=data[0]['transcription'][0][source_language],
                duration=chunk_duration_calculator(data[0]["transcription"][i]["start"], data[0]["transcription"][i]["end"]),
                # instruct="energetic and confident review speech style",
                output_path=os.path.join(output_dir, f"{filename}_chunk_{i}.wav"),
            )
            chunks_path.append(os.path.join(output_dir, f"{filename}_chunk_{i}.wav"))
            print(f"✅ Generated: {filename}_chunk_{i}.wav")
        except Exception as e:
            print(f"❌ Error generating chunk {i}: {e}")
            import traceback
            traceback.print_exc()

    pipeline.unload()
    return chunks_path