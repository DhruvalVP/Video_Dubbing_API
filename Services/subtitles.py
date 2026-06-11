import json
import subprocess
import os

def seconds_to_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def add_switchable_subtitles(video_path: str, json_path: str, source_lang: str, target_lang: str, output_path: str):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)[0]
        
    transcription = data.get("transcription", [])
    
    langs = []
    for lang in [source_lang, target_lang, "en"]:
        if lang not in langs:
            langs.append(lang)

    srt_files = []
    valid_langs = []
    
    for lang in langs:
        srt_filename = f"temp_{lang}.srt"
        
        # utf-8-sig adds a BOM, which FFmpeg heavily prefers for non-ASCII characters on Windows
        with open(srt_filename, 'w', encoding='utf-8-sig') as f:
            counter = 1
            has_content = False
            
            for chunk in transcription:
                if lang in chunk:
                    start_time = seconds_to_srt_time(chunk['start'])
                    end_time = seconds_to_srt_time(chunk['end'])
                    text = chunk[lang].strip()
                    
                    f.write(f"{counter}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
                    counter += 1
                    has_content = True
        
        # Only process files that actually have subtitle text inside them
        if has_content:
            srt_files.append(srt_filename)
            valid_langs.append(lang)
        else:
            if os.path.exists(srt_filename):
                os.remove(srt_filename)

    if not srt_files:
        raise ValueError("No valid subtitles found in the JSON to process.")

    ffmpeg_cmd = ["ffmpeg", "-y", "-i", video_path]
    
    for srt_file in srt_files:
        ffmpeg_cmd.extend(["-i", srt_file])
        
    ffmpeg_cmd.extend(["-map", "0:v", "-map", "0:a"])
    
    for i in range(len(srt_files)):
        ffmpeg_cmd.extend(["-map", f"{i+1}:s"])
        
    subtitle_codec = "mov_text" if output_path.lower().endswith(".mp4") else "srt"
    ffmpeg_cmd.extend(["-c:v", "copy", "-c:a", "copy", "-c:s", subtitle_codec])
    
    for i, lang in enumerate(valid_langs):
        ffmpeg_cmd.extend([
            f"-metadata:s:s:{i}", f"language={lang}", 
            f"-metadata:s:s:{i}", f"title={lang.upper()}"
        ])
        
    ffmpeg_cmd.append(output_path)

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    finally:
        for srt_file in srt_files:
            if os.path.exists(srt_file):
                os.remove(srt_file)


add_switchable_subtitles(
    video_path=r"mp4_translated_output\sample_1_hi\sample_1_hi.mp4",
    json_path=r"intermediate\transcriptions\sample_1.json",
    source_lang="en",
    target_lang="hi",
    output_path=r"mp4_translated_output\sample_1_hi\sample_1_hi_subtitles.mp4"
)