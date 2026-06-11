import os
from moviepy import VideoFileClip, AudioFileClip

def merge_video_audio(
    video_path: str, audio_path: str, output_path: str
) -> str:
    """Mutes the original video audio and merges it with a new audio track

    using MoviePy v2.0+ syntax.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. Load the video and the new audio track
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)

        # 2. Assign the new audio track directly using the v2.0 property syntax
        video_clip.audio = audio_clip

        # 3. Export the final video file
        video_clip.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            logger=None
        )

        # 4. Close the clips to cleanly release system resources
        video_clip.close()
        audio_clip.close()

        return os.path.abspath(output_path)

    except Exception as e:
        print(f"[ERROR] Failed to process video/audio alignment: {str(e)}")
        raise e