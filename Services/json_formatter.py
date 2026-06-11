import json
import os

def calculate_transcription_durations(filename: str, input_filepath: str, SOURCE_CODE, output_dir: str):
    """
    Reads a complex JSON file containing audio transcription data, calculates 
    the duration of each transcription segment, and saves the result to a new JSON file.

    Args:
        input_filepath (str): The path to the input JSON file.
        output_filepath (str): The path where the result JSON file will be saved.
    """
    print(f"--- Starting processing for: {input_filepath} ---")

    output_dir = os.path.join(output_dir, "transcription_durations")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Read the input JSON file
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {input_filepath}. Check file format.")
        return
    
    if not isinstance(data, list):
        print("Error: Input JSON structure must be a list of objects.")
        return

    output_data = []

    # 2. Iterate through the main data structure to extract transcriptions
    for audio_item in data:
        # Ensure the necessary keys exist
        transcriptions = audio_item.get("transcription", [])

        # 3. Process each transcription segment
        for segment in transcriptions:
            start_time = segment.get("start")
            end_time = segment.get("end")
            transcription_text = segment.get(SOURCE_CODE)

            # Validate required fields
            if start_time is not None and end_time is not None and transcription_text is not None:
                # Calculate duration
                duration = end_time - start_time
                
                # Add the result in the desired format
                output_data.append({
                    SOURCE_CODE: transcription_text,
                    "duration": round(duration, 2)
                })

    output_filepath = os.path.join(output_dir, f"{filename}.json")

    # 4. Write the result to the output JSON file
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"\n✅ Success! Processed {len(output_data)} segments.")
        print(f"Output saved successfully to: {output_filepath}")
    except IOError:
        print(f"Error: Could not write output to {output_filepath}")

    return output_data