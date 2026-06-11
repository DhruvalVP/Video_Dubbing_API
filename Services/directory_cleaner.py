import os
import shutil

def delete_contents_in_folders(base_dir):
    """
    Deletes all files and subdirectories found inside specific 
    subdirectories within the given base directory structure.

    Args:
        base_dir (str): The path to the root directory (e.g., 'intermediate').
    """
    # Define the specific subdirectories we want to clean up
    target_folders = [
        os.path.join(base_dir, 'background'),
        os.path.join(base_dir, 'clean_vocals'),
        os.path.join(base_dir, 'raw_vocals'),
        os.path.join(base_dir, 'sample_speech'),
        os.path.join(base_dir, 'transcriptions'),
        os.path.join(base_dir, 'wav_output_batch'),
        os.path.join(base_dir, 'tts_chunks'),
        "mp4_queue",
        "enhance",
    ]

    print(f"Starting cleanup in directory: {base_dir}\n")
    
    for folder_path in target_folders:
        print(f"--- Checking directory: {folder_path} ---")
        
        if not os.path.isdir(folder_path):
            print(f"Warning: Directory not found: {folder_path}. Skipping.")
            continue
            
        deleted_count = 0
        
        # Iterate over all items inside the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # Check if it's a file or a symlink
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                    print(f"  [DELETED FILE] {filename}")
                    deleted_count += 1
                
                # Check if it's a directory
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print(f"  [DELETED DIR]  {filename}/")
                    deleted_count += 1
                    
            except Exception as e:
                print(f"  [ERROR] Could not delete {file_path}. Error: {e}")
        
        if deleted_count == 0:
            print("  No items found to delete in this folder.")
        print("-" * 30 + "\n")


# --- Execution ---

# NOTE: Replace 'intermediate' with the actual path to your directory if needed.
# This function assumes the structure is relative to where you run the script.