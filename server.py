import os
import json
import asyncio
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

from main import main

# Thread pool for background processing
executor = ThreadPoolExecutor(max_workers=2)
# Thread lock to prevent concurrent file read/write issues
file_lock = threading.Lock()

QUEUE_DIR = "mp4_queue"
STATUS_FILE = "dubbing_status.json"

os.makedirs(QUEUE_DIR, exist_ok=True)

app = FastAPI(title="AI Dubbing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoItem(BaseModel):
    link: HttpUrl
    gender: str

class DubbingRequest(BaseModel):
    videos: List[VideoItem]
    target: str

class EnhanceRequest(BaseModel):
    videos: List[HttpUrl]


def download_video(url: str) -> str:
    """Synchronous download helper meant to be executed inside a background thread"""
    filename = url.split("/")[-1] or "video.mp4"
    dest = os.path.join(QUEUE_DIR, filename)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return os.path.abspath(dest)


def get_all_status():
    """Read all status messages safely using a thread lock"""
    if os.path.exists(STATUS_FILE):
        with file_lock:
            try:
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
    return []


def get_latest_status():
    """Read latest status safely"""
    status_list = get_all_status()
    return status_list[-1] if status_list else None


def clear_status():
    """Clear status safely using a thread lock"""
    with file_lock:
        try:
            with open(STATUS_FILE, 'w') as f:
                json.dump([], f)
        except IOError:
            pass


@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/clean-memory")
def health_check():
    from Services.directory_cleaner import delete_contents_in_folders

    delete_contents_in_folders('intermediate')

    return {"status": "cleaned", "timestamp": datetime.now().isoformat()}


@app.get("/processing/status")
def processing_status():
    status_list = get_all_status()
    is_processing = len(status_list) > 0
    return {
        "is_processing": is_processing,
        "status_count": len(status_list),
        "latest_status": status_list[-1] if status_list else None
    }


@app.get("/status/latest")
def get_status_latest():
    latest = get_latest_status()
    return latest if latest else {"message": "No status available"}


@app.get("/status/all")
def get_status_all():
    return {"statuses": get_all_status()}


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket):
    await websocket.accept()
    last_index = 0
    
    try:
        while True:
            status_list = get_all_status()
            
            if len(status_list) > last_index:
                for status_entry in status_list[last_index:]:
                    await websocket.send_json(status_entry)
                last_index = len(status_list)
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        # Graceful exit on disconnect
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass # Already closed


@app.post("/dub")
async def dub_videos(request: DubbingRequest):
    """Start downloading and dubbing process completely in background"""
    clear_status()
    
    # Extract data to pass to background thread
    video_inputs = [(str(v.link), v.gender.lower()) for v in request.videos]
    target_language = request.target

    # Move EVERYTHING heavy (including downloads) to the thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_full_pipeline,
        video_inputs,
        target_language
    )
    
    return {
        "status": "processing",
        "message": "Downloads and Dubbing started in background",
        "videos_queued": len(video_inputs),
        "note": "Monitor progress via /status/latest or WebSocket /ws/status"
    }


def run_full_pipeline(video_inputs, target_code):
    """Runs fully in background thread: Handles download, then runs AI Pipeline"""
    from main import status
    
    try:
        # 1. Download phase (happens in background, doesn't block FastAPI main loop)
        for url, gender in video_inputs:
            video_path  = download_video(url)

            status(f"STEP 0: VIDEO DOWNLOADED AT {video_path.upper()}")
            
            # 2. Heavy processing phase
            main(
                source_video_path=video_path,
                TARGET_CODE=target_code,
                gender=gender
            )
    except Exception as e:
        try:
            status(f"ERROR: Process failed - {str(e)}")
        except Exception:
            # Fallback if main.status itself errors out
            with file_lock:
                try:
                    with open(STATUS_FILE, 'r+') as f:
                        data = json.load(f)
                        data.append({"status": "error", "message": str(e)})
                        f.seek(0)
                        json.dump(data, f)
                except Exception:
                    pass


@app.post("/enhance")
async def enhance_videos(request: EnhanceRequest):
    """Start downloading and dubbing process completely in background"""
    clear_status()
    
    # Extract data to pass to background thread
    video_inputs = [str(v) for v in request.videos]

    # Move EVERYTHING heavy (including downloads) to the thread pool
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_enhance_pipeline,
        video_inputs,
    )
    
    
    return {
        "status": "processing",
        "message": "Downloads and Enhancing started in background",
        "videos_queued": len(video_inputs),
        "note": "Monitor progress via /status/latest or WebSocket /ws/status"
    }


def run_enhance_pipeline(video_inputs):
    """Runs fully in background thread: Handles download, then runs AI Pipeline"""
    from main import main_enhance
    from main import status
    
    try:
        # 1. Download phase (happens in background, doesn't block FastAPI main loop)
        for url in video_inputs:
            video_path  = download_video(url)

            status(f"STEP 0: VIDEO DOWNLOADED AT {video_path.upper()}")
            
            # 2. Heavy processing phase
            main_enhance(
                source_video_path=video_path
            )
    except Exception as e:
        try:
            status(f"ERROR: Process failed - {str(e)}")
        except Exception:
            # Fallback if main.status itself errors out
            with file_lock:
                try:
                    with open(STATUS_FILE, 'r+') as f:
                        data = json.load(f)
                        data.append({"status": "error", "message": str(e)})
                        f.seek(0)
                        json.dump(data, f)
                except Exception:
                    pass


if __name__ == "__main__":
    # Standardized run config assuming this file is named server.py
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)