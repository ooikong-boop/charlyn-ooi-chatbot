import os
import sys
import glob
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import argparse

# Load environment variables from .env file
load_dotenv()
client = OpenAI() # It will automatically use OPENAI_API_KEY

# Configuration
TMP_DIR = Path(".tmp")
KB_DIR = Path("knowledge_base")
TMP_DIR.mkdir(exist_ok=True)
KB_DIR.mkdir(exist_ok=True)

# Directives settings
DIRECTORIES = [
    {
        "name": "Accelerator Program",
        "path": r"N:\1.Video Work\Accelerator Program",
        "recursive": False
    },
    {
        "name": "Momentum Club",
        "path": r"N:\1.Video Work\Momentum Club",
        "recursive": True
    },
    {
        "name": "Leads on Autopilot",
        "path": r"N:\1.Video Work\Leads on Autopilot",
        "recursive": True
    }
]

def clean_tmp_dir():
    """Removes all .mp3 files in the temporary directory."""
    for f in TMP_DIR.glob("*.mp3"):
        try:
            f.unlink() 
        except OSError:
            pass

def process_video(video_path: str, program_name: str, rel_path: str):
    video_file = Path(video_path)
    # Output file logic
    # Clean up the output filename so it's a valid path
    safe_rel_path = str(rel_path).replace("\\", " - ").replace("/", " - ").replace(".mp4", ".md")
    output_dir = KB_DIR / program_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_md = output_dir / safe_rel_path
    if output_md.exists():
        print(f"Skipping already transcribed file: {output_md.name}")
        return

    print(f"\nProcessing: {video_file.name}")
    clean_tmp_dir()
    
    # Extract audio and segment it into ~20 min chunks to stay well under 25MB
    segment_pattern = str(TMP_DIR / "chunk_%03d.mp3")
    
    # We use -b:a 64k for decent speech quality but small file size.
    print(f"[{video_file.name}] Extracting and splitting audio...")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(video_file),
        "-vn", "-c:a", "libmp3lame", "-b:a", "64k",
        "-f", "segment", "-segment_time", "1200", 
        segment_pattern
    ]
    
    # Run ffmpeg (suppress output to keep console tidy, redirect to DEVNULL)
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio for {video_file.name}. Is FFmpeg missing or file corrupt?")
        return

    # Find the generated audio chunks
    chunks = sorted(list(TMP_DIR.glob("chunk_*.mp3")))
    if not chunks:
        print(f"Failed to extract any audio chunks from {video_file.name}")
        return
        
    print(f"[{video_file.name}] Audio split into {len(chunks)} chunk(s). Transcribing...")
    full_transcript = []
    
    for i, chunk_path in enumerate(chunks):
        print(f"  -> Transcribing chunk {i+1}/{len(chunks)} ({chunk_path.stat().st_size / 1024 / 1024:.1f} MB)... ", end="", flush=True)
        try:
            with open(chunk_path, "rb") as audio_file:
                # Call Whisper API with a prompt to guide spelling of specific names
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text",
                    prompt="Charlyn Ooi, business and marketing coach for nutrition professionals."
                )
                full_transcript.append(transcription)
                print("Done")
                
        except Exception as e:
            print(f"\nAPI Error during transcription: {e}")
            return
            
    # Combine everything and save to disk
    final_text = "\n\n".join(full_transcript)
    
    yaml_header = f"""---
Source Program: {program_name}
File Name: {video_file.name}
---

"""
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(yaml_header + final_text)
        
    print(f"[{video_file.name}] Finished and saved to {output_md.name}.")
    clean_tmp_dir()


def main():
    parser = argparse.ArgumentParser(description="Transcribe videos from specific training programs.")
    parser.add_argument("--program", type=str, choices=["Accelerator", "Momentum", "Leads", "All"], default="All",
                        help="Choose which program to process to keep it bite-sized.")
    
    args = parser.parse_args()
    
    # Filter directories based on user input
    directories_to_scan = []
    if args.program == "All":
        directories_to_scan = DIRECTORIES
    elif args.program == "Accelerator":
        directories_to_scan = [DIRECTORIES[0]]
    elif args.program == "Momentum":
        directories_to_scan = [DIRECTORIES[1]]
    elif args.program == "Leads":
        directories_to_scan = [DIRECTORIES[2]]

    print(f"Starting data ingestion for: {args.program}")
    print("--------------------------------------------------")
    
    for dir_config in directories_to_scan:
        program_name = dir_config["name"]
        base_path = Path(dir_config["path"])
        recursive = dir_config["recursive"]

        if not base_path.exists():
            print(f"Warning: The path {base_path} does not exist or is not attached. Skipping.")
            continue

        print(f"\n==============================================")
        print(f"Scanning program: {program_name}")
        print(f"==============================================")
        
        # Determine whether to search recursively or not
        search_pattern = "**/*.mp4" if recursive else "*.mp4"
        video_files = list(base_path.glob(search_pattern))
        
        if not video_files:
            print(f"No .mp4 files found in {base_path}.")
            continue
            
        print(f"-> Found {len(video_files)} videos in {program_name}. Beginning transcription...\n")
            
        for index, video_path in enumerate(video_files, start=1):
            print(f"\n[Video {index}/{len(video_files)}]")
            # Calculate path relative to the root program folder
            rel_path = video_path.relative_to(base_path)
            process_video(str(video_path), program_name, rel_path)
            
    print("\n==============================================")
    print("Processing complete!")
    print("==============================================")

if __name__ == "__main__":
    main()