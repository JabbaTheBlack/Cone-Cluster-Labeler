#!/usr/bin/env python3
"""
Script to filter frame files in cone_clusters folder:
- Keep frames 0-27 (already labeled)
- Keep every 10th frame starting from 250 (250, 260, 270, 280, ...)
- Delete all other frames
"""

import os
import re
from pathlib import Path

def extract_frame_number(filename):
    """Extract frame number from filename."""
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def should_keep_frame(frame_num):
    """
    Determine if a frame should be kept based on filtering rules:
    - Keep frames 0-27 (already labeled)
    - Keep frames 250, 260, 270, ... (every 10th starting from 250)
    """
    # Keep frames 0-27
    if frame_num <= 27:
        return True
    
    # Keep every 10th frame starting from 250
    if frame_num >= 250 and (frame_num - 250) % 10 == 0:
        return True
    
    return False

def main():
    # Path to cone_clusters directory
    cone_clusters_dir = Path(__file__).parent / "Dataset" / "cone_clusters"
    
    if not cone_clusters_dir.exists():
        print(f"Error: Directory {cone_clusters_dir} does not exist!")
        return
    
    print(f"Scanning directory: {cone_clusters_dir}")
    
    # Get all files
    all_files = list(cone_clusters_dir.glob("*.pcd"))
    
    # Group files by frame number
    frames_dict = {}
    for file in all_files:
        frame_num = extract_frame_number(file.name)
        if frame_num is not None:
            if frame_num not in frames_dict:
                frames_dict[frame_num] = []
            frames_dict[frame_num].append(file)
    
    # Statistics
    frames_to_keep = []
    frames_to_delete = []
    files_to_delete = []
    files_to_keep = []
    
    for frame_num in sorted(frames_dict.keys()):
        if should_keep_frame(frame_num):
            frames_to_keep.append(frame_num)
            files_to_keep.extend(frames_dict[frame_num])
        else:
            frames_to_delete.append(frame_num)
            files_to_delete.extend(frames_dict[frame_num])
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total frames found: {len(frames_dict)}")
    print(f"Frames to keep: {len(frames_to_keep)}")
    print(f"Frames to delete: {len(frames_to_delete)}")
    print(f"\nTotal files to keep: {len(files_to_keep)}")
    print(f"Total files to delete: {len(files_to_delete)}")
    
    print(f"\nFrames that will be kept:")
    print(f"  - Frames 0-27: {sum(1 for f in frames_to_keep if f <= 27)} frames")
    print(f"  - Frames >= 250 (every 10th): {sum(1 for f in frames_to_keep if f >= 250)} frames")
    
    # Show some examples of frames to keep
    print(f"\nExample frames to keep (max 20):")
    example_frames = frames_to_keep[:10] + frames_to_keep[-10:]
    print(f"  {example_frames}")
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("Do you want to proceed with deletion? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\nDeleting files...")
        deleted_count = 0
        for file in files_to_delete:
            try:
                file.unlink()
                deleted_count += 1
                if deleted_count % 100 == 0:
                    print(f"  Deleted {deleted_count}/{len(files_to_delete)} files...")
            except Exception as e:
                print(f"  Error deleting {file.name}: {e}")
        
        print(f"\n✓ Successfully deleted {deleted_count} files!")
        print(f"✓ Kept {len(files_to_keep)} files from {len(frames_to_keep)} frames")
    else:
        print("\nOperation cancelled. No files were deleted.")

if __name__ == "__main__":
    main()
