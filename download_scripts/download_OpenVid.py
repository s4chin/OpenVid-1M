import os
import subprocess
import argparse
import glob
import multiprocessing as mp
from functools import partial
import shutil

S3_BUCKET = "PLACEHOLDER"


def resize_video(input_path, output_path):
    """
    Resize video to 480p on the short side while maintaining aspect ratio
    """
    # Simple and reliable approach: scale with 480 as maximum dimension
    # This ensures short side is at most 480px while maintaining aspect ratio
    command = [
        "ffmpeg", "-i", input_path,
        "-vf", "scale=480:480:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-c:v", "libx264",      # Use H.264 codec for good compression
        "-crf", "23",           # Constant Rate Factor for quality (lower = better quality)
        "-preset", "medium",    # Encoding speed vs compression efficiency
        "-c:a", "aac",          # Audio codec
        "-b:a", "128k",         # Audio bitrate
        "-movflags", "+faststart", # Optimize for web streaming
        "-y",                   # Overwrite output file if it exists
        output_path
    ]
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error resizing {input_path}:")
        print(f"Command: {' '.join(command)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def process_single_video(args):
    """
    Worker function to process a single video - designed for multiprocessing
    Returns (success, video_file, output_path, filename) tuple
    """
    video_file, processed_folder = args
    filename = os.path.basename(video_file)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(processed_folder, f"{name}_480p{ext}")
    
    if os.path.exists(output_path):
        print(f"Processed video {output_path} already exists, skipping...")
        return (True, video_file, output_path, filename)  # Consider existing as success
        
    print(f"Processing {filename}...")
    success = resize_video(video_file, output_path)
    
    if success:
        print(f"Successfully processed {filename} -> {name}_480p{ext}")
        # Remove original to save space
        try:
            os.remove(video_file)
            print(f"Deleted original HD video: {filename}")
        except OSError as e:
            print(f"Warning: Could not delete {filename}: {e}")
    else:
        print(f"Failed to process {filename}")
    
    return (success, video_file, output_path, filename)


def process_videos_in_folder(video_folder, processed_folder, num_processes=None):
    """
    Process all videos in the video folder using multiprocessing
    """
    os.makedirs(processed_folder, exist_ok=True)
    
    # Get all video files (common extensions)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm', '*.flv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    
    if not video_files:
        print("No videos found to process.")
        return
    
    print(f"Found {len(video_files)} videos to process...")
    
    # Determine number of processes
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(video_files), 64)  # Cap at 64 to avoid overwhelming system
    
    print(f"Using {num_processes} processes for video processing...")
    
    # Prepare arguments for each worker
    worker_args = [(video_file, processed_folder) for video_file in video_files]
    
    # Process videos in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered for better memory efficiency and progress updates
        results = pool.imap_unordered(process_single_video, worker_args)
        
        # Process results as they complete
        successful = 0
        failed = 0
        for success, video_file, output_path, filename in results:
            if success:
                successful += 1
            else:
                failed += 1
    
    print(f"Video processing complete: {successful} successful, {failed} failed")


def zip_and_upload_to_s3(processed_folder, part_number):
    """
    Zip the processed folder and upload to S3
    Returns True if successful, False otherwise
    """
    if not os.path.exists(processed_folder) or not os.listdir(processed_folder):
        print(f"Processed folder {processed_folder} is empty or doesn't exist, skipping zip/upload")
        return False
    
    # Create zip file name
    zip_filename = f"video_480p_part_{part_number}.zip"
    zip_path = os.path.join(os.path.dirname(processed_folder), zip_filename)
    
    try:
        # Create zip file
        print(f"Creating zip file: {zip_filename}")
        shutil.make_archive(zip_path[:-4], 'zip', processed_folder)  # Remove .zip extension as make_archive adds it
        
        if not os.path.exists(zip_path):
            print(f"Error: Zip file {zip_path} was not created")
            return False
        
        # Upload to S3
        s3_path = f"s3://{S3_BUCKET}/kelkar/dataset/{zip_filename}"
        upload_command = ["aws", "s3", "cp", zip_path, s3_path]
        
        print(f"Uploading {zip_filename} to S3...")
        result = subprocess.run(upload_command, check=True, capture_output=True, text=True)
        print(f"Successfully uploaded {zip_filename} to {s3_path}")
        
        # Clean up: remove the zip file and processed folder to save space
        try:
            os.remove(zip_path)
            print(f"Deleted local zip file: {zip_path}")
            
            shutil.rmtree(processed_folder)
            print(f"Deleted processed folder: {processed_folder}")
        except OSError as e:
            print(f"Warning: Could not clean up files: {e}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error uploading {zip_filename} to S3:")
        print(f"Command: {' '.join(upload_command)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error creating zip or uploading {zip_filename}: {e}")
        return False


def download_files(output_directory, num_processes=None):
    zip_folder = os.path.join(output_directory, "download")
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(zip_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    error_log_path = os.path.join(zip_folder, "download_log.txt")

    for i in range(1, 100):
        # Create separate folder for each part's processed videos
        processed_folder = os.path.join(output_directory, f"video_480p_part_{i}")
        os.makedirs(processed_folder, exist_ok=True)
        
        url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_{i}.zip"
        file_path = os.path.join(zip_folder, f"OpenVidHD_part_{i}.zip")
        if os.path.exists(file_path):
            print(f"file {file_path} exits.")
            continue

        command = ["wget", "-O", file_path, url]
        unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
        try:
            subprocess.run(command, check=True)
            print(f"file {url} saved to {file_path}")
            subprocess.run(unzip_command, check=True)
            
            # Process videos after each successful extraction
            print(f"Processing videos from part {i}...")
            process_videos_in_folder(video_folder, processed_folder, num_processes)
            
            # Zip processed folder and upload to S3
            print(f"Zipping and uploading part {i} to S3...")
            success = zip_and_upload_to_s3(processed_folder, i)

            if success:
                print(f"Cleaning up original downloaded files for part {i}...")
                try:
                    os.remove(file_path)
                    print(f"Deleted original downloaded zip: {file_path}")
                except OSError as e:
                    print(f"Warning: Could not delete original zip {file_path}: {e}")
            
        except subprocess.CalledProcessError as e:
            error_message = f"file {url} download failed: {e}\n"
            print(error_message)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(error_message)
            
            part_urls = [
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_{i}_part_aa",
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVidHD/OpenVidHD_part_{i}_part_ab"
            ]

            for part_url in part_urls:
                part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
                if os.path.exists(part_file_path):
                    print(f"file {part_file_path} exits.")
                    continue

                part_command = ["wget", "-O", part_file_path, part_url]
                try:
                    subprocess.run(part_command, check=True)
                    print(f"file {part_url} saved to {part_file_path}")
                except subprocess.CalledProcessError as part_e:
                    part_error_message = f"file {part_url} download failed: {part_e}\n"
                    print(part_error_message)
                    with open(error_log_path, "a") as error_log_file:
                        error_log_file.write(part_error_message)
            file_path = os.path.join(zip_folder, f"OpenVidHD_part_{i}.zip")
            cat_command = "cat " + os.path.join(zip_folder, f"OpenVidHD_part_{i}_part*") + " > " + file_path
            unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
            os.system(cat_command)
            try:
                subprocess.run(unzip_command, check=True)
                # Process videos after successful extraction from concatenated parts
                print(f"Processing videos from part {i} (concatenated)...")
                process_videos_in_folder(video_folder, processed_folder, num_processes)
                
                # Zip processed folder and upload to S3
                print(f"Zipping and uploading part {i} (concatenated) to S3...")
                success = zip_and_upload_to_s3(processed_folder, i)

                if success:
                    print(f"Cleaning up original downloaded files for part {i}...")
                    try:
                        # Delete concatenated file
                        os.remove(file_path)
                        print(f"Deleted concatenated zip: {file_path}")

                        # Delete parts
                        part_files_pattern = os.path.join(zip_folder, f"OpenVidHD_part_{i}_part_*")
                        part_files = glob.glob(part_files_pattern)
                        for part_file in part_files:
                            os.remove(part_file)
                            print(f"Deleted part file: {part_file}")

                    except OSError as e:
                        print(f"Warning: Could not delete original downloaded files for part {i}: {e}")
            except subprocess.CalledProcessError as unzip_e:
                print(f"Failed to unzip concatenated file for part {i}: {unzip_e}")
    
    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        command = ["wget", "-O", data_path, data_url]
        subprocess.run(command, check=True)

    # delete zip files
    # delete_command = "rm -rf " + zip_folder
    # os.system(delete_command)


if __name__ == '__main__':
    # Required for multiprocessing on Windows and some other platforms
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, continue
        pass
    
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--output_directory', type=str, help='Path to the dataset directory', default="/path/to/dataset")
    parser.add_argument('--num_processes', type=int, help='Number of processes for video processing', default=None)
    args = parser.parse_args()
    download_files(args.output_directory, args.num_processes)
