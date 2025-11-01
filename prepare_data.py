import os
import cv2
import random
import shutil
from pathlib import Path
import argparse

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def count_frames_and_fps_per_video(folder_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    video_info = {}
    total_frames = 0
    
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path '{folder_path}' does not exist.")
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename.lower())
            if ext in video_extensions:
                cap = cv2.VideoCapture(file_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    video_info[filename] = {'frames': frame_count, 'fps': fps}
                    total_frames += frame_count
                    cap.release()
                else:
                    print(f"{Colors.WARNING}Warning: Could not open video file '{file_path}'.{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Video info (frames and FPS):{Colors.ENDC}")
    for filename, info in video_info.items():
        print(f"  {Colors.OKBLUE}{filename}: {info['frames']} frames at {info['fps']:.2f} FPS{Colors.ENDC}")
    print(f"{Colors.OKGREEN}\nTotal number of frames: {total_frames}{Colors.ENDC}")
    return video_info

def sample_frames_from_videos(folder_path, sample_fps=1.0, output_dir='sampled_frames'):

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v', '.webm'}
    video_paths = {}
    total_extracted = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename.lower())
            if ext in video_extensions:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"{Colors.WARNING}Warning: Could not open '{file_path}'.{Colors.ENDC}")
                    continue
                
                orig_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if sample_fps is None:
                    frame_interval = 1
                    target_fps = orig_fps
                else:
                    frame_interval = max(1, int(orig_fps / sample_fps))
                    target_fps = orig_fps / frame_interval
                
                frame_paths = []
                frame_idx = 0
                sampled_time = 0.0
                extracted_count = 0
                
                print(f"{Colors.OKCYAN}Extracting from {filename} ({total_frames} total frames, interval={frame_interval})...{Colors.ENDC}")
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_idx % frame_interval == 0:
                        video_stem = Path(filename).stem
                        image_path = os.path.join(output_dir, f"{video_stem}_frame_{frame_idx:06d}_{sampled_time:.2f}s.jpg")
                        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        frame_paths.append(image_path)
                        extracted_count += 1
                    
                    frame_idx += 1
                    if sample_fps is None:
                        sampled_time += 1.0 / orig_fps
                    else:
                        sampled_time += 1.0 / target_fps
                
                cap.release()
                video_paths[filename] = frame_paths
                total_extracted += extracted_count
                print(f"  {Colors.OKGREEN}Extracted {extracted_count} frames from {filename}{Colors.ENDC}")
    
    print(f"{Colors.OKGREEN}\nTotal sampled frames: {total_extracted}{Colors.ENDC}")
    for video, paths in video_paths.items():
        print(f"  {Colors.OKBLUE}{video}: {len(paths)} frames{Colors.ENDC}")
    
    return video_paths

def train_test_split_samples(video_paths, test_size=0.2, random_state=42):
    random.seed(random_state)
    train_paths = {}
    test_paths = {}
    
    for video, paths in video_paths.items():
        random.shuffle(paths)
        n_test = int(len(paths) * test_size)
        test_paths[video] = paths[:n_test]
        train_paths[video] = paths[n_test:]
    
    total_train = sum(len(p) for p in train_paths.values())
    total_test = sum(len(p) for p in test_paths.values())
    print(f"{Colors.OKGREEN}Train: {total_train} frames | Test: {total_test} frames{Colors.ENDC}")
    
    return train_paths, test_paths

def organize_frames(video_paths, base_dir):
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    for video_name, paths in video_paths.items():
        for src in paths:
            dst = os.path.join(images_dir, os.path.basename(src))
            shutil.copy2(src, dst)
    
    total_frames = sum(len(paths) for paths in video_paths.values())
    print(f"{Colors.OKGREEN}Organized {total_frames} frames into {images_dir}  {Colors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample frames from videos with configurable FPS and split.")
    parser.add_argument("input_dir", type=str, help="Path to the input folder containing videos.")
    parser.add_argument("output_dir", type=str, help="Path to the output folder for sampled frames (will contain train/ and test/ subdirs).")
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}{Colors.BOLD}Analyzing videos...{Colors.ENDC}")
    video_info = count_frames_and_fps_per_video(args.input_dir)
    
    while True:
        try:
            sample_fps_input = input(f"{Colors.OKCYAN}\nEnter sample FPS (e.g., 1.0 for 1 FPS, or 'None' for all frames): {Colors.ENDC}").strip()
            if sample_fps_input.lower() == 'none':
                sample_fps = None
            else:
                sample_fps = float(sample_fps_input)
            break
        except ValueError:
            print(f"{Colors.FAIL}Invalid input. Please enter a valid number or 'None'.{Colors.ENDC}")
    
    while True:
        try:
            test_size = float(input(f"{Colors.OKCYAN}Enter test split ratio (e.g., 0.2 for 20% test): {Colors.ENDC}"))
            if 0 < test_size < 1:
                break
            else:
                print(f"{Colors.WARNING}Must be between 0 and 1 (exclusive).{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.FAIL}Invalid input. Please enter a valid number between 0 and 1.{Colors.ENDC}")

    print(f"{Colors.HEADER}\nStarting frame extraction at {sample_fps if sample_fps is not None else 'all frames'} FPS...{Colors.ENDC}")
    video_paths = sample_frames_from_videos(args.input_dir, sample_fps=sample_fps, output_dir=args.output_dir)

    if len(video_paths) > 0:
        train_paths, test_paths = train_test_split_samples(video_paths, test_size=test_size)
        train_base = os.path.join(args.output_dir, 'train')
        test_base = os.path.join(args.output_dir, 'test')
        organize_frames(train_paths, train_base)
        organize_frames(test_paths, test_base)
        temp_dir = args.output_dir
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)

        print(f"{Colors.OKGREEN}\nFrames organized: train/ and test/ subdirs in {args.output_dir} {Colors.ENDC}")
        print(f"{Colors.OKBLUE}Structure: {args.output_dir}/{{train|test}}/images/video_name_frame_*.jpg{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}No frames were sampled. Check your inputs.{Colors.ENDC}")