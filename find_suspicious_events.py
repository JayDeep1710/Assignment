import os
import cv2
from collections import deque
from ultralytics import YOLO
import imageio
import argparse

def save_video(buffer, folder_path, name):
    os.makedirs(folder_path, exist_ok=True)
    filename = os.path.join(folder_path, f"{name}.mp4")
    height, width = buffer[0].shape[:2]
    writer = imageio.get_writer(filename, fps=5, codec="libx264", format="ffmpeg")
    for f in buffer:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()

def custom_annotate(frame, results):
    class_names = ['Person', 'Normal wrist', 'wrist with product', 'Suspicious']
    suspicious_class_id = 3
    person_class_id = 0
    annotated = frame.copy()
    suspicious_detected = False
    if results[0].boxes is not None:
        for box in results[0].boxes:
            if int(box.cls) == suspicious_class_id:
                suspicious_detected = True
                break
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls)
            if cls == suspicious_class_id:
                continue
            if suspicious_detected and cls == person_class_id:
                continue
            color = (0, 255, 0)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            
            conf = float(box.conf)
            label = f"{class_names[cls]} {conf:.2f}"
            font_scale = 1.0
            thickness = 3
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(annotated, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)  # Filled background
            cv2.putText(annotated, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    if results[0].boxes is not None and suspicious_detected:
        for box in results[0].boxes:
            cls = int(box.cls)
            if cls == suspicious_class_id:
                color = (0, 0, 255)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
                
                conf = float(box.conf)
                label = f"{class_names[cls]} {conf:.2f}"
                font_scale = 1.0
                thickness = 3
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                cv2.rectangle(annotated, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, -1)  # Filled background
                cv2.putText(annotated, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return annotated

def process_video(video_path, model_path, output_dir='suspicious_events', suspicious_class_id=3, pre_sec=3, post_sec=5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    print(f"Processing video at {fps} FPS")
    
    pre_frames = int(pre_sec * fps)
    post_frames = int(post_sec * fps)
    
    raw_buffer = deque(maxlen=pre_frames)
    annotated_buffer = deque(maxlen=pre_frames)
    
    post_remaining = 0
    clip_raw = None
    clip_annotated = None
    event_num = 1
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, verbose=False)
        annotated = custom_annotate(frame, results)

        suspicious = False
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if int(box.cls) == suspicious_class_id:
                    suspicious = True
                    break

        if suspicious and post_remaining == 0:
            clip_raw = list(raw_buffer) + [frame]
            clip_annotated = list(annotated_buffer) + [annotated]
            post_remaining = post_frames
            print(f"Suspicious detected at frame {frame_idx}, starting clip {event_num}")
        elif post_remaining > 0:
            clip_raw.append(frame)
            clip_annotated.append(annotated)
            post_remaining -= 1
            if post_remaining == 0:
                folder_path = os.path.join(output_dir, f"suspicious_{event_num:03d}")
                save_video(clip_annotated, folder_path, "annotated")
                save_video(clip_raw, folder_path, "raw")
                print(f"Saved clip {event_num} to {folder_path}")
                event_num += 1
        raw_buffer.append(frame)
        annotated_buffer.append(annotated)
        frame_idx += 1
    
    cap.release()
    if post_remaining > 0 and clip_raw is not None:
        folder_path = os.path.join(output_dir, f"suspicious_{event_num:03d}")
        save_video(clip_annotated, folder_path, "annotated")
        save_video(clip_raw, folder_path, "raw")
        print(f"Saved partial clip {event_num} to {folder_path}")
    
    print(f"Processing complete. Saved {event_num - 1} clips.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for suspicious events using YOLO.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("model_path", help="Path to the YOLO model file.")
    args = parser.parse_args()
    process_video(args.video_path, args.model_path)