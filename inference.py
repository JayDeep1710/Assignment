import argparse
import cv2
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 and show only selected classes.")
    p.add_argument("--video", required=True, help="Path to video file or webcam index (0,1,...). Use '0' for webcam.")
    p.add_argument("--classes", required=True, help="Comma-separated list of class ids to show, e.g. 0,2")
    return p.parse_args()

def str_to_class_list(s):
    return [int(x) for x in s.split(",") if x.strip()]

def main():
    args = parse_args()
    allowed_classes = str_to_class_list(args.classes)

    video_source = int(args.video) if args.video.isdigit() else args.video
    model = YOLO("best.pt")

    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=allowed_classes)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Filtered", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
