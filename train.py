import os
from pathlib import Path
from ultralytics import YOLO

def main():
    repo_root = Path(__file__).resolve().parent
    data_yaml = repo_root / "data.yaml"
    project_dir = repo_root / "runs"         

    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")

    model = YOLO("yolo11s.pt")

    model.train(
        data=str(data_yaml),
        epochs=80,           # Number of epochs
        imgsz=640,           # Image size (640x640)
        batch=16,            # Batch size
        fliplr=0.5,          # 50% chance of horizontal flip
        flipud=0.0,          # No vertical flip
        degrees=5,           # Random rotation ±5 degrees
        shear=10,            # Shear ±10 degrees
        name="yolo11s_custom_train",  # Optional: custom run name
        patience=50,         # Early stopping patience (optional)
        device=""            # Auto-detect GPU/CPU; use device=0 for specific GPU
    )
if __name__ == "__main__":
    main()
