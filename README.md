# Plexor_Assignment

---
## Setup Instructions

Follow these steps to set up the project locally on your machine.

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/yourusername/Personal_project.git
cd Personal_project
```

---

### 2. Create a Virtual Environment

It’s recommended to create a virtual environment to manage dependencies cleanly.

#### For macOS/Linux:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

#### For Windows:

```bash
python -m venv myenv
myenv\Scripts\activate
```

---

### 3. Install Dependencies

Once the environment is activated, install all required Python packages:

```bash
pip install -r requirements.txt
```

---
---

### 3. Download data

The data is stored in google drive and can be accessed using the folllowing link

```bash
https://drive.google.com/drive/folders/1o_Gp-goYRbYKZpYHVQ2ERUnoXIz-ZL1o?usp=sharing
```

---



## 🧾 Project Structure

A sample layout of this repository:

```
Personal_project/
│
├── myenv/                 # Virtual environment (auto-created)
├── requirements.txt       # Project dependencies
├── README.md              # Setup and usage guide
├── main.py                # Entry point (replace with your actual script)
├── src/                   # Source code (modules, utils, etc.)
├── data/                  # Dataset or input files (if applicable)
└── models/                # Trained models or weights (if applicable)
```

---
# Data Creation `prepare_data.py`

## Overview

`prepare_data.py` is a utility script that extracts and samples frames from video files to create datasets for training machine learning models. It helps organize frames into structured `train/` and `test/` folders, making it easier to label and use for model training. 

**Note :** skip this step if you already have annotated data downloaded from google drive
## Features

* Extracts frames from videos at a specified FPS (or all frames).
* Automatically splits frames into train and test sets.
* Displays frame count and FPS for each input video.
* Simple command-line interface for for sampling rate and test ratio.

## Usage

Run the script from the command line:
```bash
python prepare_date.py data/videos data/videos/extracted_frames
```
To run on other video source use:
```bash
python prepare_data.py /path/to/videos /path/to/output_dir
```



You’ll be prompted to:
1. Enter the sampling FPS (e.g., `2.0`).
2. Enter the test split ratio (e.g., `0.2`).

The extracted frames will be organized as:

```
output_dir/
├─ train/images
└─ test/images
```
```
output_dir/
├─ train/
│  ├─ images
│    ├─ video1_frame_00001.jpg
│  ├─ ...
├─ test/
│  ├─ images
│    ├─ video1_frame_00034.jpg
│  ├─ ...
```