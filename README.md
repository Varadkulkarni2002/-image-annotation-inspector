# AutoLabeler – Template Matching Annotation Tool

AutoLabeler is a **desktop GUI tool** for semi-automatic dataset annotation in Computer Vision projects.
It propagates bounding box labels from a **single reference image** to a batch of similar images using **OpenCV template matching**.

The tool is designed for **controlled environments** where object scale, orientation, and appearance are relatively consistent.

---

## Project Motivation

Manual annotation is inefficient for datasets collected from:
- Fixed cameras
- Industrial inspection setups
- Assembly lines
- Screen captures or UI datasets

In such cases, deep learning-based auto-labeling is often unnecessary.
AutoLabeler provides a **lightweight, classical CV alternative** that is fast, explainable, and easy to debug.

---

## Key Capabilities

- Label propagation using template matching
- GUI-based workflow (no CLI arguments)
- Supports common annotation formats
- Preview detection before batch execution
- Non-destructive processing
- Threaded execution to keep UI responsive

---

## Supported Formats

### Input
- YOLO (`.txt`)
- Pascal VOC (`.xml`)
- JSON (LabelMe-style)

### Output
- YOLO
- Pascal VOC (XML)
- JSON

---

## System Overview

The application follows this pipeline:

1. Parse reference annotation file
2. Extract bounding box from reference image
3. Crop template from reference image
4. Apply `cv2.matchTemplate` on target images
5. Filter detections using confidence threshold
6. Serialize output annotations in selected format

---

## Installation

### Requirements
- Python 3.7+
- pip

### Setup

Clone the repository:
```bash
git clone https://github.com/Varadkulkarni2002/Varadkulkarni2002-AutoLabeler-TemplateMatching.git
cd Varadkulkarni2002-AutoLabeler-TemplateMatching
```

Install dependencies:
```bash
pip install opencv-python numpy Pillow
```

> Note: `tkinter` is bundled with most Python distributions.
> On Linux:
```bash
sudo apt install python3-tk
```

---

## Running the Application

```bash
python autolabeller.py
```

No command-line arguments are required.

---

## Usage Workflow

### 1. Reference Selection
- Select a **reference image** containing the object
- Select its **annotation file**
- (YOLO only) Provide `classes.txt`

### 2. Target Dataset
- Select the directory containing unlabeled images

### 3. Output Configuration
- Select output directory
- Choose annotation format

### 4. Execution
- Adjust confidence threshold (recommended: `0.8`)
- Use **Preview Sample** to verify detection
- Start batch auto-labeling

---

## Output Structure

```
output_directory/
├── images/
│   ├── img_001.jpg
│   └── img_002.jpg
└── labels/
    ├── img_001.txt
    └── img_002.txt
```

(Structure adapts automatically for XML / JSON outputs.)

---

## Limitations

This tool uses **template matching**, not learned representations.

- Object scale must be similar to reference
- Rotation invariance is limited
- Sensitive to major lighting changes

### Recommended Use Cases
- Fixed-camera datasets
- Industrial CV pipelines
- Dataset bootstrapping
- Rapid annotation for POCs

Not recommended for:
- In-the-wild datasets
- Highly diverse object appearances

---

## Future Improvements

- Multi-scale template matching
- Multiple reference templates per class
- Manual review and correction interface
- Match confidence visualization

---

## License

This project is licensed under the MIT License.

---

## Author

Varad Kulkarni  
Applied AI / Computer Vision
