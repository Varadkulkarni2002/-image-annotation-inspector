# Universal Visual Inspector

Universal Visual Inspector is a **desktop GUI tool** for visual inspection, review, and manual correction of image annotations used in Computer Vision datasets.

It is built for engineers who need to **validate dataset quality**, debug annotations, and visually inspect images before or after training — not for automatic labeling or model inference.

---

## What This Tool Is

- A **visual inspection and annotation review tool**
- A way to **open images + labels together**
- A utility to **verify, inspect, and manually mark datasets**
- A companion tool for auto-labelers and training pipelines

---

## What This Tool Is NOT

- ❌ Not an auto-labeling tool  
- ❌ Not a model training or inference system  
- ❌ Not an AI-powered labeling platform  

This tool focuses on **clarity, control, and correctness**.

---

## Key Features

- Desktop GUI (Tkinter-based)
- Supports YOLO, Pascal VOC (XML), and JSON annotations
- Overlay bounding boxes on images
- Manual navigation through datasets (Previous / Next / Go to index)
- Zoom, pan, and fullscreen inspection
- Manual drawing and erasing for visual markup
- Batch folder inspection support
- Non-destructive (original files are never modified)

---

## Supported Formats

### Image Formats
- `.jpg`
- `.jpeg`
- `.png`

### Annotation Formats
- YOLO (`.txt`)
- Pascal VOC (`.xml`)
- JSON

---

## Installation

### Requirements
- Python 3.7+
- pip

### Python Dependencies
```bash
pip install -r requirements.txt
```

> **Note:**  
> `tkinter` is included with most Python installations.  
> On Linux systems:
```bash
sudo apt install python3-tk
```

---

## Running the Application

From the project root:
```bash
python visual_inspection.py
```

This launches the GUI window.

---

## Usage

### Manual Inspection Mode
1. Select an image file
2. Select its corresponding annotation file
3. (Optional) Select `classes.txt` for YOLO labels
4. Click **Process Single Image**

### Automated Folder Inspection Mode
1. Select a root folder containing images and labels
2. The tool pairs images with available annotations
3. Navigate through the dataset using Next / Previous / Go

---

## Recommended Dataset Structure

```
dataset/
├── images/
│   ├── image_001.jpg
│   └── image_002.jpg
└── labels/
    ├── image_001.txt
    └── image_002.txt
```

Flat folder structures are also supported.

---

## Controls

- **Pan:** Left-click drag or middle mouse drag
- **Zoom:** Ctrl + Mouse Wheel
- **Draw Mode:** Left-click drag
- **Erase Mode:** Drag over drawn elements
- **Fullscreen:** F11
- **Exit Fullscreen:** Esc

---

## Typical Use Cases

- Dataset sanity checks before training
- Reviewing auto-labeled datasets
- Industrial vision inspection workflows
- Debugging incorrect bounding boxes
- Academic and research dataset validation

---

## Limitations

- Does not modify or rewrite annotation files
- Manual drawing is for visual inspection only
- Designed for single-user, local workflows

---

## License

MIT License

---

## Author

Varad Kulkarni  
Applied AI / Computer Vision
