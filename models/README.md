# PCB Detect AI - Model Directory

Place your custom-trained YOLOv11m PyTorch weights file in this directory and name it `best.pt`.

## Directory Path:
`models/best.pt`

## Model Format:
* **Framework**: PyTorch (`.pt`)
* **Ultralytics YOLO Compatibility**: The code expects a standard YOLOv11m model exported using the `ultralytics` package.
* **Input Size**: The model should be trained with an input size of `640x640`.
* **Classes**: The default UI supports 6 defect classes:
  1. `Missing Hole`
  2. `Mouse Bite`
  3. `Open Circuit`
  4. `Short`
  5. `Spur`
  6. `Spurious Copper`

## How it works:
When the application starts, it checks if `models/best.pt` is present:
- **Found**: The app loads the PyTorch model using `from ultralytics import YOLO` and runs active detection predictions on your uploaded PCB boards.
- **Not Found**: The app runs in simulator mode with an intelligent, deterministic layout generator. This allows testing all pages, graphs, webcam interfaces, and history tables out-of-the-box.
