# PCB Detect AI - Industrial Vision Inspection Suite

PCB Detect AI is a production-grade web application built to analyze Printed Circuit Board (PCB) images for manufacturing defects using YOLOv11 models. It features a modern, high-fidelity UI/UX styled like premium commercial AI SaaS interfaces.

---

## 🚀 Key Features

* **Interactive Dashboard**: High-level production metrics, KPI metrics cards, and yield status indicators.
* **Single Image Detection**: Upload a PCB board image to run real-time defect localization. Highlights anomalies side-by-side with original panels and shows classification confidence scales.
* **Batch Inspection**: Upload and inspect multiple PCB boards in parallel with progress bar updates and CSV export.
* **Live Webcam Stream**: Support for physical webcams with real-time frame rates and prediction statistics. Falls back to a mock conveyor belt video stream in server environments.
* **Analytics Deep-Dive**: Defect frequency counts, monthly timelines, weekday inspection load heatmaps, and spatial defect density plots.
* **Detection History Logs**: Search, filter, delete, and view side-by-side history records stored in an SQLite persistence database.
* **Custom Config Options**: Custom sliders for confidence thresholds, IoU overlays, bounding box visual properties (thickness and text sizing), and GPU acceleration toggles.

---

## 🛠️ Tech Stack

* **Front-end**: Streamlit (layout grids, container objects, custom CSS stylesheets)
* **Model Inference**: Ultralytics YOLOv11 & PyTorch
* **Computer Vision**: OpenCV (`cv2`) & Pillow (`PIL`)
* **Data Processing**: Pandas & NumPy
* **Analytics Rendering**: Plotly Express & Plotly Graph Objects
* **Database**: SQLite3

---

## 📁 Folder Structure

```text
PCB_detect/
│
├── app.py                     # Main application routing script
│
├── pages/                     # Page view component structures
│   ├── dashboard.py           # Dashboard widgets and summaries
│   ├── image_detection.py     # Single-board inspector panel
│   ├── batch_detection.py     # Multi-image high-throughput panel
│   ├── live_camera.py         # Real-time webcam acquisition loop
│   ├── history.py             # Database log viewer and search filters
│   ├── analytics.py           # Advanced plots and spatial defect density heatmaps
│   ├── model_info.py          # Model specs and accuracy cards
│   └── settings.py            # Custom slider settings
│
├── utils/                     # Supporting utilities
│   ├── predict.py             # Model execution and fallback simulation engine
│   ├── charts.py              # Visual analytics Plotly chart loaders
│   ├── database.py            # SQLite log database manager and seeder
│   └── helpers.py             # CSS inject helpers and image encoders
│
├── assets/                    # Styling sheets
│   └── style.css              # Custom SaaS style overrides
│
├── models/                    # Model folder for custom weights
│   └── README.md
│
├── history/                   # Folder for saving uploaded scans
│
├── requirements.txt           # Package dependencies
└── README.md                  # System instruction and setup guide
```

---

## ⚙️ Running Locally

### 1. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/macOS
```

### 2. Install Packages
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Load Custom Weights
To replace the default simulation mode with your custom-trained YOLO model, place your trained YOLO PyTorch weights at:
`models/best.pt`

The system diagnostics sidebar will automatically recognize the weights and swap execution targets at runtime.
