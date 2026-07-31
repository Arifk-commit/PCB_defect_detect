import os
import streamlit as st

# Default Hugging Face repository and model filename settings
HF_REPO_ID = "Arifk-commit/PCB_defect_detect_YOLO_V11m"
HF_FILENAME = "best_11m.pt"

# Local model cache paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')


def get_local_pt_path():
    """
    Scans the `models/` directory for ANY `.pt` weights file.
    Prefers 'best.pt' if present, otherwise returns the first `.pt` file found.
    """
    if os.path.exists(MODEL_DIR):
        pt_files = [
            os.path.join(MODEL_DIR, f)
            for f in os.listdir(MODEL_DIR)
            if f.endswith('.pt') and not f.startswith('.')
        ]
        if pt_files:
            # Prioritize best.pt if it exists among files
            best_match = [f for f in pt_files if os.path.basename(f).lower() == 'best.pt']
            return best_match[0] if best_match else pt_files[0]
    return MODEL_PATH


@st.cache_resource(show_spinner=False)
def get_model():
    """
    Returns a cached, initialized instance of the YOLO model.
    Checks if ANY `.pt` model file exists in `models/`.
    - If any `.pt` file is found locally: Loads directly from disk (NO Hugging Face call).
    - If no `.pt` file is found: Automatically downloads from Hugging Face as a fallback.
    """
    # 1. Verify Ultralytics YOLO library is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[MODEL LOADER] [ERROR] Ultralytics YOLO module not found. Check requirements.txt.")
        return None

    # 2. LOCAL FIRST: Scan for ANY .pt file inside models/
    target_path = get_local_pt_path()

    if os.path.exists(target_path):
        try:
            print(f"[MODEL LOADER] [OK] Found local model file '{target_path}'. Loading directly from disk (no download needed)...")
            model = YOLO(target_path)
            return model
        except Exception as e:
            print(f"[MODEL LOADER] [ERROR] Error loading local model file '{target_path}': {e}")
            return None

    # 3. FALLBACK: Only fetch from Hugging Face if NO .pt file exists locally in models/
    print(f"[MODEL LOADER] No local .pt file found in '{MODEL_DIR}'. Downloading '{HF_FILENAME}' from Hugging Face repo '{HF_REPO_ID}'...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        from huggingface_hub import hf_hub_download
        import shutil
        
        cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        shutil.copy(cached_path, MODEL_PATH)
        print(f"[MODEL LOADER] [OK] Model weights downloaded and saved to: {MODEL_PATH}")
        
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        print(f"[MODEL LOADER] [ERROR] Failed to download model weights from Hugging Face Hub: {e}")
        return None
