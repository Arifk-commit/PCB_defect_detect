import os
import streamlit as st

# Default Hugging Face repository and model filename settings
HF_REPO_ID = "AyushamM/Yolo26_pcbcheck"
HF_FILENAME = "PCBCheck_best.pt"

# Local model cache paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')

@st.cache_resource(show_spinner=False)
def get_model():
    """
    Returns a cached, initialized instance of the YOLO model.
    Checks if `models/best.pt` exists. If not, automatically downloads it from Hugging Face.
    Returns None if any failure occurs (initiating simulator fallback).
    """
    # 1. Verify Ultralytics YOLO library is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[MODEL LOADER] [ERROR] Ultralytics YOLO module not found.")
        return None

    # 2. Check if weights file exists locally
    if not os.path.exists(MODEL_PATH):
        print(f"[MODEL LOADER] Weights not found at {MODEL_PATH}. Initiating Hugging Face download...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            
            cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
            shutil.copy(cached_path, MODEL_PATH)
            print(f"[MODEL LOADER] [OK] Model weights saved to: {MODEL_PATH}")
            
        except Exception as e:
            print(f"[MODEL LOADER] [ERROR] Failed to download model weights from Hugging Face: {e}")
            return None

    # 3. Instantiate and return the YOLO model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"[MODEL LOADER] [OK] YOLO model loaded successfully from disk cache.")
            return model
    except Exception as e:
        print(f"[MODEL LOADER] [ERROR] Error instantiating YOLO model file: {e}")
            
    return None
