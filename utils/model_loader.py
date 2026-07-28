import os
import streamlit as st

# Default Hugging Face repository and model filename settings
HF_REPO_ID = "Arifk-commit/PCB_defect_detect"
HF_FILENAME = "best.pt"

# Local model cache paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')

@st.cache_resource
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
        print("[MODEL LOADER] [ERROR] Ultralytics YOLO module not found. Check requirements.txt.")
        return None

    # 2. Check if weights file exists locally
    if not os.path.exists(MODEL_PATH):
        print(f"[MODEL LOADER] Weights not found at {MODEL_PATH}. Initiating Hugging Face download...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            
            # Show download visual overlay in Streamlit if running under Streamlit UI runtime
            if st.runtime.exists():
                st.info(f"🔄 Downloading YOLOv11 model weights from Hugging Face repository `{HF_REPO_ID}`...")
                
            # Download file from HF Hub
            cached_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
            shutil.copy(cached_path, MODEL_PATH)
            
            if st.runtime.exists():
                st.success("✓ Model weights downloaded and cached successfully!")
            print(f"[MODEL LOADER] [OK] Model weights saved to: {MODEL_PATH}")
            
        except Exception as e:
            # Handle download/network failures gracefully
            err_msg = f"Failed to download model weights from Hugging Face Hub: {e}"
            print(f"[MODEL LOADER] [ERROR] {err_msg}")
            
            if st.runtime.exists():
                st.warning(f"⚠️ {err_msg}. Falling back to simulation mode.")
            return None

    # 3. Instantiate and return the YOLO model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"[MODEL LOADER] [OK] YOLO model loaded successfully from disk cache.")
            return model
    except Exception as e:
        err_msg = f"Error instantiating YOLO model file: {e}"
        print(f"[MODEL LOADER] [ERROR] {err_msg}")
        if st.runtime.exists():
            st.error(f"❌ {err_msg}")
            
    return None
