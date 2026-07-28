import sys
import os
from PIL import Image

def run_checks():
    print("--- PCB Detect AI Code Verification ---")
    
    # 1. Check directories & paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_dir)
    print(f"Project directory added to system path: {project_dir}")
    
    # 2. Test imports
    try:
        from utils.database import init_db, get_history
        from utils.predict import predict_image
        from utils.charts import prepare_dataframe, create_pie_chart
        from pages.dashboard import show_dashboard
        from pages.image_detection import show_image_detection
        from pages.batch_detection import show_batch_detection
        from pages.live_camera import show_live_camera
        from pages.history import show_history
        from pages.analytics import show_analytics
        from pages.model_info import show_model_info
        from pages.settings import show_settings
        print("[OK] Success: All custom python modules imported successfully.")
    except Exception as e:
        print(f"[ERROR] Import failure: {e}")
        return False
        
    # 3. Test database setup and seeder
    try:
        init_db()
        history_list = get_history()
        print(f"[OK] Success: Database initialized and seeded. Found {len(history_list)} records.")
        if len(history_list) == 0:
            print("[ERROR] Seeding did not insert records.")
            return False
    except Exception as e:
        print(f"[ERROR] Database initialization failure: {e}")
        return False
        
    # 4. Test prediction pipeline
    try:
        test_img = Image.new('RGB', (640, 640), color='#0F5132')
        pred, conf, boxes, proc_time, anno = predict_image(test_img)
        print(f"[OK] Success: Prediction pipeline run completed.")
        print(f"  - Prediction Label: {pred}")
        print(f"  - Confidence: {conf}")
        print(f"  - Defects count: {len(boxes)}")
        print(f"  - Latency: {proc_time} ms")
        print(f"  - Output annotated frame dims: {anno.size}")
    except Exception as e:
        print(f"[ERROR] Prediction execution failure: {e}")
        return False
        
    print("\n[OK] Verification Successful! PCB Detect AI modules compile correctly.")
    return True

if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
