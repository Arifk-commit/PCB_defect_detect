import sqlite3
import os
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'history', 'detections.db')
HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'history')

# Ensure the history directory exists
os.makedirs(HISTORY_DIR, exist_ok=True)

def init_db():
    """Initializes the database schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            defect_count INTEGER NOT NULL,
            defects_list TEXT,
            confidence REAL NOT NULL,
            inference_time REAL NOT NULL,
            timestamp TEXT NOT NULL,
            original_image_path TEXT,
            annotated_image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_placeholder_images(filename_base, prediction, defects):
    """Creates mock original and annotated images to serve as historical records."""
    img_size = (300, 300)
    
    # Original Image (Standard green PCB texture)
    orig_img = Image.new('RGB', img_size, color='#0F5132') # Deep green PCB
    draw_orig = ImageDraw.Draw(orig_img)
    
    # Draw standard PCB gold traces and pads
    for i in range(5):
        draw_orig.line([(20, 50 * i + 30), (280, 50 * i + 30)], fill='#D4AF37', width=3) # Traces
        draw_orig.rectangle([(50 * i + 40, 50 * i + 20), (50 * i + 65, 50 * i + 45)], fill='#D4AF37') # Pads
        draw_orig.ellipse([(50 * i + 48, 50 * i + 28), (50 * i + 57, 50 * i + 37)], fill='#1E1E1E') # Through-holes
        
    # Annotated Image
    anno_img = orig_img.copy()
    draw_anno = ImageDraw.Draw(anno_img)
    
    defect_colors = {
        'Missing Hole': '#EF4444',
        'Mouse Bite': '#EF4444',
        'Open Circuit': '#EF4444',
        'Short': '#EF4444',
        'Spur': '#EF4444',
        'Spurious Copper': '#EF4444'
    }
    
    # If defective, draw bounding boxes around the "defects"
    if prediction == 'Defective' and defects:
        for index, defect in enumerate(defects):
            # Pick a semi-random spot depending on the index
            x = 40 + (index * 60) % 200
            y = 60 + (index * 70) % 200
            w, h = 40, 40
            
            # Draw defect bounding box
            color = defect_colors.get(defect, '#EF4444')
            draw_anno.rectangle([(x, y), (x+w, y+h)], outline=color, width=3)
            # Label
            draw_anno.rectangle([(x, y-15), (x+80, y)], fill=color)
            draw_anno.text((x+2, y-14), defect, fill='#FFFFFF')
            
    # Save files
    orig_filename = f"orig_{filename_base}"
    anno_filename = f"anno_{filename_base}"
    
    orig_path = os.path.join(HISTORY_DIR, orig_filename)
    anno_path = os.path.join(HISTORY_DIR, anno_filename)
    
    orig_img.save(orig_path)
    anno_img.save(anno_path)
    
    return orig_path, anno_path

def seed_mock_data(conn):
    """Seeds the database with 60 realistic PCB detection entries spread over the last 30 days."""
    cursor = conn.cursor()
    defect_types = ['Missing Hole', 'Mouse Bite', 'Open Circuit', 'Short', 'Spur', 'Spurious Copper']
    
    now = datetime.now()
    
    for i in range(60):
        # Determine prediction: ~35% Defective, ~65% Healthy
        is_defective = random.random() < 0.35
        prediction = 'Defective' if is_defective else 'Healthy'
        
        num_defects = 0
        defects_found = []
        if is_defective:
            num_defects = random.randint(1, 3)
            defects_found = random.sample(defect_types, num_defects)
            
        defects_str = ','.join(defects_found) if defects_found else ''
        confidence = round(random.uniform(0.75, 0.99) if not is_defective else random.uniform(0.65, 0.94), 2)
        inference_time = round(random.uniform(15.0, 45.0), 1)
        
        # Distribute over the last 30 days
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        mins_ago = random.randint(0, 59)
        timestamp = (now - timedelta(days=days_ago, hours=hours_ago, minutes=mins_ago)).strftime('%Y-%m-%d %H:%M:%S')
        
        filename = f"pcb_inspect_{i:03d}.png"
        
        # Create mock images
        orig_path, anno_path = create_placeholder_images(filename, prediction, defects_found)
        
        cursor.execute('''
            INSERT INTO history (
                filename, prediction, defect_count, defects_list, confidence, 
                inference_time, timestamp, original_image_path, annotated_image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filename, prediction, num_defects, defects_str, confidence, 
              inference_time, timestamp, orig_path, anno_path))
        
    conn.commit()

def add_detection(filename, prediction, defect_count, defects_list, confidence, inference_time, original_image, annotated_image):
    """Saves a detection result along with physical images and database entry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the original and annotated images
    orig_filename = f"orig_{time_str}_{filename}"
    anno_filename = f"anno_{time_str}_{filename}"
    
    orig_path = os.path.join(HISTORY_DIR, orig_filename)
    anno_path = os.path.join(HISTORY_DIR, anno_filename)
    
    original_image.save(orig_path)
    annotated_image.save(anno_path)
    
    defects_str = ','.join(defects_list) if defects_list else ''
    
    cursor.execute('''
        INSERT INTO history (
            filename, prediction, defect_count, defects_list, confidence, 
            inference_time, timestamp, original_image_path, annotated_image_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (filename, prediction, defect_count, defects_str, confidence, 
          inference_time, timestamp, orig_path, anno_path))
    
    conn.commit()
    conn.close()

def get_history(search_query="", date_filter=None, confidence_min=0.0, defect_type_filter="All", prediction_filter="All"):
    """Fetches records matching query and criteria filters."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    query = "SELECT * FROM history WHERE 1=1"
    params = []
    
    if search_query:
        query += " AND filename LIKE ?"
        params.append(f"%{search_query}%")
        
    if date_filter:
        query += " AND DATE(timestamp) = DATE(?)"
        params.append(date_filter.strftime('%Y-%m-%d'))
        
    if confidence_min > 0.0:
        query += " AND confidence >= ?"
        params.append(confidence_min)
        
    if prediction_filter != "All":
        query += " AND prediction = ?"
        params.append(prediction_filter)
        
    if defect_type_filter != "All":
        query += " AND defects_list LIKE ?"
        params.append(f"%{defect_type_filter}%")
        
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in rows]
    
    conn.close()
    return results

def delete_record(record_id):
    """Deletes a record from the database and its associated image files."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT original_image_path, annotated_image_path FROM history WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    
    if row:
        orig_path, anno_path = row
        try:
            if orig_path and os.path.exists(orig_path):
                os.remove(orig_path)
            if anno_path and os.path.exists(anno_path):
                os.remove(anno_path)
        except Exception as e:
            print(f"Error deleting image files: {e}")
            
    cursor.execute("DELETE FROM history WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()

def clear_all_history():
    """Deletes all history records and all image files inside the history directory."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT original_image_path, annotated_image_path FROM history")
    rows = cursor.fetchall()
    
    for row in rows:
        orig_path, anno_path = row
        try:
            if orig_path and os.path.exists(orig_path):
                os.remove(orig_path)
            if anno_path and os.path.exists(anno_path):
                os.remove(anno_path)
        except Exception as e:
            print(f"Error deleting image file: {e}")
            
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()
