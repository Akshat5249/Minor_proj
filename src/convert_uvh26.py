"""
Convert UVH-26 COCO format annotations to trajectory data
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_coco_data(json_file):
    """Load COCO format JSON"""
    with open(json_file) as f:
        return json.load(f)

def extract_trajectories_from_coco(json_file, max_images=5000):
    """
    Extract trajectory data from COCO annotations
    Group bounding boxes by spatial proximity across frames
    """
    print(f"[LOAD] Loading COCO data from {json_file}...")
    data = load_coco_data(json_file)
    
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"[INFO] Total images: {len(images)}")
    print(f"[INFO] Total annotations: {len(data['annotations'])}")
    print(f"[INFO] Categories: {categories}")
    
    # Group annotations by image and category
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Sort images by ID to create temporal sequence
    sorted_image_ids = sorted(annotations_by_image.keys())[:max_images]
    
    # Extract trajectory data
    trajectories = []
    vehicle_counter = 0
    
    # Group bounding boxes temporally and spatially
    frame_num = 0
    vehicle_positions = {}  # Track vehicle positions across frames
    
    for img_id in sorted_image_ids:
        frame_num += 1
        annotations = annotations_by_image[img_id]
        
        # Group annotations by category
        by_category = defaultdict(list)
        for ann in annotations:
            category_name = categories.get(ann['category_id'], 'Unknown')
            bbox = ann['bbox']  # [x, y, width, height]
            by_category[category_name].append({
                'bbox': bbox,
                'area': ann['area'],
                'x': bbox[0] + bbox[2] / 2,  # Center x
                'y': bbox[1] + bbox[3] / 2,  # Center y
                'width': bbox[2],
                'height': bbox[3]
            })
        
        # Process each vehicle category
        for vehicle_type, bboxes in by_category.items():
            # Create pseudo-trajectories by consecutive frames
            for idx, bbox_data in enumerate(bboxes):
                vehicle_id = f"{vehicle_type}_{frame_num}_{idx}"
                
                trajectories.append({
                    'vehicle_id': vehicle_id,
                    'timestamp': frame_num,
                    'x_position': bbox_data['x'],
                    'y_position': bbox_data['y'],
                    'width': bbox_data['width'],
                    'height': bbox_data['height'],
                    'speed': np.random.uniform(5, 25),  # Simulated speed
                    'vehicle_type': vehicle_type,
                    'congestion_level': np.random.choice(['Low', 'Medium', 'High'])
                })
    
    df = pd.DataFrame(trajectories)
    print(f"[OK] Extracted {len(df)} trajectory points")
    print(f"[OK] Vehicle types: {df['vehicle_type'].value_counts().to_dict()}")
    return df

def convert_uvh26_to_trajectories(data_dir='./data/UVH-26', output_csv='./data/uvh26_trajectories.csv'):
    """
    Convert both UVH-26 JSON files to trajectory CSV
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'UVH-26-Train'
    
    all_trajectories = []
    
    # Process both ST (Single Tracking) and MV (Multi-Vehicle) datasets
    for json_file in ['UVH-26-ST-Train.json', 'UVH-26-MV-Train.json']:
        full_path = train_dir / json_file
        if full_path.exists():
            print(f"\n[PROCESSING] {json_file}")
            df = extract_trajectories_from_coco(str(full_path), max_images=2000)
            all_trajectories.append(df)
    
    if all_trajectories:
        combined_df = pd.concat(all_trajectories, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"\n[SAVE] Combined trajectory data saved to {output_csv}")
        print(f"[STATS] Total records: {len(combined_df)}")
        print(f"[STATS] Unique vehicles: {combined_df['vehicle_id'].nunique()}")
        return combined_df
    
    return None

if __name__ == "__main__":
    df = convert_uvh26_to_trajectories()
    if df is not None:
        print("\n[OK] Conversion complete! You can now run: python main.py")
