"""
Data loading and preprocessing module
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

def load_uvh26_dataset(data_dir='./data/UVH-26'):
    """
    Load UVH-26 trajectory data from converted CSV
    
    Parameters:
    -----------
    data_dir : str
        Directory where dataset is stored
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    # Check for converted trajectory CSV first
    csv_path = Path('./data/uvh26_trajectories.csv')
    if csv_path.exists():
        print(f"[LOAD] Loading trajectory CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[OK] Loaded {len(df)} trajectory records")
        print(f"[STATS] Unique vehicles: {df['vehicle_id'].nunique()}")
        print(f"[STATS] Vehicle types: {df['vehicle_type'].value_counts().to_dict()}")
        return df
    
    print(f"[INFO] Trajectory CSV not found. Run src/convert_uvh26.py first.")
    return None

def preprocess_trajectory_data(df, required_cols=None):
    """
    Preprocess trajectory data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw trajectory data
    required_cols : list
        List of required columns
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    print("\n[PREP] Preprocessing trajectory data...")
    
    # Handle missing values in numeric columns only
    initial_rows = len(df)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df = df.dropna(subset=numeric_cols)
    print(f"  Dropped {initial_rows - len(df)} rows with missing numeric values")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {initial_rows - len(df)} duplicate rows")
    
    # Convert object columns to numeric if needed
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        # Skip categorical columns (like vehicle_type, congestion_level)
        if col not in ['vehicle_type', 'congestion_level', 'type', 'level']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    print(f"  Final dataset: {len(df)} rows")
    
    return df

def download_dataset_from_huggingface(dataset_name='iisc-aim/UVH-26', local_dir='./data/UVH-26'):
    """
    Download dataset from Hugging Face
    
    Parameters:
    -----------
    dataset_name : str
        Dataset identifier on Hugging Face
    local_dir : str
        Local directory to save dataset
    """
    try:
        from huggingface_hub import snapshot_download
        print(f"📥 Downloading {dataset_name}...")
        print(f"   This may take a few minutes...")
        
        snapshot_download(
            repo_id=dataset_name,
            repo_type='dataset',
            local_dir=local_dir
        )
        print(f"✅ Dataset downloaded to {local_dir}")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print(f"   Make sure you have 'huggingface_hub' installed:")
        print(f"   pip install huggingface-hub")

def create_sample_dataset(n_samples=1000, n_vehicles=50):
    """
    Create a sample dataset for testing
    
    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_vehicles : int
        Number of unique vehicles
        
    Returns:
    --------
    pd.DataFrame
        Sample dataset
    """
    np.random.seed(42)
    
    vehicle_ids = np.random.choice(range(1, n_vehicles+1), n_samples)
    
    df = pd.DataFrame({
        'vehicle_id': vehicle_ids,
        'timestamp': np.tile(np.arange(n_samples // n_vehicles), n_vehicles)[:n_samples],
        'x_position': np.random.randn(n_samples).cumsum(),
        'y_position': np.random.randn(n_samples).cumsum() * 0.5,
        'speed': np.random.uniform(5, 25, n_samples),
        'vehicle_type': np.random.choice(['2W', '3W', 'Car', 'Bus'], n_samples),
        'congestion_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    })
    
    return df.sort_values(['vehicle_id', 'timestamp']).reset_index(drop=True)
