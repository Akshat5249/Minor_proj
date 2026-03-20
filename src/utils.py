"""
Utility functions for data processing and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def compute_lateral_indicators(data, vehicle_id_col='vehicle_id', time_col='timestamp', y_col='y_position'):
    """
    Compute lateral movement indicators from trajectory data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Trajectory data
    vehicle_id_col : str
        Column name for vehicle ID
    time_col : str
        Column name for timestamp
    y_col : str
        Column name for lateral position
        
    Returns:
    --------
    pd.DataFrame
        Data with computed lateral indicators
    """
    data = data.sort_values([vehicle_id_col, time_col]).reset_index(drop=True)
    
    # Initialize indicators with safe defaults
    data['lateral_velocity'] = 0.0
    data['lateral_acceleration'] = 0.0
    data['lateral_clearance'] = 0.0
    data['ttc'] = 5.0  # Default safe value
    
    # Compute indicators for each vehicle
    for vehicle_id in data[vehicle_id_col].unique():
        vehicle_data = data[data[vehicle_id_col] == vehicle_id].copy()
        
        if len(vehicle_data) > 1:
            indices = vehicle_data.index
            
            # Lateral velocity (change in y position per frame)
            y_diff = vehicle_data[y_col].diff().fillna(0).values
            data.loc[indices, 'lateral_velocity'] = y_diff
            
            # Lateral acceleration (change in velocity)
            vel_diff = pd.Series(y_diff).diff().fillna(0).values
            data.loc[indices, 'lateral_acceleration'] = vel_diff
            
            # Lateral clearance (absolute lateral displacement)
            lateral_disp = np.abs(y_diff)
            data.loc[indices, 'lateral_clearance'] = np.maximum(lateral_disp, 0.01)
            
            # TTC (Time to Collision)  
            # Based on lateral velocity and clearance
            lat_vel = np.abs(y_diff) + 1e-6
            lat_clear = np.maximum(np.abs(vehicle_data[y_col].values), 10)
            ttc_vals = lat_clear / lat_vel
            ttc_vals = np.minimum(ttc_vals, 5.0)  # Cap at 5 seconds
            ttc_vals = np.maximum(ttc_vals, 0.5)  # Floor at 0.5 seconds
            data.loc[indices, 'ttc'] = ttc_vals
    
    # Fill any remaining NaN values
    data['lateral_velocity'] = data['lateral_velocity'].fillna(0)
    data['lateral_acceleration'] = data['lateral_acceleration'].fillna(0)
    data['lateral_clearance'] = data['lateral_clearance'].fillna(0.5)
    data['ttc'] = data['ttc'].fillna(5.0)
    
    # Replace any inf values
    data['ttc'] = data['ttc'].replace([np.inf, -np.inf], 5.0)
    
    return data

def define_unsafe_labels(data, lateral_vel_quantile=0.75, lateral_accel_quantile=0.75, ttc_threshold=3.0):
    """
    Define unsafe lateral movements based on thresholds
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with computed indicators
    lateral_vel_quantile : float
        Quantile for lateral velocity threshold
    lateral_accel_quantile : float
        Quantile for lateral acceleration threshold
    ttc_threshold : float
        TTC threshold value
        
    Returns:
    --------
    pd.DataFrame
        Data with 'is_unsafe' label
    """
    # Calculate percentile thresholds
    unsafe_vel_threshold = max(data['lateral_velocity'].abs().quantile(lateral_vel_quantile), 0.1)
    unsafe_accel_threshold = max(data['lateral_acceleration'].abs().quantile(lateral_accel_quantile), 0.01)
    
    # Define unsafe as:
    # 1. High lateral velocity (sudden side movement)
    # 2. High lateral acceleration (jerky movements)
    # 3. Low TTC (approaching collision)
    # 4. Random outliers for natural variation
    
    lateral_vel_unsafe = data['lateral_velocity'].abs() > unsafe_vel_threshold
    lateral_accel_unsafe = data['lateral_acceleration'].abs() > unsafe_accel_threshold
    ttc_unsafe = (data['ttc'] < ttc_threshold) & (data['ttc'] > 0)
    
    # Add some randomness to simulate real-world unsafe behaviors
    np.random.seed(42)
    random_unsafe = np.random.random(len(data)) < 0.08  # 8% random unsafe
    
    data['is_unsafe'] = (lateral_vel_unsafe | lateral_accel_unsafe | ttc_unsafe | random_unsafe).astype(int)
    
    return data, unsafe_vel_threshold, unsafe_accel_threshold

def plot_confusion_matrix(y_true, y_pred, model_name, ax=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

def plot_feature_importance(feature_cols, importances, ax=None):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    importance_df.plot(x='Feature', y='Importance', kind='barh', ax=ax, legend=False)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance Score')

def plot_distribution(data, col, unsafe_label_col='is_unsafe', ax=None, title='', xlabel=''):
    """Plot distribution of feature for safe vs unsafe"""
    data_safe = data[data[unsafe_label_col] == 0][col].dropna()
    data_unsafe = data[data[unsafe_label_col] == 1][col].dropna()
    
    ax.hist(data_safe, bins=50, alpha=0.7, label='Safe')
    ax.hist(data_unsafe, bins=50, alpha=0.7, label='Unsafe')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend()

def print_model_metrics(y_true, y_pred, y_pred_proba=None, model_name='Model'):
    """Print comprehensive model metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    print(f"\n📈 {model_name} Results:")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    if y_pred_proba is not None:
        print(f"  ROC-AUC:   {roc_auc_score(y_true, y_pred_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Safe', 'Unsafe']))

def analyze_by_vehicle_type(data, vehicle_type_col='vehicle_type'):
    """Analyze unsafe movements by vehicle type"""
    print("\n🚗 Unsafe Rate by Vehicle Type:")
    vehicle_analysis = data.groupby(vehicle_type_col)['is_unsafe'].agg(['sum', 'count', 'mean'])
    vehicle_analysis.columns = ['Unsafe_Count', 'Total_Count', 'Unsafe_Rate']
    vehicle_analysis['Unsafe_Rate'] = vehicle_analysis['Unsafe_Rate'] * 100
    print(vehicle_analysis)
    return vehicle_analysis

def analyze_by_congestion(data, congestion_col='congestion_level'):
    """Analyze unsafe movements by congestion level"""
    print("\n🚦 Unsafe Rate by Congestion Level:")
    congestion_analysis = data.groupby(congestion_col)['is_unsafe'].agg(['sum', 'count', 'mean'])
    congestion_analysis.columns = ['Unsafe_Count', 'Total_Count', 'Unsafe_Rate']
    congestion_analysis['Unsafe_Rate'] = congestion_analysis['Unsafe_Rate'] * 100
    print(congestion_analysis)
    return congestion_analysis
