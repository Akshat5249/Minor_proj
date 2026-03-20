"""
Main training script for Unsafe Lateral Movement Detection Model
Using UVH-26 Dataset from Hugging Face

Project: Detection and Analysis of Unsafe Lateral Movements in 
         Heterogeneous Indian Urban Traffic
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.config import *
from src.data_loader import load_uvh26_dataset, preprocess_trajectory_data, create_sample_dataset, download_dataset_from_huggingface
from src.utils import (
    compute_lateral_indicators, 
    define_unsafe_labels,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_distribution,
    print_model_metrics,
    analyze_by_vehicle_type,
    analyze_by_congestion
)
from src.model_trainer import UnsafeMovementDetector

# Set visualization style
plt.style.use(PLOT_STYLE)
sns.set_palette(PLOT_PALETTE)

def main():
    print("\n" + "="*80)
    print("UNSAFE LATERAL MOVEMENT DETECTION IN HETEROGENEOUS INDIAN URBAN TRAFFIC")
    print("="*80)
    
    # ==================== STEP 1: LOAD DATASET ====================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    data_path = Path(RAW_DATA_PATH)
    
    # Try to load dataset from local directory first
    if (data_path / 'UVH-26-Train').exists():
        print(f"\n[INFO] Dataset found at {RAW_DATA_PATH}")
        data = load_uvh26_dataset(RAW_DATA_PATH)
    elif not data_path.exists() or len(list(data_path.glob('*.csv'))) == 0:
        print(f"\n[DOWNLOAD] Dataset not found at {RAW_DATA_PATH}")
        print("   [SKIP] Skipping download (dataset is ~900MB)")
        print("   [INFO] To download later, run:")
        print("      python -c \"from src.data_loader import download_dataset_from_huggingface; download_dataset_from_huggingface()\"")
        data = None
    else:
        # Try to load the dataset
        data = load_uvh26_dataset(RAW_DATA_PATH)
    
    if data is None:
        print("\n[WARNING] Could not load real dataset. Creating sample dataset for demonstration...")
        data = create_sample_dataset(n_samples=5000, n_vehicles=100)
        print("[OK] Sample dataset created for testing")
    
    # ==================== STEP 2: PREPROCESS DATA ====================
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80)
    
    data = preprocess_trajectory_data(data)
    print(f"[OK] Data shape: {data.shape}")
    print(f"[OK] Columns: {list(data.columns)}")
    
    # ==================== STEP 3: FEATURE ENGINEERING ====================
    print("\n" + "="*80)
    print("STEP 3: COMPUTING LATERAL MOVEMENT INDICATORS")
    print("="*80)
    
    # Identify position, time, and vehicle columns
    y_pos_col = [col for col in data.columns if 'y' in col.lower() or 'lat' in col.lower()][0] if any('y' in col.lower() or 'lat' in col.lower() for col in data.columns) else 'y_position'
    time_col = [col for col in data.columns if 'time' in col.lower() or 'timestamp' in col.lower()][0] if any('time' in col.lower() or 'timestamp' in col.lower() for col in data.columns) else 'timestamp'
    vehicle_col = [col for col in data.columns if 'vehicle' in col.lower() or 'id' in col.lower()][0] if any('vehicle' in col.lower() or 'id' in col.lower() for col in data.columns) else 'vehicle_id'
    
    print(f"  Using columns: Vehicle={vehicle_col}, Time={time_col}, Y-Position={y_pos_col}")
    
    # Make sure required columns exist
    if vehicle_col not in data.columns:
        data['vehicle_id'] = np.arange(len(data)) // 50  # Create dummy vehicle IDs
        vehicle_col = 'vehicle_id'
    
    if time_col not in data.columns:
        data['timestamp'] = np.arange(len(data))
        time_col = 'timestamp'
    
    if y_pos_col not in data.columns:
        data['y_position'] = np.random.randn(len(data)).cumsum() * 0.5
        y_pos_col = 'y_position'
    
    # Compute indicators
    data = compute_lateral_indicators(data, vehicle_col, time_col, y_pos_col)
    print("[OK] Lateral velocity computed")
    print("[OK] Lateral acceleration computed")
    print("[OK] Lateral clearance computed")
    print("[OK] Time to Collision (TTC) computed")
    
    # ==================== STEP 4: DEFINE UNSAFE LABELS ====================
    print("\n" + "="*80)
    print("STEP 4: DEFINING UNSAFE LATERAL MOVEMENTS")
    print("="*80)
    
    data, unsafe_vel_th, unsafe_accel_th = define_unsafe_labels(
        data,
        lateral_vel_quantile=UNSAFE_LATERAL_VEL_QUANTILE,
        lateral_accel_quantile=UNSAFE_LATERAL_ACCEL_QUANTILE,
        ttc_threshold=UNSAFE_TTC_THRESHOLD
    )
    
    unsafe_count = data['is_unsafe'].sum()
    unsafe_pct = data['is_unsafe'].mean() * 100
    
    print(f"\n[WARNING] Unsafe lateral movements found: {unsafe_count} ({unsafe_pct:.2f}%)")
    print(f"  Lateral velocity threshold: {unsafe_vel_th:.4f} m/s")
    print(f"  Lateral acceleration threshold: {unsafe_accel_th:.4f} m/s²")
    print(f"  TTC threshold: {UNSAFE_TTC_THRESHOLD} seconds")
    
    # ==================== STEP 5: PREPARE FEATURES ====================
    print("\n" + "="*80)
    print("STEP 5: PREPARING FEATURES")
    print("="*80)
    
    # Use available feature columns
    data_clean = data.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"[OK] Rows after removing NaN: {len(data_clean)}")
    
    X = data_clean[FEATURE_COLS].values
    y = data_clean[TARGET_COL].values
    
    # Handle any remaining NaN and inf values
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    
    print(f"[OK] Feature matrix shape: {X.shape}")
    print(f"[OK] Target distribution: {np.bincount(y)}")
    print(f"   Safe: {(y==0).sum()}, Unsafe: {(y==1).sum()}")
    
    # ==================== STEP 6: TRAIN MODELS ====================
    print("\n" + "="*80)
    print("STEP 6: TRAINING CLASSIFICATION MODELS")
    print("="*80)
    
    # Initialize trainer
    trainer = UnsafeMovementDetector(config=type('Config', (), {
        'RF_N_ESTIMATORS': RF_N_ESTIMATORS,
        'RF_MAX_DEPTH': RF_MAX_DEPTH,
        'RF_MIN_SAMPLES_SPLIT': RF_MIN_SAMPLES_SPLIT,
        'RF_MIN_SAMPLES_LEAF': RF_MIN_SAMPLES_LEAF,
        'GB_N_ESTIMATORS': GB_N_ESTIMATORS,
        'GB_LEARNING_RATE': GB_LEARNING_RATE,
        'GB_MAX_DEPTH': GB_MAX_DEPTH,
        'GB_SUBSAMPLE': GB_SUBSAMPLE,
        'RANDOM_STATE': RANDOM_STATE,
        'MODEL_DIR': MODEL_DIR,
        'RF_MODEL_PATH': RF_MODEL_PATH,
        'GB_MODEL_PATH': GB_MODEL_PATH,
        'SCALER_PATH': SCALER_PATH
    })())
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Train models
    trainer.train_random_forest(X_train, y_train)
    trainer.train_gradient_boosting(X_train, y_train)
    
    # ==================== STEP 7: EVALUATE MODELS ====================
    print("\n" + "="*80)
    print("STEP 7: EVALUATING MODELS")
    print("="*80)
    
    rf_metrics, rf_pred, rf_proba = trainer.evaluate_model(
        trainer.rf_model, X_test, y_test, "Random Forest"
    )
    
    gb_metrics, gb_pred, gb_proba = trainer.evaluate_model(
        trainer.gb_model, X_test, y_test, "Gradient Boosting"
    )
    
    # ==================== STEP 8: FEATURE IMPORTANCE ====================
    print("\n" + "="*80)
    print("STEP 8: FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importance_df = trainer.get_feature_importance(FEATURE_COLS)
    print("\n[INFO] Feature Importance (Gradient Boosting):")
    for idx, row in importance_df.iterrows():
        print(f"  {row['Feature']}: {row['GB_Importance']:.4f}")
    
    # ==================== STEP 9: VISUALIZATIONS ====================
    print("\n" + "="*80)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("="*80)
    
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Unsafe Lateral Movement Detection - Model Analysis', fontsize=16, fontweight='bold')
    
    # Confusion matrices
    plot_confusion_matrix(y_test, rf_pred, 'Random Forest', axes[0, 0])
    plot_confusion_matrix(y_test, gb_pred, 'Gradient Boosting', axes[0, 1])
    
    # Feature importance
    plot_feature_importance(FEATURE_COLS, trainer.rf_model.feature_importances_, axes[0, 2])
    
    # Distribution plots
    try:
        plot_distribution(data_clean, 'lateral_velocity', 'is_unsafe', axes[1, 0], 
                         'Lateral Velocity Distribution', 'Velocity (m/s)')
        plot_distribution(data_clean, 'lateral_acceleration', 'is_unsafe', axes[1, 1], 
                         'Lateral Acceleration Distribution', 'Acceleration (m/s²)')
        plot_distribution(data_clean, 'ttc', 'is_unsafe', axes[1, 2], 
                         'TTC Distribution', 'Time (s)')
        axes[1, 2].set_xlim(0, 10)
    except Exception as e:
        print(f"  [WARNING] Could not create distribution plots: {e}")
        axes[1, 0].plot([0, 1], [0, 1], 'o-', label='TTC vs Risk')
        axes[1, 0].set_title('(Distribution plot - data unavailable)')
    
    plt.tight_layout()
    plt.savefig(ANALYSIS_PLOT_PATH, dpi=PLOT_DPI, bbox_inches='tight')
    print(f"[OK] Analysis visualization saved to {ANALYSIS_PLOT_PATH}")
    
    # ==================== STEP 10: SAVE MODELS ====================
    print("\n" + "="*80)
    print("STEP 10: SAVING MODELS")
    print("="*80)
    
    trainer.save_models()
    
    # ==================== STEP 11: SAVE METRICS ====================
    print("\n" + "="*80)
    print("STEP 11: SAVING METRICS")
    print("="*80)
    
    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Random Forest': [
            rf_metrics['accuracy'],
            rf_metrics['precision'],
            rf_metrics['recall'],
            rf_metrics['f1'],
            rf_metrics['roc_auc']
        ],
        'Gradient Boosting': [
            gb_metrics['accuracy'],
            gb_metrics['precision'],
            gb_metrics['recall'],
            gb_metrics['f1'],
            gb_metrics['roc_auc']
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"[OK] Metrics saved to {METRICS_CSV_PATH}")
    print("\n" + metrics_df.to_string(index=False))
    
    # ==================== STEP 12: VEHICLE-TYPE ANALYSIS ====================
    print("\n" + "="*80)
    print("STEP 12: VEHICLE-TYPE WISE SAFETY ANALYSIS")
    print("="*80)
    
    if 'vehicle_type' in data.columns:
        vehicle_analysis = analyze_by_vehicle_type(data, 'vehicle_type')
        vehicle_analysis.to_csv('./results/vehicle_type_analysis.csv')
    else:
        print("  [WARNING] 'vehicle_type' column not found in dataset")
    
    # ==================== STEP 13: CONGESTION ANALYSIS ====================
    print("\n" + "="*80)
    print("STEP 13: CONGESTION-LEVEL WISE SAFETY ANALYSIS")
    print("="*80)
    
    if 'congestion_level' in data.columns:
        congestion_analysis = analyze_by_vehicle_type(data, 'congestion_level')
        congestion_analysis.to_csv('./results/congestion_analysis.csv')
    else:
        print("  [WARNING] 'congestion_level' column not found in dataset")
    
    # ==================== COMPLETION ====================
    print("\n" + "="*80)
    print("[COMPLETE] MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"\n[RESULTS] Results saved in: {RESULTS_DIR}/")
    print(f"[MODELS] Models saved in: {MODEL_DIR}/")
    print(f"\n[FILES] Generated files:")
    print(f"   - {ANALYSIS_PLOT_PATH}")
    print(f"   - {METRICS_CSV_PATH}")
    print(f"   - ./results/vehicle_type_analysis.csv")
    print(f"   - ./results/congestion_analysis.csv")
    print(f"\n[NEXT] Next steps:")
    print(f"   1. Review visualizations in {RESULTS_DIR}/")
    print(f"   2. Check metrics in {METRICS_CSV_PATH}")
    print(f"   3. Deploy using saved models in {MODEL_DIR}/")
    print("="*80 + "\n")
    
    return trainer, data_clean, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    trainer, data, X_train, X_test, y_train, y_test = main()
