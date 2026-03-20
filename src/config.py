"""
Configuration file for Unsafe Lateral Movement Detection Model
"""

# DATASET PATHS
DATA_DIR = './data'
DATASET_NAME = 'iisc-aim/UVH-26'
RAW_DATA_PATH = './data/UVH-26'

# MODEL PARAMETERS
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
STRATIFY = True

# UNSAFE MOVEMENT THRESHOLDS
UNSAFE_LATERAL_VEL_QUANTILE = 0.85  # 85th percentile
UNSAFE_LATERAL_ACCEL_QUANTILE = 0.85
UNSAFE_TTC_THRESHOLD = 2.0  # seconds

# RANDOM FOREST PARAMETERS
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2

# GRADIENT BOOSTING PARAMETERS
GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 5
GB_SUBSAMPLE = 0.8

# MODEL SAVE PATHS
MODEL_DIR = './models'
RF_MODEL_PATH = './models/rf_unsafe_detector.pkl'
GB_MODEL_PATH = './models/gb_unsafe_detector.pkl'
SCALER_PATH = './models/scaler.pkl'

# RESULTS PATHS
RESULTS_DIR = './results'
ANALYSIS_PLOT_PATH = './results/model_analysis.png'
METRICS_CSV_PATH = './results/model_metrics.csv'

# VISUALIZATION PARAMETERS
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_PALETTE = 'husl'
PLOT_DPI = 300

# FEATURE COLUMNS
FEATURE_COLS = ['lateral_velocity', 'lateral_acceleration', 'lateral_clearance', 'ttc']
TARGET_COL = 'is_unsafe'

# VEHICLE TYPES (for analysis)
VEHICLE_TYPES = ['2W', '3W', 'Car', 'Bus']

# CONGESTION LEVELS
CONGESTION_LEVELS = ['Low', 'Medium', 'High']
