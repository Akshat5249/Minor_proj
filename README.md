# 🚗 Unsafe Lateral Movement Detection in Heterogeneous Indian Urban Traffic

**Detection and Analysis of Unsafe Lateral Movements Using UVH-26 Dataset**

## 👋 New to Machine Learning?

If you are completely new to ML and want a plain-English explanation of what this project does, start here: **[BEGINNER_GUIDE.md](./BEGINNER_GUIDE.md)**

---

## 📋 Project Overview

This project develops a **data-driven framework** for detecting and analyzing unsafe lateral movements in heterogeneous Indian urban traffic using trajectory data from the **UVH-26 dataset**.

### 🎯 Key Objectives

1. Extract and preprocess vehicle trajectory data from UVH-26 dataset
2. Compute lateral movement indicators:
   - Lateral velocity (v_y)
   - Lateral acceleration (a_y)
   - Lateral clearance
   - Time to Collision (TTC)
3. Define quantitative thresholds for unsafe lateral maneuvers
4. Develop ML models to classify Safe vs Unsafe lateral movements
5. Perform vehicle-type-wise and congestion-level-wise safety analysis
6. Generate actionable insights for ITS deployment

---

## 📁 Project Structure

```
ML MODEL/
├── data/                      # Dataset directory
│   └── UVH-26/               # Downloaded dataset
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code modules
│   ├── config.py            # Configuration parameters
│   ├── data_loader.py       # Dataset loading utilities
│   ├── utils.py             # Helper functions
│   └── model_trainer.py     # Model training class
├── models/                  # Saved trained models
│   ├── rf_unsafe_detector.pkl
│   ├── gb_unsafe_detector.pkl
│   └── scaler.pkl
├── results/                 # Analysis outputs
│   ├── model_analysis.png
│   ├── model_metrics.csv
│   ├── vehicle_type_analysis.csv
│   └── congestion_analysis.csv
├── main.py                  # Main training script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

These dependency versions are selected to work cleanly with modern Python, including Python 3.12.

### Step 2: Download Dataset

The script will automatically download the UVH-26 dataset from Hugging Face on first run:

```bash
python main.py
```

Or manually download:

```bash
python -c "from src.data_loader import download_dataset_from_huggingface; download_dataset_from_huggingface()"
```

### Step 3: Run Training

```bash
python main.py
```

This will:
- Load and preprocess the UVH-26 dataset
- Compute lateral movement indicators
- Define unsafe lateral movement labels
- Train Random Forest and Gradient Boosting classifiers
- Evaluate model performance
- Generate visualizations and reports
- Save trained models

---

## 📊 Model Performance

Expected metrics:
- **Accuracy**: 85-92%
- **Precision**: 0.80-0.90
- **Recall**: 0.75-0.88
- **F1-Score**: 0.78-0.89
- **ROC-AUC**: 0.88-0.95

---

## 🔍 Key Features

### 1. **Lateral Movement Indicators**
- Lateral velocity: Rate of lateral position change
- Lateral acceleration: Rate of lateral velocity change
- Lateral clearance: Safety distance indicator
- TTC (Time to Collision): Safety metric

### 2. **Machine Learning Models**
- **Random Forest Classifier**: Fast inference, good interpretability
- **Gradient Boosting Classifier**: Higher accuracy, complex patterns

### 3. **Safety Analysis**
- Vehicle-type-wise unsafe rate (2W, 3W, Car, Bus)
- Congestion-level-wise analysis (Low, Medium, High)
- Risk hotspot identification

---

## 📈 Output Files

After running `main.py`, you'll get:

### 1. **model_analysis.png**
- Confusion matrices for both models
- Feature importance chart
- Lateral velocity/acceleration/TTC distributions

### 2. **model_metrics.csv**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC for both models

### 3. **vehicle_type_analysis.csv**
- Unsafe maneuver rate by vehicle type
- Risk index by vehicle category

### 4. **congestion_analysis.csv**
- Safety metrics by traffic congestion level
- Peak-hour vs non-peak analysis

### 5. **Saved Models**
- `rf_unsafe_detector.pkl`: Random Forest model
- `gb_unsafe_detector.pkl`: Gradient Boosting model
- `scaler.pkl`: Feature scaler for inference

---

## 💻 Usage Examples

### Training with Custom Parameters

Edit `src/config.py` to adjust:
```python
UNSAFE_LATERAL_VEL_QUANTILE = 0.85      # Threshold sensitivity
UNSAFE_LATERAL_ACCEL_QUANTILE = 0.85
UNSAFE_TTC_THRESHOLD = 2.0              # Seconds

RF_N_ESTIMATORS = 100                   # Model parameters
GB_LEARNING_RATE = 0.1
```

### Making Predictions on New Data

```python
import joblib
from src.model_trainer import UnsafeMovementDetector

# Load models
rf_model = joblib.load('./models/rf_unsafe_detector.pkl')
scaler = joblib.load('./models/scaler.pkl')

# Prepare new data (4 features)
X_new = [[0.5, 0.2, 1.5, 3.0]]  # [lat_vel, lat_accel, lat_clearance, ttc]

# Scale and predict
X_scaled = scaler.transform(X_new)
prediction = rf_model.predict(X_scaled)
probability = rf_model.predict_proba(X_scaled)

print(f"Prediction: {'UNSAFE' if prediction[0] else 'SAFE'}")
print(f"Confidence: {probability[0][prediction[0]]:.2%}")
```

---

## 🎓 Stakeholder Benefits

| Stakeholder | Use Case |
|-------------|----------|
| ITS Admin | Real-time risk alert generation |
| Traffic Police | Targeted enforcement planning |
| Urban Planners | Infrastructure redesign decisions |
| Smart City Authority | AI-based surveillance deployment |
| Researchers | Mixed traffic modeling validation |

---

## 📚 Dataset Information

**UVH-26 Dataset**
- Source: [Hugging Face](https://huggingface.co/datasets/iisc-aim/UVH-26)
- Trajectories from Indian urban roads
- Mixed vehicle types (2W, 3W, 4W)
- Various traffic conditions

---

## 🔧 Troubleshooting

### `pip install -r requirements.txt` fails on `pandas`
Older dependency pins can trigger source builds on Python 3.12 and fail during wheel preparation. This repository now uses Python 3.12-compatible package versions in `requirements.txt`.

### Dataset not downloading?
```bash
pip install --upgrade huggingface-hub
```

### Memory error?
Reduce sample size in `main.py`:
```python
data = data.sample(frac=0.5)  # Use 50% of data
```

### Models not training?
Check `src/config.py` for reasonable parameter ranges.

---

## 📞 Support

For issues or questions:
1. Check dataset CSV column names
2. Verify feature columns in `src/config.py`
3. Review error messages in console output

---

## 📝 Citation

If using this project, cite:

```
Detection and Analysis of Unsafe Lateral Movements 
in Heterogeneous Indian Urban Traffic using UVH-26 Dataset
```

---

## ✅ Checklist

- [x] Dataset download capability
- [x] Feature engineering (lateral indicators)
- [x] Unsafe label definition
- [x] Random Forest classifier
- [x] Gradient Boosting classifier
- [x] Model evaluation and metrics
- [x] Feature importance analysis
- [x] Visualizations
- [x] Vehicle-type analysis
- [x] Congestion analysis
- [x] Model serialization (save/load)

---

**Last Updated**: March 2026  
**Status**: ✅ Production Ready
