"""
Model training module for unsafe lateral movement detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class UnsafeMovementDetector:
    """Train and evaluate models for unsafe lateral movement detection"""
    
    def __init__(self, config):
        self.config = config
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.metrics = {}
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """
        Prepare and split data
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target
        test_size : float
            Test split ratio
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        X_train, X_test, y_train, y_test : arrays
            Split and scaled data
        """
        print("\n📊 Preparing data...")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        
        print(f"[PREP] Training set: {X_train.shape}")
        print(f"[PREP] Test set: {X_test.shape}")
        print(f"[PREP] Unsafe samples in train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
        print(f"[PREP] Unsafe samples in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """Train Random Forest model"""
        print("\n🤖 Training Random Forest Classifier...")
        
        params = {
            'n_estimators': self.config.RF_N_ESTIMATORS,
            'max_depth': self.config.RF_MAX_DEPTH,
            'min_samples_split': self.config.RF_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': self.config.RF_MIN_SAMPLES_LEAF,
            'random_state': self.config.RANDOM_STATE,
            'n_jobs': -1
        }
        params.update(kwargs)
        
        self.rf_model = RandomForestClassifier(**params)
        self.rf_model.fit(X_train, y_train)
        print("[TRAIN] Random Forest training complete!")
        
        return self.rf_model
    
    def train_gradient_boosting(self, X_train, y_train, **kwargs):
        """Train Gradient Boosting model"""
        print("\n🤖 Training Gradient Boosting Classifier...")
        
        params = {
            'n_estimators': self.config.GB_N_ESTIMATORS,
            'learning_rate': self.config.GB_LEARNING_RATE,
            'max_depth': self.config.GB_MAX_DEPTH,
            'subsample': self.config.GB_SUBSAMPLE,
            'random_state': self.config.RANDOM_STATE
        }
        params.update(kwargs)
        
        self.gb_model = GradientBoostingClassifier(**params)
        self.gb_model.fit(X_train, y_train)
        print("[TRAIN] Gradient Boosting training complete!")
        
        return self.gb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Evaluate model and return metrics
        
        Parameters:
        -----------
        model : sklearn model
            Trained model
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        model_name : str
            Model name for display
            
        Returns:
        --------
        dict : Evaluation metrics
        """
        print(f"\n[EVAL] Evaluating {model_name}...")
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"  [OK] Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  [OK] Precision: {metrics['precision']:.4f}")
        print(f"  [OK] Recall:    {metrics['recall']:.4f}")
        print(f"  [OK] F1-Score:  {metrics['f1']:.4f}")
        print(f"  [OK] ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        self.metrics[model_name] = metrics
        
        return metrics, y_pred, y_pred_proba
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from models"""
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'RF_Importance': self.rf_model.feature_importances_,
            'GB_Importance': self.gb_model.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        print("\n[INFO] Feature Importance (Random Forest):")
        for idx, row in importance_df.iterrows():
            print(f"  {row['Feature']}: {row['RF_Importance']:.4f}")
        
        return importance_df
    
    def save_models(self):
        """Save trained models"""
        Path(self.config.MODEL_DIR).mkdir(exist_ok=True)
        
        joblib.dump(self.rf_model, self.config.RF_MODEL_PATH)
        joblib.dump(self.gb_model, self.config.GB_MODEL_PATH)
        joblib.dump(self.scaler, self.config.SCALER_PATH)
        
        print(f"\n[SAVE] Models saved to {self.config.MODEL_DIR}/")
    
    def load_models(self):
        """Load trained models"""
        self.rf_model = joblib.load(self.config.RF_MODEL_PATH)
        self.gb_model = joblib.load(self.config.GB_MODEL_PATH)
        self.scaler = joblib.load(self.config.SCALER_PATH)
        
        print(f"\n[LOAD] Models loaded from {self.config.MODEL_DIR}/")
    
    def predict(self, X, use_ensemble=True):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like
            Features (raw, not scaled)
        use_ensemble : bool
            Use ensemble voting
            
        Returns:
        --------
        predictions : array
            Model predictions
        """
        X_scaled = self.scaler.transform(X)
        
        if use_ensemble:
            rf_pred = self.rf_model.predict_proba(X_scaled)[:, 1]
            gb_pred = self.gb_model.predict_proba(X_scaled)[:, 1]
            ensemble_pred = (rf_pred + gb_pred) / 2
            return (ensemble_pred > 0.5).astype(int)
        else:
            return self.rf_model.predict(X_scaled)
