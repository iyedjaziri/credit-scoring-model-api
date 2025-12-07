import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from loguru import logger

# Configuration
DATA_PATH = "data/processed/train_prepared.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
PICKLE_PATH = os.path.join(MODEL_DIR, "model.pkl")
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")

def train():
    logger.info("Loading Data...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Data Shape: {df.shape}")
    
    # Clean column names (LightGBM stricture)
    import re
    df = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # Coerce all columns to numeric, raising errors if not possible (or coerce to NaN)
    # The error indicated specific object columns.
    logger.info("Ensuring all columns are numeric...")
    for col in df.columns:
        if df[col].dtype == 'object':
             try:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
             except Exception as e:
                 logger.warning(f"Could not convert {col} to numeric: {e}. Dropping it.")
                 df = df.drop(columns=[col])

    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    # Train/Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle Imbalance (Optional but recommended for this dataset)
    # Using simple weighting for LGBM, or sampling?
    # User's previous notebook used sampling. 
    # Let's use Class Weights in LGBM which is standard for full data.
    
    logger.info("Training Model...")
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=34,
        colsample_bytree=0.9,
        subsample=0.8,
        max_depth=8,
        reg_alpha=0.04,
        reg_lambda=0.07,
        min_split_gain=0.02,
        min_child_weight=39,
        silent=-1,
        verbose=-1,
        class_weight='balanced',
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)], 
        eval_metric='auc',
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )
    
    # Validation Metrics
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    f1 = f1_score(y_val, y_pred)
    
    logger.info(f"Validation AUC: {auc:.4f}")
    logger.info(f"Validation F1: {f1:.4f}")
    
    # Save Pickle
    logger.info(f"Saving Pickle to {PICKLE_PATH}...")
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(clf, f)
        
    # Convert to ONNX
    logger.info("Converting to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = onnxmltools.convert_lightgbm(clf, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, ONNX_PATH)
    logger.info(f"Saved ONNX to {ONNX_PATH}")

if __name__ == "__main__":
    train()
