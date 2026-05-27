# MLProject/modelling.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ── Argument Parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators',     type=int, default=200)
parser.add_argument('--max_depth',        type=int, default=10)
parser.add_argument('--min_samples_split',type=int, default=2)
args = parser.parse_args()

# ── DagsHub + MLflow Setup ─────────────────────────
import os

# Kalau env variable sudah di-set (GitHub Actions), skip dagshub.init
if os.environ.get('MLFLOW_TRACKING_URI'):
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
else:
    # Lokal — pakai dagshub.init seperti biasa
    dagshub.init(
        repo_owner='projet752',
        repo_name='mlsystem-heart-disease',
        mlflow=True
    )

mlflow.set_experiment("heart-disease-ci")

mlflow.set_experiment("heart-disease-ci")

# ── Load Data ─────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(base_dir, 'heart_preprocessing/heart_train.csv'))
test_df  = pd.read_csv(os.path.join(base_dir, 'heart_preprocessing/heart_test.csv'))

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test  = test_df.drop('target', axis=1)
y_test  = test_df['target']

# ── Training + Manual Logging ─────────────────────────────────────────────────
with mlflow.start_run():

    # Log params
    mlflow.log_param("n_estimators",      args.n_estimators)
    mlflow.log_param("max_depth",         args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)

    # Train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall",    rec)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("roc_auc",   auc)

    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")

    # Artefak 1: Confusion Matrix
    os.makedirs('artifacts', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['No Disease','Disease']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    cm_path = 'artifacts/confusion_matrix.png'
    plt.savefig(cm_path); plt.close()
    mlflow.log_artifact(cm_path)

    # Artefak 2: ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0,1],[0,1], color='gray', linestyle='--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve'); ax.legend()
    plt.tight_layout()
    roc_path = 'artifacts/roc_curve.png'
    plt.savefig(roc_path); plt.close()
    mlflow.log_artifact(roc_path)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("\n✅ Training selesai! Artefak tersimpan di DagsHub.")