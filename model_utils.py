# model_utils.py
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE # <-- Tambahkan import ini
from sklearn.metrics import (
    classification_report, roc_auc_score, 
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import streamlit as st # Untuk logging

@st.cache_data
def split_and_resample(X, y, method='undersampling'): # <-- Tambahkan parameter 'method'
    """
    Melakukan split data dan resampling (hanya pada data training).
    """
    # 1. Split data menjadi train dan test (Test set asli, imbalanced)
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 2. Resample HANYA pada data training
    if method == 'undersampling':
        sampler = RandomUnderSampler(random_state=42)
        st.write("Menerapkan Random Under-Sampling (RUS) pada data latih...")
    elif method == 'oversampling':
        sampler = SMOTE(random_state=42)
        st.write("Menerapkan SMOTE (Over-sampling) pada data latih...")
    else:
        st.write("Tidak ada resampling.")
        return X_train_orig, y_train_orig, X_test, y_test
            
    X_train_res, y_train_res = sampler.fit_resample(X_train_orig, y_train_orig)
    
    # Kembalikan semua set data
    return X_train_res, y_train_res, X_test, y_test

@st.cache_resource
def train_ann_model(X_train, y_train):
    """
    Membuat dan melatih model ANN baru berdasarkan input.
    Menggunakan @st.cache_resource agar model tidak dilatih ulang
    kecuali input (data latih) berubah.
    """
    # Arsitektur ANN sesuai rencana Anda
    model = MLPClassifier(
        hidden_layer_sizes=(14,),
        activation='tanh',
        max_iter=1000,
        random_state=42,
        early_stopping=True # Mencegah overfitting & menghemat waktu
    )
    
    model.fit(X_train, y_train)
    return model

def get_predictions(_model, X_data):
    """Mendapatkan probabilitas dan kelas prediksi."""
    # Pastikan urutan kolom X_data sama dengan saat model dilatih
    X_data_reordered = X_data.reindex(columns=_model.feature_names_in_, fill_value=0)
    
    probs = _model.predict_proba(X_data_reordered)[:, 1]
    preds = _model.predict(X_data_reordered)
    return probs, preds

def generate_evaluation_metrics(_model, X_test, y_test):
    """
    Menghasilkan semua metrik evaluasi berdasarkan data Test yang ASLI.
    """
    # Dapatkan prediksi khusus untuk test set
    probs, preds = get_predictions(_model, X_test)
    
    results = {}
    
    # 1. Classification Report
    try:
        results['report'] = classification_report(y_test, preds, output_dict=True)
    except Exception:
        # Menangani kasus jika test set tidak memiliki kedua kelas (jarang terjadi)
        results['report'] = {"error": "Gagal membuat report, cek distribusi Y_test."}

    # 2. AUC Score
    try:
        results['auc'] = roc_auc_score(y_test, probs)
    except Exception:
        results['auc'] = 0.0

    # 3. Confusion Matrix Plot
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax_cm, cmap=plt.cm.Blues)
    ax_cm.set_title("Confusion Matrix (Test Set)")
    results['cm_plot'] = fig_cm
    
    # 4. ROC Curve Plot
    fig_roc, ax_roc = plt.subplots()
    try:
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"AUC = {results['auc']:.4f}")
    except Exception:
        ax_roc.text(0.5, 0.5, "Gagal membuat ROC", ha='center')
        
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve (Test Set)')
    ax_roc.legend()
    results['roc_plot'] = fig_roc
    
    return results