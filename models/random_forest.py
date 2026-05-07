import mne
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

data_path = os.path.join('../DataSet/eeg_dataset', 'balanced_data-epo.fif')

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Please run Phase 1 script first!")
else:
    epochs = mne.read_epochs(data_path, preload=True)
    
    print("Extracting frequency features (PSD)...")

    psds, freqs = mne.time_frequency.psd_array_multitaper(
        epochs.get_data(), sfreq=256, fmin=0.5, fmax=40, verbose=False
    )
    
    # Flatten the features
    X = psds.reshape(len(psds), -1) 
    y = epochs.events[:, -1]

    class_names = ['Music', 'Coffee', 'Perfume']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate Random Forest
    print("--- Evaluating Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_predictions = cross_val_predict(rf_model, X, y, cv=cv)

    print("\nClassification Report:")
    report = classification_report(y, rf_predictions, target_names=class_names)
    print(report)

    # Plot Confusion Matrix
    cm = confusion_matrix(y, rf_predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix: Random Forest')
    plt.ylabel('Actual Stimulus')
    plt.xlabel('Predicted Stimulus')
    plt.tight_layout()
    plt.show()