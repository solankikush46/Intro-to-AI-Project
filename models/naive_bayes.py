import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore') # Hides MNE deprecation warnings

# 1. LOAD BALANCED DATA
print("Loading balanced dataset...")
epochs = mne.read_epochs('../DataSet/eeg_dataset/balanced_data-epo.fif', preload=True, verbose=False)

y_labels = epochs.events[:, -1] 
class_names = ['Music', 'Coffee', 'Perfume']

# 2. FEATURE ENGINEERING: Power Spectral Density (PSD)
print("\nExtracting PSD features for Naïve Bayes...")
psds, freqs = mne.time_frequency.psd_array_welch(
    epochs.get_data(), sfreq=256, fmin=0.5, fmax=40.0, n_per_seg=256, verbose=False
)

# Flatten the features
X_features = psds.reshape(len(psds), -1) 

# --- IMPROVEMENT 1: LOG TRANSFORM ---
# PSD values are heavily skewed. Log10 forces them into a Gaussian/Normal distribution
X_log = np.log10(X_features + 1e-10) # Added 1e-10 to prevent log(0) errors
print(f"Feature extraction complete! Feature matrix shape: {X_log.shape}")

# 3. MACHINE LEARNING PIPELINE
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- IMPROVEMENT 2: SCALING & PCA ---
# Standardize the data, then use PCA to ensure features are perfectly independent.
# n_components=0.95 means PCA will keep 95% of the data's variance while dropping noise.
nb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('gnb', GaussianNB())
])

print("\n--- Evaluating Optimized Gaussian Naïve Bayes ---")
nb_predictions = cross_val_predict(nb_pipeline, X_log, y_labels, cv=cv, n_jobs=-1)

print("\nClassification Report:")
print(classification_report(y_labels, nb_predictions, target_names=class_names))

# 4. PLOT CONFUSION MATRIX
cm = confusion_matrix(y_labels, nb_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Optimized Naïve Bayes')
plt.ylabel('Actual Stimulus')
plt.xlabel('Predicted Stimulus')
plt.tight_layout()
plt.show()