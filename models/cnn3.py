import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras import regularizers
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DATA LOADING & DUAL-FEATURE EXTRACTION
# ==========================================
print("--- Loading Shuffled Preprocessed Data ---")
data_file = '../DataSet/eeg_dataset/shuffled_balanced_data-epo.fif'
epochs = mne.read_epochs(data_file, preload=True, verbose=False)
class_names = ['Music', 'Coffee', 'Perfume']

# FEATURE A: RAW SIGNALS (For CNN Branch)
print("Extracting Raw Signals...")
X_raw_mne = epochs.get_data() 
y_raw = epochs.events[:, -1] - 1 # 0, 1, 2
X_raw = np.transpose(X_raw_mne, (0, 2, 1)) # (Samples, Timesteps, Channels)

# FEATURE B: PSD FEATURES (For Dense Branch - Same as your RF)
print("Extracting PSD Features (The 'Random Forest' logic)...")
# Calculate PSD using Multitaper (standard for EEG)
spectrum = epochs.compute_psd(method='multitaper', fmin=0.5, fmax=40, verbose=False)

# 2. Get the actual power data and the frequency list
psds = spectrum.get_data() # shape: (n_epochs, n_channels, n_freqs)
freqs = spectrum.freqs

# 3. Reshape for the model (same as before)
X_psd = psds.reshape(len(epochs), -1)

# ==========================================
# 2. SCALING & SPLITTING
# ==========================================
print("Scaling Features...")
# Scale Raw Data
scaler_raw = StandardScaler()
X_raw_flat = X_raw.reshape(-1, X_raw.shape[-1])
X_raw_scaled = scaler_raw.fit_transform(X_raw_flat).reshape(X_raw.shape)

# Scale PSD Data
scaler_psd = StandardScaler()
X_psd_scaled = scaler_psd.fit_transform(X_psd)

# Split into Training and Testing
# We use a single split first to ensure speed, then you can loop for K-Fold
X_train_raw, X_test_raw, X_train_psd, X_test_psd, y_train, y_test = train_test_split(
    X_raw_scaled, X_psd_scaled, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# ==========================================
# 3. BUILDING THE HYBRID ARCHITECTURE
# ==========================================
def create_fusion_model(raw_shape, psd_shape, num_classes):
    # BRANCH 1: The CNN (Captures Timing/Wiggles)
    input_raw = Input(shape=raw_shape, name="Raw_Signal_Input")
    x1 = Conv1D(32, 7, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_raw)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Dropout(0.3)(x1)
    x1 = Conv1D(64, 5, activation='relu')(x1)
    x1 = Flatten()(x1)
    
    # BRANCH 2: The PSD Network (Captures Rhythms/Alpha-Beta)
    input_psd = Input(shape=psd_shape, name="PSD_Input")
    x2 = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(input_psd)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.4)(x2)
    
    # THE FUSION: Combine both "eyes"
    combined = concatenate([x1, x2])
    
    # FINAL DECISION HEAD
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z) # Aggressive dropout to force fusion learning
    output = Dense(num_classes, activation='softmax')(z)
    
    model = Model(inputs=[input_raw, input_psd], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_fusion_model(X_raw.shape[1:], (X_psd.shape[1],), len(class_names))

# ==========================================
# 4. TRAINING ON THE MILL
# ==========================================
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

print("\n--- Training Hybrid Fusion Model ---")
history = model.fit(
    [X_train_raw, X_train_psd], y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 5. FINAL EVALUATION
# ==========================================
print("\n--- Generating Fusion Results ---")
y_pred_probs = model.predict([X_test_raw, X_test_psd])
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nHYBRID FUSION CNN REPORT:")
print(classification_report(y_test, y_pred, target_names=class_names))

print("\n--- Plotting Training History ---")
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
plt.title('Model Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('Model Loss over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

# Save the plot as an image file to download from the cluster
plt.savefig('fusion_training_curves.png', dpi=300, bbox_inches='tight')
print("Saved training curves to 'fusion_training_curves.png'")

plt.show()