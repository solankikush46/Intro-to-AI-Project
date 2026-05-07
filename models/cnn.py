import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# GPU CONFIGURATION
# ==========================================
print("--- Checking GPU Status ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable Memory Growth to prevent VRAM allocation crashes on Windows
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Detected and Configured: {len(gpus)} GPU(s) available.")
    except RuntimeError as e:
        print(f"GPU Configuration Error: {e}")
else:
    print("⚠️ No GPU detected. TensorFlow will train on the CPU.")

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def plot_cnn_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    sns.set_theme(style="whitegrid")

    ax1.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
    ax1.set_title('1D-CNN Accuracy per Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='lower right')

    ax2.plot(history.history['loss'], label='Training Loss', color='#1f77b4', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
    ax2.set_title('1D-CNN Loss per Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('cnn_training_curves.png', dpi=300, bbox_inches='tight')
    print("Saved training curves to cnn_training_curves.png")
    plt.show()

def create_1d_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax') 
    ])
    
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================
print("\n--- Loading Preprocessed Data ---")
# Load the perfectly balanced file we made in Phase 1
epochs = mne.read_epochs('../DataSet/eeg_dataset/shuffled_balanced_data-epo.fif', preload=True, verbose=False)

class_names = ['Music', 'Coffee', 'Perfume']

# ==========================================
# 2. BRIDGING DATA FOR DEEP LEARNING
# ==========================================
print("--- Formatting Data for CNN ---")
# Extract raw brainwave matrices
X_mne = epochs.get_data()

# Extract labels (1=Music, 2=Coffee, 3=Perfume)
y_raw = epochs.events[:, -1] 
# Neural networks need labels to start at 0 (0, 1, 2)
y_encoded = y_raw - 1 

# MNE shape is (samples, channels, timesteps). 
# Keras CNN requires (samples, timesteps, channels).
X_raw = np.transpose(X_mne, (0, 2, 1))

samples, timesteps, channels = X_raw.shape

# Deep Learning requires standardized numerical ranges
scaler = StandardScaler()
X_flat = X_raw.reshape(-1, channels)
X_scaled_flat = scaler.fit_transform(X_flat)
X_ready = X_scaled_flat.reshape(samples, timesteps, channels)

print(f"CNN Input Shape Ready: {X_ready.shape}")
print(f"Encoded Labels Ready: {np.unique(y_encoded)} mapping to {class_names}")

# ==========================================
# 3. EVALUATING 1D-CNN ARCHITECTURE
# ==========================================
print("\n--- Training 1D-CNN with K-Fold Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []
input_shape = (timesteps, channels)
num_classes = len(class_names)
final_history = None

for fold, (train_idx, test_idx) in enumerate(cv.split(X_ready, y_encoded)):
    print(f"\nTraining Fold {fold + 1}/5...")
    X_train, X_test = X_ready[train_idx], X_ready[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    model = create_1d_cnn(input_shape, num_classes)
    
    # Train the model (TensorFlow will automatically route this to the GPU if available)
    final_history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# ==========================================
# 4. FINAL RESULTS & VISUALIZATIONS
# ==========================================
print("\n==========================================")
print("1D-CNN Final Classification Report:")
print("==========================================")
print(classification_report(all_y_true, all_y_pred, target_names=class_names))

if final_history:
    plot_cnn_training_curves(final_history)

cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Used Greens to distinguish from RF/NB
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: 1D-CNN')
plt.ylabel('Actual Stimulus')
plt.xlabel('Predicted Stimulus')
plt.tight_layout()
plt.show()