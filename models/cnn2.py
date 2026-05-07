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
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# GPU CONFIGURATION
# ==========================================
print("--- Checking GPU Status ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
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
    print("\nSaved training curves to cnn_training_curves.png")
    plt.show()

def create_1d_cnn(input_shape, num_classes):
    """
    Optimized Architecture: 3 Conv Layers + Batch Normalization + Dropout.
    Designed to capture complex EEG dependencies while preventing overfitting.
    """
    model = Sequential([
        # Layer 1: Basic Waveform Features
        Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Layer 2: Mid-level Rhythmic Patterns
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        # Layer 3: High-level Abstract Dependencies
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        
        # Decision Head
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5), # Crucial for forcing the model to learn general patterns
        
        Dense(num_classes, activation='softmax') 
    ])
    
    # Lower learning rate for better stability with deeper layers
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 1. LOAD PREPROCESSED DATA
# ==========================================
print("\n--- Loading Shuffled Preprocessed Data ---")
# Using the shuffled file ensures fair validation splits
data_file = '../DataSet/eeg_dataset/shuffled_balanced_data-epo.fif'
epochs = mne.read_epochs(data_file, preload=True, verbose=False)

class_names = ['Music', 'Coffee', 'Perfume']

# ==========================================
# 2. DATA FORMATTING
# ==========================================
print("--- Formatting Data for Deep Learning ---")
X_mne = epochs.get_data()
y_raw = epochs.events[:, -1] 
y_encoded = y_raw - 1 # Map 1,2,3 to 0,1,2

# Change shape from (samples, channels, timesteps) to (samples, timesteps, channels)
X_raw = np.transpose(X_mne, (0, 2, 1))
samples, timesteps, channels = X_raw.shape

# Standardization
scaler = StandardScaler()
X_flat = X_raw.reshape(-1, channels)
X_scaled_flat = scaler.fit_transform(X_flat)
X_ready = X_scaled_flat.reshape(samples, timesteps, channels)

print(f"CNN Input Shape: {X_ready.shape}")

# ==========================================
# 3. TRAINING WITH K-FOLD CROSS-VALIDATION
# ==========================================
print("\n--- Training Optimized 1D-CNN ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []
input_shape = (timesteps, channels)
num_classes = len(class_names)
final_history = None

# Early Stopping: The safety valve that stops training if validation loss stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

for fold, (train_idx, test_idx) in enumerate(cv.split(X_ready, y_encoded)):
    print(f"\n" + "="*30)
    print(f"TRAINING FOLD {fold + 1}/5")
    print("="*30)
    
    X_train, X_test = X_ready[train_idx], X_ready[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    model = create_1d_cnn(input_shape, num_classes)
    
    # Train Fold
    history = model.fit(
        X_train, y_train, 
        epochs=30, 
        batch_size=64, 
        validation_split=0.2, 
        callbacks=[early_stop], 
        verbose=1
    )
    
    # Store history of the final fold for plotting
    final_history = history
    
    # Evaluation
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# ==========================================
# 4. FINAL REPORTING
# ==========================================
print("\n" + "#"*40)
print("1D-CNN FINAL OVERALL PERFORMANCE")
print("#"*40)
report = classification_report(all_y_true, all_y_pred, target_names=class_names)
print(report)

# Save results to text file
with open("cnn_final_metrics.txt", "w") as f:
    f.write("OPTIMIZED 1D-CNN RESULTS\n")
    f.write("========================\n")
    f.write(report)

if final_history:
    plot_cnn_training_curves(final_history)

# Final Confusion Matrix Visualization
plt.figure(figsize=(7, 5))
cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Final Confusion Matrix: Optimized 1D-CNN')
plt.ylabel('Actual Stimulus')
plt.xlabel('Predicted Stimulus')
plt.tight_layout()
plt.show()