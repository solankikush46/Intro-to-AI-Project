import mne
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore') 

# Path Setup
input_path = os.path.join("eeg_dataset", "balanced_data-epo.fif")
output_path = os.path.join("eeg_dataset", "shuffled_balanced_data-epo.fif")

print(f"--- Loading Chronological Dataset ---")
try:
    epochs = mne.read_epochs(input_path, preload=True, verbose=False)
    total_samples = len(epochs)
    print(f"Successfully loaded {total_samples} samples.")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Generate random indices
print("\n--- Shuffling Data ---")
np.random.seed(42)
shuffle_indices = np.random.permutation(total_samples)

# Apply shuffle to MNE object
epochs_shuffled = epochs[shuffle_indices]

# Save new file
print("\n--- Saving Shuffled Dataset ---")
epochs_shuffled.save(output_path, overwrite=True)

# Verification
first_10_labels = epochs_shuffled.events[:10, -1]
print(f"SUCCESS! File saved as: {output_path}")
print(f"Proof of shuffle (First 10 Labels): {first_10_labels}")