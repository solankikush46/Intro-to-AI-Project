import mne
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore') # Hides MNE deprecation warnings

# 1. PATH SETUP
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

# 2. GENERATE RANDOM INDICES
print("\n--- Shuffling Data ---")
# We set a random seed (42) so that if you ever need to re-run this, 
# it scrambles the data in the exact same way.
np.random.seed(42)
shuffle_indices = np.random.permutation(total_samples)

# 3. APPLY SHUFFLE TO MNE OBJECT
epochs_shuffled = epochs[shuffle_indices]

# 4. SAVE NEW FILE
print("\n--- Saving Shuffled Dataset ---")
epochs_shuffled.save(output_path, overwrite=True)

# 5. VERIFICATION
# Let's check the first 10 labels to prove it actually scrambled them!
first_10_labels = epochs_shuffled.events[:10, -1]
print(f"SUCCESS! File saved as: {output_path}")
print(f"Proof of shuffle (First 10 Labels): {first_10_labels}")
print("(If it was chronological, these would all be '1')")