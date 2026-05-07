import mne
import os
import glob
import pandas as pd
import numpy as np

# 1. PATH SETUP
data_dir = "raw_dataset"
output_path = os.path.join("eeg_dataset", "balanced_data-epo.fif")

csv_files = glob.glob(os.path.join(data_dir, "**/EEG_recording.csv"), recursive=True)
sfreq = 256 

music_list = []
coffee_list = []
perfume_list = []

print(f"Found {len(csv_files)} subject recordings. Starting chronological slicing...")

for eeg_path in csv_files:
    parent_dir = os.path.dirname(eeg_path)
    subject_id = os.path.basename(parent_dir).upper()
    nback_path = os.path.join(parent_dir, "n_back_responses.csv")
    
    try:
        # Load and Clean
        df = pd.read_csv(eeg_path)
        eeg_data = df.iloc[:, 1:6].values.T 
        info = mne.create_info(ch_names=['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX'], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(eeg_data, info, verbose=False)
        raw.filter(l_freq=0.5, h_freq=40.0, verbose=False)
        
        # ICA Cleaning
        ica = mne.preprocessing.ICA(n_components=4, method='infomax', fit_params=dict(extended=True), random_state=42)
        ica.fit(raw, verbose=False)
        ica.apply(raw, verbose=False)
        
        # Epoching
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=1.0, preload=True, verbose=False)
        epochs.drop_bad() 
        
        # --- CHRONOLOGICAL SLICING LOGIC ---
        total_epochs = len(epochs)
        third = total_epochs // 3
        
        if subject_id.startswith('A'):
            # A-Series: Baseline (1st third), Music (2nd and 3rd thirds)
            music_epochs = epochs[third:] # Keep the last 2/3rds
            music_epochs.events[:, -1] = 1
            music_epochs.metadata = None 
            music_epochs.selection = np.arange(len(music_epochs))
            music_list.append(music_epochs)
            print(f"{subject_id} processed: Extracted {len(music_epochs)} Music samples.")
            
        elif subject_id.startswith('B'):
            # B-Series: Baseline, Coffee, Perfume (Order varies per subject)
            if not os.path.exists(nback_path):
                print(f"{subject_id}: Missing response CSV. Skipping.")
                continue
                
            nback_df = pd.read_csv(nback_path)
            
            # Find the starting row index to determine which came first
            c_idx = nback_df['CoffeeSession'].dropna().index[0] if 'CoffeeSession' in nback_df.columns and nback_df['CoffeeSession'].count() > 0 else 9999
            p_idx = nback_df['PerfumeSession'].dropna().index[0] if 'PerfumeSession' in nback_df.columns and nback_df['PerfumeSession'].count() > 0 else 9999
            
            if c_idx == 9999 and p_idx == 9999:
                print(f"⚠️ {subject_id}: No Coffee or Perfume data found. Skipping.")
                continue
            
            # Slice the middle and final thirds
            if c_idx < p_idx: # Coffee happened before Perfume
                coffee_epochs = epochs[third:2*third]
                perfume_epochs = epochs[2*third:]
            else:             # Perfume happened before Coffee
                perfume_epochs = epochs[third:2*third]
                coffee_epochs = epochs[2*third:]
                
            # Tag Coffee (Label 2)
            coffee_epochs.events[:, -1] = 2
            coffee_epochs.metadata = None
            coffee_epochs.selection = np.arange(len(coffee_epochs))
            coffee_list.append(coffee_epochs)
            
            # Tag Perfume (Label 3)
            perfume_epochs.events[:, -1] = 3
            perfume_epochs.metadata = None
            perfume_epochs.selection = np.arange(len(perfume_epochs))
            perfume_list.append(perfume_epochs)
                
            print(f"{subject_id} processed: Extracted {len(coffee_epochs)} Coffee, {len(perfume_epochs)} Perfume.")
            
    except Exception as e:
        print(f"Error on {subject_id}: {e}")


# 3. BALANCING & FINAL CONCATENATION
print("\n--- DATASET DIAGNOSTICS ---")
print(f"Total Music Blocks:   {len(music_list)}")
print(f"Total Coffee Blocks:  {len(coffee_list)}")
print(f"Total Perfume Blocks: {len(perfume_list)}")

if music_list and coffee_list and perfume_list:
    print("\nExtracting raw matrices to bypass MNE concatenation errors...")
    
    # Helper function to safely merge a list of Epochs into pure Numpy arrays
    def extract_data(ep_list):
        data = np.concatenate([ep.get_data() for ep in ep_list], axis=0)
        events = np.concatenate([ep.events for ep in ep_list], axis=0)
        return data, events

    # Extract pure numbers (No MNE tracking baggage)
    m_data, m_events = extract_data(music_list)
    c_data, c_events = extract_data(coffee_list)
    p_data, p_events = extract_data(perfume_list)
    
    # Calculate how many Music samples to keep
    target_n = int((len(c_data) + len(p_data)) / 2)
    print(f"⚖️ Balancing: Reducing Music from {len(m_data)} to {target_n} samples...")
    
    np.random.seed(42)
    keep_idx = np.random.choice(len(m_data), target_n, replace=False)
    keep_idx.sort() # Keep chronological order
    
    m_data_balanced = m_data[keep_idx]
    m_events_balanced = m_events[keep_idx]
    
    # Final Combine with Numpy
    print("Merging all data into final dataset...")
    final_data = np.concatenate([m_data_balanced, c_data, p_data], axis=0)
    
    # Rebuild the events array cleanly to prevent any overlap issues
    total_samples = len(final_data)
    final_events = np.zeros((total_samples, 3), dtype=int)
    final_events[:, 0] = np.arange(total_samples) # Clean, sequential sample IDs
    final_events[:, 2] = np.concatenate([
        m_events_balanced[:, 2], 
        c_events[:, 2], 
        p_events[:, 2]
    ])
    
    # Wrap back into a fresh MNE object for saving
    info = music_list[0].info
    final_dataset = mne.EpochsArray(final_data, info, events=final_events, verbose=False)
    
    if not os.path.exists("eeg_dataset"): os.makedirs("eeg_dataset")
    final_dataset.save(output_path, overwrite=True)
    
    print(f"\nSUCCESS! All 3 classes mapped, balanced, and saved.")
    print(f"Final Counts -> Music: {len(m_data_balanced)}, Coffee: {len(c_data)}, Perfume: {len(p_data)}")
    print("You are ready to train your models!")
else:
    print("Error: Missing data from one or more classes.")