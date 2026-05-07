import wfdb
import os

data_dir = 'eeg_dataset'
# Downloads the "Regulation of Brain Cognitive States" dataset 
wfdb.dl_database('brain-wearable-monitoring', dl_dir=data_dir)
print(f"Download complete! Files saved in: {os.path.abspath(data_dir)}")