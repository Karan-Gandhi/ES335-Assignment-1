import os
import pandas as pd
import tsfel
from pathlib import Path

# Base directory path
base_dir = 'Combined/Train'

# Output directory for TSFEL features
output_base_dir = 'TSFEL_3axes_allfeatures'

# Activities to consider
activities = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']

# Load data for each activity and process each subject individually
for activity in activities:
    activity_dir = os.path.join(base_dir, activity)
    output_activity_dir = os.path.join(output_base_dir, activity)
    Path(output_activity_dir).mkdir(parents=True, exist_ok=True)
    
    subject_files = [f for f in os.listdir(activity_dir) if f.endswith('.csv')]
    
    for file in subject_files:
        file_path = os.path.join(activity_dir, file)
        df = pd.read_csv(file_path)
        
        # Apply TSFEL feature extraction
        cfg = tsfel.get_features_by_domain()  # Adjust this based on the domains you need
        features = tsfel.time_series_features_extractor(cfg, df, freq=50)
        
        # Construct the output filename
        subject_id = file.split('.')[0]  # Assuming file naming convention allows this
        output_file = os.path.join(output_activity_dir, f'{subject_id}.csv')
        
        # Save the features to the output file
        features.to_csv(output_file, index=False)
