import os
import pandas as pd
import tsfel
from pathlib import Path

base_dir = 'Datasets/Combined/Train'
output_base_dir = 'Datasets/TSFEL_3axes_allfeatures/Train'

activities = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']

for activity in activities:
    activity_dir = os.path.join(base_dir, activity)
    output_activity_dir = os.path.join(output_base_dir, activity)
    Path(output_activity_dir).mkdir(parents=True, exist_ok=True)
    subject_files = [f for f in os.listdir(activity_dir) if f.endswith('.csv')]
    for file in subject_files:
        file_path = os.path.join(activity_dir, file)
        df = pd.read_csv(file_path).iloc[100:600, :]
        cfg = tsfel.get_features_by_domain() 
        # print(cfg)
        for domain in cfg:
            for feature in cfg[domain]:
                cfg[domain][feature]['use'] = 'yes' # use all features, even ones disabled by default

        features = tsfel.time_series_features_extractor(cfg, df, fs=50) # sampling rate 50 Hz
        subject_id = file.split('.')[0]
        output_file = os.path.join(output_activity_dir, f'{subject_id}.csv')
        features.to_csv(output_file, index=False)
