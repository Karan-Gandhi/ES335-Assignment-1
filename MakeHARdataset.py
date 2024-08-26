import os
import numpy as np
import pandas as pd

# Define the base filepath for the dataset
filepath = os.path.join(os.path.abspath('..'), 'Datasets', 'UCI HAR Dataset')

# Load feature names from 'features.txt'
with open(os.path.join(filepath, 'features.txt'), 'r') as f:
    lines = f.readlines()

# Initialize an empty list to store feature names and a dictionary to handle duplicates
features = []
exists = {}

# Process each line in the features file to extract and clean feature names
for line in lines:
    # Clean and format the feature names, replacing unwanted characters
    new_line = line.replace('\n', '').replace('-', '_')[line.find(' ') + 1:]
    if new_line in exists:
        # Handle duplicate feature names (e.g., fBodyAcc_bandsEnergy()_1,8)
        features.append(new_line + "_" + str(exists[new_line]))
        exists[new_line] += 1
    else:
        features.append(new_line)
        exists[new_line] = 1

# Function to generate the dataset (train/test) based on the folder name
def generate_dataset(folder: str = 'train') -> tuple[np.ndarray, np.ndarray]:
    dataset_path = os.path.join(filepath, folder)
    
    # Load the feature data from the specified folder (X_train.txt or X_test.txt)
    print(f"Loading feature data from {dataset_path}...")
    with open(os.path.join(dataset_path, f'X_{folder}.txt')) as X_file:
        lines = X_file.readlines()
    X = []
    for line in lines:
        # Split each line into individual feature values and convert to float64
        row = np.array(line.split(), dtype=np.float64)
        X.append(row)
    X = np.array(X)
    # print(f"Feature data loaded. Shape: {X.shape}")

    # Load the subject data from the specified folder (subject_train.txt or subject_test.txt)
    # print(f"Loading subject data from {dataset_path}...")
    with open(os.path.join(dataset_path, f'subject_{folder}.txt')) as subj_file:
        subject_lines = subj_file.readlines()
    subjects = np.array([int(subj.strip()) for subj in subject_lines], dtype=np.int32)
    # print(f"Subject data loaded. Shape: {subjects.shape}")

    # Load the activity labels from the specified folder (y_train.txt or y_test.txt)
    # print(f"Loading activity labels from {dataset_path}...")
    with open(os.path.join(dataset_path, f'y_{folder}.txt')) as y_file:
        lines = y_file.readlines()
    y = np.array([int(line.strip()) for line in lines], dtype=np.int32) - 1  # Adjust for 1-based indexing
    # print(f"Activity labels loaded. Shape: {y.shape}")

    # Create a DataFrame with features, subject, and activity label
    X_df = pd.DataFrame(X, columns=features)
    X_df['Subject'] = pd.Series(subjects)
    X_df['y'] = pd.Series(y)

    # Print the final shape of the DataFrame for debugging
    # print(f"Final dataset shape (with subject and activity): {X_df.shape}")
    # print(f"First few rows:\n{X_df.head()}")
      
    # Group by Subject and y, then count the occurrences to understand the data distribution
    # print("Number of rows per subject for each activity label in the test dataset:")
    subject_activity_counts = X_df.groupby(['Subject', 'y']).size().unstack(fill_value=0)
    # print(subject_activity_counts)

    # Step 1: Determine the minimum number of rows for each activity label across all subjects
    min_rows_per_activity = X_df.groupby(['Subject', 'y']).size().min()
    # print(f"Minimum number of rows per activity across all subjects: {min_rows_per_activity}")

    # Initialize lists to store trimmed data and labels
    trimmed_data = []
    y_labels = []
    subjects = X_df['Subject'].unique()
    
    for subject in subjects:
        subject_data = X_df[X_df['Subject'] == subject]
        subject_activities = []
        
        for activity in sorted(subject_data['y'].unique()):
            # Extract a consistent number of rows per activity and subject
            # Exclude 'Subject' and 'y' columns
            activity_data = subject_data[subject_data['y'] == activity].iloc[:min_rows_per_activity, :-2].values  
            subject_activities.append(activity_data)
            y_labels.append(activity)  # Append the activity label
            
        # Stack the activities data for each subject
        trimmed_data.append(np.stack(subject_activities))  
            
    # Convert the list of trimmed data and labels to numpy arrays
    X_trimmed, y_trimmed = np.stack(trimmed_data), np.array(y_labels)

    # Flatten the first two dimensions (subjects and activities) of X_trimmed
    X_flattened = X_trimmed.reshape(-1, X_trimmed.shape[2], X_trimmed.shape[3])
    
    # Return the flattened features and labels for further processing
    return X_flattened, y_trimmed

X_train, y_train = generate_dataset('train')
X_test, y_test = generate_dataset('test')
# Find the common number of timeseries rows between X_train and X_test
min_columns = min(X_train.shape[1], X_test.shape[1])

# Slice both X_train and X_test to have the same number of timeseries rows
X_train = X_train[:, :min_columns]
X_test = X_test[:, :min_columns]

if __name__ == "__main__":
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
# After this operation, X_train and X_test will have the same shape[1]
