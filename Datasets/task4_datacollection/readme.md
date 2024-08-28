# Data Processing for Accelerometer Data

This repository contains scripts used to process accelerometer data for six activities. The data processing involved organizing CSV files into activity-specific directories, renaming files based on subject identifiers, and further restructuring the filenames within each directory.

## Datasets Overview
- unprocessed: This dataset contains the raw g-force data captured directly from the device, sampled at the highest available frequency, which varies between 190 and 210 Hz.

- processed: The dataset has been downsampled to 50 Hz by segmenting the data into 20 ms intervals and computing the average acceleration within each segment.
- processed_trimmed: In this dataset, the first 4.5 seconds (225 rows) of the time series data have been removed, as well as the final 0.5 seconds (25 rows), to mitigate any noise present at the start and end of the recordings.
- raw_dataset: This dataset is divided into training and testing subsets. The testing subset specifically contains activity data for Yash Kokane, while the training subset includes data from all other subjects.
- TSFEL_features: This dataset includes 1173 features extracted using TSFEL (Time Series Feature Extraction Library) from the raw_dataset. It does not contain any train-test splits.
- TSFEL_dataset: This dataset also includes 1173 features extracted using TSFEL from the raw_dataset, but it is organized with predefined train-test splits.

## Directory Structure

The data consists of accelerometer readings corresponding to six activities:

- **LAYING**
- **SITTING**
- **STANDING**
- **WALKING**
- **WALKING_DOWNSTAIRS**
- **WALKING_UPSTAIRS**

Each activity directory contains CSV files representing accelerometer data for multiple subjects.

## Data Formatting

- **Nishchay (N):**
  - **Walking:**
    - Nishchay's four samples for the walking activity are mapped to `Subject_1`, `Subject_2`, `Subject_3`, and `Subject_4`.
  - **Other Activities:**
    - The same mapping pattern is applied to other activities such as Laying, Sitting, Standing, Walking Downstairs, and Walking Upstairs.

- **Karan (K):**
  - **Walking:**
    - Karan's four samples for the walking activity are mapped to `Subject_5`, `Subject_6`, `Subject_7`, and `Subject_8`.
  - **Other Activities:**
    - The same mapping pattern is applied to other activities such as Laying, Sitting, Standing, Walking Downstairs, and Walking Upstairs.

- **Yash (Y):**
  - **Walking:**
    - Yash's four samples for the walking activity are mapped to `Subject_9`, `Subject_10`, `Subject_11`, and `Subject_12`.
  - **Other Activities:**
    - The same mapping pattern is applied to other activities such as Laying, Sitting, Standing, Walking Downstairs, and Walking Upstairs.
