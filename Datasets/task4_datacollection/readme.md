# Data Processing for Accelerometer Data

This repository contains scripts used to process accelerometer data for six activities. The data processing involved organizing CSV files into activity-specific directories, renaming files based on subject identifiers, and further restructuring the filenames within each directory.

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
    - Karan's four samples for the walking activity are mapped to `Subject_1`, `Subject_2`, `Subject_3`, and `Subject_4`.
  - **Other Activities:**
    - The same mapping pattern is applied to other activities such as Laying, Sitting, Standing, Walking Downstairs, and Walking Upstairs.

- **Yash (Y):**
  - **Walking:**
    - Yash's four samples for the walking activity are mapped to `Subject_1`, `Subject_2`, `Subject_3`, and `Subject_4`.
  - **Other Activities:**
    - The same mapping pattern is applied to other activities such as Laying, Sitting, Standing, Walking Downstairs, and Walking Upstairs.

## Steps for Data Processing

### Step 1: Organizing and Renaming Files by Activity and Subject

The first script organizes the CSV files into directories based on the activity they represent. Additionally, it renames the files based on the subject performing the activity.

- **Activity Mapping**: The script identifies the activity type by matching the prefix in the filename (`lay`, `sit`, etc.) to its corresponding activity (`LAYING`, `SITTING`, etc.).
- **Subject Mapping**: The first letter of the filename (`n`, `k`, `y`) is mapped to the subject performing the activity (`Nishchay`, `Karan`, `Yash`).
- **File Naming**: Files are renamed as `Subject_1_1.csv`, `Subject_1_2.csv`, etc., where the suffix number distinguishes multiple samples from the same subject.

### Step 2: Renaming Files Within Each Activity Directory

After organizing the data into directories and renaming the files, the next script simplifies the filenames by removing the suffixes:

- **Renaming**: Within each activity directory, files named `Subject_1_1.csv`, `Subject_1_2.csv`, etc., are renamed to `Subject_1.csv`, `Subject_2.csv`, and so on. This is done to distinguish different samples for each subject.
# Data Processing for Accelerometer Data

This repository contains scripts used to process raw accelerometer data into a structured format suitable for analysis. The data processing involved downsampling, renaming columns, and rounding values to a specified precision.

## Directory Structure

The data is organized into the following directory structure:

- **`raw/`**: Contains the original, unprocessed CSV files.
- **`processed/`**: Contains the processed CSV files after downsampling and renaming.

## Processing Steps

### 1. Downsampling the Data

- **Original Sampling Rate**: The raw data was collected at an approximate frequency of 200 Hz.
- **Downsampling**: The data was downsampled to 50 Hz by averaging every 20 ms worth of data. This involved grouping data points within each 20 ms interval and calculating the average value for each axis (`gFx`, `gFy`, `gFz`).

### 2. Renaming Columns

After downsampling, the following column renaming was applied:

- **Original Columns**:
  - `gFx`: Represents the acceleration in the X-axis.
  - `gFy`: Represents the acceleration in the Y-axis.
  - `gFz`: Represents the acceleration in the Z-axis.
  
- **New Columns**:
  - `accx`: Represents the downsampled acceleration in the X-axis.
  - `accy`: Represents the downsampled acceleration in the Y-axis.
  - `accz`: Represents the downsampled acceleration in the Z-axis.

### 3. Rounding Values

- **Precision**: The final values in the processed CSV files were rounded to 7 decimal places to maintain consistency and precision across the dataset.

## Final Output

The processed CSV files are stored in the `processed/` directory, maintaining the same directory structure as the original `raw/` directory. Each processed file reflects the downsampled and renamed data ready for further analysis.
