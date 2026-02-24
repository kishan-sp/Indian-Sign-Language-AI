# ISL Recognition System
 
This repository contains the Indian Sign Language (ISL) Recognition System, which uses MediaPipe for hand landmark extraction and a trained LSTM model for real-time sign language prediction.
 
## Requirements
 
Ensure you have Python 3.9 - 3.11 installed.
 
Install the required dependencies using:
 
```bash
pip install -r requirements.txt
```
 
### Additional Prerequisites
- The system requires a webcam to work in real-time prediction mode.
- `hand_landmarker.task` file in the project's root folder (required by MediaPipe).
 
---
 
## How to Build and Train the Model
 
The training process consists of converting raw data (images, videos, or `.npy` sequences) into unified sequences and then training the model.
 
### Step 1: Data Conversion
Convert your data sources into a unified structure (126-dimensional hand keypoint `.npy` files).
Ensure your data is placed in the `archive/` or `NP_Data/` folders following this specific structure:
 
```text
📁 archive/
 ├── 📁 dataset - Gesture Speech/   (Static Images)
 │    ├── 📁 hello/
 │    │    ├── image1.jpg
 │    │    └── ...
 │    └── 📁 bye/
 │
 ├── 📁 1/                          (Raw Video Files)
 │    ├── 📁 person1/
 │    │    ├── video1.mp4
 │    │    └── ...
 │    └── 📁 person2/
 └── 📁 2/
      └── ...
 
📁 NP_Data/                         (Existing 1662-dim .npy Keypoints)
 ├── 📁 hello/
 │    ├── 📁 0/                     (Sequence number)
 │    │    ├── 0.npy                (Frame 0)
 │    │    ├── 1.npy                (Frame 1)
 │    │    └── ...
 │    └── 📁 1/
 └── 📁 bye/
      └── ...
```
 
```bash
python convert_to_npy.py
```
This script will:
- Process the dataset.
- Clean and map existing data into the `NP_Data_Combined/` directory.
- Generate `label_map.json` which maps signs to zero-indexed classes.
 
**Optional flags:**
- `--images-only`: Only process the static image dataset.
- `--videos-only`: Only process the video archive dataset.
- `--npy-only`: Migrate existing NP_Data cleanly.
 
### Step 2: Model Training
Once the combined dataset is prepared in `NP_Data_Combined/`, you can train the LSTM model.
 
```bash
python model_training.py
```
This script will:
- Preprocess data (via `data_preprocessing.py`).
- Split data into train and test sets.
- Build and train a 3-layer LSTM.
- Save the trained model as `isl_lstm_model.h5`.
- Output accuracy curves (`accuracy_plot.png`), loss curves (`loss_plot.png`), and a confusion matrix (`confusion_matrix.png`).
 
---
 
## How to Run Real-Time Prediction
 
Once the model (`isl_lstm_model.h5`) is trained and saved in your root directory, you can start the webcam application to predict signs in real time.
 
```bash
python realtime_test.py
```
 
### Controls in Real-Time App
- **Wait for prediction**: Follow the instructions on the screen to perform a sign.
- **Press `c`**: To start capturing a 2-second sequence. Once the capture meter fills up, it will predict your sign.
- **Press `q`**: To exit the real-time application.
 
## Project Structure
- `convert_to_npy.py`: Script to convert multiple types of data sources to combined numpy arrays.
- `data_preprocessing.py`: Automatically resizes sequences, pads/trims, and prepares the arrays for training.
- `model_training.py`: Builds and trains the LSTM model, logging graphs and the `isl_lstm_model.h5` file.
- `realtime_test.py`: Interactive app using OpenCV and the trained LSTM for ISL prediction.
- `fix_my_data.py`: A helper script (if required) to fix missing `.npy` data.
 
 
