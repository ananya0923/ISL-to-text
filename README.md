## Sign Language to Text Translation

# Dataset

ISL-CSLRT (Indian Sign Language) Dataset is which is Sign language video and sentence text pair. The dataset is divided into Training (781 videos), Validation(377 videos) and Testing (468 videos) dataset.

# Model Architecture

The project is built on hybri model of 3D-CNN and LSTM. 3D CNN focuses on extracting spacial and temporal features. LSTM converts those features into text tokens.

# Execution Steps:

1. **Clone Repository**:
   ```bash
   git clone https://github.com/Aishbs/Signtext.git
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv myven
   ```

3. **Activate Environment**:
   - On Windows:
     ```bash
     myven/Scripts/activate
     ```
   - On macOS/Linux:
     ```bash
     source myven/bin/activate
     ```

4. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Preprocess Dataset**:
   - Extract frames from videos:
     ```bash
     python frame.py train_videos train_frames
     python frame.py val_videos val_frames
     python frame.py test_videos test_frames
     ```

6. **Train the Model**:
   ```bash
   python gesture_model_training.py -d train_frames -e val_frames -b batch_size -l learning_rate -ep epochs
   ```
   Replace `batch_size`, `learning_rate`, and `epochs` with appropriate values for your dataset.

7. **Evaluate the Model**:
   ```bash
   python gesture_model_predictions.py -m model_dir -d test_frames
   ```
   Provide the path to your trained model (`model_dir`) and the testing frames dataset (`test_frames`).

8. **Run the Application**:
   ```bash
   python gestures_live_predictions.py -m model_dir
   ```
   Provide the path to your best-trained model (`model_dir`).
