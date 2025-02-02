import tensorflow as tf
from argparse import ArgumentParser
from logger_handler import Logger
import numpy as np
import os
import pyautogui
import cv2

# Command Line argument parser.
parser = ArgumentParser(description='Gesture model prediction code')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

# List of required CL arguments.
required_args.add_argument('-m', "--model",
                           help="Gesture model directory",
                           required=True)

# Input arguments
args = parser.parse_args()

# Train directory path
model_dir = args.model

# Load labels
labels = os.listdir('train_videos')
print(labels)
labels.sort()

logger = Logger('log_train', "live_pred.txt").build()

# My model
class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Convolutions
        self.convloution1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last', padding='SAME')
        self.pooling1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool1")
        self.convloution2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv2", data_format='channels_last', padding='SAME')
        self.pooling2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool2")

        # LSTM & Flatten
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(len(labels), activation='softmax', name="output")

    def call(self, x):
        x = self.convloution1(x)
        x = self.pooling1(x)
        x = self.convloution2(x)
        x = self.pooling2(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)

# Initialize and compile the model
new_model = Conv3DModel()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001, epsilon=0.001))

# Load model weights
new_model.load_weights(os.path.join(model_dir, 'final_weights.weights.h5'))

# Resize frames
def resize_image(image):
    image = cv2.resize(image, (64, 64))
    return image

def preprocess_image(img):
    img = resize_image(img)
    return img

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

to_predict = []
classe = ''

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Process the frame
    processed = preprocess_image(frame)
    to_predict.append(processed)

    if len(to_predict) == 6:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = np.expand_dims(frame_to_predict, axis=0)

        # Make prediction
        predict = new_model.predict(frame_to_predict)
        classe = labels[np.argmax(predict)]

        to_predict = []

    # Display the prediction on the frame
    cv2.putText(frame, classe, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('Gesture = ', classe)
