import tensorflow as tf
from argparse import ArgumentParser
from datetime import datetime
from dataGenerator import DataGenerator
from logger_handler import Logger
import matplotlib.image as img
import numpy as np
import os
import cv2

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Command Line argument parser.
parser = ArgumentParser(description='Gesture model prediction code')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

# List of required CL arguments.
required_args.add_argument('-m', "--model",
                           help="Gesture model directory",
                           required=True)

required_args.add_argument('-d', "--test_dir",
                           help="Testing directory",
                           required=False)

required_args.add_argument('-i', "--image_file",
                           help="Image file",
                           required=False)

# Input arguments
args = parser.parse_args()

# Train directory path
model_dir = args.model

# Test data directory
test_dir = args.test_dir

date_time = (datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
logger_file = "audio_logs_" + date_time + ".txt"

labels = os.listdir(test_dir)
labels.sort()

generator = DataGenerator()

test_data, test_labels = generator.load_data(test_dir)

logger = Logger('log_train', "Testing Logs" + date_time + ".txt").build()

# My model
class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Convolutions
        self.conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1", data_format='channels_last', padding='SAME')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool1")
        self.conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv2", data_format='channels_last', padding='SAME')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last', name="pool2")

        # LSTM & Flatten
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(len(labels), activation='softmax', name="output")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)

new_model = Conv3DModel()
new_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=0.001))

new_model.load_weights(os.path.join(model_dir, 'final_weights.weights.h5'))

# Resize frames
def resize_image(image):
    image = img.imread(image)
    image = cv2.resize(image, (64, 64))
    return image

def preprocess_image(img):
    img = resize_image(img)
    return img

predict = new_model.predict(test_data)

# Maximum probability in given predictions
y_pred = np.argmax(predict, axis=1)
for i, pred in enumerate(y_pred[:10]):
    logger.info(f"Prediction for test sample {i}: {labels[pred]}")
    logger.info(f"Label for test sample {i}: {labels[test_labels[i]]}")
