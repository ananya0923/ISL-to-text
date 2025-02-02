
import cv2
import os
import pickle
from os.path import join, exists
import argparse
from tqdm import tqdm
import numpy as np

hc = []

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def convert(gesture_folder, target_folder):
    rootPath = os.getcwd()
    majorData = os.path.abspath(target_folder)
    frame_size=(224, 224)

    if not exists(majorData):
        os.makedirs(majorData)

    gesture_folder = os.path.abspath(gesture_folder)

    os.chdir(gesture_folder)
    gestures = os.listdir(os.getcwd())

    print("Source Directory containing gestures: %s" % (gesture_folder))
    print("Destination Directory containing frames: %s\n" % (majorData))

    for gesture in tqdm(gestures, unit='actions', ascii=True):
        gesture_path = os.path.join(gesture_folder, gesture)
        os.chdir(gesture_path)

        gesture_frames_path = os.path.join(majorData, gesture)
        if not os.path.exists(gesture_frames_path):
            os.makedirs(gesture_frames_path)

        videos = os.listdir(os.getcwd())
        videos = [video for video in videos if(os.path.isfile(video))]
        vcount = 0

        for video in tqdm(videos, unit='videos', ascii=True):
            vcount += 1
            name = os.path.abspath(video)

            video_frame = os.path.join(gesture_frames_path, str(vcount))
            if not os.path.exists(video_frame):
                os.makedirs(video_frame)
            os.chdir(video_frame)
            cap = cv2.VideoCapture(name)  # capturing input video
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            os.chdir(video_frame)
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, frame_size)
                frame = frame[:, :, [2, 1, 0]]

                framename = os.path.splitext(video)[0]
                framename = framename + "_frame_" + str(count) + ".jpeg"

                hc.append([join(video_frame, framename), gesture, frameCount])

                if not os.path.exists(framename):
                    lastFrame = frame
                    cv2.imwrite(framename, frame)

                count += 1

            os.chdir(gesture_path)
            cap.release()
            cv2.destroyAllWindows()

    os.chdir(rootPath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Individual Frames from gesture videos.')
    parser.add_argument('gesture_folder', help='Path to folder containing folders of videos of different gestures.')
    parser.add_argument('target_folder', help='Path to folder where extracted frames should be kept.')
    args = parser.parse_args()
    convert(args.gesture_folder, args.target_folder)
