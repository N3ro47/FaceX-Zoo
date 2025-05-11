#!/bin/python3

import argparse
import json
import sys
sys.path.append('.')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import os
import yaml
import cv2
import numpy as np
import torch
import simpleaudio as sa

from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

def play_sound_async(sound_file):
    def _play():
        global is_playing
        with play_lock:
            if not is_playing:
                is_playing = True
                wave_obj = sa.WaveObject.from_wave_file(sound_file)
                play_obj = wave_obj.play()
                play_obj.wait_done()
                is_playing = False

    if(sound_file != ""):
        threading.Thread(target=_play, daemon=True).start()

def hex_to_rgb(hex_color):
    """Convert hex color string to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

parser = argparse.ArgumentParser(description='Script with config and mode options')

# Add arguments
parser.add_argument('--load', type=str, required=True,
                    help='Path to the configuration file')
parser.add_argument('-m', '--mode', type=str, required=True,
                    choices=['recognition', 'image_taker'],
                    help='Operation mode: image_taker, recognition')

# Parse arguments
args = parser.parse_args()

mode = args.mode

data = None

with open(args.load, 'r') as f:
    data = json.load(f)

print(data)
cap=None
name="Image taker"

if (data['input_source']['type'] == "Video File"):
    cap = cv2.VideoCapture(data.input_source.video_path)
elif (data['input_source']['type'] == "Live Camera"):
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)

image_counter = 0
model_path = 'models'

scene = 'non-mask'
model_category = 'face_detection'
model_name =  model_conf[scene][model_category]

if not os.path.isdir("tmp"):
    os.mkdir("tmp")

logger.info('Start to load the face detection model...')

try:
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
except Exception as e:
    logger.error('Failed to parse model configuration file!')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Successfully parsed the model configuration file model_meta.json!')

try:
    model, cfg = faceDetModelLoader.load_model()
except Exception as e:
    logger.error('Model loading failed!')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Successfully loaded the face detection model!')


faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)

if (args.mode == 'image_taker'):

    img_counter=0
    ret, frame = cap.read()
    height, width, _ = frame.shape

    while True:
        ret, frame = cap.read()
        org_frame = frame.copy()

        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.putText(frame, "Image counter: " + str(img_counter), (0,height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Capture mode', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.imwrite("tmp/capt_img" + str(img_counter) + ".png", org_frame)
            img_counter += 1
elif (args.mode == 'recognition'):
    in_files = [f for f in os.listdir("tmp")]

    fil_counter=0
    fil_cap=len(in_files)
    cur_img = cv2.imread("tmp/" + str(in_files[fil_counter]))

    if cur_img is None:
        logger.error("Can't read images")
        sys.exit(-1)

    if (fil_cap == 0):
        logger.error("No images in tmp folder")
        sys.exit(-1)

    model_category = 'face_alignment'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face landmark model...')
    try:
        faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
        model, cfg = faceAlignModelLoader.load_model()
        faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face landmark model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    # face recognition model setting.
    model_category = 'face_recognition'
    model_name =  model_conf[scene][model_category]
    logger.info('Start to load the face recognition model...')
    try:
        faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
        model, cfg = faceRecModelLoader.load_model()
        faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
    except Exception as e:
        logger.error('Failed to load face recognition model.')
        logger.error(e)
        sys.exit(-1)
    else:
        logger.info('Success!')

    face_cropper = FaceRecImageCropper()

    while True:
        ret, frame = cap.read()
        comb = np.concatenate((frame, cur_img), axis=1)

        if not ret:
            print("Error: Failed to capture image.")
            break

        try:
            dets = faceDetModelHandler.inference_on_image(frame)
        except Exception as e:
           logger.error('Face detection failed!')
           logger.error(e)
           sys.exit(-1)

        bboxs = dets

        score=0
        face_nums=0
        try:
            dets = faceDetModelHandler.inference_on_image(comb)
            face_nums = dets.shape[0]
            if face_nums != 2:
                logger.info('Input image should contain two faces to compute similarity!')
            feature_list = []
            for i in range(face_nums):
                landmarks = faceAlignModelHandler.inference_on_image(comb, dets[i])
                landmarks_list = []
                for (x, y) in landmarks.astype(np.int32):
                    landmarks_list.extend((x, y))
                cropped_image = face_cropper.crop_image_by_mat(comb, landmarks_list)
                feature = faceRecModelHandler.inference_on_image(cropped_image)
                feature_list.append(feature)
            score = np.dot(feature_list[0], feature_list[1])
        except Exception as e:
            logger.error('Pipeline failed!')
            logger.error(e)
            sys.exit(-1)

        if (score < data['recognition']['similarity_threshold']):
            if (data['alerts']['sound']['enable_match'] == True):
                play_sound_async(data['alerts']['sound']['match_sound'])
            if (data['visualization']['show_boxes'] == True):
                for box in bboxs:
                    _box = list(map(int, box))
            cv2.rectangle(comb, (_box[0], _box[1]), (_box[2], _box[3]), hex_to_rgb(data['visualization']['no_match_color']), 2)
            if (data['visualization']['show_similarity'] == True):
                cv2.putText(comb, "Similarity: {:0.2f}%".format(score*100), (_box[0] + 10, _box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,hex_to_rgb(data['visualization']['no_match_color']), 1, cv2.LINE_AA)
            if (data['visualization']['show_confidence'] == True):
                cv2.putText(comb, "Threshold: {:0.2f}%".format(data['recognition']['similarity_threshold']*100), (_box[0] + 10, _box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,hex_to_rgb(data['visualization']['no_match_color']), 1, cv2.LINE_AA)
        else:
            if (data['alerts']['sound']['enable_no_match'] == True):
                play_sound_async(data['alerts']['sound']['no_match_sound'])
            if (data['visualization']['show_boxes'] == True):
                for box in bboxs:
                    _box = list(map(int, box))
            cv2.rectangle(comb, (_box[0], _box[1]), (_box[2], _box[3]), hex_to_rgb(data['visualization']['match_color']), 2)
            if (data['visualization']['show_similarity'] == True):
                cv2.putText(comb, "Similarity: {:0.2f}%".format(score*100), (_box[0] + 10, _box[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,hex_to_rgb(data['visualization']['match_color']), 1, cv2.LINE_AA)
            if (data['visualization']['show_confidence'] == True):
                cv2.putText(comb, "Threshold: {:0.2f}%".format(data['recognition']['similarity_threshold']*100), (_box[0] + 10, _box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,hex_to_rgb(data['visualization']['match_color']), 1, cv2.LINE_AA)

        if (data['alerts']['visual']['show_text'] == True):
            if (score < data['recognition']['similarity_threshold']):
                cv2.putText(comb, data['alerts']['visual']['no_match_text'], (_box[0] + 10, _box[3] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,hex_to_rgb(data['visualization']['no_match_color']), 1, cv2.LINE_AA)
            else:
                cv2.putText(comb, data['alerts']['visual']['match_text'], (_box[0] + 10, _box[3] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,hex_to_rgb(data['visualization']['match_color']), 1, cv2.LINE_AA)


        cv2.imshow('Recognition mode', comb)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            fil_counter-=1
            fil_counter%=fil_cap
            cur_img = cv2.imread("tmp/" + str(in_files[fil_counter]))
        elif key == ord('n'):
            fil_counter+=1
            fil_counter%=fil_cap
            cur_img = cv2.imread("tmp/" + str(in_files[fil_counter]))

    cap.release()
    cv2.destroyAllWindows()
