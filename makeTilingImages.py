import math
import cv2
import requests
import argparse
import folium
import torch
import keys
import glob
import re
import numpy as np
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
from SuperGluePretrainedNetwork.models.matching import Matching
import time
from kornia.feature import LoFTR
import pickle

model_path = 'car_detection_model.pt'
model_conf = 0.65
dimOfImage = 1664
dimOfResizedImage = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imgPixelCenter = (dimOfResizedImage / 2, dimOfResizedImage / 2)

latAtZoom20 = 40.01298216829920000 - 40.01232546913910000


model = Yolov8DetectionModel(
        model_path = model_path,
        confidence_threshold=model_conf,
        device= 'mps' if torch.has_mps else 'cpu'
        )

#this runs sliced window inference, this provides better results for ariel drone footage
def inference_with_sahi(img):
    result = get_sliced_prediction(
            img,
            model,
            slice_height = 832,
            slice_width = 832,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2)
    return result

#this lets us determine where to grab the images and meta data
def parseArgs():
    parser = argparse.ArgumentParser(description='Collect values to determine GPS position')
    parser.add_argument('--framesDir', type=str, default='sampleData/images', help='where to get drone images from')
    parser.add_argument('--dataDir', type=str, default='sampleData/params', help='where to get drone data from for each frame')
    parser.add_argument('--cacheDir', type=str, default='sampleData/cachedDetections', help='where to cache detections for each frame')
    parser.add_argument('--filterCars', type=bool, default=True, help='whether or not to filter cars')
    parser.add_argument('--filterRoads', type=bool, default=True, help='whether or not to filter roads')
    parser.add_argument('--SuperGlue', type=bool, default=False, help='True for SuperGlue, False for LoFTR')

    args = parser.parse_args()
    print('directory with frames: ', args.framesDir)
    print('directory with gps data: ', args.dataDir)
    print('directory with cachedDetections: ', args.cacheDir)
    print('filtering cars: ', args.filterCars)
    print('filtering roads: ', args.filterRoads)
    print('using superglue: ', args.SuperGlue)
    
    return args

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

def drawPredictionsOnImage(image, droneImage, results):
    scaleX = droneImage.shape[1]/ image.shape[1] 
    scaleY = droneImage.shape[0]/ image.shape[0] 
    for result in results.object_prediction_list:
        p1 = (int(result.bbox.minx * scaleX), int(result.bbox.miny * scaleY))
        p2 = (int(result.bbox.maxx * scaleX), int(result.bbox.maxy * scaleY))
        
        cv2.rectangle(droneImage, p1, p2, (0,0,255), 5)

def main():
    args = parseArgs()
    frameFiles = findFiles(args.framesDir)

    np.random.shuffle(frameFiles)
    for i in range(4):
        frame = frameFiles[i]
            
        droneImage = cv2.imread(frame)
        
        image = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (dimOfImage, dimOfImage))
        
        # Its important to use binary mode
        cachedDetectionsPath = frame.replace(args.framesDir, args.cacheDir).replace('.png', '')
        results = None
        try:
            detectionsfile = open(cachedDetectionsPath, 'rb')     
            results = pickle.load(detectionsfile)
            detectionsfile.close()
            print(f'for {frame} cached detections found')
        except:
            results = inference_with_sahi(image)
            
            dbfile = open(cachedDetectionsPath, 'ab')
            pickle.dump(results, dbfile)                     
            dbfile.close()
            
            print(f'for {frame} cached detections not found, new cached detections saved')
        
        drawPredictionsOnImage(image, droneImage, results)
        cv2.imwrite(f'detections_{i}.png',droneImage)
        
    # results1 = inference_with_sahi(image1)
    # results2 = inference_with_sahi(image2)
    # results3 = inference_with_sahi(image3)
    # results4 = inference_with_sahi(image4)
    
    
    # drawPredictionsOnImage(image1, results1)
    # drawPredictionsOnImage(image2, results2)
    # drawPredictionsOnImage(image3, results3)
    # drawPredictionsOnImage(image4, results4)
    
    # cv2.imwrite('image1.png', cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('image2.png', cv2.cvtColor(image2, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('image3.png', cv2.cvtColor(image3, cv2.COLOR_RGB2BGR))
    # cv2.imwrite('image4.png', cv2.cvtColor(image4, cv2.COLOR_RGB2BGR))
    
    
    

if __name__ == "__main__":
    main()
