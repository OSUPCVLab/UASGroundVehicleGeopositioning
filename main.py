import math
import os
import cv2
import requests
import folium
import torch
import keys
import glob
import re
import numpy as np
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
import pickle
from deep_sort_realtime.deepsort_tracker import DeepSort
import argsForParameters
import transformations
from googleMapsApi import grabNewGoogleMapsImage, latAtZoom20, googleMapsImageNeedsToUpdate

model_path = 'car_detection_model.pt'
model_conf = 0.65
dimOfImage = 1664

imgPixelCenter = (transformations.dimOfResizedImage / 2, transformations.dimOfResizedImage / 2)

mapPath = 'currentLocation.png'
roadMaskPath = 'roadMask.png'
buildingMaskPath = 'buildingMask.png'
detectionsMaskPath = 'detectionsMask.png'

# "mobilenet",
#     "torchreid",
#     "clip_RN50",
#     "clip_RN101",
#     "clip_RN50x4",
#     "clip_RN50x16",
#     "clip_ViT-B/32",
#     "clip_ViT-B/16",

tracker = DeepSort(max_age=50, embedder="clip_RN50", embedder_gpu=False)

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

#this grabs the meta data for a single frame
def getParams(filePath):
    gpsFile = open(filePath)
    
    line = gpsFile.readline()
    pos = re.findall(r'[-+]?\d*\.?\d+', line)
    pos = [float(i) for i in pos]
    
    line = gpsFile.readline()
    # height = int(re.findall(r'[-+]?\d*\.?\d+', line)[0])
    
    line = gpsFile.readline()
    rot = int(re.findall(r'[-+]?\d*\.?\d+', line)[0])
    
    return pos, rot

# this generates a mask for detection results, lets us filter out detection results
def createDetectionsMask(detectionsMaskPath, detectionImageShape, originalImageShape, detections, resize, rotation):
    mask = np.ones(detectionImageShape)
    mask.fill(255)
    for detection in detections:
        bbox = detection.bbox
        mask[int(bbox.miny):int(bbox.maxy), int(bbox.minx):int(bbox.maxx)] = 0
    
    mask = cv2.resize(mask, (originalImageShape[1], originalImageShape[0]))
    
    if rotation != 0:
        mask, _ = transformations.rotateImage(mask, rotation)
    
    w_new, h_new = resize[0], resize[1]
    h_old, w_old = mask.shape
    h_new = int(h_old * w_new / w_old)
    mask = cv2.resize(mask.astype('float32'), (w_new, h_new))
    
    cv2.imwrite(detectionsMaskPath, mask)

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

def getDetections(cachedDetectionsPath, frameName, image):
    results = None
    if os.path.isfile(cachedDetectionsPath):
        detectionsFile = open(cachedDetectionsPath, 'rb')     
        results = pickle.load(detectionsFile)
        detectionsFile.close()
    
        print(f'for {frameName} cached detections found')
    
    else:
        results = inference_with_sahi(image)
        
        detectionsFile = open(cachedDetectionsPath, 'ab')
        pickle.dump(results, detectionsFile)                     
        detectionsFile.close()
        
        print(f'for {frameName} cached detections not found, new cached detections saved')
    
    return results

def getTracking(cachedTrackingPath, results, droneImage, frameName):
    tracks = None
    
    if os.path.isfile(cachedTrackingPath):
        trackingFile = open(cachedTrackingPath, 'rb')     
        tracks = pickle.load(trackingFile)
        trackingFile.close()
    
        print(f'for {frameName} cached tracking found')
    
    else:
        originalImg_h, originalImg_w, _ = droneImage.shape
        resultsList = []
        # loop over the detections
        for result in results.object_prediction_list:
            xmin, ymin, xmax, ymax = int(result.bbox.minx), int(result.bbox.miny), int(result.bbox.maxx), int(result.bbox.maxy)
            xmin, ymin, xmax, ymax = convertPredictionsToImageSpace(xmin, ymin, xmax, ymax, originalImg_w, originalImg_h)
            resultsList.append([[xmin, ymin, xmax - xmin, ymax - ymin], str(result.score.value), result.category])

        tracks = tracker.update_tracks(resultsList, frame=droneImage)
        
        trackingFile = open(cachedTrackingPath, 'ab')
        pickle.dump(tracks, trackingFile)                     
        trackingFile.close()
        
        print(f'for {frameName} cached tracking not found, new cached tracking saved')
    
    return tracks

#this calculates the GPS position using pixel positions and the expected image size
def calculateGPSPosOfObject(center, imgPixelCenter, pos):
    x = center[0]
    y = center[1]
    
    xDistFromCenter = (x - imgPixelCenter[0])
    yDistFromCenter = (imgPixelCenter[1] - y)
    
    lat = pos[0]
    long = pos[1]
    
    long_factor = 1 / math.cos(lat * math.pi/180)
    
    xcord = xDistFromCenter*latAtZoom20*long_factor/transformations.dimOfResizedImage + long
    ycord = yDistFromCenter*latAtZoom20/transformations.dimOfResizedImage + lat
    
    return (ycord, xcord)
    
def getPixelPositionInMapImage(minx, miny, maxx, maxy, transform, usingHomography):
    xDet = minx + ((maxx - minx) / 2)
    yDet = miny + ((maxy - miny) / 2)
    
    positionInOriginal = np.array([xDet, yDet, 1]).T
    
    x, y = None, None
    positionInMap = np.matmul(transform, positionInOriginal)
    if usingHomography:
        # make x, y homogeneous
        x, y = positionInMap[0] / positionInMap[2], positionInMap[1] / positionInMap[2]
    else:
        x, y = positionInMap[0], positionInMap[1]
    
    return (int(x), int(y))
    
def convertPredictionsToImageSpace(minx, miny, maxx, maxy, originalImg_w, originalImg_h):
    minx = originalImg_w * minx / dimOfImage    
    maxx = originalImg_w * maxx / dimOfImage    
    miny = originalImg_h * miny / dimOfImage    
    maxy = originalImg_h * maxy / dimOfImage
    return minx, miny, maxx, maxy
    
def getRandomColors(numColors=500):
    rng = np.random.default_rng()
    colorValues = rng.choice(16777215, numColors, replace=False)
    return [hex(color).replace('0x', '#').ljust(7, '0') for color in colorValues]
    
def drawPositionsOnMap(setsOfFrames: dict, colors, carMap, saveFileName):
    for setNum in setsOfFrames:
        for track_id in setsOfFrames[setNum]:
            
            positions = setsOfFrames[setNum][track_id]
            
            if len(positions) > 1:
                for pos in positions:
                    folium.CircleMarker(pos, radius=1, color=colors[track_id], fill_color=colors[track_id]).add_to(carMap)

def fileNames(args):
    carsText = 'filtering_cars' if args.filterCars else 'not_filtering_cars'
    roadsText = 'filtering_roads' if args.filterRoads else 'not_filtering_roads'
    buildingsText = 'filter_buildings' if args.filterBuildings else 'not_filter_buildings'
    SGText = 'SuperGlue' if args.SuperGlue else 'not_SuperGlue'
    LoFTRText = 'LoFTR' if args.LoFTR else 'not_LoFTR'
    HomographyText = 'Homography' if args.homography else 'Affine2D'
    
    base = f'run_{carsText}_{roadsText}_{buildingsText}_{SGText}_{LoFTRText}_{HomographyText}'
    saveHTMLName = f'htmlResults/{base}.html'
    savePickleName = f'GPSResults/{base}'
    
    return saveHTMLName, savePickleName

def main():
    args = argsForParameters.parseArgs(verbose=True)
    
    saveHTMLName, savePickleName = fileNames(args)
    print('saving to ', saveHTMLName, 'and', savePickleName)
    
    frameFiles = findFiles(args.framesDir)

    firstPosFile = frameFiles[0].replace(args.framesDir, args.dataDir).replace('png', 'txt')
    pos, _  = getParams(firstPosFile)
    grabNewGoogleMapsImage(pos, mapPath, roadMaskPath, buildingMaskPath)
    
    carMap = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20, 
                tiles='cartodbpositron', width=1280, height=960)
    
    lastUpdatedPos = pos
    colors = getRandomColors()
    setIdx = 0
    setsOfFrames = dict()
    setsOfFrames[0] = dict()
    
    for i, frameName in enumerate(frameFiles):
        
        fileFound = True
        paramPath = None
        pos = None
        rot = None
        try:
            paramPath = frameName.replace(args.framesDir, args.dataDir).replace('png', 'txt')
            pos, rot = getParams(paramPath)
        except:
            fileFound = False
            print(paramPath, 'not found')
        
        if fileFound:
            if googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
                setIdx += 1
                
                print('grabbing new google maps image')
                grabNewGoogleMapsImage(pos, mapPath, roadMaskPath, buildingMaskPath)
                lastUpdatedPos = pos
                
                colors = getRandomColors()
                
                setsOfFrames[setIdx] = dict()
                
                tracker.delete_all_tracks()
            
            droneImage = cv2.imread(frameName)
            imageForDetection = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
            imageForDetection = cv2.resize(imageForDetection, (dimOfImage, dimOfImage))
            
            numCars = 0
            
            cachedDetectionsPath = frameName.replace(args.framesDir, args.cacheDetDir).replace('.png', '')
            results = getDetections(cachedDetectionsPath, frameName, imageForDetection)
            createDetectionsMask(detectionsMaskPath, imageForDetection.shape[0:2], droneImage.shape[0:2], results.object_prediction_list, transformations.opt['resize'], rot)
            
            transform = transformations.findTransform(frameName, mapPath, roadMaskPath, buildingMaskPath, detectionsMaskPath, rot, args)
            
            cachedTrackingPath = frameName.replace(args.framesDir, args.cacheTrackDir).replace('.png', '')
            tracks = getTracking(cachedTrackingPath, results, droneImage, frameName)
            
            for track in tracks:
                track_id = int(track.track_id)
                
                if track.is_tentative() and track_id not in setsOfFrames[setIdx]:
                    setsOfFrames[setIdx][track_id] = []
                
                if (track.is_confirmed() or track.is_tentative()) and track.time_since_update < 1:
                    ltrb = track.to_ltrb()
                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                    
                    center = getPixelPositionInMapImage(xmin, ymin, xmax, ymax, transform, args.homography)
                    dis = calculateGPSPosOfObject(center, imgPixelCenter, pos)
                    
                    setsOfFrames[setIdx][track_id].append(dis)
                    
                    numCars += 1
                
            print(f'found {numCars} cars in {frameName}')
        
    drawPositionsOnMap(setsOfFrames, colors, carMap, saveHTMLName)
    
    detections = open(savePickleName, 'ab')
    pickle.dump(setsOfFrames, detections)                     
    detections.close()
    
    carMap.save(saveHTMLName)

if __name__ == "__main__":
    main()
