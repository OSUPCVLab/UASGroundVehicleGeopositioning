import cv2
import requests
import argparse
import folium
import torch
import keys
import glob
import re
import numpy as np
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import read_image


model_path = 'car_detection_model.pt'
model_conf = 0.65
mapPath = 'currentLocation.png'
dim = 2496
device = 'cuda' if torch.cuda.is_available() else 'cpu'
adjustmentFactor = 0.000000015
xFactor = 1.3
imgPixelCenter = (320, 320)

opt = {'nms_radius' : 4,
        'keypoint_threshold' : 0.005,
        'max_keypoints' : 2048,
        'superglue' : 'outdoor',
        'sinkhorn_iterations' : 20,
        'match_threshold' : 0.8,
        'resize' : [640, 640],
        'resize_float' : True}

config = {
    'superpoint': {
        'nms_radius': opt['nms_radius'],
        'keypoint_threshold': opt['keypoint_threshold'],
        'max_keypoints': opt['max_keypoints']
    },
    'superglue': {
        'weights': opt['superglue'],
        'sinkhorn_iterations': opt['sinkhorn_iterations'],
        'match_threshold': opt['match_threshold'],
    }
}
matching = Matching(config).eval().to(device)

do_match = True

def inference_with_sahi(img):
    model = Yolov5DetectionModel(
        model_path = model_path,
        confidence_threshold=model_conf,
        device=device
        )
    result = get_sliced_prediction(
            img,
            model,
            slice_height = 832,
            slice_width = 832,
            overlap_height_ratio = 0.2,
            overlap_width_ratio = 0.2)
    return result

def parseArgs():
    parser = argparse.ArgumentParser(description='Collect values to determine GPS position')
    parser.add_argument('--framesDir', type=str, default='residential_01/frames', help='where to get drone images from')
    parser.add_argument('--dataDir', type=str, default='residential_01/paras', help='where to get drone data from for reach frame')

    args = parser.parse_args()
    print('directory with frames:', args.framesDir)
    print('directory with gps data', args.dataDir)
    
    return args

def getParams(filePath):
    
    gpsFile = open(filePath)
    
    line = gpsFile.readline()
    pos = re.findall(r'[-+]?\d*\.?\d+', line)
    pos = [float(i) for i in pos]
    
    line = gpsFile.readline()
    height = int(re.findall(r'[-+]?\d*\.?\d+', line)[0])
    
    return pos, height

def grabNewGoogleMapsImage(pos, fileName):
    response = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=1920x1080&maptype=satellite&key={keys.GOOGLE_API_KEY}')
    with open(fileName, 'wb') as file:
        file.write(response.content)

def googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
    return np.sqrt((lastUpdatedPos[0] - pos[0])**2 + (lastUpdatedPos[1] - pos[1])**2) > 0.0002

def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

def keyPointsWithSuperGlue(srcPath, dstPath):
    _, inp0, _ = read_image(
        srcPath, device, opt['resize'], 0, opt['resize_float'])
    _, inp1, _ = read_image(
        dstPath, device, opt['resize'], 0, opt['resize_float'])

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, _ = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    # WILL NEED LATER
    src = np.float32(mkpts0).reshape(-1,1,2)
    dst = np.float32(mkpts1).reshape(-1,1,2)
    
    return src, dst

def findHomographyUsingNN(srcPath, dstPath):
   
    src, dst = keyPointsWithSuperGlue(srcPath, dstPath)
    
    image0 = cv2.imread(srcPath)
    image1 = cv2.imread(dstPath)

    image0 = cv2.resize(image0, [640, 640])
    image1 = cv2.resize(image1, [640, 640])

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    
    return H

def calculateGPSPosOfObject(center, imgPixelCenter, pos, height):
    x = center[0]
    y = center[1]
    
    xDistFromCenter = (x - imgPixelCenter[0])
    yDistFromCenter = (imgPixelCenter[1] - y)
    
    lat = pos[0]
    long = pos[1]
    
    xcord = xFactor*xDistFromCenter*adjustmentFactor*height + long
    ycord = yDistFromCenter*adjustmentFactor*height + lat
    
    return (ycord, xcord)
    
def main():
    args = parseArgs()
    
    frameFiles = findFiles(args.framesDir)

    firstPosFile = frameFiles[0].replace(args.framesDir, args.dataDir).replace('png', 'txt')
    pos, _  = getParams(firstPosFile)
    grabNewGoogleMapsImage(pos, 'currentLocation.png')
    
    map = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20, 
                tiles='cartodbpositron', width=1280, height=960)
    
    lastUpdatedPos = pos
    for frame in frameFiles:
        
        fileFound = True
        paramPath = None
        pos = None
        height = None
        try:
            paramPath = frame.replace(args.framesDir, args.dataDir).replace('png', 'txt')
            pos, height = getParams(paramPath)
        except:
            fileFound = False
        
        if fileFound:
            if googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
                grabNewGoogleMapsImage(pos, 'currentLocation.png')
            
            droneImage = cv2.imread(frame)
            
            image = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (dim, dim))
            
            results = inference_with_sahi(image)
            
            H = findHomographyUsingNN(frame, mapPath)
            
            c = 0
            
            for result in results.object_prediction_list:
                x1 = result.bbox.minx + ((result.bbox.maxx - result.bbox.minx) / 2)
                y1 = result.bbox.miny + ((result.bbox.maxy - result.bbox.miny) / 2)
                
                x1 = 640 * x1 / 2496
                y1 = 640 * y1 / 2496
                
                x = (x1*H[0,0] + y1*H[0,1] + H[0,2])/(x1*H[2,0] + y1*H[2,1] + H[2,2])
                y = (x1*H[1,0] + y1*H[1,1] + H[1,2])/(x1*H[2,0] + y1*H[2,1] + H[2,2])
                
                center = (int(x), int(y))
                
                dis = calculateGPSPosOfObject(center, imgPixelCenter, pos, height)
                
                folium.CircleMarker(dis, radius=1, color='#0080bb', fill_color='#0080bb').add_to(map)
                
                c += 1
                
            print(f'added {c} cars')
        
        map.save('superglue_multi.html')
        
        
            
    
    

if __name__ == "__main__":
    main()
