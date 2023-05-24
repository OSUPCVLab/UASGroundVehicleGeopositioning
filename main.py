import math
import os
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
from kornia.feature import LoFTR
import pickle
from deep_sort_realtime.deepsort_tracker import DeepSort

model_path = 'car_detection_model.pt'
model_conf = 0.65
dimOfImage = 1664
dimOfResizedImage = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imgPixelCenter = (dimOfResizedImage / 2, dimOfResizedImage / 2)

latAtZoom20 = 40.01298216829920000 - 40.01232546913910000

mapPath = 'currentLocation.png'
mapsMaskPath = 'mapMask.png'
detectionsMaskPath = 'detectionsMask.png'

opt = {'nms_radius' : 4,
        'keypoint_threshold' : 0.005,
        'max_keypoints' : 2048,
        'superglue' : 'outdoor',
        'sinkhorn_iterations' : 20,
        'match_threshold' : 0.6,
        'resize' : [dimOfResizedImage, dimOfResizedImage],
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
SuperGlueMatcher = Matching(config).eval().to(device)
LoFTRMatcher = LoFTR(pretrained="outdoor")

tracker = DeepSort(max_age=50)

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
    # parser.add_argument('--framesDir', type=str, default='sampleTrackingTestData/images', help='where to get drone images from')
    # parser.add_argument('--dataDir', type=str, default='sampleTrackingTestData/params', help='where to get drone data from for each frame')
    # parser.add_argument('--cacheDir', type=str, default='sampleTrackingTestData/cachedDetections', help='where to cache detections for each frame')
    parser.add_argument('--framesDir', type=str, default='sampleData/images', help='where to get drone images from')
    parser.add_argument('--dataDir', type=str, default='sampleData/params', help='where to get drone data from for each frame')
    parser.add_argument('--cacheDir', type=str, default='sampleData/cachedDetections', help='where to cache detections for each frame')
    parser.add_argument('--filterCars', type=bool, default=True, help='whether or not to filter cars')
    parser.add_argument('--filterRoads', type=bool, default=True, help='whether or not to filter roads')
    parser.add_argument('--SuperGlue', type=bool, default=True, help='True for SuperGlue, False for LoFTR')

    args = parser.parse_args()
    print('directory with frames: ', args.framesDir)
    print('directory with gps data: ', args.dataDir)
    print('directory with cachedDetections: ', args.cacheDir)
    print('filtering cars: ', args.filterCars)
    print('filtering roads: ', args.filterRoads)
    print('using superglue: ', args.SuperGlue)
    
    return args

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

#this grabs new google maps images when needed
def grabNewGoogleMapsImage(pos, fileName, maskPath):
    try:
        satMapRequest = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=640x640&maptype=satellite&key={keys.GOOGLE_API_KEY}', stream=True).raw
        terrainMapRequest = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=640x640&maptype=terrain&key={keys.GOOGLE_API_KEY}', stream=True).raw
        
        if satMapRequest.status == 200 and terrainMapRequest.status == 200:
                
            #turn the responses into images
            satMapImage = np.asarray(bytearray(satMapRequest.read()), dtype="uint8")
            satMapImage = cv2.imdecode(satMapImage, cv2.IMREAD_COLOR)
            terrainMapImage = np.asarray(bytearray(terrainMapRequest.read()), dtype="uint8")
            terrainMapImage = cv2.imdecode(terrainMapImage, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(terrainMapImage, cv2.COLOR_BGR2GRAY)

            # apply thresholding to convert grayscale to binary image
            _,thresh = cv2.threshold(gray,253,255,0)
            
            kernel = np.ones((25, 25), np.uint8)
            mask = cv2.dilate(thresh, kernel, iterations=4)
            
            cv2.imwrite(maskPath, mask)
            
            cv2.imwrite(fileName, satMapImage)
            
            print('Successfully updated maps reference image')
        else:
            print('\nERROR(s):',satMapRequest.data, '\n', terrainMapRequest.data, '\n')
            exit()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

# this generates a mask for detection results, lets us filter out detection results
def createDetectionsMask(detectionsMaskPath, detectionImageShape, originalImageShape, detections, resize, rotation):
    mask = np.ones(detectionImageShape)
    mask.fill(255)
    for detection in detections:
        bbox = detection.bbox
        mask[int(bbox.miny):int(bbox.maxy), int(bbox.minx):int(bbox.maxx)] = 0
    
    mask = cv2.resize(mask, (originalImageShape[1], originalImageShape[0]))
    
    if rotation != 0:
        mask, _ = rotateImage(mask, rotation)
    
    w_new, h_new = resize[0], resize[1]
    h_old, w_old = mask.shape
    h_new = int(h_old * w_new / w_old)
    mask = cv2.resize(mask.astype('float32'), (w_new, h_new))
    
    cv2.imwrite(detectionsMaskPath, mask)

def rotateImage(image, angle):
    height, width = image.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, -angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_image = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h))
    return rotated_image, rotation_mat

#this checks to see if the google maps image needs to be updated
def googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
    return np.sqrt((lastUpdatedPos[0] - pos[0])**2 + (lastUpdatedPos[1] - pos[1])**2) > 0.1 * latAtZoom20

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

#this grabs the image as a tensor, it also rotates it if needed
def getImageAsTensor(path, device, resize, rotation):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    rot_mat = np.array([[1.0, 0.0, 0.0],
                        [-0.0, 1.0, 0.0]])
    
    if rotation != 0:
        image, rot_mat = rotateImage(image, rotation)
    
    w_new, h_new = resize[0], resize[1]
    h_old, w_old = image.shape
    h_new = int(h_old * w_new / w_old)
    image = cv2.resize(image.astype('float32'), (w_new, h_new))

    imageAsTensor = torch.from_numpy(image/255.0).float()[None, None].to(device)
    
    return image.astype('uint8'), imageAsTensor, rot_mat, h_old, w_old

# this takes a mask, the matching points, and the points to which the mask will be applied to
def applyMaskToPoints(maskPath, matchingPoints, pointsToApplyMaskTo):
    # read in the mask
    mapsMask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    
    # make a copy of the key points
    matchingPointsCopy = np.copy(matchingPoints)
    pointsToApplyMaskToCopy = np.copy(pointsToApplyMaskTo)
    
    # grab the indexes of the points we want to filter out using the mask
    indexes = pointsToApplyMaskToCopy.astype(int)
    
    # apply the mask
    mkpts0 = matchingPointsCopy[mapsMask[indexes[:,1], indexes[:,0]] != 0]
    mkpts1 = pointsToApplyMaskToCopy[mapsMask[indexes[:,1], indexes[:,0]] != 0]
    
    return mkpts0, mkpts1

#this grabs the key points using superglue, it returns a list of matching points
def keyPointsWithSuperGlue(batch):
    # Perform the matching.
    pred = SuperGlueMatcher(batch)
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
    matches, _ = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    
    # return src, dst, altpoints
    return mkpts0, mkpts1

def keyPointsWithLoFTR(batch):
    # Inference with LoFTR and get prediction
    with torch.no_grad():
        LoFTRMatcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

    mkpts1 = [mkpts1[i] for i in range(len(mconf)) if mconf[i] > 0.6]
    mkpts0 = [mkpts0[i] for i in range(len(mconf)) if mconf[i] > 0.6]
    
    return mkpts0, mkpts1

def getUnrotationMat(rot_mat):
    padded = np.array([rot_mat[0,:], rot_mat[1,:], [0.0,0.0,1.0]])
    inv_rot = np.linalg.inv(padded.astype(np.float32)).astype(np.float32)[0:2,:]
    return inv_rot

#this finds the key points and then calculates the homography
def findHomographyUsingNN(srcPath, dstPath, mapsMaskPath, detectionsMaskPath, rot, args):
    image0, img0, rot_0, h_old, w_old = getImageAsTensor(
        srcPath, device, opt['resize'], rot)
    image1, img1, _, _, _ = getImageAsTensor(
        dstPath, device, opt['resize'], 0)
   
    batch = {'image0': img0, 'image1': img1}
   
    if args.SuperGlue:
        mkpts0, mkpts1 = keyPointsWithSuperGlue(batch)
    else:
        mkpts0, mkpts1 = keyPointsWithLoFTR(batch)
    
    # mkpts0SG, mkpts1SG = keyPointsWithSuperGlue(batch)
    # mkpts0LFTR, mkpts1LFTR = keyPointsWithLoFTR(batch)
    
    # mkpts0 = np.concatenate((mkpts0SG, mkpts0LFTR))
    # mkpts1 = np.concatenate((mkpts1SG, mkpts1LFTR))
    
    # apply masks to matching points
    if args.filterRoads:
        mkpts0, mkpts1 = applyMaskToPoints(mapsMaskPath, mkpts0, mkpts1)
    if args.filterCars:
        mkpts1, mkpts0 = applyMaskToPoints(detectionsMaskPath, mkpts1, mkpts0)
    
    # src = np.float32(mkpts0).reshape(-1,1,2)
    # for p in src:
    #     cv2.circle(image0, (int(p[0,0]), int(p[0,1])), 3, (255,5,255), 3)
    # cv2.imshow('original',image0)
    
    # image0_original = cv2.imread(srcPath)
    
    # unscale points
    mkpts0[:,0] = mkpts0[:,0] * w_old / opt['resize'][0]
    mkpts0[:,1] = mkpts0[:,1] * w_old / opt['resize'][1]
    
    # unrotate points
    unrot_mat = getUnrotationMat(rot_0)
    mkpts0WithOnes = np.append(mkpts0,np.ones([mkpts0.shape[0],1]),1)
    mkpts0 = np.matmul(unrot_mat, mkpts0WithOnes.T).T
    
    src = np.float32(mkpts0).reshape(-1,1,2)
    dst = np.float32(mkpts1).reshape(-1,1,2)
    
    # for p in src:
    #     cv2.circle(image0_original, (int(p[0,0]), int(p[0,1])), 3, (255,5,255), 3)
    # cv2.imshow('new',image0_original)
    # cv2.waitKey()
        
    # mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    # image1[mask < 100] = (0,0,0)
    # for p in dst:
    #     cv2.circle(image1, (int(p[0,0]), int(p[0,1])), 3, (5,255,255), 3)
    # for p in src:
    #     cv2.circle(image0, (int(p[0,0]), int(p[0,1])), 3, (255,5,255), 3)
    
    
    # cv2.imshow('dots',image1)
    # cv2.imshow('dots2',image0)
    # cv2.waitKey()

    # image0 = cv2.resize(image0, [dimOfResizedImage, dimOfResizedImage])
    # image1 = cv2.resize(image1, [dimOfResizedImage, dimOfResizedImage])

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    
    return H

def getDetections(cachedDetectionsPath, frame, image):
    results = None
    if os.path.isfile(cachedDetectionsPath):
        detectionsfile = open(cachedDetectionsPath, 'rb')     
        results = pickle.load(detectionsfile)
        detectionsfile.close()
    
        print(f'for {frame} cached detections found')
    
    else:
        results = inference_with_sahi(image)
        
        detectionsfile = open(cachedDetectionsPath, 'ab')
        pickle.dump(results, detectionsfile)                     
        detectionsfile.close()
        
        print(f'for {frame} cached detections not found, new cached detections saved')
    
    return results

#this calculates the GPS position using pixel positions and the expected image size
def calculateGPSPosOfObject(center, imgPixelCenter, pos):
    x = center[0]
    y = center[1]
    
    xDistFromCenter = (x - imgPixelCenter[0])
    yDistFromCenter = (imgPixelCenter[1] - y)
    
    lat = pos[0]
    long = pos[1]
    
    long_factor = 1 / math.cos(lat * math.pi/180)
    
    xcord = xDistFromCenter*latAtZoom20*long_factor/dimOfResizedImage + long
    ycord = yDistFromCenter*latAtZoom20/dimOfResizedImage + lat
    
    return (ycord, xcord)
    
def getPixelPositionInMapImage(minx, miny, maxx, maxy, H):
    xDet = minx + ((maxx - minx) / 2)
    yDet = miny + ((maxy - miny) / 2)
    
    positionInOriginal = np.array([xDet, yDet, 1]).T
    positionInMap = np.matmul(H, positionInOriginal)
    
    # make x, y homogeneous
    x, y = positionInMap[0] / positionInMap[2], positionInMap[1] / positionInMap[2]
    
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
    return [hex(color).replace('0x', '#') for color in colorValues]
    
def drawPositionsOnMap(setsOfFrames: dict, colors, carMap, saveFileName):
    for setNum in setsOfFrames:
        for track_id in setsOfFrames[setNum]:
            
            positions = setsOfFrames[setNum][track_id]
            
            if len(positions) > 1:
                for pos in positions:
                    folium.CircleMarker(pos, radius=1, color=colors[track_id], fill_color=colors[track_id]).add_to(carMap)

def main():
    args = parseArgs()
    carsText = 'filtering_cars' if args.filterCars else 'not_filtering_cars'
    roadsText = 'filtering_roads' if args.filterRoads else 'not_filtering_roads'
    featureMatchingModelText = 'SuperGlue' if args.SuperGlue else 'LoFTR'
    # featureMatchingModelText = 'SuperGlue_LoFTR'
    saveFileName = f'run_{carsText}_{roadsText}_{featureMatchingModelText}.html'
    print('saving results to: ', saveFileName)
    
    frameFiles = findFiles(args.framesDir)

    firstPosFile = frameFiles[0].replace(args.framesDir, args.dataDir).replace('png', 'txt')
    pos, _  = getParams(firstPosFile)
    grabNewGoogleMapsImage(pos, mapPath, mapsMaskPath)
    
    carMap = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20, 
                tiles='cartodbpositron', width=1280, height=960)
    
    lastUpdatedPos = pos
    colors = getRandomColors()
    setIdx = 0
    setsOfFrames = dict()
    setsOfFrames[0] = dict()
    
    for i, frame in enumerate(frameFiles):
        
        fileFound = True
        paramPath = None
        pos = None
        rot = None
        try:
            paramPath = frame.replace(args.framesDir, args.dataDir).replace('png', 'txt')
            pos, rot = getParams(paramPath)
        except:
            fileFound = False
            print(paramPath, 'not found')
        
        if fileFound:
            if googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
                setIdx += 1
                
                print('grabbing new google maps image')
                grabNewGoogleMapsImage(pos, mapPath, mapsMaskPath)
                lastUpdatedPos = pos
                
                colors = getRandomColors()
                
                setsOfFrames[setIdx] = dict()
                
                tracker.delete_all_tracks()
            
            droneImage = cv2.imread(frame)
            
            image = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
            originalImg_h, originalImg_w, _ = image.shape
            image = cv2.resize(image, (dimOfImage, dimOfImage))
            
            # Its important to use binary mode
            cachedDetectionsPath = frame.replace(args.framesDir, args.cacheDir).replace('.png', '')
            results = getDetections(cachedDetectionsPath, frame, image)
            
            createDetectionsMask(detectionsMaskPath, image.shape[0:2], droneImage.shape[0:2], results.object_prediction_list, opt['resize'], rot)
            
            H = findHomographyUsingNN(frame, mapPath, mapsMaskPath, detectionsMaskPath, rot, args)
            
            numCars = 0
            
            resultsList = []
            # loop over the detections
            for result in results.object_prediction_list:
                xmin, ymin, xmax, ymax = int(result.bbox.minx), int(result.bbox.miny), int(result.bbox.maxx), int(result.bbox.maxy)
                xmin, ymin, xmax, ymax = convertPredictionsToImageSpace(xmin, ymin, xmax, ymax, originalImg_w, originalImg_h)
                resultsList.append([[xmin, ymin, xmax - xmin, ymax - ymin], str(result.score.value), result.category])
            
            tracks = tracker.update_tracks(resultsList, frame=droneImage)
            # loop over the tracks
            for track in tracks:
                track_id = int(track.track_id)
                if track.is_tentative() and track_id not in setsOfFrames[setIdx]:
                    setsOfFrames[setIdx][track_id] = []
                # if the track is not confirmed, ignore it
                if (track.is_confirmed() or track.is_tentative()) and track.time_since_update < 1:
                    ltrb = track.to_ltrb()
                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                    
                    center = getPixelPositionInMapImage(xmin, ymin, xmax, ymax, H)
                    dis = calculateGPSPosOfObject(center, imgPixelCenter, pos)
                    
                    setsOfFrames[setIdx][track_id].append(dis)
                    
                    numCars += 1
                
            print(f'found {numCars} cars in {frame}')
        
    drawPositionsOnMap(setsOfFrames, colors, carMap, saveFileName)
    carMap.save(saveFileName)

if __name__ == "__main__":
    main()
