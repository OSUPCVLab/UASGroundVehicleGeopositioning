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
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
from SuperGluePretrainedNetwork.models.matching import Matching
import matplotlib.pyplot as plt


model_path = 'car_detection_model.pt'
model_conf = 0.65
mapPath = 'currentLocation.png'
dimOfImage = 2496
dimOfResizedImage = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imgPixelCenter = (dimOfResizedImage / 2, dimOfResizedImage / 2)

latAtZoom20 = 40.01298216829920000 - 40.01232546913910000

opt = {'nms_radius' : 4,
        'keypoint_threshold' : 0.005,
        'max_keypoints' : 2048,
        'superglue' : 'outdoor',
        'sinkhorn_iterations' : 20,
        'match_threshold' : 0.8,
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
matching = Matching(config).eval().to(device)

do_match = True

#this runs sliced window inference, this provides better results for ariel drone footage
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

#this lets us determine where to grab the images and meta data
def parseArgs():
    parser = argparse.ArgumentParser(description='Collect values to determine GPS position')
    parser.add_argument('--framesDir', type=str, default='sampleData/images', help='where to get drone images from')
    parser.add_argument('--dataDir', type=str, default='sampleData/params', help='where to get drone data from for reach frame')
    # parser.add_argument('--framesDir', type=str, default='sampleData/images', help='where to get drone images from')
    # parser.add_argument('--dataDir', type=str, default='sampleData/params', help='where to get drone data from for reach frame')

    args = parser.parse_args()
    print('directory with frames:', args.framesDir)
    print('directory with gps data', args.dataDir)
    
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
def grabNewGoogleMapsImage(pos, fileName):
    response = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=1920x1080&maptype=satellite&key={keys.GOOGLE_API_KEY}')
    if response.status_code == 200:    
        with open(fileName, 'wb') as file:
            file.write(response.content)
    else:
        print('\nERROR:',response.text,'\n')
        exit() 

#this checks to see if the google maps image needs to be updated
def googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
    return np.sqrt((lastUpdatedPos[0] - pos[0])**2 + (lastUpdatedPos[1] - pos[1])**2) > 0.0002

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

#this grabs the image as a tensor, it also rotates it if needed
def getImageAsTensor(path, device, resize, rotation):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    w_new, h_new = resize[0], resize[1]

    image = cv2.resize(image.astype('float32'), (w_new, h_new))

    if rotation != 0:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -rotation, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    imageAsTensor = torch.from_numpy(image/255.0).float()[None, None].to(device)
    
    return imageAsTensor

#this grabs the key points using superglue, it returns a list of matching points
def keyPointsWithSuperGlue(srcPath, dstPath, rot):
    inp0 = getImageAsTensor(
        srcPath, device, opt['resize'], rot)
    inp1 = getImageAsTensor(
        dstPath, device, opt['resize'], 0)

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
    
    src = np.float32(mkpts0).reshape(-1,1,2)
    dst = np.float32(mkpts1).reshape(-1,1,2)
    
    return src, dst

#this finds the key points and then calculates the homography
def findHomographyUsingNN(srcPath, dstPath, rot):
   
    src, dst = keyPointsWithSuperGlue(srcPath, dstPath, rot)
    
    # image0 = cv2.imread(srcPath)
    # image1 = cv2.imread(dstPath)

    # image0 = cv2.resize(image0, [dimOfResizedImage, dimOfResizedImage])
    # image1 = cv2.resize(image1, [dimOfResizedImage, dimOfResizedImage])

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    
    return H

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

def computeCascadingHomography(H_a, H_b, newDetections, pastDetections):
    # blankImageNew = np.zeros((dimOfResizedImage,dimOfResizedImage))
    # blankImagePast = np.zeros((dimOfResizedImage,dimOfResizedImage))
    pts_src = []
    pts_dst = []
    for detection in newDetections:
        if detection[0] < dimOfResizedImage and detection[1] < dimOfResizedImage: 
            # blankImageNew[detection[1], detection[0]] = 1
            pts_src.append([detection[1], detection[0]])
    
    
    if H_b is not None and pastDetections is not None:
        valid_dst_points = []
        for detection in pastDetections:
            if detection[0] < dimOfResizedImage and detection[1] < dimOfResizedImage:
                # blankImageNew[detection[1], detection[0]] = 0.5
                valid_dst_points.append([detection[1], detection[0]])
                
        
        for src_point in pts_src:
            closestPoint = pastDetections[closest_point_idx(src_point, valid_dst_points)]
            pts_dst.append([closestPoint[0], closestPoint[1]])
        
        for a,b in zip(pts_src, pts_dst):
            plt.plot(a,b)
        plt.savefig('fig.png')
        
        H_a_to_b, _ = cv2.findHomography(np.array(pts_src), np.array(pts_dst))
        
        if H_a_to_b is not None:
            modified_detections = []
            for src_point in pts_src:
                modified_detections.append(applyHomographyToPoint(H_a_to_b, src_point[0], src_point[1]))
            newDetections = modified_detections
        
        
    
    # cv2.imshow('points', blankImageNew)
    # cv2.waitKey()
    return H_a, newDetections

def applyHomographyToPoint(H, x1, y1):
    # compute new pixel positions using homography
    x = (x1*H[0,0] + y1*H[0,1] + H[0,2])/(x1*H[2,0] + y1*H[2,1] + H[2,2])
    y = (x1*H[1,0] + y1*H[1,1] + H[1,2])/(x1*H[2,0] + y1*H[2,1] + H[2,2])
    
    return (int(x), int(y))

def closest_point_idx(point_a, points_b):
    points_b = np.asarray(points_b)
    dist_2 = np.sum((points_b - point_a)**2, axis=1)
    return np.argmin(dist_2)
    
    
def main():
    args = parseArgs()
    
    frameFiles = findFiles(args.framesDir)

    firstPosFile = frameFiles[0].replace(args.framesDir, args.dataDir).replace('png', 'txt')
    pos, _  = getParams(firstPosFile)
    grabNewGoogleMapsImage(pos, 'currentLocation.png')
    
    map = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20, 
                tiles='cartodbpositron', width=1280, height=960)
    
    lastUpdatedPos = pos
    
    colors= ['#0080bb','#aa0000', '#00aa00', '#aaaa00', '#999999', '#010101', '#8000bb']
    
    H_a = None
    H_b = None
    pastDetections = None
    newDetections = None
    i = 0
    
    for frame in frameFiles:
        
        fileFound = True
        paramPath = None
        pos = None
        rot = None
        
        H_b = H_a
        pastDetections = newDetections
        
        try:
            paramPath = frame.replace(args.framesDir, args.dataDir).replace('png', 'txt')
            pos, rot = getParams(paramPath)
        except:
            fileFound = False
        
        if fileFound:
            if googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
                print('grabbing new google maps image')
                lastUpdatedPos = pos
                grabNewGoogleMapsImage(pos, 'currentLocation.png')
                H_b = None
                pastDetections = None
                i += 1
            
            droneImage = cv2.imread(frame)
            
            image = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (dimOfImage, dimOfImage))
            
            results = inference_with_sahi(image)
            
            rot_mat = cv2.getRotationMatrix2D(imgPixelCenter, -rot, 1.0)
            
            H_a = findHomographyUsingNN(frame, mapPath, rot)
            newDetections = []
            
            for result in results.object_prediction_list:
                x1 = result.bbox.minx + ((result.bbox.maxx - result.bbox.minx) / 2)
                y1 = result.bbox.miny + ((result.bbox.maxy - result.bbox.miny) / 2)
                
                x1 = dimOfResizedImage * x1 / dimOfImage
                y1 = dimOfResizedImage * y1 / dimOfImage
                
                # rotate the detections
                rotated_point = rot_mat.dot(np.array((x1,y1) + (1,)))
                x1, y1 = int(rotated_point[0]), int(rotated_point[1])
                
                newDetections.append(applyHomographyToPoint(H_a, x1, y1))
                
            H_a, newDetections = computeCascadingHomography(H_a, H_b, newDetections, pastDetections)
            
            c = 0
            for detection in newDetections:
                
                dis = calculateGPSPosOfObject(detection, imgPixelCenter, pos)
                
                folium.CircleMarker(dis, radius=1, color=colors[i], fill_color=colors[i]).add_to(map)
                
                c += 1
                
            print(f'found {c} cars in {frame}')
        
    map.save('multiple_frames__larger_area_cascading_homography.html')

if __name__ == "__main__":
    main()
