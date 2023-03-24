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
import matplotlib.cm as cm
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import make_matching_plot
import time


model_path = 'car_detection_model.pt'
model_conf = 0.65
dimOfImage = 2496
dimOfResizedImage = 640
device = 'cuda' if torch.cuda.is_available() else 'cpu'
imgPixelCenter = (dimOfResizedImage / 2, dimOfResizedImage / 2)

latAtZoom20 = 40.01298216829920000 - 40.01232546913910000

mapPath = 'currentLocation.png'
maskPath = 'mask.png'

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
def grabNewGoogleMapsImage(pos, fileName, maskPath):
    try:
        satMapRequest = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=1920x1080&maptype=satellite&key={keys.GOOGLE_API_KEY}', stream=True).raw
        terrainMapRequest = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=1920x1080&maptype=terrain&key={keys.GOOGLE_API_KEY}', stream=True).raw
        
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

            # set those pixels to black
            # terrainMapImage[np.where(
            # (img_dilation[:, :, 0] < 254) & 
            # (img_dilation[:, :, 1] < 254) & 
            # (img_dilation[:, :, 2] < 254)
            # )] = [0, 0, 0]
            
            cv2.imwrite(fileName, satMapImage)
            
            print('Successfully updated maps reference image')
        else:
            print('\nERROR(s):',satMapRequest.text, '\n', terrainMapRequest.text, '\n')
            exit()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    
        

#this checks to see if the google maps image needs to be updated
def googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
    return np.sqrt((lastUpdatedPos[0] - pos[0])**2 + (lastUpdatedPos[1] - pos[1])**2) > 0.001

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

#this grabs the image as a tensor, it also rotates it if needed
def getImageAsTensor(path, device, resize, rotation):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageGrey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    w_new, h_new = resize[0], resize[1]

    image = cv2.resize(image.astype('float32'), (w_new, h_new))
    imageGrey = cv2.resize(imageGrey.astype('float32'), (w_new, h_new))
    
    if rotation != 0:
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -rotation, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    imageAsTensor = torch.from_numpy(imageGrey/255.0).float()[None, None].to(device)
    
    # cv2.imshow('junk',image/255)
    # cv2.waitKey()
    
    return image/255.0, imageAsTensor

#this grabs the key points using superglue, it returns a list of matching points
def keyPointsWithSuperGlue(srcPath, dstPath, maskPath, rot):
    image0, inp0 = getImageAsTensor(
        srcPath, device, opt['resize'], rot)
    image1, inp1 = getImageAsTensor(
        dstPath, device, opt['resize'], 0)

    if do_match:
        # Perform the matching.
        pred = matching({'image0': inp0, 'image1': inp1})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

    # Keep the matching keypoints.
    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    
    mkpts0Copy = np.copy(mkpts0)
    mkpts1Copy = np.copy(mkpts1)
    
    mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    
    indexes = mkpts1.astype(int)
    
    # s = time.time()
    # indxs = np.full(mkpts0.shape[0], True)
    # for i in range(mkpts0.shape[0]):
    #     point = mkpts1[i]
    #     if mask[int(point[1]), int(point[0])] < 100:
    #         indxs[i] = False
        
    # altpoints = mkpts1Copy[~indxs]
    # mkpts0 = mkpts0Copy[indxs]
    # mkpts1 = mkpts1Copy[indxs]
    # f = time.time()
    # print('method 1:', f-s)
    
    # s = time.time()
    # altpoints = mkpts1Copy[mask[indexes[:,1], indexes[:,0]] < 100]
    mkpts0 = mkpts0Copy[mask[indexes[:,1], indexes[:,0]] > 100]
    mkpts1 = mkpts1Copy[mask[indexes[:,1], indexes[:,0]] > 100]
    # f = time.time()
    # print('method 2:', f-s)
        
    # altpoints = np.float32(altpoints).reshape(-1,1,2)
    color = cm.jet(mconf[mask[indexes[:,1], indexes[:,0]] > 100])
    make_matching_plot(
            image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
            [], 'matchesWithMask', False,
            False, False, 'Matches', [])
    
    src = np.float32(mkpts0).reshape(-1,1,2)
    dst = np.float32(mkpts1).reshape(-1,1,2)
    
    # return src, dst, altpoints
    return src, dst

#this finds the key points and then calculates the homography
def findHomographyUsingNN(srcPath, dstPath, maskPath, rot):
   
    src, dst = keyPointsWithSuperGlue(srcPath, dstPath, maskPath, rot)
    
    # image0 = cv2.imread(srcPath)
    # image1 = cv2.imread(dstPath)
    # mask = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    # image1[mask < 100] = (0,0,0)
    # for p in dst:
    #     cv2.circle(image1, (int(p[0,0]), int(p[0,1])), 3, (5,255,255), 3)
    # for p in altpoint:
    #     cv2.circle(image1, (int(p[0,0]), int(p[0,1])), 3, (255,5,255), 3)
    
    
    # cv2.imshow('dots',image1)
    # cv2.waitKey()

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
    
def main():
    args = parseArgs()
    
    frameFiles = findFiles(args.framesDir)

    firstPosFile = frameFiles[0].replace(args.framesDir, args.dataDir).replace('png', 'txt')
    pos, _  = getParams(firstPosFile)
    grabNewGoogleMapsImage(pos, mapPath, maskPath)
    
    map = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20, 
                tiles='cartodbpositron', width=1280, height=960)
    
    lastUpdatedPos = pos
    
    colors= ['#0080bb','#aa0000', '#00aa00', '#aaaa00', '#999999', '#010101', '#8000bb']
    colorIdx = 0
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
        
        if fileFound:
            if googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
                print('grabbing new google maps image')
                grabNewGoogleMapsImage(pos, mapPath, maskPath)
                colorIdx += 1
                lastUpdatedPos = pos
            
            droneImage = cv2.imread(frame)
            
            image = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (dimOfImage, dimOfImage))
            
            results = inference_with_sahi(image)
            
            rot_mat = cv2.getRotationMatrix2D(imgPixelCenter, -rot, 1.0)
            H = findHomographyUsingNN(frame, mapPath, maskPath, rot)
            
            c = 0
            
            for result in results.object_prediction_list:
                x1 = result.bbox.minx + ((result.bbox.maxx - result.bbox.minx) / 2)
                y1 = result.bbox.miny + ((result.bbox.maxy - result.bbox.miny) / 2)
                
                x1 = dimOfResizedImage * x1 / dimOfImage
                y1 = dimOfResizedImage * y1 / dimOfImage
                
                # rotate the detections
                rotated_point = rot_mat.dot(np.array((x1,y1) + (1,)))
                x1,y1 = int(rotated_point[0]), int(rotated_point[1])
                
                # compute new pixel positions using homography
                x = (x1*H[0,0] + y1*H[0,1] + H[0,2])/(x1*H[2,0] + y1*H[2,1] + H[2,2])
                y = (x1*H[1,0] + y1*H[1,1] + H[1,2])/(x1*H[2,0] + y1*H[2,1] + H[2,2])
                
                center = (int(x), int(y))
                
                dis = calculateGPSPosOfObject(center, imgPixelCenter, pos)
                
                folium.CircleMarker(dis, radius=1, color=colors[colorIdx], fill_color=colors[colorIdx]).add_to(map)
                
                c += 1
                
            print(f'found {c} cars in {frame}')
        
    map.save('multiple_frames_croppingMapsImage_larger_unsegmented_method.html')

if __name__ == "__main__":
    main()
