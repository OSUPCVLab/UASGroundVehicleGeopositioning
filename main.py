import os
import cv2
import folium
from sklearn import decomposition
import torch
import glob
import re
import numpy as np
from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction
import pickle
from deep_sort_realtime.deepsort_tracker import DeepSort
import argsForParameters
import transformations
from googleMapsApi import grabNewGoogleMapsImage, googleMapsImageNeedsToUpdate
import matplotlib.pyplot as plt

model_path = 'car_detection_model.pt'
model_conf = 0.65
dimOfImage = 1664

imgPixelCenter = (transformations.dimOfResizedImage / 2,
                  transformations.dimOfResizedImage / 2)

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

tracker = DeepSort(max_age=30, embedder="clip_RN50", embedder_gpu=False)

model = Yolov8DetectionModel(
    model_path=model_path,
    confidence_threshold=model_conf,
    device='mps' if torch.backends.mps.is_available() else 'cpu'
)

# this runs sliced window inference, this provides better results for ariel drone footage


def inference_with_sahi(img):
    result = get_sliced_prediction(
        img,
        model,
        slice_height=832,
        slice_width=832,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2)
    return result

# this grabs the meta data for a single frame


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

# this returns the files in a sorted list


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

        print(
            f'for {frameName} cached detections not found, new cached detections saved')

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
            xmin, ymin, xmax, ymax = int(result.bbox.minx), int(
                result.bbox.miny), int(result.bbox.maxx), int(result.bbox.maxy)
            xmin, ymin, xmax, ymax = convertPredictionsToImageSpace(
                xmin, ymin, xmax, ymax, originalImg_w, originalImg_h)
            resultsList.append(
                [[xmin, ymin, xmax - xmin, ymax - ymin], str(result.score.value), result.category])

        tracks = tracker.update_tracks(resultsList, frame=droneImage)

        trackingFile = open(cachedTrackingPath, 'ab')
        pickle.dump(tracks, trackingFile)
        trackingFile.close()

        print(
            f'for {frameName} cached tracking not found, new cached tracking saved')

    return tracks

# this calculates the GPS position using pixel positions and the expected image size


def calculateGPSPosOfObject(center, imgPixelCenter, pos, zoom):
    x = center[0]
    y = center[1]

    xDistFromCenter = (x - imgPixelCenter[0])
    yDistFromCenter = (imgPixelCenter[1] - y)

    lat = pos[0]
    long = pos[1]

    degrees_per_meter = 360 / (2 * np.pi * 6378137)
    meters_per_pixel = 156543.03392 / (2 ** zoom)
    lat_factor = np.cos(lat * np.pi / 180)
    # image_shape = map_image.shape
    delta_lat = yDistFromCenter * degrees_per_meter * meters_per_pixel * lat_factor
    delta_long = xDistFromCenter * degrees_per_meter * meters_per_pixel

    xcord = delta_long + long
    ycord = delta_lat + lat

    return (ycord, xcord)


def getPixelPositionInMapImage(minx, miny, maxx, maxy, transform, usingHomography):
    xDet = minx + ((maxx - minx) / 2)
    yDet = miny + ((maxy - miny) / 2)

    positionInOriginal = np.array([xDet, yDet, 1]).T

    x, y = None, None
    positionInMap = np.matmul(transform, positionInOriginal)
    if usingHomography:
        # make x, y homogeneous
        x, y = positionInMap[0] / \
            positionInMap[2], positionInMap[1] / positionInMap[2]
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


def drawPositionsOnMap(setsOfFrames: dict, colors, carMap):
    for setNum in setsOfFrames:
        for track_id in setsOfFrames[setNum]:

            positions = setsOfFrames[setNum][track_id]

            if len(positions) > 1:
                for pos in positions:
                    folium.CircleMarker(
                        pos, radius=1, color=colors[track_id], fill_color=colors[track_id]).add_to(carMap)


def fileNames(args):
    carsText = 'filtering_cars' if args.filterCars else 'not_filtering_cars'
    roadsText = 'filtering_roads' if args.filterRoads else 'not_filtering_roads'
    buildingsText = 'filter_buildings' if args.filterBuildings else 'not_filter_buildings'
    SGText = 'SuperGlue' if args.SuperGlue else 'not_SuperGlue'
    LoFTRText = 'LoFTR' if args.LoFTR else 'not_LoFTR'
    HomographyText = 'Homography' if args.homography else 'Affine2D'

    base = f'run_{carsText}_{roadsText}_{buildingsText}_{SGText}_{LoFTRText}_{HomographyText}'
    saveHTMLName = f'larger_set_testHtmlResults/{base}.html'
    savePickleName = f'larger_set_testGPSResults/{base}'

    return saveHTMLName, savePickleName


def need_to_delete_tracks(last_param_path, param_path):
    last_area = last_param_path[last_param_path.find("params")+7:-7]
    new_area = param_path[param_path.find("params")+7:-7]
    return last_area != new_area


def clean_with_PCA(X, make_circular=True, zero_first_component=False):
    # apply PCA to the data, uncorrelate it
    pca = decomposition.PCA(2).fit(X)
    uncorrelated_data = pca.transform(X)

    assert pca.explained_variance_ratio_[0] >= pca.explained_variance_ratio_[
        1], "PCA component 1 has a lower variance than component 2"

    # scale the data if needed
    if make_circular:
        min_minor = uncorrelated_data[:, 1].min()
        max_minor = uncorrelated_data[:, 1].max()
        min_major = uncorrelated_data[:, 0].min()
        max_major = uncorrelated_data[:, 0].max()
        r_minor = (np.abs(max_minor) + np.abs(min_minor))/2
        r_major = (np.abs(max_major) + np.abs(min_major))/2
        uncorrelated_data[:, 0] = uncorrelated_data[:, 0] * r_minor / r_major
    # zero the first component if needed
    if zero_first_component:
        uncorrelated_data[:, 0] = np.zeros_like(uncorrelated_data[:, 0])

    data_placed_back = pca.inverse_transform(uncorrelated_data)

    return data_placed_back


def as_radians(degrees):
    return degrees * np.pi / 180


def get_YX_in_meters_from_center(center, p):
    # gives a position in meters
    deltaLatitude = p[0] - center[0]
    deltaLongitude = p[1] - center[1]

    latitudeCircumference = 40075160 * np.cos(as_radians(center[0]))

    resultX = deltaLongitude * latitudeCircumference / 360
    resultY = deltaLatitude * 40008000 / 360

    return resultY, resultX


def convert_to_meters(positions_in_geo, geo_center):
    positions_in_meters = []
    for i in range(positions_in_geo.shape[0]):
        positions_in_meters.append(
            get_YX_in_meters_from_center(geo_center, positions_in_geo[i]))

    positions_in_meters = np.asarray(positions_in_meters)

    return positions_in_meters


def convert_back_to_geo(pos_in_meters, geo_center):
    lat_circ = 40075160 * np.cos(as_radians(geo_center[0]))
    long = (pos_in_meters[1] * 360 / lat_circ) + geo_center[1]
    lat = (pos_in_meters[0] * 360 / 40008000) + geo_center[0]

    return lat, long


def clean_sets_of_frames_with_PCA(setsOfFrames):
    for setNum in setsOfFrames:
        for track_id in setsOfFrames[setNum]:
            positions_in_geo = np.asarray(setsOfFrames[setNum][track_id])
            geo_center = positions_in_geo.mean(axis=0)

            positions_in_meters = convert_to_meters(
                positions_in_geo, geo_center)

            positions_in_meters_cleaned = clean_with_PCA(positions_in_meters)

            new_positions_in_geo = []
            for i in range(positions_in_geo.shape[0]):
                new_positions_in_geo.append(
                    convert_back_to_geo(positions_in_meters_cleaned[i], geo_center))
            new_positions_in_geo = np.asarray(new_positions_in_geo)


def main():
    args = argsForParameters.parseArgs(verbose=True)

    print('Using: ', 'mps' if torch.backends.mps.is_available() else 'cpu')

    zoom = 20

    saveHTMLName, savePickleName = fileNames(args)
    print('saving to ', saveHTMLName, 'and', savePickleName)

    frameFiles = findFiles(args.framesDir)

    firstPosFile = frameFiles[0].replace(
        args.framesDir, args.dataDir).replace('png', 'txt')
    pos, _ = getParams(firstPosFile)
    grabNewGoogleMapsImage(pos, mapPath, roadMaskPath, buildingMaskPath, zoom)

    carMap = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20,
                        tiles='cartodbpositron', width=1280, height=960)

    google_maps_pos = pos
    colors = getRandomColors()
    setIdx = 0
    setsOfFrames = dict()
    setsOfFrames[0] = dict()
    last_param_path = ""

    for i, frameName in enumerate(frameFiles):

        fileFound = True
        paramPath = None
        pos = None
        rot = None
        try:
            paramPath = frameName.replace(
                args.framesDir, args.dataDir).replace('png', 'txt')
            pos, rot = getParams(paramPath)
        except:
            fileFound = False
            print(paramPath, 'not found')

        if fileFound:

            if need_to_delete_tracks(last_param_path, paramPath):
                tracker.delete_all_tracks()
                print('deleting tracks, setting new area')
                colors = getRandomColors()
                setIdx += 1
                setsOfFrames[setIdx] = dict()

            last_param_path = paramPath

            if googleMapsImageNeedsToUpdate(google_maps_pos, pos):

                print('grabbing new google maps image')
                # grabNewGoogleMapsImage(
                #     pos, mapPath, roadMaskPath, buildingMaskPath, zoom)
                google_maps_pos = pos

            droneImage = cv2.imread(frameName)
            imageForDetection = cv2.cvtColor(droneImage, cv2.COLOR_BGR2RGB)
            imageForDetection = cv2.resize(
                imageForDetection, (dimOfImage, dimOfImage))

            numCars = 0

            cachedDetectionsPath = frameName.replace(
                args.framesDir, args.cacheDetDir).replace('.png', '')
            results = getDetections(
                cachedDetectionsPath, frameName, imageForDetection)
            # originalH, originalW, _ = droneImage.shape
            # detectionH, detectionW, _ = imageForDetection.shape
            # for result in results.object_prediction_list:
            #     x_min, x_max, y_min, y_max = result.bbox.minx, result.bbox.maxx, result.bbox.miny, result.bbox.maxy
            #     x_scale = originalW / detectionW
            #     y_scale = originalH / detectionH
            #     x_min, x_max, y_min, y_max = x_min * x_scale, x_max * \
            #         x_scale, y_min * y_scale, y_max * y_scale
            #     cv2.rectangle(droneImage, (int(x_min), int(
            #         y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
            # cv2.imwrite("sampleDetections.png", droneImage)

            createDetectionsMask(detectionsMaskPath, imageForDetection.shape[0:2], droneImage.shape[
                                 0:2], results.object_prediction_list, transformations.opt['resize'], rot)

            transform = transformations.findTransform(
                frameName, mapPath, roadMaskPath, buildingMaskPath, detectionsMaskPath, rot, args, verbose=True)

            cachedTrackingPath = frameName.replace(
                args.framesDir, args.cacheTrackDir).replace('.png', '')
            tracks = getTracking(cachedTrackingPath,
                                 results, droneImage, frameName)

            for track in tracks:
                track_id = int(track.track_id)

                if track.is_tentative() and track_id not in setsOfFrames[setIdx]:
                    setsOfFrames[setIdx][track_id] = []

                if (track.is_confirmed() or track.is_tentative()) and track.time_since_update < 1:
                    ltrb = track.to_ltrb()
                    xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                        ltrb[1]), int(ltrb[2]), int(ltrb[3])

                if transform is not None:
                    center = getPixelPositionInMapImage(
                        xmin, ymin, xmax, ymax, transform, args.homography)
                    dis = calculateGPSPosOfObject(
                        center, imgPixelCenter, google_maps_pos, zoom)

                    setsOfFrames[setIdx][track_id].append(dis)

                    numCars += 1

            print(f'found {numCars} cars in {frameName}')

    drawPositionsOnMap(setsOfFrames, colors, carMap)

    detections = open(savePickleName, 'wb')
    pickle.dump(setsOfFrames, detections)
    detections.close()

    carMap.save(saveHTMLName)


if __name__ == "__main__":
    main()
