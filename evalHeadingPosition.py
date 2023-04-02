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

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

def arrow_points_calculate(ini_lat, ini_long, heading):
    # length_scale = 0.00012
    length_scale = 0.0005
    sides_scale = 0.000025
    sides_angle = 15

    latA= ini_lat
    longA = ini_long

    latB = length_scale * math.cos(math.radians(heading)) + latA
    longB = length_scale * math.sin(math.radians(heading)) + longA

    latC = sides_scale * math.cos(math.radians(heading + 180 - sides_angle)) + latB
    longC = sides_scale * math.sin(math.radians(heading + 180 - sides_angle)) + longB

    latD = sides_scale * math.cos(math.radians(heading + 180 + sides_angle)) + latB
    longD = sides_scale * math.sin(math.radians(heading + 180 + sides_angle)) + longB

    pointA = (latA, longA)
    pointB = (latB, longB)
    pointC = (latC, longC)
    pointD = (latD, longD)

    point = [pointA, pointB, pointC, pointD, pointB]
    return point

def main():
    posFiles = findFiles('sampleGPSPos')

    firstPosFile = posFiles[0]
    pos, _  = getParams(firstPosFile)
    
    map = folium.Map(location=[float(pos[0]), float(pos[1])], zoom_start=20, 
                tiles='cartodbpositron', width=1280, height=960)
    
    for posFile in posFiles:
        pos, rot = getParams(posFile)
        
        if rot != 0:
            # points = arrow_points_calculate(pos[0], pos[1], rot)
            # folium.PolyLine(locations=points, color="purple", weight=0.5).add_to(
            #         map)
            folium.CircleMarker((pos[0], pos[1]), radius=1, color="blue", fill_color='blue').add_to(map)
        
        
    map.save('gpsData.html')

if __name__ == "__main__":
    main()
