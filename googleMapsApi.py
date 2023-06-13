#this grabs new google maps images when needed
import cv2
import numpy as np
import requests
import keys

latAtZoom20 = 40.01298216829920000 - 40.01232546913910000

def grabNewGoogleMapsImage(pos, fileName, roadMaskPath, buildingMaskPath):
    try:
        satMapRequest = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=640x640&maptype=satellite&key={keys.GOOGLE_API_KEY}', stream=True).raw
        roadmapRequest = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=640x640&maptype=roadmap&key={keys.GOOGLE_API_KEY}', stream=True).raw
        
        if satMapRequest.status == 200 and roadmapRequest.status == 200:
                
            #turn the responses into images
            satMapImage = np.asarray(bytearray(satMapRequest.read()), dtype="uint8")
            satMapImage = cv2.imdecode(satMapImage, cv2.IMREAD_COLOR)
            roadmapImage = np.asarray(bytearray(roadmapRequest.read()), dtype="uint8")
            roadmapImage = cv2.imdecode(roadmapImage, cv2.IMREAD_COLOR)
            
            gray = cv2.cvtColor(roadmapImage, cv2.COLOR_BGR2GRAY)
            
            getRoadMask(gray, roadMaskPath)
            getBuildingMask(gray, buildingMaskPath)
            
            cv2.imwrite(fileName, satMapImage)
            
            print('Successfully updated maps reference image')
        else:
            print('\nERROR(s):',satMapRequest.data, '\n', roadmapRequest.data, '\n')
            exit()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)

def getRoadMask(mapImage, roadMaskPath):
    _,thresh = cv2.threshold(mapImage,253,255,0)
            
    kernel = np.ones((25, 25), np.uint8)
    mask = cv2.dilate(thresh, kernel, iterations=4)
    
    cv2.imwrite(roadMaskPath, mask)

def getBuildingMask(mapImage, buildingMaskPath):
    # apply thresholding to convert grayscale to binary image
    _,thresh1 = cv2.threshold(mapImage,240,255,0)
    _,thresh2 = cv2.threshold(mapImage,243,255,0)
    
    e_kernel = np.ones((3, 3), np.uint8)
    d_kernel = np.ones((9, 9), np.uint8)
    img_erosion = cv2.erode(thresh1 - thresh2, e_kernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, d_kernel, iterations=1)
    
    cv2.imwrite(buildingMaskPath, 255-img_dilation)
    

#this checks to see if the google maps image needs to be updated
def googleMapsImageNeedsToUpdate(lastUpdatedPos, pos):
    return np.sqrt((lastUpdatedPos[0] - pos[0])**2 + (lastUpdatedPos[1] - pos[1])**2) > 0.1 * latAtZoom20