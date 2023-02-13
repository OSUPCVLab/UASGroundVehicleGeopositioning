import cv2
import numpy as np
import requests
import argparse
import keys
import re

def parseArgs():
    parser = argparse.ArgumentParser(description='Collect values to determine GPS position')
    parser.add_argument('--framesDir', type=str, default='sampleData/images', help='where to get drone images from')
    parser.add_argument('--dataDir', type=str, default='sampleData/params', help='where to get drone data from for reach frame')

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
    mapTypes = [
        # 'roadmap',
        # 'satellite',
        # 'hybrid',
        'terrain'
        ]
    for map in mapTypes:
        url = f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=1920x1080&scale=2&maptype={map}&key={keys.GOOGLE_API_KEY}'
        # response = requests.get(f'https://maps.googleapis.com/maps/api/staticmap?center={pos[0]},{pos[1]}&zoom=20&size=1920x1080&scale=2&maptype={map}&key={keys.GOOGLE_API_KEY}')
        resp = requests.get(url, stream=True).raw
        

        
        if resp.status == 200:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)   
            
            # for testing
            cv2.imshow('image',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite(f'{map}_altGoogleImage.png', image)
            
        else:
            print('\nERROR:',resp.text,'\n')
            exit()
            
            
def filterMapImage():
    segmentedImage = cv2.imread('terrain_altGoogleImage.png')
    realImage = cv2.imread('satellite_altGoogleImage.png')
    white = np.where(
    (segmentedImage[:, :, 0] > 253) & 
    (segmentedImage[:, :, 1] > 253) & 
    (segmentedImage[:, :, 2] > 253)
    )
    mask = np.zeros(realImage.shape)
    mask[white] = 255
    cv2.imwrite('mask.png', mask)
    kernel = np.ones((35, 35), np.uint8)
    img_dilation = cv2.dilate(mask, kernel, iterations=4)

    # set those pixels to black
    realImage[np.where(
    (img_dilation[:, :, 0] < 254) & 
    (img_dilation[:, :, 1] < 254) & 
    (img_dilation[:, :, 2] < 254)
    )] = [0, 0, 0]
    
    cv2.imwrite('segmented.png', realImage)

if __name__ == "__main__":
    pos, _  = getParams('sampleData/params/000000.txt')
    grabNewGoogleMapsImage(pos, 'altGoogleImage.png')
    # filterMapImage()
