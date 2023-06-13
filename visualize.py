import cv2
import numpy as np

def showKeyPointsOnFrame(frame, pts):    
    frameCopy = np.copy(frame)
    ptsFormated = np.float32(pts).reshape(-1,1,2)
    for p in ptsFormated:
        cv2.circle(frameCopy, (int(p[0,0]), int(p[0,1])), 3, (255,5,255), 3)
    cv2.imshow('points', frameCopy)
    cv2.imwrite('matchingpoints0_noFilter.png', frameCopy)
    cv2.waitKey()