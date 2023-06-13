import os
import pickle
import cv2
import numpy as np
from SuperGluePretrainedNetwork.models.matching import Matching
from kornia.feature import LoFTR
import torch
import visualize

dimOfResizedImage = 640

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

SuperGlueMatcher = Matching(config).eval().to(device)
LoFTRMatcher = LoFTR(pretrained="outdoor")

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

#this grabs the image as a tensor, it also rotates it if needed
def getImageAsTensor(path, device, resize, rotation):
    image = cv2.imread(str(path))
    # image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    
    rot_mat = np.array([[1.0, 0.0, 0.0],
                        [-0.0, 1.0, 0.0]])
    
    if rotation != 0:
        image, rot_mat = rotateImage(image, rotation)
    
    w_new, h_new = resize[0], resize[1]
    h_old, w_old = image.shape[0:2]
    # h_old, w_old = image.shape
    h_new = int(h_old * w_new / w_old)
    image = cv2.resize(image.astype('float32'), (w_new, h_new))

    imageAsTensor = torch.from_numpy(image/255.0).float()[None, None].to(device)
    
    return image.astype('uint8'), imageAsTensor, rot_mat, h_old, w_old, h_new, w_new

def getMkpts(srcPath, dstPath, rot, args, verbose = False):
    mkpts0, mkpts1, w_old, h_old, rot_0, h_new, w_new = None, None, None, None, None, None, None
    cacheFolderSG =  args.cache_mkptsDir + 'SuperGlue'
    cachedMkptsPathSG = srcPath.replace(args.framesDir, cacheFolderSG).replace('.png', '')
    cacheFolderLFTR =  args.cache_mkptsDir + 'LoFTR'
    cachedMkptsPathLFTR = srcPath.replace(args.framesDir, cacheFolderLFTR).replace('.png', '')
    
    keyPointSource = []
    if args.SuperGlue:
        keyPointSource.append((cachedMkptsPathSG, keyPointsWithSuperGlue))
    if args.LoFTR:
        keyPointSource.append((cachedMkptsPathLFTR, keyPointsWithLoFTR))
    
    for cachePath, func in keyPointSource:
        if os.path.isfile(cachePath):
            detectionsFile = open(cachePath, 'rb')     
            mkpts0_tmp, mkpts1_tmp, w_old, h_old, rot_0, h_new, w_new = pickle.load(detectionsFile)
            detectionsFile.close()
            
            if verbose: print(f'for {srcPath} cached matching points found')
        
        else:
            image0, img0, rot_0, h_old, w_old, h_new, w_new = getImageAsTensor(
            srcPath, device, opt['resize'], rot)
            image1, img1, _, _, _, _, _ = getImageAsTensor(
                dstPath, device, opt['resize'], 0)
            
            batch = {'image0': img0, 'image1': img1}
        
            mkpts0_tmp, mkpts1_tmp = func(batch)
            
            detectionsFile = open(cachePath, 'ab')
            pickle.dump((mkpts0_tmp, mkpts1_tmp, w_old, h_old, rot_0, h_new, w_new), detectionsFile)                     
            detectionsFile.close()
            
            if verbose: print(f'for {srcPath} cached matching points not found, new cached matching points saved')

        # visualize.showKeyPointsOnFrame(cv2.imread(dstPath), mkpts1_tmp)
        if verbose: print('found', len(mkpts0_tmp), 'points')
        if mkpts0 is None:
            mkpts0 = np.asarray(mkpts0_tmp)
            mkpts1 = np.asarray(mkpts1_tmp)
        else:
            mkpts0 = np.asarray(np.concatenate((mkpts0, mkpts0_tmp)))
            mkpts1 = np.asarray(np.concatenate((mkpts1, mkpts1_tmp)))
        # visualize.showKeyPointsOnFrame(cv2.imread(dstPath), mkpts1)
        
    return mkpts0, mkpts1, w_old, h_old, rot_0, h_new, w_new

def getUnrotationMat(rot_mat):
    padded = np.array([rot_mat[0,:], rot_mat[1,:], [0.0,0.0,1.0]])
    inv_rot = np.linalg.inv(padded.astype(np.float32)).astype(np.float32)[0:2,:]
    return inv_rot

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

#this finds the key points and then calculates the transform (homography or affine2D)
def findTransform(srcPath, dstPath, roadMaskPath, buildingMaskPath, detectionsMaskPath, rot, args, verbose = False):
    mkpts0, mkpts1, w_old, h_old, rot_0, h_new, w_new = getMkpts(srcPath, dstPath, rot, args)
    
    if verbose: print('found', mkpts0.shape[0], 'points')
    
    # apply masks to matching points
    # if args.filterRoads:
    #     mkpts0, mkpts1 = applyMaskToPoints(roadMaskPath, mkpts0, mkpts1)
    # if args.filterBuildings:
    #     mkpts0, mkpts1 = applyMaskToPoints(buildingMaskPath, mkpts0, mkpts1)
    # if args.filterCars:
    #     mkpts1, mkpts0 = applyMaskToPoints(detectionsMaskPath, mkpts1, mkpts0)
    image0, img0, rot_0, h_old, w_old, h_new, w_new = getImageAsTensor(
            srcPath, device, opt['resize'], rot)
    visualize.showKeyPointsOnFrame(image0, mkpts0)
    
    
    # unscale points
    mkpts0[:,0] = mkpts0[:,0] * w_old / w_new
    mkpts0[:,1] = mkpts0[:,1] * h_old / h_new
    
    # unrotate points
    unrot_mat = getUnrotationMat(rot_0)
    mkpts0WithOnes = np.append(mkpts0,np.ones([mkpts0.shape[0],1]),1)
    mkpts0 = np.matmul(unrot_mat, mkpts0WithOnes.T).T
    
    src = np.float32(mkpts0).reshape(-1,1,2)
    dst = np.float32(mkpts1).reshape(-1,1,2)
    
    transform = None
    
    if args.homography:
        transform, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    else:
        transform, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, maxIters=5)
    
    return transform