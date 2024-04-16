import argparse


def evalBool(s):
    return s != 'False'

# this lets us determine where to grab the images and meta data


def parseArgs(verbose=False):
    parser = argparse.ArgumentParser(
        description='Collect values to determine GPS position')
    # parser.add_argument('--framesDir', type=str, default='isoData/images', help='where to get drone images from')
    # parser.add_argument('--dataDir', type=str, default='isoData/params', help='where to get drone data from for each frame')
    # parser.add_argument('--cacheDetDir', type=str, default='isoData/cachedDetections', help='where to cache detections for each frame')
    # parser.add_argument('--cacheTrackDir', type=str, default='isoData/cachedTracking', help='where to cache tracking for each frame')
    # parser.add_argument('--cache_mkptsDir', type=str, default='isoData/cachedmkpts_', help='where to cache keypoints for each frame')
    parser.add_argument('--framesDir', type=str, default='larger_set/images',
                        help='where to get drone images from')
    parser.add_argument('--dataDir', type=str, default='larger_set/params',
                        help='where to get drone data from for each frame')
    parser.add_argument('--cacheDetDir', type=str, default='larger_set/cachedDetections',
                        help='where to cache detections for each frame')
    parser.add_argument('--cacheTrackDir', type=str, default='larger_set/cachedTracking',
                        help='where to cache tracking for each frame')
    parser.add_argument('--cache_mkptsDir', type=str, default='larger_set/cachedmkpts_',
                        help='where to cache keypoints for each frame')
    parser.add_argument('--filterCars', default=False,
                        type=evalBool, help='whether or not to filter cars')
    parser.add_argument('--filterRoads', default=False,
                        type=evalBool, help='whether or not to filter roads')
    parser.add_argument('--filterBuildings', default=False,
                        type=evalBool, help='whether or not to filter buildings')
    parser.add_argument('--SuperGlue', default=True, type=evalBool,
                        help='whether or not to use SuperGlue features')
    parser.add_argument('--LoFTR', default=True, type=evalBool,
                        help='whether or not to use LoFTR features')
    parser.add_argument('--homography', default=True, type=evalBool,
                        help='True for homography, False for affine2D')

    args = parser.parse_args()

    if not args.SuperGlue and not args.LoFTR:
        print('\nERROR: tried to use neither SuperGlue nor LoFTR to match features\n')
        exit()
    if verbose:
        print('directory with frames: ', args.framesDir)
        print('directory with gps data: ', args.dataDir)
        print('directory with cached detection results: ', args.cacheDetDir)
        print('directory with cached tracking results: ', args.cacheTrackDir)
        print('directory with cached keypoint results: ', args.cache_mkptsDir)
        print('filtering cars: ', args.filterCars)
        print('filtering roads: ', args.filterRoads)
        print('filtering buildings: ', args.filterBuildings)
        print('using SuperGlue: ', args.SuperGlue)
        print('using LoFTR: ', args.LoFTR)
        print('using homography: ', args.homography)

    return args
