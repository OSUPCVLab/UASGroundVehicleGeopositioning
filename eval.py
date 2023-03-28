import argparse
import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster

# grabs the file to evaluate
def parseArgs():
    parser = argparse.ArgumentParser(description='Determine how well the method performs')
    parser.add_argument('--fileToEvaluate', type=str, default='multiple_frames_croppingMapsImage_larger_unsegmented_method.html', help='where to get predicted car locations from from')

    args = parser.parse_args()
    print('file with predictions:', args.fileToEvaluate)
    
    return args

def main():
    args = parseArgs()
    
    file = open(args.fileToEvaluate)
    points = []
    
    line = file.readline()
    
    while line:
        if line.__contains__('L.circleMarker('):
            line = file.readline()
            vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
            if vals.__len__() != 2:
                print('found not 2 values in line', line, 'found these values', vals)
                exit()
            points.append((float(vals[0]),float(vals[1])))
            
        
        line = file.readline()
        
    points = np.array(points)
    mean = points.mean(axis=0)
    stdev = points.std(axis=0)
    normalizedPoints = np.zeros(points.shape)
    for i in range(normalizedPoints.shape[0]):
        normalizedPoints[i] = (points[i] - mean)/stdev
    
    clusters = 21 + 19 + 44 + 35 + 28 + 23
    kmeans = cluster.KMeans(clusters)
    kmeans.fit(normalizedPoints)
    print(kmeans.inertia_)


    # sns.scatterplot(x = points[:,0], y = points[:,1], hue=kmeans.labels_)

    
    # plt.scatter(normalizedPoints[:,1], normalizedPoints[:,0])
    # plt.scatter(normalizedPoints[:,1], normalizedPoints[:,0])
    # plt.show()

if __name__ == "__main__":
    main()
