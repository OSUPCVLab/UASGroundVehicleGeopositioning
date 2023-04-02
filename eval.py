import argparse
import glob
import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster
from tabulate import tabulate
import csv
import copy



debug = True

# grabs the file to evaluate
def parseArgs():
    parser = argparse.ArgumentParser(description='Determine how well the method performs')
    parser.add_argument('--filesToEvaluateDir', type=str, default='results', help='where to get predicted car locations files from')

    args = parser.parse_args()
    if debug: print('folder with predictions:', args.filesToEvaluateDir)
    
    return args

def asRadians(degrees):
    return degrees * np.pi / 180

def getYXpos(mean, p):
    # deltaLatitude = p.latitude - mean.latitude
    # deltaLongitude = p.longitude - mean.longitude
    deltaLatitude = p[0] - mean[0]
    deltaLongitude = p[1] - mean[1]
    
    # latitudeCircumference = 40075160 * np.cos(asRadians(mean.latitude))
    latitudeCircumference = 40075160 * np.cos(asRadians(mean[0]))
    
    resultX = deltaLongitude * latitudeCircumference / 360
    resultY = deltaLatitude * 40008000 / 360
    
    return resultY, resultX


def runAnalysisOnFile(fileName):
    file = open(fileName)
    points = []
    colors = []
    
    line = file.readline()
    if debug: print('for ', fileName) 
    
    while line:
        if line.__contains__('L.circleMarker('):
            line = file.readline()
            vals = re.findall(r"[-+]?(?:\d*\.*\d+)", line)
            if vals.__len__() != 2:
                print('found not 2 values in line', line, 'found these values', vals)
                exit()
            line = file.readline()
            indx = line.index('\"color\": \"#')
            color = line[indx+10:indx+17]
            colors.append(color)
            points.append((float(vals[0]),float(vals[1])))
            
        
        line = file.readline()
        
    points = np.array(points)
    colors = np.array(colors)
    
    mean_lat_long = points.mean(axis=0)
    for i in range(points.shape[0]):
        points[i] = getYXpos(mean_lat_long, points[i])
    
    normalizedPoints = points
    numberClusters = 21 + 17 + 43 + 35 + 28 + 21
    
    dbscan = cluster.DBSCAN(eps = 5, min_samples=2)
    dbscan.fit(points)
    
    if debug:
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        
        plt.scatter(normalizedPoints[:,1], normalizedPoints[:,0], s = 3, c=colors)
        plt.savefig('originalPoints.png')
        
        outlierIdx = dbscan.labels_ == -1 
        plt.scatter(normalizedPoints[outlierIdx,1], normalizedPoints[outlierIdx,0], s=45, c='purple', marker='x')
        plt.savefig('outliers.png')
        
        plt.show()
    
    pointsRemoved = np.count_nonzero(dbscan.labels_ == -1)
    if debug: print('removed ', pointsRemoved, ' points')
    if debug: print(normalizedPoints.shape)
    normalizedPoints = normalizedPoints[dbscan.labels_ != -1]
    colors = colors[dbscan.labels_ != -1]
    if debug: print(normalizedPoints.shape)
    
    # mean_meters = normalizedPoints.mean(axis=0)
    # stdev_meters = normalizedPoints.std(axis=0)
    # for i in range(normalizedPoints.shape[0]):
    #     normalizedPoints[i] = (normalizedPoints[i] - mean_meters)/stdev_meters
    
    kmeans = cluster.KMeans(numberClusters, n_init='auto')
    kmeans.fit(normalizedPoints)
    
    if debug: print('sum of stdev is ', kmeans.inertia_)

    if debug:
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        
        plt.scatter(normalizedPoints[:,1], normalizedPoints[:,0], s = 3, c=colors)
        plt.savefig('removedoutliers.png')
        
        plt.scatter(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,0], s=15, c='pink', marker='+')
        plt.savefig('computedClusterCenters.png')
        
        plt.show()
        
    return kmeans.inertia_, pointsRemoved

def getRow(fileName: str):
    if 'run_filtering_cars_filtering_roads' in fileName:
        return 4
    if 'run_filtering_cars_not_filtering_roads' in fileName:
        return 3
    if 'run_not_filtering_cars_filtering_roads' in fileName:
        return 2
    if 'run_not_filtering_cars_not_filtering_roads' in fileName:
        return 1

def getColumn(fileName: str):
    if 'SuperGlue_LoFTR' in fileName:
        return 3
    if 'LoFTR' in fileName:
        return 1
    if 'SuperGlue' in fileName:
        return 2

def runFullAnalysis():
    args = parseArgs()
    
    files = glob.glob(f'{args.filesToEvaluateDir}/*')
    files.sort()
    
    errorData = [["Filtering Performed", "LoFTR", "SuperGlue", "LoFTR + SuperGlue"],
            ['None', None, None, None],
            ['Road', None, None, None],
            ['Remove Cars', None, None, None],
            ['Road + Remove Cars', None, None, None]]
    
    removedData = copy.deepcopy(errorData)
    
    for file in files:
        totalError, pointsRemoved = runAnalysisOnFile(file)
        errorData[getRow(file)][getColumn(file)] = totalError
        removedData[getRow(file)][getColumn(file)] = pointsRemoved
        
    
    print('\nSum of Squared Distances for clusters (lower is better):')
    print(tabulate(errorData,headers='firstrow'))
    print('\nNumber of outliers removed:')
    print(tabulate(removedData,headers='firstrow'))
    
    with open("errorData.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(errorData)
    
    with open("removedData.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(removedData)

def main():
    args = parseArgs()
    files = glob.glob(f'{args.filesToEvaluateDir}/*')
    files.sort()
    file = files[1]
    runAnalysisOnFile(file)
    
    
if __name__ == "__main__":
    main()
