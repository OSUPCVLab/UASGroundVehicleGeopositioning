import argparse
import glob
import pickle
import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster
from tabulate import tabulate
import csv
import copy
from scipy.stats import mannwhitneyu, ttest_ind, chi

debug = False

# grabs the file to evaluate
def parseArgs():
    parser = argparse.ArgumentParser(description='Determine how well the method performs')
    parser.add_argument('--filesToEvaluateDir', type=str, default='GPSResults', help='where to get predicted car locations files from')

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

def readInCachedData(cachedFilePath):
    cacheFile = open(cachedFilePath, 'rb')     
    setsOfFrames = pickle.load(cacheFile)
    cacheFile.close()
    
    return setsOfFrames

def getRandomColors(numColors=500):
    rng = np.random.default_rng()
    colorValues = rng.choice(16777215, numColors, replace=False)
    return [hex(color).replace('0x', '#').ljust(7, '0') for color in colorValues]

def getCenter(setsOfFrames):
    allPoints = []
    for setNum in setsOfFrames:
        for track_id in setsOfFrames[setNum]:
            positions = setsOfFrames[setNum][track_id]
            
            for position in positions:
                allPoints.append(position)
    allPoints = np.asarray(allPoints)
    return allPoints.mean(axis=0)

def normalizeDataAndFilterOutliers(setsOfFrames):
    mean_lat_long = getCenter(setsOfFrames)
    filteredSetsOfFrames = dict()
    for setNum in setsOfFrames:
        filteredSetsOfFrames[setNum] = dict()
        for track_id in setsOfFrames[setNum]:
            positions = setsOfFrames[setNum][track_id]
            
            numPositions = len(positions)
            if numPositions > 1:
                filteredSetsOfFrames[setNum][track_id] = []
                for i in range(numPositions):
                    pos = getYXpos(mean_lat_long, setsOfFrames[setNum][track_id][i])
                    filteredSetsOfFrames[setNum][track_id].append(pos)
                    setsOfFrames[setNum][track_id][i] = pos
            else:
                pos = getYXpos(mean_lat_long, setsOfFrames[setNum][track_id][0])
                setsOfFrames[setNum][track_id][0] = pos
    
    if debug:
        colors = getRandomColors()
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')
        idx = 0
        for setNum in setsOfFrames:
            for track_id in setsOfFrames[setNum]:
                positions = setsOfFrames[setNum][track_id]
                
                if len(positions) > 1:
                    positions = np.asarray(positions)
                    plt.scatter(positions[:,1], positions[:,0], s = 3, c=colors[idx])
                else:
                    positions = np.asarray(positions)
                    plt.scatter(positions[:,1], positions[:,0], s = 3, c=colors[idx])
                idx += 1
        plt.savefig('originalPoints.png',bbox_inches='tight')
        plt.show()
        
        for setNum in filteredSetsOfFrames:
            for track_id in filteredSetsOfFrames[setNum]:
                positions = filteredSetsOfFrames[setNum][track_id]
                
                if len(positions) > 1:
                    positions = np.asarray(positions)
                    plt.scatter(positions[:,1], positions[:,0], s = 3, c=colors[idx])
                idx += 1
        plt.savefig('outliers.png',bbox_inches='tight')
        plt.show()
        
    return filteredSetsOfFrames
    
def runAnalysisOnFile(fileName):
    setsOfFrames = readInCachedData(fileName)
    
    normalizedPoints = normalizeDataAndFilterOutliers(setsOfFrames)
    
    frameAvgDist = []
    numOutliers = 0
    for setNum in normalizedPoints:
        for track_id in normalizedPoints[setNum]:
            positions = normalizedPoints[setNum][track_id]
            
            if len(positions) > 1:
                center = np.mean(positions, axis=0)
                for position in positions:
                    dist = np.linalg.norm(position - center)
                    if dist < 20:
                        frameAvgDist.append(dist)
                    else:
                        numOutliers += 1
            else:
                print('ERROR')
                exit()
    # plt.hist(frameAvgDist)
    # plt.show()
    return np.array(frameAvgDist), numOutliers

def getRow(fileName: str):
    row = 0
    
    if 'not_filtering_roads' in fileName:
        row += 4
    if 'not_filter_buildings' in fileName:
        row += 2
    if 'not_filtering_cars' in fileName:
        row += 1
    
    return row
    
    

def getColumn(fileName: str):
    col = 0
    
    if 'not_SuperGlue' in fileName:
        col += 0
    elif 'not_LoFTR' in fileName:
        col += 2
    else:
        col += 4
        
    if 'Affine2D' in fileName:
        col += 1
    
    return col

def shouldCompare(distNameA, distNameB):
    indexA = distNameA.index('KP')
    indexB = distNameB.index('KP')
    return (indexA == indexB and distNameA[:indexA] == distNameB[:indexB]) or (distNameA[indexA:] == distNameB[indexB:])

#this returns the files in a sorted list
def findFiles(framesDir):
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files

def runFullAnalysis():
    args = parseArgs()
    
    files = glob.glob(f'{args.filesToEvaluateDir}/*')
    files.sort()
    
    files = findFiles('GPSResults')
    
    baseFormat = [["Road", "Building", "Car", "LoFTR", "LoFTR", "SuperGlue", "SuperGlue", "LoFTR + SuperGlue", "LoFTR + SuperGlue"],
    # baseFormat = [["Road Filtering", "Building Filtering", "Car Filtering", "LoFTR", "LoFTR", "SuperGlue", "SuperGlue", "LoFTR + SuperGlue", "LoFTR + SuperGlue"],
            ['True', 'True', 'True', None, None, None, None, None, None],
            ['True', 'True', 'False', None, None, None, None, None, None],
            ['True', 'False', 'True', None, None, None, None, None, None],
            ['True', 'False', 'False', None, None, None, None, None, None],
            ['False', 'True', 'True', None, None, None, None, None, None],
            ['False', 'True', 'False', None, None, None, None, None, None],
            ['False', 'False', 'True', None, None, None, None, None, None],
            ['False', 'False', 'False', None, None, None, None, None, None],
            [None, None, None, 'homography', 'affine2D', 'homography', 'affine2D', 'homography', 'affine2D']
    ]
    
    removedData = copy.deepcopy(baseFormat)
    meanError = copy.deepcopy(baseFormat)
    names = copy.deepcopy(baseFormat)
    
    distributions = []
    meanDistributions = []
    distributionNames = []
    
    kmeansRuns = 200
    
    for file in files:
        fileName = file.replace('GPSResults/run_', '').replace('filtering_', '').replace('filter_', '').replace('Homography', 'H').replace('Affine2D', 'A')
        fileName = fileName.replace('not', 'n').replace('SuperGlue', 'SG').replace('LoFTR', 'L').replace('cars', 'c').replace('roads', 'r').replace('buildings', 'b')
        names[getRow(file)+1][getColumn(file)+3] = fileName
    print('\nFile Name Check:')
    print(tabulate(names,headers='firstrow'))
    
    for file in files:
        print('running analysis on ', file)
        frameAvgDist, numOutliers = runAnalysisOnFile(file)
        removedData[getRow(file)+1][getColumn(file)+3] = numOutliers
        meanError[getRow(file)+1][getColumn(file)+3] = '{:.3f} ± {:.3f}'.format(frameAvgDist.mean(), np.std(frameAvgDist) )
        
        
    print('\nNumber of outliers removed:')
    print(tabulate(removedData,headers='firstrow'))
    print('\nMean Distance from Cluster:')
    print(tabulate(meanError,headers='firstrow'))
    
    with open("removedData.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(removedData)
        
    with open("meanData.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(meanError)
        
    
    exit()
    l = len(distributions)
    
    for distType in ['Mean Distance', 'Distance']:
        distribution = distributions if distType == 'Distance' else meanDistributions
        meandist = '' if distType == 'Distance' else ' per Cluster'
        MWScoresText = np.chararray((l, l), itemsize = 13, unicode=True)    
        MWScores = np.zeros((l, l))    
            
        for i in range(l):
            for j in range(i, l):
                if shouldCompare(distributionNames[j], distributionNames[i]):
                    ps = []
                    for k in range(kmeansRuns):
                        _, p = mannwhitneyu(distribution[i][k], distribution[j][k])
                        ps.append(p)
                    ps = np.asarray(ps)
                    MWScoresText[i,j] = ' {:.2f}\n± {:.2f}'.format(ps.mean(), np.std(ps) )
                    MWScores[i,j] = ps.mean()
        
        fig, ax = plt.subplots()
        
        fig.set_figheight(17)
        fig.set_figwidth(20)
        
        ax.matshow(MWScores, cmap=plt.cm.Blues, alpha=0.3)
        ax.set_yticklabels(distributionNames, weight='bold', fontsize=24)
        ax.set_yticks(range(l))
        
        ax2 = ax.secondary_yaxis("right")
        ax2.set_yticklabels(distributionNames, weight='bold', fontsize=24)
        ax2.set_yticks(range(l))
        for i in range(l): ax2.get_yticklabels()[i].set_color("white")
        
        ax.set_xticklabels(['\n' + d for d in distributionNames], weight='bold', fontsize=24)
        ax.set_xticks(range(l))
        ax.xaxis.set_ticks_position('bottom')
        
        for i in [0,1,2,3]: ax.get_xticklabels()[i].set_color("red")
        for i in [0,1,2,3]: ax.get_xticklabels()[i + 4].set_color("green")
        for i in [0,1,2,3]: ax.get_xticklabels()[i + 8].set_color("blue")
        
        for i in [0,1,2,3]: ax.get_yticklabels()[i].set_color("red")
        for i in [0,1,2,3]: ax.get_yticklabels()[i + 4].set_color("green")
        for i in [0,1,2,3]: ax.get_yticklabels()[i + 8].set_color("blue")
        
        fig.autofmt_xdate(rotation=90)
        for i in range(MWScores.shape[0]):
            for j in range(i,MWScores.shape[1]):
                if shouldCompare(distributionNames[j], distributionNames[i]):
                    ax.text(x=j, y=i,s=MWScoresText[i, j], va='center', ha='center', size='xx-large', weight='bold')
        
        plt.title(f'Mann-Whitney Test p scores for Distributions of\n{distType} to Cluster Center{meandist}\nAverage over {kmeansRuns} iterations', fontsize=32, weight='bold')
        plt.savefig(f'Mann-Whitney_{distType}.png'.replace(' ', '_'), bbox_inches='tight')

def main():
    # args = parseArgs()
    # files = glob.glob(f'{args.filesToEvaluateDir}/*')
    # files.sort()
    # file = files[1]
    # runAnalysisOnFile('GPSResults/run_filtering_cars_filtering_roads_filter_buildings_SuperGlue_LoFTR_Homography')
    runFullAnalysis()
    
if __name__ == "__main__":
    main()
