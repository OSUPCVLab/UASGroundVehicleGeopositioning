import argparse
import glob
import re
from matplotlib import pyplot as plt
import numpy as np
from sklearn import cluster
from tabulate import tabulate
import csv
import copy
from scipy.stats import mannwhitneyu, ttest_ind, chi

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

def readInData(fileName):
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
    
    return points, colors

def normalizeDataAndFilterOutliers(points, colors):
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
        plt.savefig('originalPoints.png',bbox_inches='tight')
        
        outlierIdx = dbscan.labels_ == -1 
        plt.scatter(normalizedPoints[outlierIdx,1], normalizedPoints[outlierIdx,0], s=45, c='purple', marker='x')
        plt.savefig('outliers.png',bbox_inches='tight')
        
        plt.show()
    
    pointsRemoved = np.count_nonzero(dbscan.labels_ == -1)
    if debug: print('removed ', pointsRemoved, ' points')
    if debug: print(normalizedPoints.shape)
    normalizedPoints = normalizedPoints[dbscan.labels_ != -1]
    colors = colors[dbscan.labels_ != -1]
    if debug: print(normalizedPoints.shape)
    
    return numberClusters, normalizedPoints, pointsRemoved, colors

def plotKMeans(normalizedPoints, colors, cluster_centers, dists, fileName):
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    
    plt.scatter(normalizedPoints[:,1], normalizedPoints[:,0], s = 3, c=colors)
    plt.savefig('removedoutliers.png',bbox_inches='tight')
    
    plt.scatter(cluster_centers[:,1], cluster_centers[:,0], s=15, c='pink', marker='+')
    plt.savefig('computedClusterCenters.png',bbox_inches='tight')
    
    plt.show()
    
    plt.hist(dists, 100)
    plt.title(fileName)
    plt.show()

def runAnalysisOnFile(fileName, timesToRunKMeans = 1):
    points, colors = readInData(fileName)
    
    numberClusters, normalizedPoints, pointsRemoved, colors = normalizeDataAndFilterOutliers(points, colors)
    
    aggregateClusterMeanDistances = []
    aggregateClusterDistances = []
    
    for i in range(timesToRunKMeans):
        kmeans = cluster.KMeans(numberClusters, n_init=1)
        kmeans.fit(normalizedPoints)
        
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        dists = np.zeros(labels.shape)
        for p in range(len(normalizedPoints)):
            center = centers[labels[p]]
            dists[p] = np.linalg.norm(normalizedPoints[p] - center)
        
        clusterMeanDistance = np.zeros(centers.shape[0])
        for c in range(len(centers)):
            clusterDists = dists[labels == c]
            clusterMeanDistance[c] = np.mean(clusterDists)
            
        aggregateClusterDistances.append(dists)
        aggregateClusterMeanDistances.append(clusterMeanDistance)
        
        if debug:
            plotKMeans(normalizedPoints, colors, kmeans.cluster_centers_, dists, fileName)
        
    return pointsRemoved, np.array(aggregateClusterMeanDistances), np.array(aggregateClusterDistances)

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
    
    
fileNameToHeader = {
    'results/run_filtering_cars_filtering_roads_LoFTR.html': 'CF, RF, KP: Lo',
    'results/run_not_filtering_cars_filtering_roads_LoFTR.html': 'RF, KP: Lo',
    'results/run_filtering_cars_filtering_roads_SuperGlue.html': 'CF, RF, KP: SG',
    'results/run_not_filtering_cars_filtering_roads_SuperGlue.html': 'RF, KP: SG',
    'results/run_filtering_cars_filtering_roads_SuperGlue_LoFTR.html': 'CF, RF, KP: SG + Lo',
    'results/run_not_filtering_cars_filtering_roads_SuperGlue_LoFTR.html': 'RF, KP: SG + Lo',
    'results/run_filtering_cars_not_filtering_roads_LoFTR.html': 'CF, KP: Lo',
    'results/run_not_filtering_cars_not_filtering_roads_LoFTR.html': 'NF, KP: Lo',
    'results/run_filtering_cars_not_filtering_roads_SuperGlue.html': 'CF, KP: SG',
    'results/run_not_filtering_cars_not_filtering_roads_SuperGlue.html': 'NF, KP: SG',
    'results/run_filtering_cars_not_filtering_roads_SuperGlue_LoFTR.html': 'CF, KP: SG + Lo',
    'results/run_not_filtering_cars_not_filtering_roads_SuperGlue_LoFTR.html': 'NF, KP: SG + Lo'
}

def shouldCompare(distNameA, distNameB):
    indexA = distNameA.index('KP')
    indexB = distNameB.index('KP')
    return (indexA == indexB and distNameA[:indexA] == distNameB[:indexB]) or (distNameA[indexA:] == distNameB[indexB:])


def runFullAnalysis():
    args = parseArgs()
    
    files = glob.glob(f'{args.filesToEvaluateDir}/*')
    files.sort()
    
    files = [
        'results/run_filtering_cars_filtering_roads_LoFTR.html',
        'results/run_not_filtering_cars_filtering_roads_LoFTR.html',
        'results/run_filtering_cars_not_filtering_roads_LoFTR.html',
        'results/run_not_filtering_cars_not_filtering_roads_LoFTR.html',
        'results/run_filtering_cars_filtering_roads_SuperGlue.html',
        'results/run_not_filtering_cars_filtering_roads_SuperGlue.html',
        'results/run_filtering_cars_not_filtering_roads_SuperGlue.html',
        'results/run_not_filtering_cars_not_filtering_roads_SuperGlue.html',
        'results/run_filtering_cars_filtering_roads_SuperGlue_LoFTR.html',
        'results/run_not_filtering_cars_filtering_roads_SuperGlue_LoFTR.html',
        'results/run_filtering_cars_not_filtering_roads_SuperGlue_LoFTR.html',
        'results/run_not_filtering_cars_not_filtering_roads_SuperGlue_LoFTR.html',
    ]
    
    baseFormat = [["Filtering Performed", "LoFTR", "SuperGlue", "LoFTR + SuperGlue"],
            ['None', None, None, None],
            ['Road', None, None, None],
            ['Remove Cars', None, None, None],
            ['Road + Remove Cars', None, None, None]]
    
    removedData = copy.deepcopy(baseFormat)
    meanError = copy.deepcopy(baseFormat)
    
    distributions = []
    meanDistributions = []
    distributionNames = []
    
    kmeansRuns = 200
    
    for file in files:
        print('running analysis on ', file)
        pointsRemoved, clusterMeanDistance, dists = runAnalysisOnFile(file, kmeansRuns)
        metric = clusterMeanDistance.flatten()
        removedData[getRow(file)][getColumn(file)] = pointsRemoved
        meanError[getRow(file)][getColumn(file)] = '{:.3f} ± {:.3f}'.format(metric.mean(), np.std(metric) )
        
        meanDistributions.append(clusterMeanDistance)
        distributions.append(dists)
        
        distributionName = fileNameToHeader[file]
        distributionNames.append(distributionName)
        
        bins=np.arange(0, 12 + 0.5, 0.5)
        plt.hist(dists.flatten(), bins)
        plt.xlabel('Distance to Cluster Center (meters)')
        plt.ylabel('Frequency of Bin')
        plt.title('{} Distance Distribution'.format(distributionName))
        plt.savefig('imagesForPaper/{}_dists.png'.format(distributionName.replace(' ', '_').replace(',','').replace(':', '_is').replace('+', 'and')))
        plt.close()
        # plt.show()
        
        bins=np.arange(0, 7 + 0.5, 0.25)
        plt.hist(clusterMeanDistance.flatten(), bins)
        plt.xlabel('Mean Distance to Cluster Center (meters)')
        plt.ylabel('Frequency of Bin')
        plt.title('{} Mean Distance Distribution'.format(distributionName))
        plt.savefig('imagesForPaper/{}_meanDists.png'.format(distributionName.replace(' ', '_').replace(',','').replace(':', '_is').replace('+', 'and')))
        plt.close()
        # plt.show()
        
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
    runAnalysisOnFile('results/run_filtering_cars_filtering_roads_LoFTR.html')
    # runFullAnalysis()
    
if __name__ == "__main__":
    main()
