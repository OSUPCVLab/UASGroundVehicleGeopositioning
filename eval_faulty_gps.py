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
from scipy.stats import mannwhitneyu, ttest_ind, wilcoxon, ttest_rel
from sklearn.cluster import DBSCAN
from sklearn import decomposition

debug = False


def parseArgs():
    parser = argparse.ArgumentParser(
        description='Determine how well the method performs')
    # parser.add_argument('--filesToEvaluateDir', type=str, default='larger_set_testGPSResults',
    #                     help='where to get predicted car locations files from')
    # parser.add_argument('--csv_save_dir', type=str, default='csv_results',
    #                     help='where to put csv results')
    parser.add_argument('--filesToEvaluateDir', type=str, default='gps_errors_GPS_results',
                        help='where to get predicted car locations files from')
    parser.add_argument('--csv_save_dir', type=str, default='csv_results_faulty_gps',
                        help='where to put csv results')

    args = parser.parse_args()
    if debug:
        print('folder with predictions:', args.filesToEvaluateDir)

    return args


def asRadians(degrees):
    return degrees * np.pi / 180


def getYXpos(mean, p):
    # gives a position in meters
    deltaLatitude = p[0] - mean[0]
    deltaLongitude = p[1] - mean[1]

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


def convert_data_to_meters(setsOfFrames):

    mean_lat_long = getCenter(setsOfFrames)
    filteredSetsOfFrames = dict()

    for setNum in setsOfFrames:
        filteredSetsOfFrames[setNum] = dict()

        for track_id in setsOfFrames[setNum]:
            positions = setsOfFrames[setNum][track_id]

            numPositions = len(positions)
            filteredSetsOfFrames[setNum][track_id] = []

            for i in range(numPositions):
                pos = getYXpos(
                    mean_lat_long, setsOfFrames[setNum][track_id][i])
                filteredSetsOfFrames[setNum][track_id].append(np.asarray(pos))
                setsOfFrames[setNum][track_id][i] = np.asarray(pos)

    return filteredSetsOfFrames


def runAnalysisOnFile(fileName, gt_data_in_meters):
    setsOfFrames = readInCachedData(fileName)

    normalizedPoints = convert_data_to_meters(setsOfFrames)

    frame_dist_diff = []
    for setNum in normalizedPoints:
        for track_id in normalizedPoints[setNum]:
            for position, position_gt in zip(normalizedPoints[setNum][track_id], gt_data_in_meters[setNum][track_id]):
                dist = np.linalg.norm(position - position_gt)
                frame_dist_diff.append(dist)

    return np.asarray(frame_dist_diff)


def findFiles(framesDir):
    # this returns the files in a sorted list
    files = glob.glob(f'{framesDir}/*')
    files.sort()
    return files


def print_table_in_latex_format(file_name, table):
    f = open(file_name, "w")
    f.write("\n\\begin{table*}[htbp]")
    f.write("\n\centering")
    f.write("\n\\begin{tabular}{@{}")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c @{}}")
    f.write("\n\\toprule")
    f.write("\n\multicolumn{3}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{Filtering Performed}} & \multicolumn{6}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{Feature Extraction Method}}                                                                                                                         \\\\ \midrule")
    f.write(
        "\n\\textbf{GPS Error} & \\textbf{Deviation}  \\\\ \midrule")
    for i in reversed(range(6)):
        f.write("\n")
        table_row = table[i+1]
        f.write(str(table_row[0]))
        f.write(" & ")
        f.write(str(table_row[1]))
        f.write("\\\\")
    f.write("\n\cmidrule(l){4-9}")
    f.write("\n\end{tabular}")
    f.write("\n\caption{TODO CAPTION}")
    f.write("\n\label{TODO LABEL}")
    f.write("\n\end{table*}")
    f.close()


def get_row(file_name):
    if 'err_0' in file_name:
        return 1
    if 'err_10' in file_name:
        return 4
    if 'err_1' in file_name:
        return 2
    if 'err_50' in file_name:
        return 8
    if 'err_5' in file_name:
        return 3
    if 'err_20' in file_name:
        return 5
    if 'err_30' in file_name:
        return 6
    if 'err_40' in file_name:
        return 7
    if 'err_60' in file_name:
        return 9
    print("ERROR")
    exit()


def get_dist(file_name):
    if 'err_0' in file_name:
        return 0
    if 'err_10' in file_name:
        return 10
    if 'err_1' in file_name:
        return 1
    if 'err_50' in file_name:
        return 50
    if 'err_5' in file_name:
        return 5
    if 'err_20' in file_name:
        return 20
    if 'err_30' in file_name:
        return 30
    if 'err_40' in file_name:
        return 40
    if 'err_60' in file_name:
        return 60
    print("ERROR")
    exit()


def runFullAnalysis():
    args = parseArgs()

    files = glob.glob(f'{args.filesToEvaluateDir}/*')
    files = [file for file in files if 'reported_gps' not in file]

    # TODO: this is trash code
    gt_data_path = 'gps_errors_GPS_results/err_0_run_not_filtering_cars_not_filtering_roads_not_filter_buildings_SuperGlue_LoFTR_Homography'
    gt_data = readInCachedData(gt_data_path)
    gt_data_in_meters = convert_data_to_meters(gt_data)
    percentiles = [15, 50, 85]
    list_of_vals_at_percentiles = []
    list_of_errors = []

    errors = ['0', '1', '5', '10', '20', '30', '40', '50', '60',]

    for i, error in enumerate(errors):
        file = f'gps_errors_GPS_results/err_{error}_run_not_filtering_cars_not_filtering_roads_not_filter_buildings_SuperGlue_LoFTR_Homography'
        frame_dist_diff = runAnalysisOnFile(file, gt_data_in_meters)

        dist = get_dist(file)
        vals_at_percentiles = []
        for i in percentiles:
            vals_at_percentiles.append(np.percentile(frame_dist_diff, i))
        list_of_vals_at_percentiles.append(vals_at_percentiles)
        list_of_errors.append(dist)

    list_of_vals_at_percentiles = np.asarray(list_of_vals_at_percentiles)
    # pick

    # print_table_in_latex_format(
    #     f"{args.csv_save_dir}/{args.filesToEvaluateDir}_mean_dist_to_gt_latex.txt", mean_dist_to_gt)
    # with open(f"{args.csv_save_dir}/{args.filesToEvaluateDir}_mean_dist_to_gt.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(mean_dist_to_gt)

    # print_table_in_latex_format(
    #     f"{args.csv_save_dir}/{args.filesToEvaluateDir}_p_score_to_gt_latex.txt", p_score_to_gt)
    # with open(f"{args.csv_save_dir}/{args.filesToEvaluateDir}_p_score_to_gt.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(p_score_to_gt)

    plt.errorbar(list_of_errors,
                 list_of_vals_at_percentiles[:, 1],
                 list_of_vals_at_percentiles[:, [0, 2]].T,
                 capsize=5,
                 fmt="r--o",
                 ecolor="black",
                 label='15th, 50th, and 85th percentiles')
    plt.xlabel('Maximum GPS Error (m)', fontsize=15)
    plt.ylabel('Deviation from \nOriginal Predicted Position (m)', fontsize=15)
    plt.title('Change in Predicted Position\n with a Faulty GPS', fontsize=20)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid()
    plt.legend()
    plt.savefig('distances.png', bbox_inches='tight')


def main():
    # args = parseArgs()
    # files = glob.glob(f'{args.filesToEvaluateDir}/*')
    # files.sort()
    # file = files[1]
    # runAnalysisOnFile('GPSResults/run_not_filtering_cars_filtering_roads_not_filter_buildings_SuperGlue_not_LoFTR_Homography')
    runFullAnalysis()


if __name__ == "__main__":
    main()
