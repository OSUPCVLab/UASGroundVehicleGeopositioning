import argparse
import glob
import pickle
from matplotlib import pyplot as plt
import numpy as np
from tabulate import tabulate
import csv
import copy
from scipy.stats import ttest_ind
from sklearn.cluster import DBSCAN
from sklearn import decomposition

debug = False
car_major_axis = 4.48


def parseArgs():
    parser = argparse.ArgumentParser(
        description='Determine how well the method performs')
    parser.add_argument('--filesToEvaluateDir', type=str, default='larger_set_testGPSResults',
                        help='where to get predicted car locations files from')
    parser.add_argument('--csv_save_dir', type=str, default='csv_results',
                        help='where to put csv results')
    # parser.add_argument('--filesToEvaluateDir', type=str, default='gps_errors_GPS_results',
    #                     help='where to get predicted car locations files from')
    # parser.add_argument('--csv_save_dir', type=str, default='csv_results_faulty_gps',
    #                     help='where to put csv results')

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
                    pos = getYXpos(
                        mean_lat_long, setsOfFrames[setNum][track_id][i])
                    filteredSetsOfFrames[setNum][track_id].append(pos)
                    setsOfFrames[setNum][track_id][i] = pos
            elif numPositions == 1:
                pos = getYXpos(
                    mean_lat_long, setsOfFrames[setNum][track_id][0])
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
                    plt.scatter(positions[:, 1],
                                positions[:, 0], s=3, c=colors[idx])
                else:
                    positions = np.asarray(positions)
                    plt.scatter(positions[:, 1],
                                positions[:, 0], s=3, c=colors[idx])
                idx += 1
        plt.savefig('originalPoints.png', bbox_inches='tight')
        plt.show()

        for setNum in filteredSetsOfFrames:
            for track_id in filteredSetsOfFrames[setNum]:
                positions = filteredSetsOfFrames[setNum][track_id]

                if len(positions) > 1:
                    positions = np.asarray(positions)
                    plt.scatter(positions[:, 1],
                                positions[:, 0], s=3, c=colors[idx])
                idx += 1
        plt.savefig('outliers.png', bbox_inches='tight')
        plt.show()

    return filteredSetsOfFrames


def clean_with_PCA(X, make_circular=True, zero_first_component=False):
    # apply PCA to the data, uncorrelate it
    pca = decomposition.PCA(2).fit(X)
    uncorrelated_data = pca.transform(X)

    assert pca.explained_variance_ratio_[0] >= pca.explained_variance_ratio_[
        1], "PCA component 1 has a lower variance than component 2"

    # scale the data if needed
    if make_circular:
        min_minor = uncorrelated_data[:, 1].min()
        max_minor = uncorrelated_data[:, 1].max()
        min_major = uncorrelated_data[:, 0].min()
        max_major = uncorrelated_data[:, 0].max()
        r_minor = (np.abs(max_minor) + np.abs(min_minor))/2
        r_major = (np.abs(max_major) + np.abs(min_major))/2
        uncorrelated_data[:, 0] = uncorrelated_data[:, 0] * r_minor / r_major
    # zero the first component if needed
    if zero_first_component:
        uncorrelated_data[:, 0] = np.zeros_like(uncorrelated_data[:, 0])

    data_placed_back = pca.inverse_transform(uncorrelated_data)

    return data_placed_back


def get_biggest_cluster_of_points(positions: np.ndarray):
    # return the best cluster center, # of points removed, total number of points, number of clusters

    # run dbscan
    db = DBSCAN(
        eps=2*car_major_axis).fit(positions)
    labels = db.labels_

    # remove outlier label
    labels_clean = labels[labels != -1]

    # find the biggest cluster label, use it to find the points that correspond to the biggest cluster label
    best_cluster_points = None
    num_points_in_biggest_cluster = 0
    values, counts = np.unique(labels_clean, return_counts=True)
    if counts.size > 0:
        biggest_cluster_label = values[np.argmax(counts)]
        best_cluster_points = positions[labels == biggest_cluster_label, :]
        num_points_in_biggest_cluster = best_cluster_points.shape[0]

    # find other metrics to report
    total_num_of_points = positions.shape[0]
    num_points_removed = total_num_of_points - num_points_in_biggest_cluster
    num_clusters = values.shape[0]

    return best_cluster_points, num_points_removed, total_num_of_points, num_clusters


def runAnalysisOnFile(fileName):
    setsOfFrames = readInCachedData(fileName)

    normalizedPoints = normalizeDataAndFilterOutliers(setsOfFrames)

    frameAvgDist = []
    agg_points_removed = 0
    agg_points_considered = 0
    agg_num_clusters = []
    num_cars = 0
    total_cars_located = 0
    for setNum in normalizedPoints:
        for track_id in normalizedPoints[setNum]:
            positions = np.asarray(normalizedPoints[setNum][track_id])

            if positions.shape[0] > 1:
                best_cluster_points, num_points_removed, total_num_of_points, num_clusters_found = get_biggest_cluster_of_points(
                    positions)

                agg_points_removed += num_points_removed
                agg_points_considered += total_num_of_points
                agg_num_clusters.append(num_clusters_found)
                num_cars += 1

                if best_cluster_points is not None:
                    total_cars_located += 1

                    best_cluster_points -= best_cluster_points.mean(
                        axis=0)
                    best_cluster_points_cleaned = clean_with_PCA(
                        best_cluster_points)

                    for position in best_cluster_points_cleaned:
                        dist = np.linalg.norm(position)
                        frameAvgDist.append(dist)

            else:
                print('ERROR')
                exit()
    # plt.hist(frameAvgDist)
    # plt.show()
    return np.asarray(frameAvgDist), agg_points_removed, agg_points_considered, np.asarray(agg_num_clusters), num_cars, total_cars_located


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
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c ")
    f.write("\n>{\columncolor[HTML]{FFFFFF}}c @{}}")
    f.write("\n\\toprule")
    f.write("\n\multicolumn{3}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{Filtering Performed}} & \multicolumn{6}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{Feature Extraction Method}}                                                                                                                         \\\\ \midrule")
    f.write(
        "\n\\textbf{Road} & \\textbf{Building} & \\textbf{Vehicle} & \multicolumn{2}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{LoFTR}} & \multicolumn{2}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{SuperGlue}} & \multicolumn{2}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{LoFTR + SuperGlue}} \\\\ \midrule")
    for i in reversed(range(8)):
        f.write("\n")
        table_row = table[i+1]
        for j in range(3):
            if table_row[j] == 'True':
                f.write("\\textbf{\checkmark} & ")
            else:
                f.write("\\textbf{} & ")
        for j in range(5):
            f.write(str(table_row[j+3]))
            f.write(" & ")
        f.write(str(table_row[8]))
        f.write("\\\\")
    f.write("\n\cmidrule(l){4-9}")
    f.write("\n\multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}} & \multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}} & \multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}} & \\textbf{Homography}          & \\textbf{2D Affine}          & \\textbf{Homography}              & \\textbf{2D Affine}          & \\textbf{Homography}                 & \\textbf{2D Affine}                 \\\\ \cmidrule(l){4-9} ")
    f.write("\n\multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}} & \multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}} & \multicolumn{1}{l}{\cellcolor[HTML]{FFFFFF}} & \multicolumn{6}{c}{\cellcolor[HTML]{FFFFFF}\\textbf{Transformation}} \\\\ \cmidrule(l){4-9}")
    f.write("\n\end{tabular}")
    f.write("\n\caption{TODO CAPTION}")
    f.write("\n\label{TODO LABEL}")
    f.write("\n\end{table*}")
    f.close()


def file_name_to_dict(file: str):
    rep = {}

    rep['cars'] = not "not_filtering_cars" in file
    rep['roads'] = not "not_filtering_roads" in file
    rep['buildings'] = not "not_filter_buildings" in file
    rep['SG'] = not "not_SuperGlue" in file
    rep['LF'] = not "not_LoFTR" in file
    rep['homography'] = "Homography" in file

    return rep


def file_dict_to_name(file_dict: dict):
    name = ""

    for key in file_dict:
        name += key + '_' + str(file_dict[key])

    return name


def should_compare(dist_dict_1, dist_dict_2):
    # score each change, if there is only 1 change then we should compare
    # if a parameter is unchanged, it is a 0, otherwise it is a 1
    # if the sum of changes is 1, then we should compare, otherwise we should not

    cars_point = 0 if dist_dict_1['cars'] == dist_dict_2['cars'] else 1
    roads_point = 0 if dist_dict_1['roads'] == dist_dict_2['roads'] else 1
    buildings_point = 0 if dist_dict_1['buildings'] == dist_dict_2['buildings'] else 1

    SG_changed = not (dist_dict_1['SG'] == dist_dict_2['SG'])
    LF_changed = not (dist_dict_1['LF'] == dist_dict_2['LF'])
    kp_point = 1 if SG_changed or LF_changed else 0

    homography_point = 0 if dist_dict_1['homography'] == dist_dict_2['homography'] else 1

    return 1 == (cars_point + roads_point + buildings_point + kp_point + homography_point)


def what_is_different(dist_dict_1, dist_dict_2):
    # score each change, if there is only 1 change then we should compare
    # if a parameter is unchanged, it is a 0, otherwise it is a 1
    # if the sum of changes is 1, then we should compare, otherwise we should not
    different = ""

    different += 'cars' if dist_dict_1['cars'] != dist_dict_2['cars'] else ''
    different += 'roads' if dist_dict_1['roads'] != dist_dict_2['roads'] else ''
    different += 'buildings' if dist_dict_1['buildings'] != dist_dict_2['buildings'] else ''
    different += 'homography' if dist_dict_1['homography'] != dist_dict_2['homography'] else ''
    different += 'SG' if dist_dict_1['SG'] != dist_dict_2['SG'] else ''
    different += 'LF' if dist_dict_1['LF'] != dist_dict_2['LF'] else ''

    return different


def runFullAnalysis():
    args = parseArgs()

    files = glob.glob(f'{args.filesToEvaluateDir}/*')
    files.sort()

    baseFormat = [["Road", "Building", "Car", "LoFTR", "LoFTR", "SuperGlue", "SuperGlue", "LoFTR + SuperGlue", "LoFTR + SuperGlue"],
                  ['True', 'True', 'True', None, None, None, None, None, None],
                  ['True', 'True', 'False', None, None, None, None, None, None],
                  ['True', 'False', 'True', None, None, None, None, None, None],
                  ['True', 'False', 'False', None, None, None, None, None, None],
                  ['False', 'True', 'True', None, None, None, None, None, None],
                  ['False', 'True', 'False', None, None, None, None, None, None],
                  ['False', 'False', 'True', None, None, None, None, None, None],
                  ['False', 'False', 'False', None, None, None, None, None, None],
                  [None, None, None, 'homography', 'affine2D',
                   'homography', 'affine2D', 'homography', 'affine2D']
                  ]

    mean_dist_to_cluster = copy.deepcopy(baseFormat)
    per_outliers_removed = copy.deepcopy(baseFormat)
    num_clusters = copy.deepcopy(baseFormat)
    per_cars_cluster_found = copy.deepcopy(baseFormat)

    distributions = []
    distribution_dicts = []

    for file in files:
        file_row = getRow(file) + 1
        file_col = getColumn(file)+3
        # print('running analysis on ', file)
        frameAvgDist, agg_points_removed, agg_points_considered, agg_num_clusters, num_cars, total_cars_located = runAnalysisOnFile(
            file)

        distributions.append(frameAvgDist)
        distribution_dict = file_name_to_dict(file)
        distribution_dicts.append(distribution_dict)

        per_outliers_removed[file_row][file_col] = agg_points_removed / \
            agg_points_considered
        per_cars_cluster_found[file_row][file_col] = total_cars_located / num_cars
        mean_dist_to_cluster[file_row][file_col] = '{:.3f} ± {:.3f}'.format(
            frameAvgDist.mean(), np.std(frameAvgDist))
        num_clusters[file_row][file_col] = '{:.3f} ± {:.3f}'.format(
            agg_num_clusters.mean(), np.std(agg_num_clusters))

    # print('\nNumber of outliers removed:')
    # print(tabulate(per_outliers_removed,headers='firstrow'))
    print('\nMean Distance from Cluster:')
    print(tabulate(mean_dist_to_cluster, headers='firstrow'))
    print('\nPercent outliers removed:')
    print(tabulate(per_outliers_removed, headers='firstrow'))
    print('Tolerance: ', car_major_axis)

    print_table_in_latex_format(
        f"{args.csv_save_dir}/{args.filesToEvaluateDir}_per_outliers_removed_latex.txt", per_outliers_removed)
    with open(f"{args.csv_save_dir}/{args.filesToEvaluateDir}_per_outliers_removed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(per_outliers_removed)

    print_table_in_latex_format(
        f"{args.csv_save_dir}/{args.filesToEvaluateDir}_mean_dist_to_cluster_center_latex.txt", mean_dist_to_cluster)
    with open(f"{args.csv_save_dir}/{args.filesToEvaluateDir}_mean_dist_to_cluster_center.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(mean_dist_to_cluster)

    print_table_in_latex_format(
        f"{args.csv_save_dir}/{args.filesToEvaluateDir}_per_cars_cluster_found_latex.txt", per_cars_cluster_found)
    with open(f"{args.csv_save_dir}/{args.filesToEvaluateDir}_per_cars_cluster_found.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(per_cars_cluster_found)

    print_table_in_latex_format(
        f"{args.csv_save_dir}/{args.filesToEvaluateDir}_num_clusters_latex.txt", num_clusters)
    with open(f"{args.csv_save_dir}/{args.filesToEvaluateDir}_num_clusters.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(num_clusters)

    l = len(distributions)

    difference_list = []
    no_significant_difference_list = []

    for i in range(l):
        for j in range(i, l):
            dist_dict_1 = distribution_dicts[j]
            dist_dict_2 = distribution_dicts[i]
            if should_compare(dist_dict_1, dist_dict_2):
                t_score = ttest_ind(distributions[i], distributions[j])
                # mann_whitney_score = mannwhitneyu(
                #     distributions[i], distributions[j])

                difference = what_is_different(dist_dict_1, dist_dict_2)
                difference_list.append(difference)

                # if t_score.pvalue > 0.01 or mann_whitney_score.pvalue > 0.01:
                if t_score.pvalue > 0.01:
                    # if mann_whitney_score.pvalue > 0.01:
                    no_significant_difference_list.append(difference)

    print('How many comparisions are statistically significant:')
    for key in distribution_dicts[0].keys():
        no_sig_count = no_significant_difference_list.count(key)
        total_count = difference_list.count(key)
        print(key, ' has ', total_count - no_sig_count, '/',
              total_count, '=', '{:.3f} '.format(1 - (no_sig_count/total_count)))


def main():
    # args = parseArgs()
    # files = glob.glob(f'{args.filesToEvaluateDir}/*')
    # files.sort()
    # file = files[1]
    # runAnalysisOnFile('GPSResults/run_not_filtering_cars_filtering_roads_not_filter_buildings_SuperGlue_not_LoFTR_Homography')
    runFullAnalysis()


if __name__ == "__main__":
    main()
