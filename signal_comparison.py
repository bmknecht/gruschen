import dbscan
import dynamic_time_warping
import metrics
import mfcc
import operator
import pre_processing
import sound_file
import statistics
import time


import numpy as np


save_preprocessed_files = False


class ProgressPrinter:
    def __init__(self, max_iterations):
        self.max_iterations = int(max_iterations)
        self.iterations = 0
        self.last_printed_percent = -1

    def iterate_and_print(self):
        self.iterations += 1
        percent = _percent(self.iterations, self.max_iterations)
        assert percent <= 100
        if percent > self.last_printed_percent:
            self.last_printed_percent = percent
            print('\r{} of {} ~ {} %'.format(self.iterations,
                                             self.max_iterations,
                                             percent),
                  end='')
        if self.iterations == self.max_iterations:
            # overwrite prev. message with whitespace
            print("\r... done" + (" "*30))


def _percent(part, total):
    assert part <= total
    return int(round(part / total * 100))


def compare_files(files):
    distances = _compute_file_distances(files)
    _print_comparison_with_correct_result(distances, files)
    # _dbscan_metrics_comparison(distances, files)
    return distances


def _compute_file_distances(files):
    characteristics = {}
    print("processing recordings:")
    progress = ProgressPrinter(len(files))
    for f in files:
        characteristics[f] = _compute_file_characteristics(f.filename)
        progress.iterate_and_print()
    distances = _metrics_of_characteristics(characteristics,
                                            files,
                                            dynamic_time_warping.norm_sqr)
    return distances


def _compute_file_characteristics(filename):
    signal = sound_file.load(filename)
    signal = pre_processing.process(signal)
    if save_preprocessed_files:
        sound_file.save(filename[:-4] + "_p.wav", signal)
    return mfcc.mfcc(signal)


def _metrics_of_characteristics(characteristics, files, metricFunction):
    characts_for_metric = {name: np.array([[_ for _ in f]
                                          for f in characteristics[name]])
                           for name in characteristics}

    distances = {f: {} for f in files}
    print("comparing files:")

    progress = ProgressPrinter((len(files)**2 - len(files)) / 2)
    for f1 in files:
        characteristics1 = characts_for_metric[f1]
        distances[f1][f1] = 0
        for f2 in files:
            if f2 not in distances[f1]:
                characteristics2 = characts_for_metric[f2]
                metric = metricFunction(characteristics1, characteristics2)
                distances[f1][f2] = metric
                distances[f2][f1] = metric
                progress.iterate_and_print()

    return distances


def _print_comparison_with_correct_result(distances, files):
    print(("/" * 10) + "\ndistances of recordings with same text spoken:\n")
    unique_texts = set([f.text for f in files])
    mean_correct_results = 0
    for text in unique_texts:
        mean = _print_comparison_single_text(text, unique_texts, distances,
                                             files)
        mean_correct_results += mean
    print("\nmean distance of same words: {}".format(mean_correct_results))
    # collect mean of all distances, excluding those of identical sounds
    mean_all_distances = statistics.mean([
        distances[name1][name2] for name1 in distances
        for name2 in distances[name1]
        if distances[name1][name2] > 1e-7
        ])
    print("mean of distances in general: {}".format(mean_all_distances))
    print("ratio: 1 to {:.3f}".format(mean_all_distances /
                                      mean_correct_results))


def _print_comparison_single_text(text, unique_texts, distances, files):
    print(text + ": ")
    files_with_text = list(filter(lambda f: f.text == text, files))

    distances_of_same_texts = [distances[f1][f2]
                               for f1 in files_with_text
                               for f2 in files_with_text if f1 != f2]
    mean = statistics.mean(distances_of_same_texts)
    print("mean: {}".format(mean))
    standard_deviation = statistics.stdev(distances_of_same_texts)
    print("standard deviation: {}".format(standard_deviation))
    print("median: {}".format(statistics.median(distances_of_same_texts)))
    print("minimum: {}".format(min(distances_of_same_texts)))
    print("maximum: {}".format(max(distances_of_same_texts)))
    return mean


# Density-based spatial clustering of applications with noise (DBSCAN)
def _dbscan_metrics_comparison(distances, files):
    user_choice = 1
    while(user_choice > 0):
        msg = ("Please give a minimum distance per " +
               "cluster for the DBSCAN algorithm: ")
        user_choice = int(input(msg))
        if user_choice > 0:
            _run_dbscan_clustering(distances, files, user_choice)


def _run_dbscan_clustering(distances, files, chosen_eps):
    clusters = dbscan.dbscan(distances, chosen_eps, 2)
    cluster_goodness = _analyze_dbscan_results(distances, files, clusters)
    _print_dbscan_clustering_goodness(cluster_goodness, len(distances))


def _print_dbscan_clustering_goodness(cluster_goodness, nodes_num):
    good_clusters, bad_clusters, total_clustered_count = cluster_goodness
    not_in_clusters = nodes_num - total_clustered_count
    print("good clusters: {}".format(good_clusters))
    print("bad clusters: {}".format(bad_clusters))
    print("not in clusters: {}".format(not_in_clusters))


def _analyze_dbscan_results(distances, files, clusters):
    good_clusters = 0
    bad_clusters = 0
    total_clustered_count = 0
    good_cluster_texts = set()
    for cluster in clusters:
        total_clustered_count += len(cluster.members)
        goodness_result = _goodness_of_cluster(cluster,
                                               good_clusters,
                                               bad_clusters)
        good_clusters = goodness_result[0]
        bad_clusters = goodness_result[1]
    return good_clusters, bad_clusters, total_clustered_count


def _goodness_of_cluster(cluster, good_clusters, bad_clusters):
    if len(cluster.members) > 1:
        goodness = _goodness_non_empty_cluster(cluster, good_clusters,
                                               bad_clusters)
        good_clusters = goodness[0]
        bad_clusters = goodness[1]
    else:
        bad_clusters += 1
    return good_clusters, bad_clusters


def _goodness_non_empty_cluster(cluster, good_clusters, bad_clusters):
    cluster_text = cluster.members.pop().text
    is_bad_cluster = False
    for member in cluster.members:
        if cluster_text != member.name:
            is_bad_cluster = True
            break
    if is_bad_cluster:
        bad_clusters += 1
    else:
        if cluster_text not in good_cluster_texts:
            good_clusters += 1
            good_cluster_texts.add(good_clusters)
        else:
            bad_clusters += 1
    return good_clusters, bad_clusters
