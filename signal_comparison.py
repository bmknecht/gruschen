import dbscan
import metrics
import mfcc
import operator
import pre_processing
import sound_file
import statistics
import time


import numpy as np


save_preprocessed_files = False


def compare_files(files):
    distances = _compute_file_distances(files)
    _analyze_metrics_of_correct_result(distances, files)
    _simple_metrics_comparison(distances, files)
    # _dbscan_metrics_comparison(distances, files)
    return distances


def _compute_file_distances(files):
    characteristics = {}
    print("processing recordings:")
    i = 0
    for f in files:
        percent = _percent(i, len(files))
        print('\r{} of {} ~ {} %'.format(i, len(files), percent), end='')
        characteristics[f] = _compute_file_characteristics(f.filename)
        i += 1
    print("\r... done" + (" "*30))  # overwrite prev. message with whitespace
    distances = _metrics_of_characteristics(characteristics,
                                            files,
                                            metrics.dynamic_time_warping_sqr)


def _percent(part, total):
    assert part <= total
    return round(part / total * 100)


def _compute_file_characteristics(filename):
    signal = sound_file.load(f.filename)
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
    percent = -1
    comparisons = 0

    for f1 in files:
        if percent != _percent(comparisons, (len(files)**2 - len(files)) / 2):
            percent = _percent(comparisons, (len(files)**2 - len(files)) / 2)
            print('\r{}%'.format(percent), end='')

        characteristics1 = characts_for_metric[f1]
        distances[f1][f1] = 0
        for f2 in files:
            if f2 not in distances[f1]:
                characteristics2 = characts_for_metric[f2]
                metric = metricFunction(characteristics1, characteristics2)
                distances[f1][f2] = metric
                distances[f2][f1] = metric
                comparisons += 1
    print("\r... done")
    return distances


def _analyze_metrics_of_correct_result(distances, files):
    print("/////////////////////////////\ncomparison of " +
          "expected with actual results:\n")
    unique_texts = set([f.text for f in files])
    mean_correct_results = 0
    for t in unique_texts:
        print(t + ": ")
        files_with_text = [f for f in files if f.text == t]
        distances_of_same_texts = [distances[f1][f2]
                                   for f1 in files_with_text
                                   for f2 in files_with_text if f1 != f2]
        mean = statistics.mean(distances_of_same_texts)
        print("mean: {}".format(mean))
        mean_correct_results += mean / len(unique_texts)
        standard_deviation = statistics.stdev(distances_of_same_texts)
        print("standard deviation: {}".format(standard_deviation))
        print("median: {}".format(statistics.median(distances_of_same_texts)))
        print("minimum: {}".format(min(distances_of_same_texts)))
        print("maximum: {}".format(max(distances_of_same_texts)))
    print("\nmean of same words: {}".format(mean_correct_results))
    # collect mean of all distances, excluding those of identical sounds
    mean_all_distances = statistics.mean([
        distances[name1][name2] for name1 in distances
        for name2 in distances[name1]
        if distances[name1][name2] > 1e-7
        ])
    print("mean of distances in general: {}".format(mean_all_distances))
    print("ratio: 1 to {:.3f}".format(mean_all_distances /
                                      mean_correct_results))


def _simple_metrics_comparison(distances, files):
    print("/////////////////////////////\nsimple metrics result:\n")
    assigned_correctly = 0
    doubted_correct_result = 0
    doubt_made_correct_result_worse = 0
    doubted_wrong_result = 0
    doubt_made_wrong_result_better = 0

    for f, dists in distances.items():
        sorted_dists = sorted(dists.items(), key=operator.itemgetter(1))
        most_similar_file, most_similar_metric = sorted_dists[1]    # skip self
        also_similar_file, also_similar_metric = sorted_dists[2]

        # statistics
        most_similar_text = most_similar_file.text
        also_similar_text = also_similar_file.text
        if f.text == most_similar_text:
            assigned_correctly += 1
            if _considered_similar_too(most_similar_metric,
                                       also_similar_metric):
                doubted_correct_result += 1
                if most_similar_text != also_similar_text:
                    doubt_made_correct_result_worse += 1
        else:
            if _considered_similar_too(most_similar_metric,
                                       also_similar_metric):
                doubted_wrong_result += 1
                if f.text == also_similar_text:
                    doubt_made_wrong_result_better += 1

    print("\n\nsummary:")
    print("{} ~ {}% correctly associated".format(assigned_correctly,
                                                 _percent(assigned_correctly,
                                                          len(distances))))
    if assigned_correctly > 0:
        doubted_correct_percent = _percent(doubted_correct_result,
                                           assigned_correctly)
        print("{} ~ {}% of which were doubted".format(doubted_correct_result,
                                                      doubted_correct_percent))
        if doubted_correct_result > 0:
            print("{} ~ {}% of which were made worse".format(
                doubt_made_correct_result_worse,
                _percent(doubt_made_correct_result_worse,
                         doubted_correct_result)))
    print("{} ~ {}% of wrong associations doubted".format(
        doubted_wrong_result,
        _percent(doubted_wrong_result,
                 len(distances) - assigned_correctly)))
    if doubted_wrong_result > 0:
        print("{} ~ {}% of which were made better".format(
            doubt_made_wrong_result_better,
            _percent(doubt_made_wrong_result_better, doubted_wrong_result)))


def _considered_similar_too(metric1, metric2):
    similarity_distance = abs(metric1 - metric2)
    denominator = metric1 if metric1 < metric2 else metric2
    return similarity_distance / denominator < 0.1


# Density-based spatial clustering of applications with noise (DBSCAN)
def _dbscan_metrics_comparison(distances, files):
    user_choice = 1
    while(user_choice > 0):
        msg = ("Please give a minimum distance per " +
               "cluster for the DBSCAN algorithm: ")
        user_choice = int(input(msg))
        if user_choice > 0:
            clusters = dbscan.dbscan(distances, user_choice, 2)

            good_clusters = 0
            bad_clusters = 0
            total_clustered_count = 0
            good_cluster_texts = set()
            for cluster in clusters:
                total_clustered_count += len(cluster.members)
                if len(cluster.members) > 1:
                    cluster_text = cluster.members.pop()
                    is_bad_cluster = False
                    for member_name in cluster.members:
                        if cluster_text != member_name:
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
                else:
                    bad_clusters += 1
            not_in_clusters = len(distances) - total_clustered_count
            print("good clusters: {}".format(good_clusters))
            print("bad clusters: {}".format(bad_clusters))
            print("not in clusters: {}".format(not_in_clusters))
