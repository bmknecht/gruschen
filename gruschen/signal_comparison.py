import json
import statistics

from . import (
    clustering,
    mfcc,
    sound_file,
)

from .metrics import dynamic_time_warping as dtw
from .metrics import dtw_delta
from .metrics import linear_stretch as linear_stretch
from .clustering import dbscan
from .preprocessing import preprocessing as prepro
from .progress_printer import ProgressPrinter


save_preprocessed_files = False


def compare_files(files):
    signals = _apply_method_to_all({f: f.filename for f in files},
                                   sound_file.load,
                                   'loading files')
    signals = _apply_method_to_all(signals,
                                   prepro.process,
                                   'preprocessing signals')
    characteristics = _apply_method_to_all(signals,
                                           mfcc.framed_power_spectrum,
                                           'computing characteristics')
    distances = _compute_signals_distances(characteristics,
                                           dtw)
    with open('distances.dict', 'w') as f:
        json.dump({d1.name: {d2.name: distances[d1][d2] for d2 in distances}
                   for d1 in distances}, f)
    _summarize_distances(distances)
    _clustering(distances, files, dbscan)
    return distances


def _apply_method_to_all(vec, method, message):
    print(message)
    progress = ProgressPrinter(len(vec))

    def loop_iteration(x):
        progress.iterate_and_print()
        return x

    return {v: loop_iteration(method(vec[v])) for v in vec}


def _compute_signals_distances(characteristics, metric):
    print("computing distances:")
    progress = ProgressPrinter((len(characteristics)**2 -
                                len(characteristics)) / 2)
    distances = {f: {f: 0} for f in characteristics}
    for f1 in characteristics:
        for f2 in characteristics:
            if f2 not in distances[f1]:
                m = metric.get_metric(characteristics[f1], characteristics[f2])
                distances[f1][f2] = distances[f2][f1] = m
                progress.iterate_and_print()
    return distances


def _summarize_distances(distances):
    print(("/" * 10) + "\ndistances of recordings with same text spoken:\n")
    unique_texts = set([d.text for d in distances])
    for text in unique_texts:
        _print_comparison_single_text(text, distances)
    correct_result_distances = [
        distances[d1][d2] for d1 in distances for d2 in distances[d1]
        if d1.text == d2.text and d1 != d2
    ]
    all_distances = [
        distances[d1][d2] for d1 in distances for d2 in distances[d1] if
        d1 != d2
    ]
    median_correct_results = statistics.median(correct_result_distances)
    print("\nmedian distance of same words: {}".format(median_correct_results))
    median_all_distances = statistics.median(all_distances)
    print("median of distances in general: {}".format(median_all_distances))
    print("ratio: 1 to {:.3f}".format(median_all_distances /
                                      median_correct_results))

    mean_correct_results = statistics.mean(correct_result_distances)
    print("\nmean distance of same words: {}".format(mean_correct_results))
    mean_all_distances = statistics.mean(all_distances)
    print("mean of distances in general: {}".format(mean_all_distances))
    print("ratio: 1 to {:.3f}".format(mean_all_distances /
                                      mean_correct_results))


def _print_comparison_single_text(text, distances):
    print(text + ": ")
    distances_of_same_texts = [distances[d1][d2]
                               for d1 in distances
                               for d2 in distances if d1 != d2 and
                               d1.text == d2.text]
    print("mean: {}".format(statistics.mean(distances_of_same_texts)))
    print("standard deviation: {}".format(statistics.stdev(
        distances_of_same_texts
    )))
    print("median: {}".format(statistics.median(distances_of_same_texts)))
    print("minimum: {}".format(min(distances_of_same_texts)))
    print("maximum: {}".format(max(distances_of_same_texts)))


def _clustering(distances, files, clustering_method):
    user_choice = 1
    while(user_choice > 0):
        msg = ("Please give a minimum distance per " +
               "cluster for the clustering algorithm: ")
        user_choice = int(input(msg))
        if user_choice > 0:
            clusters = clustering_method(distances, user_choice, 2)
            clustering.summary.summarize(distances, files, clusters)
