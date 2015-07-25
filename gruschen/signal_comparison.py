import json

import pandas
from progress.bar import Bar as ProgressBar

from .features import (
    mfcc
)
from .metrics import (
    dynamic_time_warping as dtw,
    dtw_delta,
    linear_stretch as linear_stretch
)
from . import sound_file
from .preprocessing import preprocessing as prepro


save_preprocessed_files = False


def compare_files(files):
    signals, samplerates = _load_files({f: f.filename for f in files},
                                       sound_file.load)
    signals = _preprocess_signals(signals, samplerates,
                                  prepro.process)
    features = _compute_signal_features(signals, samplerates,
                                        mfcc.extract_features)
    distances = _compute_signals_distances(features,
                                           linear_stretch)
    with open('distances.dict', 'w') as f:
        json.dump({d1.name: {d2.name: distances[d1][d2] for d2 in distances}
                   for d1 in distances}, f)
    _summarize_distances(distances)
    # _clustering(distances, files, dbscan)
    return distances


def _loop_iteration_progress(progressbar, x):
        progressbar.next()
        return x


def _load_files(filenames, method):
    progressbar = ProgressBar('loading files', max=len(filenames))
    return_values = {v: _loop_iteration_progress(progressbar,
                                                 method(filenames[v]))
                     for v in filenames}
    signals = {name: return_values[name][0] for name in return_values}
    samplerates = {name: return_values[name][1] for name in return_values}
    return signals, samplerates


def _preprocess_signals(signals, samplerates, method):
    progressbar = ProgressBar('preprocessing', max=len(signals))
    return {v: _loop_iteration_progress(progressbar,
                                        method(signals[v], samplerates[v]))
            for v in signals}


def _compute_signal_features(signals, samplerates, method):
    progressbar = ProgressBar('computing features', max=len(signals))
    return {v: _loop_iteration_progress(progressbar,
                                        method(signals[v], samplerates[v]))
            for v in signals}


def _compute_signals_distances(features, metric):
    progressbar = ProgressBar('computing distances',
                              max=(len(features)**2 - len(features)) / 2)
    distances = {f: {f: 0} for f in features}
    for f1 in features:
        for f2 in features:
            if f2 not in distances[f1]:
                m = metric.get_metric(features[f1], features[f2])
                distances[f1][f2] = distances[f2][f1] = m
                progressbar.next()
    return distances


def _summarize_distances(distances):
    print("\ndistances of recordings with same text spoken:")
    unique_texts = set([d.text for d in distances])
    for text in unique_texts:
        _print_comparison_single_text(text, distances)
    all_distances = [
        distances[d1][d2] for d1 in distances for d2 in distances[d1] if
        d1 != d2
    ]
    print("\ndistances in general:")
    print(pandas.Series(all_distances).describe())

    correct_result_distances = [
        distances[d1][d2] for d1 in distances for d2 in distances[d1]
        if d1.text == d2.text and d1 != d2
    ]
    print("\ndistances of same words:")
    print(pandas.Series(correct_result_distances).describe())


def _print_comparison_single_text(text, distances):
    print("\n" + text + ": ")
    distances_of_same_texts = [distances[d1][d2]
                               for d1 in distances
                               for d2 in distances if d1 != d2 and
                               d1.text == d2.text]
    print(pandas.Series(distances_of_same_texts).describe())


def _clustering(distances, files, clustering_method):
    user_choice = 1
    while(user_choice > 0):
        msg = ("Please give a minimum distance per " +
               "cluster for the clustering algorithm: ")
        user_choice = int(input(msg))
        if user_choice > 0:
            clusters = clustering_method(distances, user_choice, 2)
            clustering.summary.summarize(distances, files, clusters)
