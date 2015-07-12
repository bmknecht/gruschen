import dbscan
import mfcc
import operator
import pre_processing
import sound_file


def compare_files(filenames, save_preprocessed_files):
    class Sim:
        def __init__(self, name1, name2):
            self.name1 = name1
            self.name2 = name2

    characteristics = {}
    for filename in filenames:
        print("processing {}".format(filename))
        signal = sound_file.load(filename)
        signal = pre_processing.process(signal)
        if save_preprocessed_files:
            sound_file.save(filename[:-4] + "_p.wav", signal)
        characteristics[filename] = mfcc.mfcc(signal)
    metrics = _metrics_of_characteristics(characteristics, filenames)
    _simple_metrics_comparison(metrics)
    _dbscan_metrics_comparison(metrics)
    return metrics


def _metrics_of_characteristics(characteristics, filenames):
    metrics = {filename: {} for filename in filenames}
    for i, name1 in enumerate(filenames):
        characteristics1 = characteristics[name1]
        for j, name2 in enumerate(filenames):
            if i < j:
                characteristics2 = characteristics[name2]
                metric = mfcc.dynamic_time_warping_metric(characteristics1,
                                                          characteristics2)
                metrics[name1][name2] = metric
                metrics[name2][name1] = metric
    return metrics


def _simple_metrics_comparison(metrics):
    print("/////////////////////////////\nsimple metrics result:\n")
    for name1, dists in metrics.items():
        sorted_dists = sorted(dists.items(), key=operator.itemgetter(1))
        most_similar = sorted_dists[0]
        print("{}: most likely to be the same:".format(name1))
        print("{} - {}".format(most_similar[0], most_similar[1]))
        also_similar = sorted_dists[1]
        similarity_distance = also_similar[1] - most_similar[1]
        if similarity_distance / most_similar[1] < 0.1:
            print("\t also similar: {} - {}".format(also_similar[0],
                                                    also_similar[1]))
        print("\n")
    print("\n")


# Density-based spatial clustering of applications with noise (DBSCAN)
def _dbscan_metrics_comparison(metrics):
    dbscan.dbscan(metrics, 1000, 2)
