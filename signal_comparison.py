import dbscan
import mfcc
import operator
import pre_processing
import sound_file
import statistics


def compare_files(files, save_preprocessed_files):
    characteristics = {}
    print("processing recordings:")
    i = 0
    for f in files:
        print('\r{} of {} ~ {}%'.format(i,
                                        len(files),
                                        _percent(i, len(files))),
              end='')
        signal = sound_file.load(files[f].filename)
        signal = pre_processing.process(signal)
        if save_preprocessed_files:
            sound_file.save(files[f].filename[:-4] + "_p.wav", signal)
        characteristics[f] = mfcc.mfcc(signal)
        i += 1
    print("\r... done" + (" "*30))  # overwrite prev. message with whitespace
    metrics = _metrics_of_characteristics(characteristics, files)
    _analyze_metrics_of_correct_result(metrics, files)
    _simple_metrics_comparison(metrics, files)
    # _dbscan_metrics_comparison(metrics)
    return metrics


def _percent(part, total):
    assert part <= total
    return round(part / total * 100)


def _metrics_of_characteristics(characteristics, files):
    metrics = {f: {} for f in files}
    print("comparing files:")
    percent = -1
    comparisons = 0

    for f1 in files:
        if percent != _percent(comparisons, len(files)**2):
            percent = _percent(comparisons, len(files)**2)
            print('\r{}%'.format(percent), end='')

        characteristics1 = characteristics[f1]
        metrics[f1][f1] = 0
        for f2 in files:
            if f2 != f1:
                characteristics2 = characteristics[f2]
                metric = mfcc.dynamic_time_warping_metric_sqr(characteristics1,
                                                              characteristics2)
                metrics[f1][f2] = metric
                metrics[f2][f1] = metric
            comparisons += 1
    print("\r... done")
    return metrics


def _analyze_metrics_of_correct_result(metrics, files):
    print("/////////////////////////////\ncomparison of " +
          "expected with actual results:\n")
    unique_texts = set([files[f].text for f in files])
    for t in unique_texts:
        print(t + ": ")
        files_with_text = [files[f] for f in files if files[f].text == t]
        metrics_with_text = [metrics[str(f1)][str(f2)]
                             for f1 in files_with_text
                             for f2 in files if str(f1) != str(f2)]
        print("mean: {}".format(statistics.mean(metrics_with_text)))
        standard_deviation = statistics.stdev(metrics_with_text)
        print("standard deviation: {}".format(standard_deviation))
        print("median: {}".format(statistics.median(metrics_with_text)))
        print("minimum: {}".format(min(metrics_with_text)))
        print("maximum: {}".format(max(metrics_with_text)))


def _simple_metrics_comparison(metrics, files):
    print("/////////////////////////////\nsimple metrics result:\n")
    assigned_correctly = 0
    doubted_correct_result = 0
    doubt_made_correct_result_worse = 0
    doubted_wrong_result = 0
    doubt_made_wrong_result_better = 0

    for f, dists in metrics.items():
        sorted_dists = sorted(dists.items(), key=operator.itemgetter(1))
        most_similar_file, most_similar_metric = sorted_dists[1]    # skip self
        # uncomment next few lines for extended output
        # print("{}: most likely to be the same:".format(str(f)))
        # print("{} - {}".format(str(most_similar_file), most_similar_metric))
        also_similar_file, also_similar_metric = sorted_dists[2]
        # if _considered_similar_too(most_similar_metric, also_similar_metric):
        #    print("\t also similar: {} - {}".format(str(also_similar_file),
        #                                            also_similar_metric))
        # print("\n")

        # statistics
        most_similar_text = files[most_similar_file].text
        also_similar_text = files[also_similar_file].text
        if files[f].text == most_similar_text:
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
                if files[f].text == also_similar_text:
                    doubt_made_wrong_result_better += 1

    print("\n\nsummary:")
    print("{} ~ {}% correctly associated".format(assigned_correctly,
                                                 _percent(assigned_correctly,
                                                          len(metrics))))
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
                 len(metrics) - assigned_correctly)))
    if doubted_wrong_result > 0:
        print("{} ~ {}% of which were made better".format(
            doubt_made_wrong_result_better,
            _percent(doubt_made_wrong_result_better, doubted_wrong_result)))


def _considered_similar_too(metric1, metric2):
    similarity_distance = abs(metric1 - metric2)
    denominator = metric1 if metric1 < metric2 else metric2
    return similarity_distance / denominator < 0.1


# Density-based spatial clustering of applications with noise (DBSCAN)
def _dbscan_metrics_comparison(metrics):
    user_choice = 1
    while(user_choice > 0):
        msg = ("Please give a minimum distance per " +
               "cluster for the DBSCAN algorithm: ")
        user_choice = int(input(msg))
        if user_choice > 0:
            dbscan.dbscan(metrics, user_choice, 2)
