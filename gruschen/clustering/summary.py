
def summarize(distances, files, clusters):
    cluster_goodness = _analyze_results(distances, files, clusters)
    _print_clustering_goodness(cluster_goodness, len(distances))


def _print_clustering_goodness(cluster_goodness, nodes_num):
    good_clusters, bad_clusters, total_clustered_count = cluster_goodness
    not_in_clusters = nodes_num - total_clustered_count
    print("good clusters: {}".format(good_clusters))
    print("bad clusters: {}".format(bad_clusters))
    print("not in clusters: {}".format(not_in_clusters))


def _analyze_results(distances, files, clusters):
    good_clusters = 0
    bad_clusters = 0
    total_clustered_count = 0
    good_cluster_texts = set()
    for cluster in clusters:
        total_clustered_count += len(cluster.members)
        goodness_result = _goodness_of_cluster(cluster, good_clusters,
                                               bad_clusters, good_cluster_texts)
        good_clusters = goodness_result[0]
        bad_clusters = goodness_result[1]
    return good_clusters, bad_clusters, total_clustered_count


def _goodness_of_cluster(cluster, good_clusters, bad_clusters,
                         good_cluster_texts):
    if len(cluster.members) > 1:
        goodness = _goodness_non_empty_cluster(cluster, good_clusters,
                                               bad_clusters,
                                               good_cluster_texts)
        good_clusters = goodness[0]
        bad_clusters = goodness[1]
    else:
        bad_clusters += 1
    return good_clusters, bad_clusters


def _goodness_non_empty_cluster(cluster, good_clusters, bad_clusters,
                                good_cluster_texts):
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
